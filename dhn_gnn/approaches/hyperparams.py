"""Shared hyperparameter machinery: one YAML file == one run recipe.

Every approach declares its knobs as dataclasses, grouped into `model` (what the
constructor needs, and therefore what is baked into the parameter shapes) and
`train` (optimizer/schedule, which is not). The split is not cosmetic: the model
group is what `save_checkpoint` has to store for a checkpoint to be rebuildable,
while the train group only describes how it got there.

Resolution order, lowest priority first:

    dataclass defaults  ->  --config file.yaml  ->  --set key=value

The resolved config is written next to the weights as `config.yaml`, so a run is
always reproducible from its own output directory, and a thesis table can cite
the file rather than a half-remembered command line.
"""

import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import get_type_hints

import yaml


@dataclass
class TrainConfig:
    """Optimizer and schedule knobs shared by every approach."""
    epochs: int = 40
    lr: float = 1e-3
    clip: float = 1.0
    seed: int = 0
    n_train: int = 200        # subsample this many training timesteps (0 = all)
    split_every: int = 5      # every Nth timestep is held out for test
    device: str = "cpu"       # cpu default: see config.get_device for why


# --- (de)serialization ---------------------------------------------------------

def to_dict(cfg) -> dict:
    """Nested plain-dict view of a config, suitable for YAML or JSON."""
    return dataclasses.asdict(cfg)


def from_dict(cls, data: dict):
    """Rebuild a config dataclass from a nested dict, rejecting unknown keys.

    Unknown keys are an ERROR rather than a warning on purpose: a typo'd knob that
    is silently dropped trains the default and reports it as the requested run,
    which is the most expensive kind of quiet failure in a sweep.
    """
    if not is_dataclass(cls):
        return data
    known = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown key(s) {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    kwargs = {}
    for name, value in data.items():
        ftype = hints.get(name, known[name].type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = from_dict(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_yaml(cls, path: Path):
    """Read a run recipe. An empty file resolves to the defaults."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return from_dict(cls, data)


def save_yaml(cfg, path: Path) -> Path:
    """Write the fully resolved recipe beside whatever it produced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(to_dict(cfg), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


# --- command-line overrides ----------------------------------------------------

def _coerce(raw: str, ftype):
    """Cast a command-line string to the field's declared type."""
    if ftype is bool:
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"cannot read {raw!r} as a boolean")
    if ftype is int:
        return int(raw)
    if ftype is float:
        return float(raw)
    return raw


def _paths(cfg, prefix=()):
    """Every (dotted-path, owner, field-name, type) leaf in a nested config."""
    hints = get_type_hints(type(cfg))
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype):
            yield from _paths(value, prefix + (f.name,))
        else:
            yield ".".join(prefix + (f.name,)), cfg, f.name, ftype


def apply_overrides(cfg, assignments):
    """
    Apply `key=value` strings in place, e.g. "model.d_model=128" or "lr=1e-4".

    A bare key is accepted when it is unambiguous across groups, because
    `--set lr=1e-4` reads better than `--set train.lr=1e-4` and there is only one
    `lr`. When a bare name IS ambiguous the error names the qualified candidates
    rather than picking one.
    """
    leaves = list(_paths(cfg))
    for item in assignments or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()

        exact = [l for l in leaves if l[0] == key]
        if not exact:
            exact = [l for l in leaves if l[0].split(".")[-1] == key]
        if not exact:
            raise ValueError(
                f"unknown hyperparameter {key!r}. Available:\n  "
                + "\n  ".join(sorted(l[0] for l in leaves))
            )
        if len(exact) > 1:
            raise ValueError(
                f"{key!r} is ambiguous; qualify it as one of "
                f"{sorted(l[0] for l in exact)}"
            )
        _, owner, name, ftype = exact[0]
        setattr(owner, name, _coerce(raw, ftype))
    return cfg


def resolve(cls, config_file=None, overrides=None):
    """Defaults -> YAML file -> --set, in that order. The one entry point."""
    cfg = load_yaml(cls, config_file) if config_file else cls()
    return apply_overrides(cfg, overrides)


def summarize(cfg) -> str:
    """One-line 'k=v k=v' rendering, for the training banner."""
    return "  ".join(f"{p}={getattr(o, n)!r}" for p, o, n, _ in _paths(cfg))
