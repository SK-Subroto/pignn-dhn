"""Registry of learned approaches: one self-contained package each.

    dhn_gnn/approaches/<name>/
        config.py   hyperparameters (dataclasses, no torch)
        model.py    the nn.Module
        train.py    the fitting loop

Everything shared stays outside: the pressure-flow operator in dhn_gnn/physics/,
solver machinery in dhn_gnn/solvers/, sample construction in dhn_gnn/datasets.py,
checkpoints in dhn_gnn/checkpoints.py. `newton` is deliberately NOT an approach -- it is
the pure-physics control condition and has nothing to configure or fit.

Model and trainer are resolved LAZILY (by import string) so that listing the
approaches, or parsing a config file, never pays for importing torch.
"""

from dataclasses import dataclass
from importlib import import_module

from dhn_gnn.approaches.initializer.hparams import InitializerConfig
from dhn_gnn.approaches.unrolled.hparams import UnrolledConfig


@dataclass(frozen=True)
class Approach:
    name: str
    config_cls: type
    summary: str
    _model_ref: str        # "module:attribute"
    _fit_ref: str

    @staticmethod
    def _resolve(ref):
        module, attr = ref.split(":")
        return getattr(import_module(module), attr)

    def model_cls(self):
        """The nn.Module class. Imports torch on first call."""
        return self._resolve(self._model_ref)

    def fit_fn(self):
        """(model, samples, cfg, monitor) -> (model, history, extra_meta)."""
        return self._resolve(self._fit_ref)


APPROACHES = {
    "initializer": Approach(
        name="initializer",
        config_cls=InitializerConfig,
        summary="GNN once + Newton polish (current)",
        _model_ref="dhn_gnn.approaches.initializer.model:DHNInitializerSolver",
        _fit_ref="dhn_gnn.approaches.initializer.train:fit",
    ),
    "unrolled": Approach(
        name="unrolled",
        config_cls=UnrolledConfig,
        summary="GNN every step (original)",
        _model_ref="dhn_gnn.approaches.unrolled.model:DHNUnrolledSolver",
        _fit_ref="dhn_gnn.approaches.unrolled.train:fit",
    ),
}

ARCH_CHOICES = tuple(APPROACHES)


def get(name: str) -> Approach:
    if name not in APPROACHES:
        raise KeyError(f"unknown approach {name!r}; choose from {sorted(APPROACHES)}")
    return APPROACHES[name]


__all__ = ["Approach", "APPROACHES", "ARCH_CHOICES", "get",
           "InitializerConfig", "UnrolledConfig"]
