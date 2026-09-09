"""Single entry point for every approach and every report.

    python -m dhn_gnn.cli train     --arch initializer      fit a model
    python -m dhn_gnn.cli evaluate  --arch initializer      score it, write CSVs
    python -m dhn_gnn.cli benchmark                         all approaches vs PyDHN
    python -m dhn_gnn.cli compare   --arch initializer      predictions vs reference
    python -m dhn_gnn.cli config    --arch unrolled         print the resolved recipe

Three approaches are directly comparable, and every report names them the same way:

    newton       pure physics, NO learning        <- the control condition
    unrolled     GNN every step (original)
    initializer  GNN once + Newton polish         <- current

HYPERPARAMETERS
    Every knob is declared in the approach's own config dataclass
    (dhn_gnn/approaches/<arch>/hparams.py) and resolved in this order:

        dataclass defaults  ->  --config file.yaml  ->  named flags  ->  --set k=v

    Set anything without touching source:

        ... train --arch initializer --set model.d_model=128 --set lr=1e-4
        ... train --arch unrolled    --set K=25 --set model.newton_mode=diagonal

    A bare key works when it is unambiguous (--set K=25); qualify it otherwise.
    An unknown key is an error, never a silent fallback to the default.

RUNS
    --run NAME puts everything for one experiment in its own directory:

        results/<arch>/<run>/  config.yaml  model.pt  metrics.txt  pred-*.csv

    so two approaches -- or two hyperparameter settings of one approach -- can no
    longer overwrite each other's scores. The resolved config.yaml is written
    beside the weights, so a run can always be reproduced from its own output.
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dhn_gnn import approaches, config
from dhn_gnn.approaches import hyperparams
from dhn_gnn.checkpoints import save_checkpoint
from dhn_gnn.datasets import make_samples, ops_cache, timestep_split

# Named convenience flags and the config path each one drives. They exist because
# --lr 1e-4 reads better than --set train.lr=1e-4 for the knobs that change on
# nearly every run; everything else goes through --set.
_FLAG_TO_KEY = {
    "epochs": "train.epochs",
    "lr": "train.lr",
    "seed": "train.seed",
    "n_train": "train.n_train",
    "split_every": "train.split_every",
    "device": "train.device",
    # unrolled-only, kept because the docs and generated guides name them
    "k": "model.K",
    "step_scale": "model.step_scale",
    "newton_mode": "model.newton_mode",
}


def _explicitly_set(args, key):
    """True if --set touched this key, so a coupling rule must not override it."""
    return any(item.split("=", 1)[0].strip().split(".")[-1] == key
               for item in (getattr(args, "set", None) or []))


def resolve_config(args):
    """Build the run recipe: defaults -> YAML -> named flags -> --set."""
    cfg = approaches.get(args.arch).config_cls()
    if getattr(args, "config", None):
        cfg = hyperparams.load_yaml(type(cfg), args.config)

    # None means "not passed", which is how an argparse default is kept from
    # silently overwriting a value the YAML file set on purpose.
    flags = [f"{key}={getattr(args, flag)}"
             for flag, key in _FLAG_TO_KEY.items()
             if getattr(args, flag, None) is not None]
    try:
        hyperparams.apply_overrides(cfg, flags)
        hyperparams.apply_overrides(cfg, getattr(args, "set", None))
    except ValueError as exc:
        raise SystemExit(f"config error: {exc}")

    # newton_mode and newton_damping are coupled: the original approach pairs
    # diagonal Newton with 0.5 damping, while full Newton converges quadratically
    # and needs none -- carrying 0.5 over to full would halve every step. Skipped
    # when the damping was set by hand, which is the point of setting it by hand.
    model = cfg.model
    if hasattr(model, "newton_damping") and not _explicitly_set(args, "newton_damping"):
        model.newton_damping = 0.5 if model.newton_mode == "diagonal" else 1.0
    return cfg


def cmd_train(args):
    approach = approaches.get(args.arch)
    cfg = resolve_config(args)

    torch.manual_seed(cfg.train.seed)
    dev = config.get_device(cfg.train.device)
    ops = ops_cache()
    n_ts, train_ts, test_ts = timestep_split(cfg.train.split_every, cfg.train.n_train)

    print(f"arch: {args.arch} ({approach.summary})   run: {args.run}   device: {dev}")
    print(f"timesteps: {n_ts} total | train {len(train_ts)} (used) | test {len(test_ts)}")
    print(f"config: {hyperparams.summarize(cfg)}")

    model_kwargs = dataclasses.asdict(cfg.model)
    model = approach.model_cls()(ops, **model_kwargs).float().to(dev)
    samples = make_samples(ops, train_ts, device=dev)
    monitor = samples[::max(1, len(samples) // 20)]

    _t_fit = time.time()
    model, hist, extra = approach.fit_fn()(model, samples, cfg, monitor=monitor)
    train_seconds = time.time() - _t_fit

    ckpt = args.out or config.checkpoint_path(args.arch, args.run)
    save_checkpoint(model, model_kwargs, ckpt,
                    meta=dict(arch=args.arch, run=args.run,
                              split_every=cfg.train.split_every,
                              n_train=len(train_ts), epochs=cfg.train.epochs,
                              lr=cfg.train.lr, seed=cfg.train.seed,
                              train_ts=train_ts.tolist(),
                              train_seconds=float(train_seconds),
                              n_params=int(sum(p.numel() for p in model.parameters())),
                              config=hyperparams.to_dict(cfg), **extra, **hist))
    # The recipe is written only after the run survives, so a half-finished
    # directory never looks like a completed experiment.
    written = hyperparams.save_yaml(cfg, config.config_path(args.arch, args.run))
    print(f"saved recipe     -> {written}")


def cmd_config(args):
    """Print the resolved recipe without training. Cheap way to check a sweep."""
    cfg = resolve_config(args)
    print(yaml.safe_dump(hyperparams.to_dict(cfg), sort_keys=False, default_flow_style=False),
          end="")
    if args.save:
        print(f"saved -> {hyperparams.save_yaml(cfg, args.save)}")


def cmd_evaluate(args):
    from dhn_gnn.reporting import evaluate
    evaluate.main(args)


def cmd_benchmark(args):
    from dhn_gnn.reporting import benchmark
    benchmark.main(args)


def cmd_compare(args):
    from dhn_gnn.reporting import compare
    compare.main(args)


def build_parser():
    ap = argparse.ArgumentParser(prog="dhn_gnn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- shared: which approach, which run -------------------------------------
    run_opts = argparse.ArgumentParser(add_help=False)
    run_opts.add_argument("--arch", choices=approaches.ARCH_CHOICES,
                          default="initializer")
    run_opts.add_argument("--run", default="default",
                          help="run name; outputs go to results/<arch>/<run>/ "
                               "(default: %(default)s)")

    # --- shared: how the hyperparameters are set -------------------------------
    hp = argparse.ArgumentParser(add_help=False)
    hp.add_argument("--config", type=Path, default=None,
                    help="YAML recipe to start from; --set still applies on top")
    hp.add_argument("--set", action="append", metavar="KEY=VALUE", default=[],
                    help="override one hyperparameter, repeatable "
                         "(e.g. --set model.d_model=128 --set lr=1e-4)")
    # default=None throughout: it is how "not passed" stays distinguishable from
    # "passed the default", which is what lets a YAML file survive these flags.
    hp.add_argument("--epochs", type=int, default=None)
    hp.add_argument("--lr", type=float, default=None)
    hp.add_argument("--seed", type=int, default=None)
    hp.add_argument("--n-train", dest="n_train", type=int, default=None,
                    help="subsample this many training timesteps (0 = all)")
    hp.add_argument("--split-every", dest="split_every", type=int, default=None,
                    help="every Nth timestep is held out for test")
    hp.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                    help="cpu by default: at batch size 1 this model is too "
                         "small for a GPU to help (see config.get_device)")
    hp.add_argument("--k", type=int, default=None,
                    help="unrolled only: number of unrolled steps")
    hp.add_argument("--step-scale", dest="step_scale", type=float, default=None,
                    help="unrolled only: bound on the learned per-step correction")
    hp.add_argument("--newton-mode", dest="newton_mode",
                    choices=["full", "diagonal"], default=None,
                    help="unrolled only: 'diagonal' reproduces the ORIGINAL approach")

    t = sub.add_parser("train", parents=[run_opts, hp], help="fit a model")
    t.add_argument("--out", type=Path, default=None,
                   help="override the checkpoint path; point smoke runs elsewhere "
                        "so a short test cannot overwrite a real trained model")
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("config", parents=[run_opts, hp],
                       help="print the resolved recipe without training")
    c.add_argument("--save", type=Path, default=None, help="also write it here")
    c.set_defaults(func=cmd_config)

    e = sub.add_parser("evaluate", parents=[run_opts],
                       help="score a checkpoint, write CSVs")
    e.add_argument("--ckpt", type=Path, default=None)
    e.add_argument("--tol", type=float, default=config.EVAL_TOL_PA)
    e.add_argument("--max-test", type=int, default=0)
    e.add_argument("--split-every", dest="split_every", type=int, default=5)
    e.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("benchmark", help="all approaches vs PyDHN")
    b.add_argument("--n-ts", type=int, default=149)
    b.add_argument("--tol", type=float, default=config.EVAL_TOL_PA)
    b.add_argument("--split-every", dest="split_every", type=int, default=5)
    b.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    b.add_argument("--run", default="default",
                   help="which run of each approach to load (default: %(default)s)")
    b.set_defaults(func=cmd_benchmark)

    cp = sub.add_parser("compare", parents=[run_opts],
                        help="predicted CSVs vs the PyDHN reference")
    cp.set_defaults(func=cmd_compare)
    return ap


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
