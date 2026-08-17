"""Single entry point for every approach and every report.

    python -m dhn_gnn.cli train     --arch initializer      fit a model
    python -m dhn_gnn.cli evaluate  --arch initializer      score it, write CSVs
    python -m dhn_gnn.cli benchmark                         all approaches vs PyDHN
    python -m dhn_gnn.cli compare                           predictions vs reference

Three approaches are directly comparable, and every report names them the same way:

    newton       pure physics, NO learning        <- the control condition
    unrolled     GNN every step (original)
    initializer  GNN once + Newton polish         <- current
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dhn_gnn import config
from dhn_gnn.training import (load_checkpoint, make_samples, ops_cache,
                              save_checkpoint, timestep_split)
from dhn_gnn.training.initializer import train_initializer
from dhn_gnn.training.unrolled import mean_final_residual, train


def cmd_train(args):
    torch.manual_seed(args.seed)
    dev = config.get_device(args.device)
    ops = ops_cache()
    n_ts, train_ts, test_ts = timestep_split(args.split_every, args.n_train)
    print(f"timesteps: {n_ts} total | train {len(train_ts)} (used) | test {len(test_ts)}")

    if args.arch == "initializer":
        from dhn_gnn.model.initializer import DHNInitializerSolver
        kwargs = dict(config.INIT_MODEL_KWARGS)
        model = DHNInitializerSolver(ops, **kwargs).float().to(dev)
        samples = make_samples(ops, train_ts, device=dev)
        print(f"arch: initializer  n_newton={kwargs['n_newton']}  epochs={args.epochs}  "
              f"lr={args.lr}  device={dev}")
        zero_err = float(np.mean([float(a.abs().max()) for _, a, _ in samples]))
        print(f"zero-guess |c0-a*|max (no initializer): {zero_err:.4f} kg/s")
        monitor = samples[::max(1, len(samples) // 20)]
        _, hist = train_initializer(model, samples, epochs=args.epochs, lr=args.lr,
                                    log_every=max(1, args.epochs // 10), monitor=monitor)
        out = args.out or config.CHECKPOINT_INIT
        extra = dict(zero_guess_err=zero_err)
    else:
        from dhn_gnn.model.unrolled_solver import DHNUnrolledSolver
        kwargs = dict(config.MODEL_KWARGS, step_scale=args.step_scale)
        model = DHNUnrolledSolver(ops, **kwargs).float().to(dev)
        samples = make_samples(ops, train_ts, device=dev)
        print(f"arch: unrolled  K={kwargs['K']}  step_scale={args.step_scale}  "
              f"epochs={args.epochs}  lr={args.lr}  device={dev}")
        r0, _ = mean_final_residual(model, samples)
        print(f"untrained (zero-init head = pure Newton) mean residual: {r0:.1f} Pa")
        monitor = samples[::max(1, len(samples) // 20)]
        _, hist = train(model, samples, epochs=args.epochs, lr=args.lr,
                        log_every=max(1, args.epochs // 10), monitor=monitor)
        out = args.out or config.CHECKPOINT
        extra = dict(untrained_mean_residual=float(r0), step_scale=args.step_scale)

    save_checkpoint(model, kwargs, out,
                    meta=dict(arch=args.arch, split_every=args.split_every,
                              n_train=len(train_ts), epochs=args.epochs, lr=args.lr,
                              seed=args.seed, train_ts=train_ts.tolist(),
                              **extra, **hist))


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

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--split-every", type=int, default=5,
                        help="every Nth timestep is held out for test")
    common.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu",
                        help="cpu by default: at batch size 1 this model is too "
                             "small for a GPU to help (see config.get_device)")

    t = sub.add_parser("train", parents=[common], help="fit a model")
    t.add_argument("--arch", choices=["initializer", "unrolled"], default="initializer")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--n-train", type=int, default=200,
                   help="subsample this many training timesteps (0 = all)")
    t.add_argument("--step-scale", type=float, default=0.01,
                   help="unrolled only: bound on the learned per-step correction")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, default=None,
                   help="checkpoint path; point smoke runs elsewhere so a short "
                        "test cannot overwrite a real trained model")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", parents=[common], help="score a checkpoint, write CSVs")
    e.add_argument("--arch", choices=["initializer", "unrolled"], default="initializer")
    e.add_argument("--ckpt", type=Path, default=None)
    e.add_argument("--tol", type=float, default=config.EVAL_TOL_PA)
    e.add_argument("--max-test", type=int, default=0)
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("benchmark", parents=[common], help="all approaches vs PyDHN")
    b.add_argument("--n-ts", type=int, default=149)
    b.add_argument("--tol", type=float, default=config.EVAL_TOL_PA)
    b.set_defaults(func=cmd_benchmark)

    c = sub.add_parser("compare", help="predicted CSVs vs the PyDHN reference")
    c.set_defaults(func=cmd_compare)
    return ap


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
