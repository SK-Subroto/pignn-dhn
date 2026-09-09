"""Head-to-head benchmark of the solver variants against PyDHN.

Compares, on the same held-out timesteps and the same convergence tolerance:

  1. unrolled + diagonal Newton   the original architecture (GNN every step)
  2. unrolled + full Newton       same, but with the exact loop Jacobian
  3. cold  + full Newton          no learned guess at all (c0 = 0)
  4. warm  + full Newton          c0 = previous timestep's solution (PyDHN's trick)
  5. learned initializer + full Newton    c0 from one GNN pass  <- proposed
  6. pure predictor, NO Newton            the same GNN, polish removed (ablation)

Variants 3-5 are the SAME class (DHNInitializerSolver) differing only in where c0
comes from, which is exactly the comparison worth making: all three then run the
identical parameter-free Newton polish, so any difference is attributable to the
starting point and nothing else.

Variant 6 is the control in the other direction: it isolates what the learned
component achieves alone, with no exact steps behind it to absorb a mediocre guess.

PyDHN's own cost is read from the dataset's history.json. Note the asymmetry that
number hides: PyDHN reaches it by warm-starting, so it must walk the timesteps IN
ORDER. Variants 3, 5 and 6 have no such dependency and could be solved for the
whole year at once; variant 4 deliberately reintroduces it, to price what the
history is worth.

Usage:
    python -m dhn_gnn.benchmark --n-ts 149
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dhn_gnn import config
from dhn_gnn.physics import operators as netops
from dhn_gnn.approaches.initializer.model import DHNInitializerSolver
from dhn_gnn.solvers.newton import NewtonSolver
from dhn_gnn.approaches.predictor.model import DHNPredictorSolver
from dhn_gnn.approaches.unrolled.model import DHNUnrolledSolver


def _steps_to_tol(curve, tol):
    """How many steps until max|residual| first drops below tol (inf if never)."""
    for i, v in enumerate(curve):
        if v < tol:
            return i + 1
    return np.inf


def bench_unrolled(ops, samples, tol, newton_mode, damping, K):
    kw = dict(config.MODEL_KWARGS, K=K, newton_mode=newton_mode, newton_damping=damping)
    torch.manual_seed(0)
    m = DHNUnrolledSolver(ops, **kw).float()
    m.eval()
    steps, finals, t0 = [], [], time.time()
    with torch.no_grad():
        for m0, _, _ in samples:
            _, terms, _ = m(m0, tol=tol)
            curve = [t.abs().max().item() for t in terms]
            steps.append(_steps_to_tol(curve, tol)); finals.append(curve[-1])
    return dict(steps=np.array(steps, float), final=np.array(finals),
                ms=(time.time() - t0) / len(samples) * 1e3)


def bench_initializer(ops, samples, tol, mode, ckpt=None, n_newton=20):
    """mode: 'cold' (c0=0) | 'warm' (c0 = previous solution) | 'learned'."""
    torch.manual_seed(0)
    if mode == "learned":
        from dhn_gnn.checkpoints import load_checkpoint
        m, _ = load_checkpoint(ops, ckpt, DHNInitializerSolver)
        m = m.float()
    else:
        # the explicit pure-physics solver: no learned parameters at all, so the
        # cold/warm variants cannot accidentally benefit from a trained network
        m = NewtonSolver(ops, n_newton=config.INIT_MODEL_KWARGS["n_newton"]).float()
    m.eval()

    steps, finals, guess_res = [], [], []
    prev_c = None
    t0 = time.time()
    with torch.no_grad():
        for m0, _, _ in samples:
            if mode == "warm":
                c_init = prev_c if prev_c is not None else torch.zeros(m.n_free)
            elif mode == "cold":
                c_init = torch.zeros(m.n_free)
            else:
                c_init = None                      # use the learned guess
            _, terms, c_hist = m(m0, tol=tol, n_newton=n_newton, c_init=c_init)
            curve = [t.abs().max().item() for t in terms]
            guess_res.append(curve[0])
            # curve[0] is the guess BEFORE any Newton step, so subtract it to count
            # Newton steps rather than evaluations
            steps.append(_steps_to_tol(curve, tol) - 1)
            finals.append(curve[-1])
            prev_c = c_hist[-1]
    return dict(steps=np.array(steps, float), final=np.array(finals),
                guess=np.array(guess_res), ms=(time.time() - t0) / len(samples) * 1e3)


def bench_predictor(ops, samples, tol, ckpt):
    """
    The no-Newton ablation: one GNN pass and nothing else.

    Reported as ZERO solver steps, because that is literally what it spends -- the
    prediction is the answer. The interesting column for this row is therefore not
    the step count but whether it clears the tolerance at all: a variant that never
    does scores inf here, which is the result the ablation is looking for.
    """
    from dhn_gnn.checkpoints import load_checkpoint
    torch.manual_seed(0)
    m, _ = load_checkpoint(ops, ckpt, DHNPredictorSolver)
    m = m.float()
    m.eval()

    steps, finals, guess_res, t0 = [], [], [], time.time()
    with torch.no_grad():
        for m0, _, _ in samples:
            _, terms, _ = m(m0)                     # exactly one entry
            r = terms[-1].abs().max().item()
            guess_res.append(r)
            finals.append(r)
            steps.append(0.0 if r < tol else np.inf)
    return dict(steps=np.array(steps, float), final=np.array(finals),
                guess=np.array(guess_res), ms=(time.time() - t0) / len(samples) * 1e3)


def _standalone_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-ts", type=int, default=149, help="held-out timesteps to use")
    ap.add_argument("--tol", type=float, default=config.EVAL_TOL_PA)
    ap.add_argument("--split-every", type=int, default=5)
    return ap.parse_args()


def main(args=None):
    args = args if args is not None else _standalone_args()

    from dhn_gnn.datasets import make_samples
    ops = netops.build_operators()
    n_ts = len(pd.read_csv(config.MASS_FLOW_CSV, index_col=0))
    _, test_ts = config.split_timesteps(n_ts, args.split_every)
    test_ts = test_ts[:args.n_ts]
    samples = make_samples(ops, test_ts)
    print(f"benchmarking {len(samples)} held-out timesteps at tol={args.tol} Pa\n")

    run = getattr(args, "run", "default")
    # Keyed as well as ordered: the commentary below looks variants up by name, so
    # adding or skipping one cannot silently shift what those lines describe.
    rows, by_key = [], {}

    def add(key, name, result):
        rows.append((name, result))
        by_key[key] = result

    add("unrolled_diag", "1. unrolled + diagonal Newton (original)",
        bench_unrolled(ops, samples, args.tol, "diagonal", 0.5, 20))
    add("unrolled_full", "2. unrolled + full Newton",
        bench_unrolled(ops, samples, args.tol, "full", 1.0, 20))
    add("cold", "3. cold start + full Newton (no GNN)",
        bench_initializer(ops, samples, args.tol, "cold"))
    add("warm", "4. warm start + full Newton (PyDHN's trick)",
        bench_initializer(ops, samples, args.tol, "warm"))

    init_ckpt = config.resolve_checkpoint("initializer", run)
    if init_ckpt.exists():
        add("initializer", "5. learned initializer + full Newton",
            bench_initializer(ops, samples, args.tol, "learned", ckpt=init_ckpt))
    else:
        print(f"(skipping variant 5: no checkpoint at {init_ckpt})\n")

    pred_ckpt = config.resolve_checkpoint("predictor", run)
    if pred_ckpt.exists():
        add("predictor", "6. pure predictor, NO Newton (ablation)",
            bench_predictor(ops, samples, args.tol, pred_ckpt))
    else:
        print(f"(skipping variant 6: no checkpoint at {pred_ckpt})\n")

    hist = json.loads(Path(config.GEN_DATA_DIR / "history.json").read_text())
    pydhn_it = np.array(hist["hydraulics iterations"])

    hdr = f"{'variant':<44}{'steps>tol':>11}{'median':>8}{'final Pa':>12}{'ms/ts':>9}{'parallel':>10}"
    lines = ["=== Solver variants vs PyDHN ===", "",
             f"tolerance: {args.tol} Pa   timesteps: {len(samples)} (held out)", "", hdr,
             "-" * len(hdr)]
    for name, r in rows:
        ok = np.isfinite(r["steps"])
        mean_s = r["steps"][ok].mean() if ok.any() else np.inf
        med_s = np.median(r["steps"][ok]) if ok.any() else np.inf
        par = "no" if name.startswith("4.") else "yes"
        lines.append(f"{name:<44}{mean_s:>11.2f}{med_s:>8.0f}"
                     f"{np.median(r['final']):>12.2e}{r['ms']:>9.0f}{par:>10}")
    lines += [
        "-" * len(hdr),
        f"{'PyDHN (reference, warm-started)':<44}{pydhn_it.mean():>11.2f}"
        f"{np.median(pydhn_it):>8.0f}{'~46':>12}{'n/a':>9}{'no':>10}",
        "",
        "'steps>tol' = solver steps to first reach the tolerance (Newton steps only",
        "for variants 3-6; the learned guess itself is not counted as a step, so",
        "variant 6 -- which takes no Newton step at all -- scores 0 when its",
        "prediction already clears the tolerance and inf when it never does.",
        "'parallel'  = can timesteps be solved independently? Variant 4 and PyDHN",
        "chain on the previous solution, so they cannot.",
    ]

    if "initializer" in by_key and "cold" in by_key:
        g = by_key["initializer"]["guess"]; c = by_key["cold"]["guess"]
        lines += ["",
                  f"initial-guess quality (residual BEFORE any Newton step):",
                  f"  cold start   : {np.median(c):.3e} Pa",
                  f"  learned guess: {np.median(g):.3e} Pa  "
                  f"({np.median(c)/max(np.median(g),1e-12):.1f}x better)"]

    if "predictor" in by_key:
        p = by_key["predictor"]["final"]
        cleared = 100.0 * np.mean(p < args.tol)
        lines += ["",
                  "-- Is Newton necessary? (variant 6) --",
                  f"  prediction alone : {np.median(p):.3e} Pa median residual",
                  f"  clears {args.tol:g} Pa    : {cleared:.1f}% of timesteps",
                  f"  gap to tolerance : {np.median(p)/args.tol:.1f}x"]
        if "initializer" in by_key:
            lines.append(
                f"  the same network WITH the Newton polish reaches "
                f"{np.median(by_key['initializer']['final']):.3e} Pa in "
                f"{np.nanmean(np.where(np.isfinite(by_key['initializer']['steps']), by_key['initializer']['steps'], np.nan)):.2f} steps.")

    txt = "\n".join(lines)
    print(txt)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "benchmark.txt").write_text(txt, encoding="utf-8")
    print(f"\nsaved {config.RESULTS_DIR / 'benchmark.txt'}")


if __name__ == "__main__":
    main()
