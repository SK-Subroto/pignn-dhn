"""Run every approach on identical timesteps and produce the report assets.

Outputs, all under results/:

    predictions/<approach>/{mass_flow,delta_p,pressure}.csv   per-approach predictions
    figures/*.png                                             charts
    report_data.json                                          every number in the writeup

The things compared:

    pydhn        the reference simulator (its own solved CSVs)
    newton       pure physics, cold start, NO learning
    unrolled     GNN every step (the original approach)
    initializer  GNN once, then Newton (the current approach)
    predictor    GNN once, NO Newton (the ablation) -- scored only when a
                 checkpoint for it exists, since it is optional

Everything is scored on the SAME held-out timesteps, with the SAME tolerance, on
the SAME machine, so the only differences are the ones being studied.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dhn_gnn import config
from dhn_gnn.physics import operators as netops
from dhn_gnn.physics import darcy
from dhn_gnn.physics.pressure import PressureReconstructor
from dhn_gnn.approaches.initializer.model import DHNInitializerSolver
from dhn_gnn.approaches.predictor.model import DHNPredictorSolver
from dhn_gnn.solvers.newton import NewtonSolver
from dhn_gnn.approaches.unrolled.model import DHNUnrolledSolver
from dhn_gnn.datasets import make_samples

RES = config.RESULTS_DIR
FIG = RES / "figures"
PRED = RES / "predictions"

# PyDHN's cost is the wall-clock to generate the year, supplied by the data owner.
# It covers the FULL simulation (setup, controls, thermal, IO), not just the
# hydraulic solve, so it is an upper bound on the comparable work -- flagged
# wherever it is used rather than quietly presented as like-for-like.
PYDHN_HOURS_PER_YEAR = 11.0
PYDHN_N_TIMESTEPS = 8760

PALETTE = {
    "pydhn": "#6b7280",
    "newton": "#0f766e",
    "unrolled": "#b45309",
    "initializer": "#1d4ed8",
    "predictor": "#7c3aed",
}
LABEL = {
    "pydhn": "PyDHN (reference)",
    "newton": "Newton (pure physics)",
    "unrolled": "Unrolled GNN (original)",
    "initializer": "Learned initializer (ours)",
    "predictor": "Pure predictor (no Newton)",
}


# --------------------------------------------------------------------------- run
def steps_to_tol(curve, tol, records_start):
    """
    Number of SOLVER STEPS until max|residual| first drops below tol.

    The two families index their residual list differently, and getting this wrong
    silently shifts the headline metric by one step in opposite directions:

      Newton / initializer  record the starting point, so curve[i] is the state
                            after i steps -> steps = i, and steps = 0 is a real
                            outcome (the learned guess already cleared tolerance).
      unrolled              records only post-step states, so curve[i] is the state
                            after i+1 steps -> steps = i + 1.
    """
    bump = 0 if records_start else 1
    for i, v in enumerate(curve):
        if v < tol:
            return i + bump
    return len(curve) - 1 + bump      # never converged: all steps taken


def run_approach(name, model, ops, samples, timesteps, rec, p_true, tol, records_start):
    flows, dps, press, steps, finals, curves, guesses = [], [], [], [], [], [], []
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for m0, _, _ in samples:
            mdot, terms, _ = model(m0, tol=tol)
            curve = [t.abs().max().item() for t in terms]
            flows.append(mdot.numpy())
            steps.append(steps_to_tol(curve, tol, records_start))
            finals.append(curve[-1])
            curves.append(curve)
            guesses.append(curve[0])
    elapsed_ms = (time.time() - t0) / len(samples) * 1e3

    # dp / pressure computed outside the timed loop: they are reporting artifacts,
    # not part of the solve, and including them would flatter nothing consistently
    with torch.no_grad():
        for i, (m0, _, _) in enumerate(samples):
            mdot = torch.as_tensor(flows[i])
            dp = model.pipe_dp(mdot)
            anchors = torch.as_tensor(p_true.iloc[timesteps[i]].to_numpy(np.float64)[rec.anchors])
            dps.append(dp.numpy())
            press.append(rec(dp.to(torch.float64), anchors).numpy())

    return dict(name=name, flow=np.stack(flows), dp=np.stack(dps),
                pressure=np.stack(press), steps=np.array(steps, float),
                final=np.array(finals), guess=np.array(guesses),
                curves=curves, ms=elapsed_ms, records_start=records_start)


def score(r, ops, timesteps, flow_t, dp_t, p_t):
    pipe = ops.pipe_mask.numpy()
    pf, tf = r["flow"].ravel(), flow_t.iloc[timesteps].to_numpy(np.float64).ravel()
    err = pf - tf
    db = 1e-2
    dirs = np.mean((np.sign(pf) * (np.abs(pf) > db)) == (np.sign(tf) * (np.abs(tf) > db)))
    dpp = r["dp"][:, pipe].ravel()
    dpt = dp_t.iloc[timesteps].to_numpy(np.float64)[:, pipe].ravel()
    pp, pt = r["pressure"].ravel(), p_t.iloc[timesteps].to_numpy(np.float64).ravel()
    return dict(
        flow_mae=float(np.mean(np.abs(err))), flow_rmse=float(np.sqrt(np.mean(err**2))),
        flow_max=float(np.abs(err).max()),
        flow_r2=float(1 - np.sum(err**2) / np.sum((tf - tf.mean())**2)),
        dir_acc=float(dirs * 100),
        dp_mae=float(np.mean(np.abs(dpp - dpt))),
        press_mae=float(np.mean(np.abs(pp - pt))),
        steps_mean=float(r["steps"].mean()), steps_median=float(np.median(r["steps"])),
        steps_max=float(r["steps"].max()),
        resid_mean=float(r["final"].mean()), resid_median=float(np.median(r["final"])),
        pct_under_tol=float(100 * np.mean(r["final"] <= config.EVAL_TOL_PA)),
        ms_per_ts=float(r["ms"]),
        guess_resid_median=float(np.median(r["guess"])),
    )


# ----------------------------------------------------------------------- figures
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=.25, linewidth=.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_convergence(runs, tol):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for k, r in runs.items():
        L = max(len(c) for c in r["curves"])
        pad = np.array([c + [c[-1]] * (L - len(c)) for c in r["curves"]])
        # same convention as steps_to_tol: x is the number of steps TAKEN, so a
        # family that does not log its starting point begins at 1, not 0
        x = np.arange(pad.shape[1]) + (0 if r["records_start"] else 1)
        ax.plot(x, np.median(pad, 0), "-o", ms=3.5, color=PALETTE[k], label=LABEL[k])
        ax.fill_between(x, np.percentile(pad, 25, 0), np.percentile(pad, 75, 0),
                        color=PALETTE[k], alpha=.13)
    ax.axhline(tol, ls="--", lw=1.2, color="#dc2626", label=f"{tol:g} Pa tolerance")
    ax.set_yscale("log")
    _style(ax, "Convergence: loop residual per solver step", "solver step",
           "max |residual|  (Pa)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "convergence.png", dpi=160); plt.close(fig)


def fig_steps(runs, pydhn_steps):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    keys = list(runs)
    width = .8 / (len(keys) + (1 if pydhn_steps is not None else 0))
    allv = np.arange(0, max(int(r["steps"].max()) for r in runs.values()) + 2)
    for i, k in enumerate(keys):
        c = np.array([np.sum(runs[k]["steps"] == v) for v in allv]) / len(runs[k]["steps"]) * 100
        ax.bar(allv + i * width - .4, c, width, color=PALETTE[k], label=LABEL[k])
    _style(ax, "How many solver steps each approach needs",
           "steps to reach tolerance", "% of timesteps")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "steps_distribution.png", dpi=160); plt.close(fig)


def fig_parity(runs, flow_t, timesteps):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    tf = flow_t.iloc[timesteps].to_numpy(np.float64).ravel()
    lim = np.abs(tf).max() * 1.05
    for ax, (k, r) in zip(np.atleast_1d(axes), runs.items()):
        ax.hexbin(tf, r["flow"].ravel(), gridsize=55, bins="log",
                  cmap="Blues", mincnt=1, linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        err = r["flow"].ravel() - tf
        r2 = 1 - np.sum(err**2) / np.sum((tf - tf.mean())**2)
        _style(ax, f"{LABEL[k]}\nR2 = {r2:.6f}", "PyDHN flow (kg/s)", "predicted (kg/s)")
    fig.tight_layout(); fig.savefig(FIG / "flow_parity.png", dpi=160); plt.close(fig)


def fig_timing(scores, pydhn_ms):
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    keys = list(scores)
    vals = [scores[k]["ms_per_ts"] for k in keys]
    ax.barh([LABEL[k] for k in keys], vals, color=[PALETTE[k] for k in keys], height=.6)
    for i, v in enumerate(vals):
        ax.text(v * 1.05, i, f"{v:.0f} ms", va="center", fontsize=9)
    ax.axvline(pydhn_ms, ls="--", color=PALETTE["pydhn"], lw=1.4)
    ax.text(pydhn_ms * .95, -.75, f"PyDHN ~ {pydhn_ms:,.0f} ms\n(full simulation)",
            ha="right", fontsize=8, color=PALETTE["pydhn"])
    ax.set_xscale("log")
    _style(ax, "Wall-clock per timestep (log scale, CPU)", "milliseconds", "")
    fig.tight_layout(); fig.savefig(FIG / "timing.png", dpi=160); plt.close(fig)


def fig_seasonal(runs, timesteps, index):
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    months = pd.to_datetime(index[timesteps]).month
    for k, r in runs.items():
        by = [r["steps"][months == m].mean() for m in range(1, 13)]
        ax.plot(range(1, 13), by, "-o", ms=4, color=PALETTE[k], label=LABEL[k])
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=8)
    _style(ax, "Solver effort through the year (does summer break it?)",
           "", "mean steps to tolerance")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "seasonal.png", dpi=160); plt.close(fig)


def fig_guess_quality(runs):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    keys = [k for k in runs if k in ("newton", "initializer")]
    data = [np.log10(np.maximum(runs[k]["guess"], 1e-3)) for k in keys]
    parts = ax.violinplot(data, showmedians=True, widths=.7)
    for pc, k in zip(parts["bodies"], keys):
        pc.set_facecolor(PALETTE[k]); pc.set_alpha(.55)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[key].set_color("#374151"); parts[key].set_linewidth(1)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels(["cold start\n(c = 0)", "learned guess\n(one GNN pass)"], fontsize=9)
    _style(ax, "Quality of the starting point, before any Newton step",
           "", "log residual (Pa)")
    fig.tight_layout(); fig.savefig(FIG / "guess_quality.png", dpi=160); plt.close(fig)


def fig_architecture():
    """Schematic of what each approach actually executes per timestep."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2))
    rows = [
        ("unrolled", "Unrolled GNN  (original)",
         [("GNN", 1)] + [("Newton\n(diagonal)", 0)] + [("GNN", 1), ("Newton\n(diagonal)", 0)] * 2,
         "x20 steps - attention runs every step"),
        ("newton", "Newton  (pure physics)",
         [("c0 = 0", 2)] + [("Newton\n(exact)", 0)] * 4,
         "no learned parameters at all"),
        ("initializer", "Learned initializer  (ours)",
         [("GNN", 1), ("Newton\n(exact)", 0), ("Newton\n(exact)", 0)],
         "attention runs ONCE, Newton finishes"),
    ]
    for ax, (key, title, blocks, note) in zip(axes, rows):
        ax.set_xlim(0, 10.4); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0, .93, title, fontsize=11.5, weight="bold", color=PALETTE[key], va="top")
        ax.text(10.35, .93, note, fontsize=8.5, color="#6b7280", va="top", ha="right")
        x = 0.05
        fills = {0: "#e5e7eb", 1: PALETTE[key], 2: "#f3f4f6"}
        for j, (lab, kind) in enumerate(blocks):
            w = 1.45
            ax.add_patch(plt.Rectangle((x, .18), w, .5, facecolor=fills[kind],
                                       edgecolor="#9ca3af", lw=.9, zorder=2))
            ax.text(x + w / 2, .43, lab, ha="center", va="center", fontsize=8.5, zorder=3,
                    color="white" if kind == 1 else "#111827")
            if j < len(blocks) - 1:
                ax.annotate("", xy=(x + w + .28, .43), xytext=(x + w + .02, .43),
                            arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1))
            x += w + .3
        if key == "unrolled":
            ax.text(x + .05, .43, "...", fontsize=15, va="center", color="#6b7280")
    fig.suptitle("What each approach executes per timestep", fontsize=13, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(FIG / "architecture.png", dpi=160); plt.close(fig)


def fig_reduction():
    """The cycle-space reduction that makes exact Newton affordable."""
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
    steps = [
        ("1,514\nedge flows", "#e5e7eb", "the raw unknowns"),
        ("1,352\nnode balances", "#e5e7eb", "mass conservation"),
        ("12\nloop flows", PALETTE["initializer"], "what is actually solved"),
    ]
    x = .5
    for i, (lab, col, note) in enumerate(steps):
        ax.add_patch(plt.Rectangle((x, .34), 2.3, .42, facecolor=col,
                                   edgecolor="#9ca3af", lw=1, zorder=2))
        ax.text(x + 1.15, .55, lab, ha="center", va="center", fontsize=10, zorder=3,
                color="white" if i == 2 else "#111827")
        ax.text(x + 1.15, .24, note, ha="center", va="top", fontsize=8, color="#6b7280")
        if i < 2:
            ax.annotate("", xy=(x + 3.05, .55), xytext=(x + 2.4, .55),
                        arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.4))
            ax.text(x + 2.72, .64, "reduce", ha="center", fontsize=8, color="#6b7280")
        x += 3.25
    ax.text(5, .05, "mdot = mdot0 + Z.c   with   A.Z = 0   ->   mass conservation is "
                    "structural, never learned",
            ha="center", fontsize=9, color="#374151", style="italic")
    ax.set_title("Why exact Newton is cheap here: 1,514 unknowns collapse to 12",
                 fontsize=12, pad=6)
    fig.tight_layout(); fig.savefig(FIG / "reduction.png", dpi=160); plt.close(fig)


# -------------------------------------------------------------------------- main
def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-ts", type=int, default=400)
    ap.add_argument("--tol", type=float, default=config.EVAL_TOL_PA)
    ap.add_argument("--split-every", type=int, default=5)
    ap.add_argument("--run-initializer", dest="run_initializer", default="gpu",
                    help="which results/initializer/<run>/ to score")
    ap.add_argument("--run-unrolled", dest="run_unrolled", default="2022",
                    help="which results/unrolled/<run>/ to score")
    ap.add_argument("--run-predictor", dest="run_predictor", default="default",
                    help="which results/predictor/<run>/ to score; the ablation "
                         "is skipped when it does not exist")
    a = args if args is not None else ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True); PRED.mkdir(parents=True, exist_ok=True)
    ops = netops.build_operators()
    rec = PressureReconstructor(ops)

    flow_t = pd.read_csv(config.MASS_FLOW_CSV, index_col=0)[ops.edge_names]
    dp_t = pd.read_csv(config.DELTA_P_CSV, index_col=0)[ops.edge_names]
    p_t = pd.read_csv(config.NODE_PRESSURE_CSV, index_col=0)[ops.node_names]

    _, test_ts = config.split_timesteps(len(flow_t), a.split_every)
    ts = test_ts[np.linspace(0, len(test_ts) - 1, a.n_ts).round().astype(int)]
    ts = np.unique(ts)
    samples = make_samples(ops, ts)
    print(f"{len(ts)} held-out timesteps, tol={a.tol} Pa")

    from dhn_gnn.checkpoints import load_checkpoint

    # records_start marks which families log their starting point as curve[0]:
    # Newton and the initializer do, the unrolled solver does not.
    models = {"newton": (NewtonSolver(ops, n_newton=8).float(), True)}
    # Which RUN of each approach the report scores. These were the flat files
    # results/model_init_gpu.pt and results/model_unrolled_2022.pt before the
    # per-run split; naming them as runs is what lets the report be regenerated
    # against a different sweep without editing this module.
    ip = config.resolve_checkpoint("initializer", a.run_initializer)
    up = config.resolve_checkpoint("unrolled", a.run_unrolled)
    mi, ck_i = load_checkpoint(ops, ip, DHNInitializerSolver)
    models["initializer"] = (mi, True)
    mu, ck_u = load_checkpoint(ops, up, DHNUnrolledSolver)
    models["unrolled"] = (mu, False)
    print(f"initializer <- {ip}")
    print(f"unrolled    <- {up}")

    # Optional: the ablation is only scored when a checkpoint for it exists, so
    # the report still builds for anyone who has not trained one.
    order = ["unrolled", "newton", "initializer"]
    pp = config.resolve_checkpoint("predictor", a.run_predictor)
    if pp.exists():
        mp, ck_p = load_checkpoint(ops, pp, DHNPredictorSolver)
        models["predictor"] = (mp, True)
        order.append("predictor")
        print(f"predictor   <- {pp}")
    else:
        print(f"predictor   <- (none at {pp}; skipping the no-Newton ablation)")

    runs, scores = {}, {}
    for k in order:
        m, rec_start = models[k]
        print(f"  running {k} ...", flush=True)
        runs[k] = run_approach(k, m, ops, samples, ts, rec, p_t, a.tol, rec_start)
        scores[k] = score(runs[k], ops, ts, flow_t, dp_t, p_t)
        d = PRED / k; d.mkdir(parents=True, exist_ok=True)
        idx = flow_t.index[ts]
        pd.DataFrame(runs[k]["flow"], index=idx, columns=ops.edge_names).to_csv(d / "mass_flow.csv")
        pd.DataFrame(runs[k]["dp"], index=idx, columns=ops.edge_names).to_csv(d / "delta_p.csv")
        pd.DataFrame(runs[k]["pressure"], index=idx, columns=ops.node_names).to_csv(d / "pressure.csv")
        print(f"    steps={scores[k]['steps_mean']:.2f}  {scores[k]['ms_per_ts']:.0f} ms/ts"
              f"  -> {d}")

    # PyDHN reference: its own CSVs are the ground truth, so its "error" is zero by
    # construction; only its COST is a real comparison point.
    pydhn_ms = PYDHN_HOURS_PER_YEAR * 3600 / PYDHN_N_TIMESTEPS * 1e3

    print("  figures ...")
    fig_convergence(runs, a.tol)
    fig_steps(runs, None)
    fig_parity(runs, flow_t, ts)
    fig_timing(scores, pydhn_ms)
    fig_seasonal(runs, ts, flow_t.index)
    fig_guess_quality(runs)
    fig_architecture()
    fig_reduction()

    # dataset consistency, carried into the report so the caveat cannot be lost
    from dhn_gnn.reporting.compare import dataset_consistency
    dpf = pd.read_csv(config.DELTA_P_FRICTION_CSV, index_col=0)[ops.edge_names]
    cons = dataset_consistency(ops, dpf, flow_t)

    hist = json.loads((config.GEN_DATA_DIR / "history.json").read_text())
    out = dict(
        n_timesteps=int(len(ts)), tol=float(a.tol),
        window=[str(flow_t.index[ts[0]]), str(flow_t.index[ts[-1]])],
        n_edges=len(ops.edge_names), n_pipes=int(ops.pipe_mask.sum()),
        n_nodes=len(ops.node_names), n_loops=int(ops.internal_loops.numel()),
        scores=scores, pydhn=dict(ms_per_ts=pydhn_ms, hours_per_year=PYDHN_HOURS_PER_YEAR,
                                  n_timesteps=PYDHN_N_TIMESTEPS,
                                  history_entries=len(hist["hydraulics iterations"]),
                                  history_stale=len(hist["hydraulics iterations"]) != len(flow_t)),
        consistency=cons,
        # The paths actually loaded above, not a restatement of them: this field is
        # the report's record of WHICH weights produced its numbers, so a hardcoded
        # value here names a file the run never opened.
        checkpoints=dict(initializer=str(ip),
                         unrolled=str(up),
                         predictor=str(pp) if pp.exists() else None),
    )
    (RES / "report_data.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved {RES/'report_data.json'}  and {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()

