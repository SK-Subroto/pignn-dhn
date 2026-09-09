"""Train and compare every approach/variant on identical data, then vs PyDHN.

This is the experiment driver for the thesis comparison. It defines the sweep ONCE
as data, so the table in the report and the commands that produced it cannot drift:

    python -m dhn_gnn.reporting.sweep --train     fit every run that is missing
    python -m dhn_gnn.reporting.sweep --report    score them all + write the report
    python -m dhn_gnn.reporting.sweep --train --report

Every run shares n_train, split_every and seed, so the only differences between
rows are the ones being studied. --train SKIPS a run whose checkpoint already
exists, which makes the sweep resumable: a 2-3 hour job that dies at run 8 does
not restart from run 1.

Scoring note: the variants are NOT all measured the same way, and pretending they
were would be the easiest way to publish a wrong table.

  * unrolled and initializer take Newton steps, so "steps to tolerance" is the
    honest cost metric.
  * predictor takes none by construction -- it always spends exactly one GNN pass.
    Its step count is 0 when the prediction clears the tolerance and infinite when
    it never does, so the column that matters for it is `under_tol`.
  * PyDHN's own cost comes from the dataset's history.json, and it WARM-STARTS
    from the previous hour, so it must walk the year in order. Every learned
    variant here is independent per timestep and could be solved in parallel.
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dhn_gnn import approaches, config
from dhn_gnn.checkpoints import load_checkpoint
from dhn_gnn.datasets import make_samples
from dhn_gnn.physics import operators as netops
from dhn_gnn.solvers.newton import NewtonSolver

OUT = config.RESULTS_DIR / "sweep"

# Settings shared by EVERY run, so no row differs by accident.
COMMON = ["--set", "n_train=200", "--set", "split_every=5", "--set", "seed=0"]

# (arch, run name, overrides, human label) -- the sweep, as data.
SWEEP = [
    # a. unrolled: the two Jacobians
    ("unrolled", "full", ["--set", "model.newton_mode=full"],
     "unrolled, full Jacobian"),
    ("unrolled", "diagonal", ["--set", "model.newton_mode=diagonal"],
     "unrolled, diagonal Jacobian"),

    # b. initializer: how much of the graph the embedding sees
    ("initializer", "all", ["--set", "model.node_scope=all"],
     "initializer, all nodes"),
    ("initializer", "cycle0", ["--set", "model.node_scope=cycle",
                               "--set", "model.cycle_hops=0"],
     "initializer, loop nodes only"),
    ("initializer", "cycle1", ["--set", "model.node_scope=cycle",
                               "--set", "model.cycle_hops=1"],
     "initializer, loop + 1 hop"),
    ("initializer", "cycle2", ["--set", "model.node_scope=cycle",
                               "--set", "model.cycle_hops=2"],
     "initializer, loop + 2 hops"),
    ("initializer", "cycle3", ["--set", "model.node_scope=cycle",
                               "--set", "model.cycle_hops=3"],
     "initializer, loop + 3 hops"),

    # c. predictor: which objective, and whether cycle scoping helps without Newton
    ("predictor", "residual_all", ["--set", "loss=residual",
                                   "--set", "model.node_scope=all"],
     "predictor, residual loss"),
    ("predictor", "residual_cycle1", ["--set", "loss=residual",
                                      "--set", "model.node_scope=cycle",
                                      "--set", "model.cycle_hops=1"],
     "predictor, residual loss, loop + 1 hop"),
    ("predictor", "astar_all", ["--set", "loss=a_star",
                                "--set", "model.node_scope=all"],
     "predictor, a_star loss"),
]
LABEL = {(a, r): lab for a, r, _, lab in SWEEP}


# ------------------------------------------------------------------------ train
def train_all(force=False):
    todo = [(a, r, o) for a, r, o, _ in SWEEP
            if force or not config.checkpoint_path(a, r).exists()]
    done = len(SWEEP) - len(todo)
    print(f"sweep: {len(SWEEP)} runs, {done} already trained, {len(todo)} to go\n")
    for i, (arch, run, over) in enumerate(todo, 1):
        cmd = [sys.executable, "-m", "dhn_gnn.cli", "train",
               "--arch", arch, "--run", run] + COMMON + over
        print(f"[{i}/{len(todo)}] {arch}/{run}\n    {' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            # Keep going: one bad variant must not cost the whole sweep, and the
            # report simply omits a run with no checkpoint.
            print(f"    FAILED ({p.returncode}):\n{p.stdout[-1500:]}{p.stderr[-1500:]}",
                  flush=True)
            continue
        tail = [l for l in p.stdout.splitlines() if "restored" in l or "untrained" in l]
        print(f"    done in {time.time()-t0:.0f}s   " + " | ".join(tail), flush=True)


# ----------------------------------------------------------------------- report
def _steps_to_tol(curve, tol, records_start):
    """Solver steps until max|residual| first drops below tol; inf if never.

    `records_start` says whether curve[0] is the state BEFORE any step (Newton and
    the initializer log it; the unrolled solver does not). Getting this wrong
    shifts the headline number by one step, in opposite directions per family.
    """
    bump = 0 if records_start else 1
    for i, v in enumerate(curve):
        if v < tol:
            return i + bump
    return np.inf


def _accuracy(pred, true, pipe_mask):
    """Flow error against the reference solution, all edges and pipes only.

    Reported alongside the iteration count because a solver that converges fast to
    the wrong answer is not better: `steps` measures cost, this measures whether
    the answer is right.
    """
    err = pred - true
    ss = np.sum((true - true.mean()) ** 2)
    pm = pipe_mask.astype(bool)
    ep = (pred[:, pm] - true[:, pm])
    return dict(
        flow_mae=float(np.mean(np.abs(err))),
        flow_rmse=float(np.sqrt(np.mean(err ** 2))),
        flow_max=float(np.abs(err).max()),
        flow_r2=float(1 - np.sum(err ** 2) / ss),
        flow_mae_pipes=float(np.mean(np.abs(ep))),
        # sign agreement: a wrong flow DIRECTION is a qualitatively different
        # failure from a wrong magnitude, and an MAE alone hides it
        dir_acc=float(100 * np.mean(
            (np.sign(pred) * (np.abs(pred) > 1e-2)) ==
            (np.sign(true) * (np.abs(true) > 1e-2)))),
    )


def _curve_stats(curves):
    """Median convergence curve, padded so every run has the same length.

    Padding uses each run's LAST value rather than NaN: a solver that exited early
    has genuinely reached that residual and holds it, so carrying it forward is the
    honest continuation.
    """
    L = max(len(c) for c in curves)
    padded = np.array([c + [c[-1]] * (L - len(c)) for c in curves])
    return dict(median=np.median(padded, 0).tolist(),
                p25=np.percentile(padded, 25, 0).tolist(),
                p75=np.percentile(padded, 75, 0).tolist())


def score_run(arch, run, ops, samples, tol, flow_true):
    ckpt = config.checkpoint_path(arch, run)
    if not ckpt.exists():
        return None
    model, ck = load_checkpoint(ops, ckpt, approaches.get(arch).model_cls())
    model = model.float().eval()
    records_start = arch != "unrolled"

    steps, finals, guesses, flows, curves = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for m0, _, _ in samples:
            mdot, terms, _ = model(m0, tol=tol)
            curve = [t.abs().max().item() for t in terms]
            steps.append(_steps_to_tol(curve, tol, records_start))
            finals.append(curve[-1])
            guesses.append(curve[0])
            curves.append(curve)
            flows.append(mdot.numpy())
    ms = (time.time() - t0) / len(samples) * 1e3

    steps = np.array(steps, float)
    finals = np.array(finals)
    ok = np.isfinite(steps)
    meta = ck.get("meta", {})
    pred = np.stack(flows)

    # How much of the graph this configuration actually attends over. Only the
    # families that carry a node mask have it; the unrolled solver has no scope
    # setting, so reporting a number would imply a choice it never makes.
    scope = {}
    if hasattr(model, "node_mask"):
        scope = dict(nodes=int(model.node_mask.sum()),
                     nodes_total=int(model.node_mask.numel()),
                     attn_edges=int(model.attn_edge_index.shape[1]),
                     attn_edges_total=int(model.edge_index.shape[1]))

    return dict(
        arch=arch, run=run, label=LABEL.get((arch, run), f"{arch}/{run}"),
        steps_mean=float(steps[ok].mean()) if ok.any() else None,
        steps_median=float(np.median(steps[ok])) if ok.any() else None,
        steps_max=float(steps[ok].max()) if ok.any() else None,
        steps_hist={str(int(k)): int(v) for k, v in
                    zip(*np.unique(steps[ok], return_counts=True))} if ok.any() else {},
        reached_tol_pct=float(100 * ok.mean()),
        resid_median=float(np.median(finals)),
        resid_mean=float(finals.mean()),
        resid_p95=float(np.percentile(finals, 95)),
        under_tol_pct=float(100 * np.mean(finals <= tol)),
        # None for the unrolled family: it logs only post-step states, so there is
        # no "before any step" value to report and a number here would be compared
        # against the others as if there were.
        guess_resid_median=float(np.median(guesses)) if records_start else None,
        ms_per_ts=float(ms),
        n_params=int(sum(p.numel() for p in model.parameters())),
        scope=scope,
        curve=_curve_stats(curves),
        accuracy=_accuracy(pred, flow_true, ops.pipe_mask.numpy()),
        _flows=pred,
        model_kwargs=ck.get("model_kwargs", {}),
        best_epoch=meta.get("best_epoch"),
        train_seconds=meta.get("train_seconds"),
        n_train=meta.get("n_train"),
        epochs=meta.get("epochs"),
        lr=meta.get("lr"),
        untrained_resid=meta.get("untrained_mean_residual"),
        loss=meta.get("loss"),
    )


def score_newton(ops, samples, tol, flow_true, n_newton=20):
    """Pure physics, cold start: the control condition every learned row is measured
    against. No learned parameters at all, so it isolates what learning contributes."""
    m = NewtonSolver(ops, n_newton=n_newton).float().eval()
    steps, finals, guesses, flows, curves = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for m0, _, _ in samples:
            mdot, terms, _ = m(m0, tol=tol, c_init=torch.zeros(m.n_free))
            curve = [t.abs().max().item() for t in terms]
            steps.append(_steps_to_tol(curve, tol, True))
            finals.append(curve[-1]); guesses.append(curve[0])
            curves.append(curve); flows.append(mdot.numpy())
    ms = (time.time() - t0) / len(samples) * 1e3
    steps = np.array(steps, float); finals = np.array(finals)
    ok = np.isfinite(steps); pred = np.stack(flows)
    return dict(
        arch="newton", run="cold", label="Newton, cold start (no learned parameters)",
        steps_mean=float(steps[ok].mean()) if ok.any() else None,
        steps_median=float(np.median(steps[ok])) if ok.any() else None,
        steps_max=float(steps[ok].max()) if ok.any() else None,
        steps_hist={str(int(k)): int(v) for k, v in
                    zip(*np.unique(steps[ok], return_counts=True))} if ok.any() else {},
        reached_tol_pct=float(100 * ok.mean()),
        resid_median=float(np.median(finals)), resid_mean=float(finals.mean()),
        resid_p95=float(np.percentile(finals, 95)),
        under_tol_pct=float(100 * np.mean(finals <= tol)),
        guess_resid_median=float(np.median(guesses)),
        ms_per_ts=float(ms), n_params=0, scope={},
        curve=_curve_stats(curves),
        accuracy=_accuracy(pred, flow_true, ops.pipe_mask.numpy()),
        _flows=pred, model_kwargs=dict(n_newton=n_newton, newton_mode="full"),
        best_epoch=None, train_seconds=0.0, n_train=0, epochs=0, lr=None,
        untrained_resid=None, loss=None)


def environment():
    """Recorded in the report so a number can be tied to the machine that made it."""
    import platform
    return dict(python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, pandas=pd.__version__,
                platform=platform.platform(), processor=platform.processor(),
                device="cpu", threads=int(torch.get_num_threads()))


def build_report(n_ts=200, tol=None, split_every=5):
    tol = config.EVAL_TOL_PA if tol is None else tol
    OUT.mkdir(parents=True, exist_ok=True)

    ops = netops.build_operators()
    flow_t = pd.read_csv(config.MASS_FLOW_CSV, index_col=0)
    _, test_ts = config.split_timesteps(len(flow_t), split_every)
    ts = np.unique(test_ts[np.linspace(0, len(test_ts) - 1, n_ts).round().astype(int)])
    samples = make_samples(ops, ts)
    print(f"scoring on {len(ts)} HELD-OUT timesteps at tol={tol} Pa\n")

    flow_true = flow_t.iloc[ts][ops.edge_names].to_numpy(np.float64)
    rows = [score_newton(ops, samples, tol, flow_true)]
    for arch, run, _, _ in SWEEP:
        r = score_run(arch, run, ops, samples, tol, flow_true)
        if r is None:
            print(f"  (no checkpoint for {arch}/{run}; omitted)")
            continue
        rows.append(r)
        print(f"  {r['label']:42s} steps={str(r['steps_mean'])[:5]:>5s}  "
              f"resid={r['resid_median']:9.1f} Pa  {r['ms_per_ts']:6.1f} ms", flush=True)

    # Per-run predictions, on the SAME held-out timesteps as the table above, so a
    # number in the report can always be traced to the flows that produced it.
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    idx = flow_t.index[ts]
    for r in rows:
        f = r.pop("_flows", None)
        if f is None:
            continue
        name = f"{r['arch']}-{r['run']}.csv"
        pd.DataFrame(f, index=idx, columns=ops.edge_names).to_csv(pred_dir / name)
    pd.DataFrame(flow_t.iloc[ts].to_numpy(), index=idx,
                 columns=flow_t.columns).to_csv(pred_dir / "pydhn-reference.csv")
    print(f"\n  wrote {len(list(pred_dir.glob('*.csv')))} prediction CSVs -> {pred_dir}")

    hist = json.loads((config.GEN_DATA_DIR / "history.json").read_text())
    it = np.array(hist["hydraulics iterations"], float)
    pydhn = dict(steps_mean=float(it.mean()), steps_median=float(np.median(it)),
                 n_steps=int(len(it)),
                 stale=bool(len(it) != len(flow_t)))

    data = dict(n_timesteps=int(len(ts)), tol=float(tol), split_every=split_every,
                dataset=config.GEN_DATA_DIR.name,
                window=[str(flow_t.index[ts[0]]), str(flow_t.index[ts[-1]])],
                n_edges=len(ops.edge_names), n_nodes=len(ops.node_names),
                n_loops=int(ops.internal_loops.numel()),
                common=" ".join(COMMON), rows=rows, pydhn=pydhn,
                environment=environment(),
                generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "sweep.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    txt = render_text(data)
    (OUT / "sweep.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    html = render_html(data)
    (OUT / "sweep.html").write_text(html, encoding="utf-8")
    print(f"\nsaved {OUT/'sweep.json'}\n      {OUT/'sweep.txt'}\n      {OUT/'sweep.html'}")
    return data


def _fmt(v, spec=".2f", dash="-"):
    return dash if v is None else format(v, spec)


def render_text(d):
    w = 42
    hdr = (f"{'variant':<{w}}{'steps':>8}{'reached':>9}{'resid Pa':>11}"
           f"{'<=tol':>8}{'guess Pa':>11}{'ms/ts':>8}{'params':>9}")
    L = ["=== Sweep: every approach on identical held-out data ===", "",
         f"dataset {d['dataset']}   {d['n_timesteps']} held-out timesteps   "
         f"tol {d['tol']:g} Pa",
         f"network {d['n_nodes']} nodes / {d['n_edges']} edges / {d['n_loops']} loops",
         f"all runs trained with: {d['common']}", "", hdr, "-" * len(hdr)]
    for r in d["rows"]:
        L.append(f"{r['label']:<{w}}{_fmt(r['steps_mean']):>8}"
                 f"{r['reached_tol_pct']:>8.0f}%{r['resid_median']:>11.1f}"
                 f"{r['under_tol_pct']:>7.0f}%"
                 f"{_fmt(r['guess_resid_median'], '.1f'):>11}"
                 f"{r['ms_per_ts']:>8.1f}{r['n_params']:>9,}")
    p = d["pydhn"]
    L += ["-" * len(hdr),
          f"{'PyDHN (reference, WARM-started)':<{w}}{p['steps_mean']:>8.2f}"
          f"{100:>8.0f}%{'~46':>11}{100:>7.0f}%{'-':>11}{'-':>8}{0:>9,}"]
    L += ["", "Columns:",
          "  steps    = solver steps to first reach the tolerance. The predictor takes",
          "             none by construction, so it scores 0 where it succeeds; read its",
          "             '<=tol' column instead.",
          "  reached  = % of timesteps that ever reached the tolerance.",
          "  guess Pa = residual of the STARTING point, before any Newton step. For the",
          "             predictor this is also its final answer.",
          "",
          "PyDHN warm-starts from the previous hour, so it must solve the year IN ORDER.",
          "Every learned variant above is independent per timestep and can be batched.",
          ]
    if p["stale"]:
        L.append("WARNING: history.json length != dataset length; its step count is stale.")
    return "\n".join(L)


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DHN solver sweep</title><style>
:root{{--bg:#fff;--fg:#111827;--mut:#6b7280;--line:#e5e7eb;--card:#f9fafb;
--unrolled:#b45309;--initializer:#1d4ed8;--predictor:#7c3aed;--newton:#0f766e;--pydhn:#6b7280;}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0b0f14;--fg:#e5e7eb;--mut:#9ca3af;
--line:#1f2937;--card:#111827;--unrolled:#e39445;--initializer:#6ea8fa;
--predictor:#a78bfa;--newton:#31c9b6;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:2.2rem 1.2rem}}
main{{max-width:1080px;margin:0 auto}}h1{{font-size:1.6rem;margin:0 0 .3rem}}
h2{{font-size:1.15rem;margin:2.2rem 0 .6rem;padding-bottom:.3rem;border-bottom:1px solid var(--line)}}
.sub{{color:var(--mut);margin:0 0 1.6rem}}
.wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;min-width:760px;font-size:.88rem}}
th,td{{padding:.5rem .7rem;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}
thead th{{background:var(--card);font-weight:600;color:var(--mut);font-size:.78rem;
text-transform:uppercase;letter-spacing:.04em}}
tbody tr:last-child td{{border-bottom:none}}
tr.ref td{{background:var(--card);font-style:italic}}
.sw{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:.5rem}}
.best{{font-weight:700}}
.note{{color:var(--mut);font-size:.85rem;margin:.8rem 0 0}}
ul{{padding-left:1.1rem}}li{{margin:.3rem 0}}
code{{background:var(--card);padding:.1rem .35rem;border-radius:4px;font-size:.85em}}
</style></head><body><main>
<h1>District heating solvers: full sweep</h1>
<p class="sub">{n} held-out timesteps &middot; tolerance {tol:g} Pa &middot; dataset <code>{dataset}</code><br>
network {nodes} nodes / {edges} edges / {loops} independent loops &middot; {win_a} to {win_b}</p>
<h2>Every variant, same data</h2>
<div class="wrap"><table><thead><tr>
<th>variant</th><th>steps</th><th>reached tol</th><th>residual (Pa)</th>
<th>&le; tol</th><th>guess (Pa)</th><th>ms / step</th><th>params</th>
</tr></thead><tbody>
{rows}
</tbody></table></div>
<p class="note">All learned runs share <code>{common}</code>, so rows differ only in
what is being studied.</p>
<h2>How to read this</h2>
<ul>
<li><b>steps</b> &mdash; solver steps to first reach the tolerance. The
<b>predictor takes none by construction</b>: it scores 0 where it succeeds, so judge
it by <b>&le; tol</b> instead.</li>
<li><b>guess</b> &mdash; residual of the starting point, before any Newton step.
For the predictor this is also its final answer.</li>
<li><b>PyDHN warm-starts</b> from the previous hour, so it must solve the year in
order. Every learned variant here is independent per timestep and can be batched.</li>
</ul>
{caveat}
</main></body></html>
"""


def render_html(d):
    def row(r, ref=False):
        key = r["arch"] if r["arch"] in ("unrolled", "initializer", "predictor",
                                         "newton", "pydhn") else "newton"
        sw = f'<span class="sw" style="background:var(--{key})"></span>'
        cells = [_fmt(r["steps_mean"]), f"{r['reached_tol_pct']:.0f}%",
                 f"{r['resid_median']:.1f}", f"{r['under_tol_pct']:.0f}%",
                 _fmt(r["guess_resid_median"], ".1f", "&mdash;"),
                 f"{r['ms_per_ts']:.1f}",
                 f"{r['n_params']:,}"]
        cls = ' class="ref"' if ref else ""
        return (f"<tr{cls}><td>{sw}{r['label']}</td>"
                + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    body = "\n".join(row(r) for r in d["rows"])
    p = d["pydhn"]
    body += (f'\n<tr class="ref"><td><span class="sw" style="background:var(--pydhn)">'
             f'</span>PyDHN (reference, warm-started)</td><td>{p["steps_mean"]:.2f}</td>'
             f"<td>100%</td><td>~46</td><td>100%</td><td>&mdash;</td>"
             f"<td>&mdash;</td><td>0</td></tr>")

    caveat = ""
    if p["stale"]:
        caveat = ('<p class="note"><b>Warning:</b> history.json does not match the '
                  "dataset length; PyDHN's step count is stale.</p>")
    return HTML.format(
        n=d["n_timesteps"], tol=d["tol"], dataset=d["dataset"],
        nodes=d["n_nodes"], edges=d["n_edges"], loops=d["n_loops"],
        win_a=d["window"][0][:16], win_b=d["window"][1][:16],
        common=d["common"], rows=body, caveat=caveat)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="fit every missing run")
    ap.add_argument("--force", action="store_true", help="retrain even if present")
    ap.add_argument("--report", action="store_true", help="score all runs + write report")
    ap.add_argument("--n-ts", type=int, default=200, help="held-out timesteps to score on")
    ap.add_argument("--tol", type=float, default=None)
    a = ap.parse_args()
    if not (a.train or a.report):
        ap.error("nothing to do: pass --train, --report, or both")
    if a.train:
        train_all(force=a.force)
    if a.report:
        build_report(n_ts=a.n_ts, tol=a.tol)


if __name__ == "__main__":
    main()
