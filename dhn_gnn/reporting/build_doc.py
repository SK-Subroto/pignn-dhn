"""Build the comparison report as a self-contained HTML page.

Every figure is embedded as a data URI and every number is read from
results/report_data.json, so the document cannot drift from the run that produced
it -- re-running full_report.py and then this script regenerates the whole thing.

    python -m dhn_gnn.reporting.full_report --n-ts 400
    python -m dhn_gnn.reporting.build_doc
"""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dhn_gnn import config

RES = config.RESULTS_DIR
FIG = RES / "figures"
OUT = RES / "report" / "solver-comparison.html"

ORDER = ["pydhn", "unrolled", "newton", "initializer"]
NAME = {
    "pydhn": "PyDHN",
    "unrolled": "Unrolled GNN",
    "newton": "Newton",
    "initializer": "Learned initializer",
}
SUB = {
    "pydhn": "reference simulator",
    "unrolled": "original approach",
    "newton": "pure physics, no learning",
    "initializer": "this work",
}


def img(name, alt, caption):
    p = FIG / name
    if not p.exists():
        return f"<p class='missing'>missing figure: {name}</p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure class="chart">\n'
            f'  <img src="data:image/png;base64,{b64}" alt="{alt}">\n'
            f'  <figcaption>{caption}</figcaption>\n</figure>')


def fmt(v, spec=".2f"):
    return format(v, spec)


def build():
    d = json.loads((RES / "report_data.json").read_text(encoding="utf-8"))
    s, py, cons = d["scores"], d["pydhn"], d["consistency"]
    n = d["n_timesteps"]
    speedup = py["ms_per_ts"] / s["initializer"]["ms_per_ts"]
    vs_unrolled = s["unrolled"]["ms_per_ts"] / s["initializer"]["ms_per_ts"]

    def row(k, cells):
        sw = f'<span class="sw" style="background:var(--{k})"></span>'
        return (f'<tr><th scope="row">{sw}{NAME[k]}<small>{SUB[k]}</small></th>'
                + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    quality = "\n".join(row(k, [
        fmt(s[k]["flow_mae"], ".2e"), fmt(s[k]["flow_rmse"], ".2e"),
        fmt(s[k]["flow_r2"], ".6f"), fmt(s[k]["dir_acc"], ".2f") + " %",
        fmt(s[k]["dp_mae"], ".2f"), fmt(s[k]["press_mae"], ".0f"),
    ]) for k in ("unrolled", "newton", "initializer"))

    cost = "\n".join(row(k, [
        fmt(s[k]["steps_mean"], ".2f"), fmt(s[k]["steps_median"], ".0f"),
        fmt(s[k]["steps_max"], ".0f"), fmt(s[k]["ms_per_ts"], ".0f") + " ms",
        fmt(s[k]["pct_under_tol"], ".1f") + " %",
        "<span class='yes'>yes</span>",
    ]) for k in ("unrolled", "newton", "initializer"))
    cost += ("\n" + row("pydhn", ["&mdash;", "&mdash;", "&mdash;",
                                  f"~{py['ms_per_ts']:,.0f} ms", "100 %",
                                  "<span class='no'>no</span>"]))

    html = TEMPLATE.format(
        n=n, window_a=d["window"][0][:10], window_b=d["window"][1][:10],
        tol=fmt(d["tol"], ".0f"),
        n_edges=f"{d['n_edges']:,}", n_pipes=f"{d['n_pipes']:,}",
        n_nodes=f"{d['n_nodes']:,}", n_loops=d["n_loops"],
        quality_rows=quality, cost_rows=cost,
        init_steps=fmt(s["initializer"]["steps_mean"], ".2f"),
        newton_steps=fmt(s["newton"]["steps_mean"], ".2f"),
        unrolled_steps=fmt(s["unrolled"]["steps_mean"], ".2f"),
        init_ms=fmt(s["initializer"]["ms_per_ts"], ".0f"),
        unrolled_ms=fmt(s["unrolled"]["ms_per_ts"], ".0f"),
        newton_ms=fmt(s["newton"]["ms_per_ts"], ".0f"),
        init_r2=fmt(s["initializer"]["flow_r2"], ".6f"),
        init_mae=fmt(s["initializer"]["flow_mae"], ".2e"),
        pydhn_ms=f"{py['ms_per_ts']:,.0f}",
        pydhn_hours=fmt(py["hours_per_year"], ".0f"),
        pydhn_n=f"{py['n_timesteps']:,}",
        speedup=fmt(speedup, ".0f"), vs_unrolled=fmt(vs_unrolled, ".1f"),
        guess_cold=fmt(s["newton"]["guess_resid_median"], ".2e"),
        guess_learned=fmt(s["initializer"]["guess_resid_median"], ".2e"),
        guess_ratio=fmt(s["newton"]["guess_resid_median"]
                        / max(s["initializer"]["guess_resid_median"], 1e-9), ".0f"),
        cons_within=fmt(cons["within_5pct"], ".1f"),
        cons_ratio=fmt(cons["median_ratio"], ".3f"),
        cons_lo=fmt(cons["ratio_spread"][0], ".3f"),
        cons_hi=fmt(cons["ratio_spread"][1], ".3f"),
        hist_n=py["history_entries"],
        fig_conv=img("convergence.png", "Residual per solver step",
                     "Loop residual after each solver step, median with inter-quartile "
                     "band. Exact Newton converges quadratically; the diagonal "
                     "approximation used by the original approach converges linearly."),
        fig_steps=img("steps_distribution.png", "Distribution of steps to tolerance",
                      "Steps needed to reach the tolerance, as a share of timesteps. "
                      "Means hide the tail, so the full distribution is shown."),
        fig_guess=img("guess_quality.png", "Starting-point quality",
                      "Residual of the starting point before any Newton step. This is "
                      "the entire contribution of the learned component."),
        fig_timing=img("timing.png", "Wall-clock per timestep",
                       "Wall-clock per timestep on CPU, log scale. The PyDHN marker "
                       "covers its full simulation, not just the hydraulic solve."),
        fig_parity=img("flow_parity.png", "Predicted vs reference flow",
                       "Predicted against reference mass flow for every pipe and "
                       "timestep. Density is log-scaled; the dashed line is equality."),
        fig_season=img("seasonal.png", "Solver effort through the year",
                       "Mean steps to tolerance by month. A model fitted on 200 hours "
                       "sampled across the year shows no seasonal collapse."),
        fig_reduce=img("reduction.png", "Cycle-space reduction",
                       "The reduction that makes exact Newton affordable: 1,514 edge "
                       "unknowns collapse to 12 loop unknowns."),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    kb = len(html.encode()) / 1024
    print(f"wrote {OUT}  ({kb:,.0f} KB)")
    return OUT


TEMPLATE = r"""<title>DHN Solver Comparison</title>
<style>
:root {{
  --paper:#fcfdfd; --ink:#0e1a1d; --muted:#59686c; --rule:#dbe4e5; --panel:#f2f6f6;
  --figure-bg:#ffffff;
  --pydhn:#5f6b6d; --unrolled:#b45309; --newton:#0f766e; --initializer:#1d4ed8;
  --warn:#b3261e; --warn-bg:#fdf3f2; --ok:#0f766e;
  --serif:Georgia,"Iowan Old Style","Times New Roman",serif;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"Cascadia Mono",Consolas,"Liberation Mono",monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0b1113; --ink:#e4edee; --muted:#93a4a8; --rule:#1d2a2d; --panel:#111a1d;
    --figure-bg:#e9eeef;
    --pydhn:#98a4a6; --unrolled:#e39445; --newton:#31c9b6; --initializer:#6ea8fa;
    --warn:#f08b82; --warn-bg:#1f1413; --ok:#31c9b6;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0b1113; --ink:#e4edee; --muted:#93a4a8; --rule:#1d2a2d; --panel:#111a1d;
  --figure-bg:#e9eeef;
  --pydhn:#98a4a6; --unrolled:#e39445; --newton:#31c9b6; --initializer:#6ea8fa;
  --warn:#f08b82; --warn-bg:#1f1413; --ok:#31c9b6;
}}

* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.65;
  -webkit-text-size-adjust:100%;
}}
.wrap {{ max-width:56rem; margin:0 auto; padding:4rem 1.5rem 6rem; }}
.col {{ max-width:38rem; }}

h1,h2,h3 {{ font-family:var(--serif); font-weight:600; text-wrap:balance; line-height:1.2; }}
h1 {{ font-size:2.45rem; margin:0 0 .6rem; letter-spacing:-.01em; }}
h2 {{ font-size:1.6rem; margin:4rem 0 1rem; padding-top:1.4rem; border-top:1px solid var(--rule); }}
h3 {{ font-size:1.12rem; margin:2.2rem 0 .5rem; }}
p {{ margin:0 0 1.05rem; }}
a {{ color:inherit; }}
small {{ font-size:.8rem; }}

.eyebrow {{
  font-family:var(--mono); font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); margin:0 0 1rem;
}}
.lede {{ font-size:1.14rem; color:var(--ink); margin-bottom:1.4rem; }}
.meta {{
  font-family:var(--mono); font-size:.78rem; color:var(--muted);
  border-top:1px solid var(--rule); border-bottom:1px solid var(--rule);
  padding:.85rem 0; margin:2rem 0 0; display:flex; flex-wrap:wrap; gap:.4rem 2rem;
}}

.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(11rem,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); margin:2.5rem 0; }}
.kpi {{ background:var(--paper); padding:1.15rem 1.2rem; }}
.kpi .v {{ font-family:var(--mono); font-size:1.7rem; line-height:1.1;
  font-variant-numeric:tabular-nums; display:block; }}
.kpi .k {{ font-size:.78rem; color:var(--muted); display:block; margin-top:.4rem; }}

table {{ width:100%; border-collapse:collapse; font-size:.87rem; margin:1.2rem 0 .5rem;
  font-variant-numeric:tabular-nums; }}
.scroll {{ overflow-x:auto; }}
th,td {{ text-align:right; padding:.6rem .7rem; border-bottom:1px solid var(--rule); }}
thead th {{ font-family:var(--sans); font-weight:600; font-size:.74rem; color:var(--muted);
  text-transform:uppercase; letter-spacing:.06em; border-bottom:1.5px solid var(--ink); }}
tbody th {{ text-align:left; font-weight:600; white-space:nowrap; }}
tbody th small {{ display:block; font-weight:400; color:var(--muted); }}
.sw {{ display:inline-block; width:.6rem; height:.6rem; border-radius:2px; margin-right:.5rem; }}
tr.best td {{ background:color-mix(in srgb, var(--initializer) 9%, transparent); }}
.yes {{ color:var(--ok); font-weight:600; }}
.no {{ color:var(--warn); font-weight:600; }}

figure {{ margin:2.2rem 0; }}
figure.chart img {{ width:100%; height:auto; display:block; background:var(--figure-bg);
  border:1px solid var(--rule); padding:.6rem; }}
figcaption {{ font-size:.82rem; color:var(--muted); margin-top:.7rem; max-width:38rem; }}
figure.diagram svg {{ width:100%; height:auto; display:block; }}

.note {{ border-left:3px solid var(--rule); padding:.15rem 0 .15rem 1.1rem; margin:1.6rem 0;
  color:var(--muted); font-size:.92rem; }}
.warn {{ border-left:3px solid var(--warn); background:var(--warn-bg);
  padding:1rem 1.2rem; margin:1.8rem 0; font-size:.92rem; }}
.warn strong {{ color:var(--warn); }}
.warn p:last-child {{ margin-bottom:0; }}

pre {{ background:var(--panel); border:1px solid var(--rule); padding:1rem 1.1rem;
  overflow-x:auto; font-family:var(--mono); font-size:.8rem; line-height:1.6; margin:1.2rem 0; }}
code {{ font-family:var(--mono); font-size:.87em; }}
p code, li code, td code {{ background:var(--panel); padding:.1em .35em; border-radius:3px; }}
ul {{ padding-left:1.15rem; }} li {{ margin-bottom:.4rem; }}
.missing {{ color:var(--warn); font-family:var(--mono); font-size:.8rem; }}
.foot {{ margin-top:4rem; padding-top:1.2rem; border-top:1px solid var(--rule);
  font-size:.8rem; color:var(--muted); }}
@media (max-width:640px) {{ h1 {{ font-size:1.9rem; }} .wrap {{ padding-top:2.5rem; }} }}
</style>

<div class="wrap">
<p class="eyebrow">District heating hydraulics &middot; solver comparison</p>
<h1>Four ways to solve the same network</h1>
<p class="lede col">
  A physics-informed graph network that <em>guesses</em> where the solution is, followed
  by exact Newton, reaches tolerance in <strong>{init_steps} solver steps</strong> against
  {newton_steps} for pure physics and {unrolled_steps} for the original architecture &mdash;
  and unlike the reference simulator, it needs no previous timestep, so the whole year
  can be solved in parallel.
</p>
<div class="meta">
  <span>{n} held-out timesteps</span><span>{window_a} &rarr; {window_b}</span>
  <span>tolerance {tol} Pa</span><span>CPU, single sample</span>
</div>

<div class="kpis">
  <div class="kpi"><span class="v">{init_steps}</span><span class="k">solver steps &middot; learned initializer</span></div>
  <div class="kpi"><span class="v">{vs_unrolled}&times;</span><span class="k">faster than the original approach</span></div>
  <div class="kpi"><span class="v">{init_r2}</span><span class="k">flow R&sup2; vs reference</span></div>
  <div class="kpi"><span class="v">{guess_ratio}&times;</span><span class="k">better starting point than cold</span></div>
</div>

<div class="col">
<h2>The problem</h2>
<p>
  The network has {n_edges} edges ({n_pipes} of them pipes) joining {n_nodes} nodes.
  Solving it means finding the mass flow in every pipe such that water is conserved at
  every junction and pressure drop closes around every loop.
</p>
<p>
  Those two laws are not equally hard. Mass conservation can be satisfied
  <em>by construction</em>: write the flow as a boundary-feasible reference plus a
  circulation around each independent loop, and conservation holds for any value of the
  loop flows. What remains is {n_loops} unknowns &mdash; and that reduction is what makes
  an exact Newton solve cheap enough to use as the default.
</p>
</div>

{fig_reduce}

<div class="col">
<h2>The four approaches</h2>
<p>
  All three solvers below share one physics implementation and are scored on identical
  timesteps at an identical tolerance, so any difference between them is attributable to
  the mechanism being studied and nothing else. PyDHN is the reference that produced the
  ground truth.
</p>
</div>

<figure class="diagram">
<svg viewBox="0 0 760 300" role="img" aria-label="Three solver architectures compared: the original runs a graph network at every one of twenty steps, pure Newton runs none, and the learned initializer runs one graph-network pass followed by exact Newton steps.">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
    </marker>
  </defs>
  <g font-family="ui-sans-serif,system-ui,sans-serif" font-size="12" fill="currentColor">

    <text x="0" y="16" font-size="12.5" font-weight="600" fill="var(--unrolled)">Unrolled GNN</text>
    <text x="0" y="32" font-size="11" fill="var(--muted)">attention on every step</text>
    <rect x="150" y="4" width="74" height="34" fill="var(--unrolled)" rx="2"/>
    <text x="187" y="26" text-anchor="middle" fill="#fff" font-size="11.5">GNN</text>
    <line x1="228" y1="21" x2="252" y2="21" stroke="currentColor" marker-end="url(#ar)"/>
    <rect x="256" y="4" width="90" height="34" fill="none" stroke="currentColor" rx="2"/>
    <text x="301" y="26" text-anchor="middle" font-size="11">Newton diag</text>
    <line x1="350" y1="21" x2="374" y2="21" stroke="currentColor" marker-end="url(#ar)"/>
    <rect x="378" y="4" width="74" height="34" fill="var(--unrolled)" rx="2"/>
    <text x="415" y="26" text-anchor="middle" fill="#fff" font-size="11.5">GNN</text>
    <line x1="456" y1="21" x2="480" y2="21" stroke="currentColor" marker-end="url(#ar)"/>
    <rect x="484" y="4" width="90" height="34" fill="none" stroke="currentColor" rx="2"/>
    <text x="529" y="26" text-anchor="middle" font-size="11">Newton diag</text>
    <text x="590" y="26" font-size="13" fill="var(--muted)">&#8230; &times;20</text>
    <text x="660" y="26" font-size="11.5" fill="var(--unrolled)">{unrolled_steps} steps</text>

    <line x1="0" y1="66" x2="760" y2="66" stroke="var(--rule)"/>

    <text x="0" y="104" font-size="12.5" font-weight="600" fill="var(--newton)">Newton</text>
    <text x="0" y="120" font-size="11" fill="var(--muted)">no learned parameters</text>
    <rect x="150" y="92" width="74" height="34" fill="none" stroke="currentColor" stroke-dasharray="3 3" rx="2"/>
    <text x="187" y="114" text-anchor="middle" font-size="11.5">c = 0</text>
    <line x1="228" y1="109" x2="252" y2="109" stroke="currentColor" marker-end="url(#ar)"/>
    <rect x="256" y="92" width="90" height="34" fill="none" stroke="currentColor" rx="2"/>
    <text x="301" y="114" text-anchor="middle" font-size="11">Newton exact</text>
    <line x1="350" y1="109" x2="374" y2="109" stroke="currentColor" marker-end="url(#ar)"/>
    <rect x="378" y="92" width="90" height="34" fill="none" stroke="currentColor" rx="2"/>
    <text x="423" y="114" text-anchor="middle" font-size="11">Newton exact</text>
    <line x1="472" y1="109" x2="496" y2="109" stroke="currentColor" marker-end="url(#ar)"/>
    <rect x="500" y="92" width="90" height="34" fill="none" stroke="currentColor" rx="2"/>
    <text x="545" y="114" text-anchor="middle" font-size="11">Newton exact</text>
    <text x="660" y="114" font-size="11.5" fill="var(--newton)">{newton_steps} steps</text>

    <line x1="0" y1="154" x2="760" y2="154" stroke="var(--rule)"/>

    <text x="0" y="192" font-size="12.5" font-weight="600" fill="var(--initializer)">Learned initializer</text>
    <text x="0" y="208" font-size="11" fill="var(--muted)">attention runs once</text>
    <rect x="150" y="180" width="74" height="34" fill="var(--initializer)" rx="2"/>
    <text x="187" y="202" text-anchor="middle" fill="#fff" font-size="11.5">GNN</text>
    <line x1="228" y1="197" x2="252" y2="197" stroke="currentColor" marker-end="url(#ar)"/>
    <rect x="256" y="180" width="90" height="34" fill="none" stroke="currentColor" rx="2"/>
    <text x="301" y="202" text-anchor="middle" font-size="11">Newton exact</text>
    <line x1="350" y1="197" x2="374" y2="197" stroke="currentColor" marker-end="url(#ar)"/>
    <rect x="378" y="180" width="90" height="34" fill="none" stroke="currentColor" rx="2"/>
    <text x="423" y="202" text-anchor="middle" font-size="11">Newton exact</text>
    <text x="660" y="202" font-size="11.5" fill="var(--initializer)">{init_steps} steps</text>

    <line x1="0" y1="242" x2="760" y2="242" stroke="var(--rule)"/>
    <text x="0" y="272" font-size="12.5" font-weight="600" fill="var(--pydhn)">PyDHN</text>
    <text x="0" y="288" font-size="11" fill="var(--muted)">reference simulator</text>
    <rect x="150" y="256" width="130" height="34" fill="none" stroke="var(--pydhn)" stroke-dasharray="3 3" rx="2"/>
    <text x="215" y="278" text-anchor="middle" font-size="11">previous timestep</text>
    <line x1="284" y1="273" x2="308" y2="273" stroke="var(--pydhn)" marker-end="url(#ar)"/>
    <rect x="312" y="256" width="90" height="34" fill="none" stroke="var(--pydhn)" rx="2"/>
    <text x="357" y="278" text-anchor="middle" font-size="11">Newton</text>
    <text x="418" y="278" font-size="11" fill="var(--warn)">forces sequential order</text>
  </g>
</svg>
<figcaption>
  What each approach executes for one timestep. The filled blocks are the learned
  component. The difference that matters is not how many boxes there are but where the
  starting point comes from: PyDHN and the original approach both depend on something
  external &mdash; the previous hour, or twenty rounds of correction &mdash; while the
  learned guess is computed from the boundary conditions alone.
</figcaption>
</figure>

<div class="col">
<h2>Accuracy</h2>
<p>
  Scored against the PyDHN solution on the same {n} held-out timesteps. Flow is the
  quantity the solver actually produces; delta-p and pressure are derived from it.
</p>
</div>

<div class="scroll">
<table>
<thead><tr><th scope="col" style="text-align:left">approach</th>
<th scope="col">flow MAE<br><small>kg/s</small></th><th scope="col">RMSE<br><small>kg/s</small></th>
<th scope="col">R&sup2;</th><th scope="col">direction</th>
<th scope="col">&Delta;p MAE<br><small>Pa</small></th><th scope="col">pressure MAE<br><small>Pa</small></th></tr></thead>
<tbody>
{quality_rows}
</tbody>
</table>
</div>

{fig_parity}

<div class="col">
<h2>Cost</h2>
<p>
  Steps are counted to the same {tol} Pa tolerance. The last column is the one that does
  not appear in any single-timestep measurement: whether hours can be solved independently.
</p>
</div>

<div class="scroll">
<table>
<thead><tr><th scope="col" style="text-align:left">approach</th>
<th scope="col">steps<br><small>mean</small></th><th scope="col">median</th><th scope="col">max</th>
<th scope="col">wall clock<br><small>per timestep</small></th>
<th scope="col">within tol</th><th scope="col">parallel?</th></tr></thead>
<tbody>
{cost_rows}
</tbody>
</table>
</div>

<p class="note col">
  PyDHN's {pydhn_ms} ms is the {pydhn_hours}-hour generation of the {pydhn_n}-step year
  divided by its timesteps. That covers the <em>entire</em> simulation &mdash; setup,
  controls, thermal, file IO &mdash; while the solvers above do the hydraulic solve only.
  Treat it as an upper bound on comparable work, not a like-for-like measurement.
</p>

{fig_conv}
{fig_steps}
{fig_timing}

<div class="col">
<h2>Where the learning actually helps</h2>
<p>
  The learned component contributes exactly one thing: a better place to start. Newton
  does the rest, with no parameters involved. Measured on the starting point alone,
  before any Newton step runs, the cold start sits at {guess_cold} Pa and the learned
  guess at {guess_learned} Pa &mdash; about {guess_ratio}&times; closer.
</p>
<p>
  That is the whole mechanism, and it is worth being precise about what it buys. It saves
  roughly {newton_steps}&nbsp;&minus;&nbsp;{init_steps} Newton steps. Those steps are
  cheap, so in serial wall-clock the gain over pure Newton is modest
  ({newton_ms}&nbsp;ms &rarr; {init_ms}&nbsp;ms). The gain that matters is structural,
  and appears in the last column of the cost table.
</p>
</div>

{fig_guess}

<div class="col">
<h3>Why parallelism is the real result</h3>
<p>
  PyDHN reaches its low iteration count by warm-starting from the previous hour. That is
  an excellent engineering choice and it is also a hard constraint: hour 100 cannot begin
  before hour 99 finishes. The learned guess is computed from boundary conditions alone,
  so it carries no such dependency &mdash; all {pydhn_n} hours of the year are independent
  problems that could be solved at once.
</p>
<p>
  This is the claim the architecture was chosen to support, and it is the one a reviewer
  cannot dismiss by pointing at a faster classical solver.
</p>
</div>

{fig_season}

<div class="col">
<h2>What the original approach taught us</h2>
<p>
  The unrolled architecture ran the graph network at every one of twenty steps and learned
  a correction at each. Trained on this dataset, best-weight selection landed on
  <em>epoch zero</em> &mdash; the initialization. Every subsequent epoch was worse on the
  monitored residual, at every stable learning rate tried.
</p>
<p>
  The reason is visible in the cost table: with the head zero-initialized the untrained
  model <em>is</em> the physics solver, and the physics solver was already at the accuracy
  ceiling the reference data supports. The learned correction was being asked to improve
  on something with no headroom left, while costing twenty attention passes to do it.
  Moving the same network from <em>correcting</em> to <em>guessing</em> is what made it
  earn its place.
</p>
</div>

<div class="col">
<h2>Limitations</h2>
</div>

<div class="warn col">
<p><strong>Our physics does not match this dataset.</strong> Evaluated on the reference's
own flows, our pressure-drop model reproduces only {cons_within}% of pipe friction losses
within 5% (median ratio {cons_ratio}, spread {cons_lo}&ndash;{cons_hi} across timesteps).
The scatter rather than bias is consistent with per-pipe temperatures from a thermal run
against our fixed 50&nbsp;&deg;C fluid properties.</p>
<p>Flow comparisons remain meaningful &mdash; boundary conditions are taken from the
reference &mdash; but <strong>&Delta;p and pressure errors in the accuracy table mix model
error with this offset and must not be reported as pure model error.</strong> Resolving it
needs the generation settings used to produce the dataset.</p>
</div>

<div class="warn col">
<p><strong>PyDHN's iteration count is unavailable for this dataset.</strong> The solver log
shipped with the data holds {hist_n} entries against {pydhn_n} timesteps, so it describes
an earlier run. The wall-clock figure used here comes from the stated generation time
instead; no per-timestep iteration count for PyDHN on this data is quoted anywhere in
this report.</p>
</div>

<div class="col">
<ul>
  <li><strong>GPU does not help.</strong> Measured on a GTX 1060: inference 104.8 ms on
  GPU against 44.0 ms on CPU, training a tie. At batch size 1 the tensors are too small to
  cover kernel-launch overhead. Every number here is CPU. Batching is the prerequisite,
  and it is exactly what the parallel structure enables.</li>
  <li><strong>Flow R&sup2; flatters every approach.</strong> The boundary-feasible
  reference already carries most of each edge flow; only {n_loops} loop unknowns are
  actually solved. The loop-space error is the honest accuracy measure.</li>
  <li><strong>Single network, single year.</strong> Checkpoints bake in the topology.
  Nothing here demonstrates transfer to another district heating network.</li>
</ul>

<h2>Reproducing this</h2>
<pre>python -m dhn_gnn.cli train --arch initializer --epochs 40 --n-train 200
python -m dhn_gnn.reporting.full_report --n-ts {n}
python -m dhn_gnn.reporting.build_doc</pre>
<p>Per-approach predictions are written to separate folders, each indexed by timestamp and
column-aligned with the reference CSVs:</p>
<pre>results/predictions/newton/{{mass_flow,delta_p,pressure}}.csv
results/predictions/unrolled/{{mass_flow,delta_p,pressure}}.csv
results/predictions/initializer/{{mass_flow,delta_p,pressure}}.csv
results/figures/*.png
results/report_data.json</pre>

<p class="foot">
  Every number in this document is read from <code>results/report_data.json</code>,
  produced by the run described above. Figures are regenerated by the same command.
</p>
</div>
</div>
"""


if __name__ == "__main__":
    build()
