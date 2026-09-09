"""Formal technical report for the solver comparison, in HTML and .docx.

Both formats are generated from the SAME results/sweep/sweep.json, so the two
documents cannot disagree with each other or with the run that produced them:

    python -m dhn_gnn.reporting.sweep --train --report    produce the data
    python -m dhn_gnn.reporting.sweep_doc                 produce the documents

Style is deliberately plain: numbered sections, ruled tables, one accent colour
used only for figure emphasis, black text on white. Figures are drawn once with
matplotlib and written as both SVG (embedded in the HTML) and PNG (placed in the
.docx), so the two documents show identical artwork.

Naming: runs are referenced by stable identifiers (N, U1, I3, ...) defined in
RUN_ID below, and the four formulations are named for what they compute. No run
is described as original, current, proposed or preferred -- the tables carry the
comparison and the reader draws the conclusion.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dhn_gnn import config

SRC = config.RESULTS_DIR / "sweep" / "sweep.json"
OUT = config.RESULTS_DIR / "sweep" / "document"
FIG = OUT / "figures"

# Stable identifiers used throughout the document, in presentation order.
RUN_ID = {
    ("newton", "cold"): "N",
    ("unrolled", "full"): "U1",
    ("unrolled", "diagonal"): "U2",
    ("initializer", "all"): "I1",
    ("initializer", "cycle0"): "I2",
    ("initializer", "cycle1"): "I3",
    ("initializer", "cycle2"): "I4",
    ("initializer", "cycle3"): "I5",
    ("predictor", "residual_all"): "P1",
    ("predictor", "residual_cycle1"): "P2",
    ("predictor", "astar_all"): "P3",
}

# Neutral formulation names: what the method computes, not its place in a history.
FAMILY = {
    "newton": "Loop-Newton solver",
    "unrolled": "Per-step correction solver",
    "initializer": "Learned-initialisation solver",
    "predictor": "Direct-prediction solver",
}

# What each run varies, for the catalogue table. Everything else is held fixed.
VARIED = {
    "N": ("--", "reference condition; no learned parameters"),
    "U1": ("newton_mode = full", "exact loop Jacobian in the base step"),
    "U2": ("newton_mode = diagonal", "diagonal approximation of the loop Jacobian"),
    "I1": ("node_scope = all", "embedding over every node"),
    "I2": ("node_scope = cycle, cycle_hops = 0", "embedding over loop-incident nodes"),
    "I3": ("node_scope = cycle, cycle_hops = 1", "loop-incident nodes plus one ring"),
    "I4": ("node_scope = cycle, cycle_hops = 2", "loop-incident nodes plus two rings"),
    "I5": ("node_scope = cycle, cycle_hops = 3", "loop-incident nodes plus three rings"),
    "P1": ("loss = residual, node_scope = all", "residual objective, full embedding"),
    "P2": ("loss = residual, node_scope = cycle, cycle_hops = 1",
           "residual objective, restricted embedding"),
    "P3": ("loss = a_star, node_scope = all", "regression objective, full embedding"),
}

ACCENT = "#1f4e79"
GREY = ["#111111", "#555555", "#888888", "#aaaaaa", "#c8c8c8"]

# One hue per run. Derived from the Okabe-Ito qualitative palette, which stays
# distinguishable under the common colour-vision deficiencies. Family is ALSO
# encoded by line style and marker, so the figures remain readable in greyscale.
COLORS = {
    "N":  "#000000",
    "U1": "#D55E00", "U2": "#E69F00",
    "I1": "#0072B2", "I2": "#56B4E9", "I3": "#009E73",
    "I4": "#117733", "I5": "#332288",
    "P1": "#CC79A7", "P2": "#AA4499", "P3": "#882255",
}
FAMILY_STYLE = {          # (linestyle, marker)
    "newton": ("-", "o"),
    "unrolled": ((0, (6, 2)), "s"),
    "initializer": ("-", "o"),
    "predictor": ((0, (1, 1.4)), "D"),
}


def _run_style(r):
    ls, mk = FAMILY_STYLE.get(r["arch"], ("-", "o"))
    return dict(linestyle=ls, marker=mk, color=COLORS.get(rid(r), GREY[2]))


def rid(r):
    return RUN_ID.get((r["arch"], r["run"]), f"{r['arch']}/{r['run']}")


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return FIG / f"{name}.svg"


# ----------------------------------------------------------------------- figures
def fig_convergence(d):
    """Figure 1: median loop residual against solver step, one line per run."""
    _style()
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    for r in d["rows"]:
        c = r["curve"]["median"]
        st = _run_style(r)
        # x is the number of steps TAKEN. The per-step correction solver does not
        # record its pre-step state, so its curve starts at 1 rather than 0.
        x = np.arange(len(c)) + (1 if r["arch"] == "unrolled" else 0)
        # A single-point series (the prediction runs) needs a visible marker.
        ax.plot(x, c, linewidth=1.5, markersize=5.0 if len(c) == 1 else 3.0,
                label=f"{rid(r)}  {r['label']}", **st)
    ax.axhline(d["tol"], color="#000000", linewidth=0.9, linestyle=(0, (6, 3)))
    ax.annotate(f"tolerance {d['tol']:g} Pa", xy=(0.99, d["tol"]),
                xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xlabel("solver steps taken")
    ax.set_ylabel(r"median  max$|B_{int}\,\varphi(\dot m)|$   (Pa)")
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.2,
              borderaxespad=0)
    return _save(fig, "fig1_convergence")


def fig_hops(d):
    """Figure 2: effect of the embedding scope on the learned-initialisation runs."""
    _style()
    order = ["I2", "I3", "I4", "I5"]
    by = {rid(r): r for r in d["rows"]}
    runs = [by[k] for k in order if k in by]
    if not runs:
        return None
    hops = [r["model_kwargs"].get("cycle_hops", 0) for r in runs]
    steps = [r["steps_mean"] for r in runs]
    guess = [r["guess_resid_median"] for r in runs]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(hops, steps, "-o", color="#0072B2", linewidth=1.7, markersize=5,
            label="solver steps to tolerance")
    ax.set_xlabel("cycle_hops  (rings of nodes beyond the loop-incident set)")
    ax.set_ylabel("mean solver steps", color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax.set_xticks(hops)
    # Zero-based: auto-scaling this axis turns a spread of a fifth of a step into a
    # full-panel slope, which would overstate the effect the text describes as small.
    ax.set_ylim(0, max(steps + [by["I1"]["steps_mean"] if "I1" in by else 0]) * 1.25)

    # The all-node run is not a point on this axis -- it is a different setting of
    # node_scope entirely -- so it is drawn as a reference line, not a fifth marker.
    if "I1" in by:
        ax.axhline(by["I1"]["steps_mean"], color=GREY[1], linewidth=0.9,
                   linestyle=(0, (5, 3)))
        ax.annotate(f"I1, node_scope = all ({by['I1']['steps_mean']:.2f})",
                    xy=(0.02, by["I1"]["steps_mean"]), xytext=(0, 4),
                    textcoords="offset points",
                    xycoords=("axes fraction", "data"), va="bottom", fontsize=7.5,
                    color=GREY[1])

    ax2 = ax.twinx()
    ax2.plot(hops, guess, "--s", color="#D55E00", linewidth=1.4, markersize=4.2,
             label="initial-guess residual")
    ax2.set_ylabel("median initial-guess residual (Pa)", color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax2.set_ylim(0, max(guess) * 1.35)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, fontsize=7.5)
    return _save(fig, "fig2_scope")


def fig_steps(d):
    """Figure 3: solver steps to tolerance for every run, against the reference."""
    _style()
    rows = list(d["rows"])
    labels = [f"{rid(r)}  {r['label']}" for r in rows]
    vals = [r["steps_mean"] if r["steps_mean"] is not None else np.nan for r in rows]
    cols = [COLORS.get(rid(r), GREY[2]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 1.3))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=cols, edgecolor="#333333", linewidth=0.5, height=0.62)
    for yi, v, r in zip(y, vals, rows):
        # A run that never reaches the tolerance has no step count; saying so in
        # words is clearer than a bar of unbounded length. A run that takes zero
        # steps by construction needs its success rate attached, or a zero-length
        # bar reads as the fastest configuration rather than one that rarely
        # converges at all.
        if np.isnan(v):
            txt = "did not reach tolerance"
        else:
            txt = f"{v:.2f}"
            if r["reached_tol_pct"] < 99.0:
                txt += f"   (reached on {r['reached_tol_pct']:.1f} % of timesteps)"
        ax.text((0 if np.isnan(v) else v) + 0.08, yi, txt, va="center", fontsize=7.2)
    ax.axvline(d["pydhn"]["steps_mean"], color="#000000", linewidth=0.9,
               linestyle=(0, (6, 3)))
    ax.annotate(f"reference R: {d['pydhn']['steps_mean']:.2f}",
                xy=(d["pydhn"]["steps_mean"], 0.995), xycoords=("data", "axes fraction"),
                ha="left", va="top", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("mean solver steps to reach the tolerance")
    ax.grid(axis="y", visible=False)
    return _save(fig, "fig3_steps")


def fig_schematic(d):
    """Figure 4: where computation happens in each formulation."""
    _style()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.set_xlim(0, 10.9)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    BW, BH = 2.0, 0.86

    def box(x, y, text, learned=False):
        ax.add_patch(FancyBboxPatch((x, y), BW, BH,
                                    boxstyle="round,pad=0.02,rounding_size=0.07",
                                    linewidth=0.9,
                                    edgecolor=ACCENT if learned else "#333333",
                                    facecolor="#eef2f6" if learned else "white"))
        ax.text(x + BW / 2, y + BH / 2, text, ha="center", va="center", fontsize=7.2)

    def arrow(x1, x2, y):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                     mutation_scale=7, linewidth=0.8, color="#333333"))

    def feedback(x_from, x_to, y_bot, note):
        """Rectangular return path drawn BELOW its own row, so it can never cross
        another row the way a wide arc does."""
        dy = y_bot - 0.40
        ax.plot([x_from, x_from], [y_bot, dy], color="#777777", linewidth=0.7)
        ax.plot([x_from, x_to], [dy, dy], color="#777777", linewidth=0.7)
        ax.add_patch(FancyArrowPatch((x_to, dy), (x_to, y_bot), arrowstyle="-|>",
                                     mutation_scale=7, linewidth=0.7, color="#777777"))
        ax.text((x_from + x_to) / 2, dy - 0.06, note, ha="center", va="top",
                fontsize=6.9, color="#555555")

    phys = "physics\n" + r"$\varphi,\ r=B\varphi$"
    newt = "Newton step\n" + r"$J^{-1}r$"
    # last field: index of the first block the feedback path encloses. For the
    # learned-initialisation solver the network runs ONCE and only the Newton step
    # repeats, so its loop must start at block 1, not 0.
    rows = [
        (8.1, "N", "Loop-Newton solver",
         [(phys, False), (newt, False)], "repeat until tolerance", 0),
        (5.95, "U", "Per-step correction solver",
         [(phys, False), ("attention\nnetwork", True),
          ("Newton step\n+ correction", False)], "repeat K times", 0),
        (3.6, "I", "Learned-initialisation solver",
         [("attention network\n(once)", True), (newt, False)],
         "repeat until tolerance", 1),
        (1.45, "P", "Direct-prediction solver",
         [("attention network\n(once)", True)], None, 0),
    ]

    for y, tag, name, blocks, note, loop_from in rows:
        ax.text(0.05, y + BH + 0.30, tag + "    " + name, fontsize=8.4, va="center")
        ax.text(0.10, y + BH / 2, r"$\dot m_0$", fontsize=8.2, va="center")
        x = 0.82
        arrow(x, x + 0.45, y + BH / 2)
        x += 0.50
        loop_x = x
        for i_b, (label, learned) in enumerate(blocks):
            if i_b == loop_from:
                loop_x = x
            box(x, y, label, learned)
            x += BW
            if i_b < len(blocks) - 1:
                arrow(x, x + 0.45, y + BH / 2)
                x += 0.50
        arrow(x, x + 0.45, y + BH / 2)
        ax.text(x + 0.58, y + BH / 2, r"$\dot m$", fontsize=8.2, va="center")
        if note:
            feedback(x - 0.55, loop_x + 0.55, y, note)

    ax.add_patch(FancyBboxPatch((6.55, 0.02), 4.1, 0.56,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=0.7, edgecolor="#aaaaaa", facecolor="white"))
    ax.add_patch(FancyBboxPatch((6.72, 0.16), 0.26, 0.26, boxstyle="square,pad=0",
                                linewidth=0.9, edgecolor=ACCENT, facecolor="#eef2f6"))
    ax.text(7.08, 0.29, "learned parameters", fontsize=7, va="center")
    ax.add_patch(FancyBboxPatch((9.20, 0.16), 0.26, 0.26, boxstyle="square,pad=0",
                                linewidth=0.9, edgecolor="#333333", facecolor="white"))
    ax.text(9.56, 0.29, "exact physics", fontsize=7, va="center")
    return _save(fig, "fig4_schematic")


def make_figures(d):
    return {"fig1": fig_convergence(d), "fig2": fig_hops(d),
            "fig3": fig_steps(d), "fig4": fig_schematic(d)}


# ----------------------------------------------------------------- content model
# Both renderers walk the SAME block list, which is why the HTML and the .docx
# cannot drift apart. Block kinds:
#   ("h1"|"h2"|"h3", text)      ("p", text)          ("ul", [items])
#   ("table", caption, headers, rows)                ("fig", key, caption)
#   ("pre", text)               ("note", text)

def _by_id(d):
    return {rid(r): r for r in d["rows"]}


def _n(v, spec=".2f", dash="not reached"):
    return dash if v is None else format(v, spec)


def _fmt_secs(s):
    if s is None:
        return "--"
    return f"{s/60:.1f} min" if s >= 60 else f"{s:.0f} s"


def content(d):
    from dhn_gnn.reporting.sweep import SWEEP as SWEEP_REF
    R = _by_id(d)
    env = d.get("environment", {})
    tol = d["tol"]
    py = d["pydhn"]
    B = []
    A = B.append

    # ------------------------------------------------------------------ 1
    A(("h1", "1  Scope"))
    A(("p",
       "This report compares eleven solver configurations for the steady-state "
       "hydraulic problem on a single district heating network. One configuration "
       "uses no learned parameters; ten are trained. All are evaluated on the same "
       f"{d['n_timesteps']} held-out timesteps, at the same {tol:g} Pa convergence "
       "tolerance, on the same machine, so that the only differences between them "
       "are the ones under study."))
    A(("p",
       "The configurations differ along three axes, examined in Sections 7.2 to 7.4: "
       "the Jacobian used by the base solver step; the extent of the graph over which "
       "the learned embedding is computed; and the training objective. Section 6 "
       "states exactly how every reported quantity was obtained."))

    # ------------------------------------------------------------------ 2
    A(("h1", "2  Problem formulation"))
    A(("p",
       "The network is solved by the loop method. Let A be the node-edge incidence "
       "matrix, B the cycle matrix, and phi the Darcy-Weisbach pressure-drop map from "
       "edge mass flow to edge pressure drop. The unknown is the vector of independent "
       "loop flows m_loop, and the governing equation is the loop-closure condition"))
    A(("pre", "    r(m_loop)  =  B_int . phi( B^T . m_loop )  =  0"))
    A(("p",
       f"where B_int selects the {d['n_loops']} internal loops. Edge flows are "
       "recovered as mdot = mdot_0 + Z c, with Z = B_int^T and c the loop-space "
       "correction. Because A Z = 0, every configuration in this report satisfies "
       "nodal mass conservation exactly and by construction, whatever c it produces; "
       "no configuration can trade mass balance for a lower residual."))
    A(("p",
       f"The problem is therefore {d['n_loops']}-dimensional, not "
       f"{d['n_edges']}-dimensional: the network has {d['n_nodes']} nodes and "
       f"{d['n_edges']} edges, but only {d['n_loops']} degrees of freedom remain once "
       "the boundary conditions and mass conservation are imposed. This is central to "
       "interpreting the results, and is returned to in Section 8."))

    # ------------------------------------------------------------------ 3
    A(("h1", "3  Experimental setup"))
    A(("h2", "3.1  Network and data"))
    A(("table", "Table 1. Network and dataset.",
       ["Property", "Value"],
       [["Nodes", f"{d['n_nodes']:,}"],
        ["Edges", f"{d['n_edges']:,}"],
        ["Independent loops (unknowns)", f"{d['n_loops']}"],
        ["Dataset", d["dataset"]],
        ["Reference solver", "PyDHN, thermohydraulic steady state, hourly"],
        ["Evaluation window", f"{d['window'][0]} to {d['window'][1]}"],
        ["Held-out timesteps scored", f"{d['n_timesteps']}"],
        ["Convergence tolerance", f"{tol:g} Pa"]]))
    A(("p",
       "The reference solution is produced by the PyDHN simulator and is treated as "
       "ground truth throughout. Its own solver cost is reported in Section 6.8."))

    A(("h2", "3.2  Data split"))
    A(("p",
       f"Timesteps are split by index: every {d['split_every']}th timestep is held "
       "out for evaluation and never seen during training. The split is interleaved "
       "rather than contiguous because the dataset is driven by a year of real "
       "weather; a contiguous tail would constitute a different operating regime, and "
       "the resulting score would measure distribution shift rather than "
       "generalisation."))
    A(("p",
       "Training draws a fixed subsample from the training partition, identical for "
       "every configuration. The exact timestep indices used for fitting are recorded "
       "inside each checkpoint, so a score can always be checked against the data the "
       "model actually saw."))

    A(("h2", "3.3  Configuration held constant"))
    A(("p", "Every trained configuration in this report was fitted with:"))
    A(("pre", "    " + d["common"].replace("--set ", "")))
    A(("p",
       "Optimiser Adam, gradient-norm clipping at 1.0, single seed. Architecture "
       "width, head count and attention depth are identical across all trained "
       "configurations (Section 5), so parameter count is constant within a family "
       "and differences in the tables are not capacity effects."))

    A(("h2", "3.4  Computing environment"))
    A(("table", "Table 2. Software and hardware.",
       ["Component", "Version"],
       [["Python", env.get("python", "--")],
        ["PyTorch", env.get("torch", "--")],
        ["NumPy", env.get("numpy", "--")],
        ["pandas", env.get("pandas", "--")],
        ["Platform", env.get("platform", "--")],
        ["Processor", env.get("processor", "--")],
        ["Compute device", env.get("device", "cpu")],
        ["Torch threads", str(env.get("threads", "--"))]]))
    A(("p",
       "All timings in this report were measured on this configuration, single "
       "process, batch size one, CPU. They are comparable with each other and should "
       "not be read as absolute performance figures."))

    # ------------------------------------------------------------------ 4
    A(("h1", "4  Solver formulations"))
    A(("p",
       "Four formulations are evaluated. They differ in where computation happens: "
       "whether a learned component is evaluated once, once per solver step, or "
       "without any exact step following it."))
    A(("fig", "fig4", "Figure 4. Structure of the four formulations. Shaded blocks "
                      "contain learned parameters; unshaded blocks are exact physics."))

    A(("h2", "4.1  Loop-Newton solver (N)"))
    A(("p",
       "No learned parameters. Starting from c = 0, each step evaluates the physics, "
       "forms the loop residual r, and applies the Newton correction obtained by "
       "solving J dc = r, where J = B_int diag(dphi/dmdot) B_int^T. On this network J "
       f"is {d['n_loops']} x {d['n_loops']}, symmetric positive semi-definite, and is "
       "solved directly with a small ridge term to cover loops carrying near-zero "
       "flow. This configuration is the control condition: it isolates what the "
       "learned components add."))

    A(("h2", "4.2  Per-step correction solver (U)"))
    A(("p",
       "An attention network is evaluated at every one of K steps. Each step "
       "re-evaluates the physics, runs the attention stack over the graph, and applies"))
    A(("pre", "    dc = -damping * newton_step + step_scale * tanh( head_k(...) )"))
    A(("p",
       "with one output head per step, zero-initialised. The zero initialisation is "
       "material to interpreting the results: an untrained model of this form reduces "
       "exactly to the Loop-Newton solver, so training can only be credited with the "
       "difference from that starting point. Gradients are truncated between steps; "
       "back-propagating through all K applications of the quadratic loop operator is "
       "numerically unstable."))

    A(("h2", "4.3  Learned-initialisation solver (I)"))
    A(("p",
       "The attention network is evaluated once, producing a loop-space starting "
       "point c_0 in place of the zero vector; exact Newton steps then proceed as in "
       "Section 4.1. The output layer is zero-initialised, so an untrained model of "
       "this form again reduces to the Loop-Newton solver. Training is plain "
       "regression onto a_star = (Z^T Z)^-1 Z^T mdot_true, the loop-space coordinate "
       "of the reference solution, with no gradient passing through the solver."))

    A(("h2", "4.4  Direct-prediction solver (P)"))
    A(("p",
       "Architecturally identical to Section 4.3 with the number of Newton steps set "
       "to zero: the single network evaluation is the answer. Two training objectives "
       "are compared, the physics residual and the a_star regression of Section 4.3. "
       "Because no exact step follows the prediction, the residual it leaves is its "
       "final error, and the step count is not a meaningful cost measure for it "
       "(Section 6.2)."))

    # ------------------------------------------------------------------ 5
    A(("h1", "5  Run catalogue"))
    A(("p",
       "Ten configurations were trained. Each varies exactly one aspect of its "
       "formulation; everything else is held at the values in Section 3.3. The "
       "identifiers below are used throughout the remainder of the report."))
    cat = []
    for key in ["N", "U1", "U2", "I1", "I2", "I3", "I4", "I5", "P1", "P2", "P3"]:
        r = R.get(key)
        if r is None:
            continue
        varied, purpose = VARIED.get(key, ("--", "--"))
        cat.append([key, FAMILY.get(r["arch"], r["arch"]), varied, purpose])
    A(("table", "Table 3. Configurations evaluated, and the parameter each varies.",
       ["ID", "Formulation", "Parameter varied", "What the run isolates"], cat))

    A(("h2", "5.1  Parameter values"))
    A(("p",
       "The architecture arguments below are stored inside each checkpoint and were "
       "read back from it for this table, rather than restated from the launch "
       "command; a checkpoint and its description therefore cannot disagree."))
    rows5 = []
    for key in ["N", "U1", "U2", "I1", "I2", "I3", "I4", "I5", "P1", "P2", "P3"]:
        r = R.get(key)
        if r is None:
            continue
        mk = r.get("model_kwargs", {})
        arch_desc = ", ".join(f"{k} = {v}" for k, v in mk.items())
        rows5.append([key, arch_desc or "--",
                      str(r.get("epochs") or "--"),
                      str(r.get("n_train") or "--"),
                      str(r.get("lr") or "--"),
                      str(r.get("best_epoch") if r.get("best_epoch") is not None else "--"),
                      _fmt_secs(r.get("train_seconds")),
                      f"{r['n_params']:,}"])
    A(("table", "Table 4. Full parameter set and training outcome for every run. "
                "'Best epoch' is the epoch whose weights were retained; 0 means no "
                "epoch improved on the initialisation and the initial weights were "
                "kept.",
       ["ID", "Architecture arguments", "Epochs", "Train ts", "LR",
        "Best epoch", "Train time", "Parameters"], rows5))

    A(("h2", "5.2  Embedding scope actually realised"))
    A(("p",
       "The node_scope and cycle_hops settings change how much of the graph the "
       "attention stack operates on. The counts below were measured from the "
       "constructed models, not derived from the settings, and are the quantities "
       "that make Section 7.4 interpretable."))
    rows_sc = []
    for key in ["I1", "I2", "I3", "I4", "I5", "P1", "P2", "P3"]:
        r = R.get(key)
        if r is None or not r.get("scope"):
            continue
        sc = r["scope"]
        rows_sc.append([key,
                        f"{sc['nodes']:,} / {sc['nodes_total']:,}",
                        f"{100*sc['nodes']/sc['nodes_total']:.1f} %",
                        f"{sc['attn_edges']:,} / {sc['attn_edges_total']:,}",
                        f"{100*sc['attn_edges']/sc['attn_edges_total']:.1f} %"])
    if rows_sc:
        A(("table", "Table 5. Nodes and directed attention edges retained by each "
                    "embedding scope.",
           ["ID", "Nodes", "Share", "Attention edges", "Share"], rows_sc))

    # ------------------------------------------------------------------ 6
    A(("h1", "6  Measurement methodology"))
    A(("p",
       "This section defines every quantity reported in Section 7. The definitions "
       "are not uniform across formulations, and treating them as if they were is the "
       "most likely way to misread the results."))

    A(("h2", "6.1  Scoring protocol"))
    A(("p",
       f"Each configuration is run over the same {d['n_timesteps']} held-out "
       "timesteps, drawn at even spacing from the held-out partition. For each "
       "timestep the solver is given the boundary-feasible reference flow mdot_0 and "
       f"run with early exit at {tol:g} Pa. Model weights are loaded from the "
       "checkpoint together with the architecture arguments they were saved with, so "
       "no configuration is rebuilt from assumed settings. Evaluation is under "
       "torch.no_grad in eval mode."))

    A(("h2", "6.2  Definition of a solver step"))
    A(("p",
       "A solver step is one application of the update rule. The reported figure is "
       "the number of steps taken before the residual first falls below the "
       "tolerance. Two adjustments are required for this to be comparable:"))
    A(("ul", [
        "The Loop-Newton, learned-initialisation and direct-prediction formulations "
        "record their state BEFORE any step is applied, so entry i of their residual "
        "history is the state after i steps and the count is i.",
        "The per-step correction formulation records only post-step states, so entry "
        "i is the state after i+1 steps and the count is i+1. Applying the same "
        "indexing to both families would shift the headline figure by one step, in "
        "opposite directions.",
        "The direct-prediction formulation performs no solver step at all. It is "
        "recorded as zero steps where its prediction already satisfies the tolerance. "
        "Its step count is therefore not a cost comparison, and the proportion of "
        "timesteps within tolerance is the column to read for it.",
    ]))
    A(("p",
       "A configuration that never reaches the tolerance on a given timestep "
       "contributes no step count. The mean and median are taken over the timesteps "
       "that did converge, and the proportion that converged is reported separately "
       "as 'reached'. Averaging a censored value into the mean would otherwise make a "
       "configuration look better the more often it failed."))

    A(("h2", "6.3  Residual"))
    A(("p",
       "The residual of a state is the largest absolute loop-closure violation,"))
    A(("pre", "    residual  =  max_i | ( B_int . phi(mdot) )_i |     [Pa]"))
    A(("p",
       "taken over the internal loops. It is zero exactly at the solution. The tables "
       "report its median and mean over the evaluated timesteps, and the proportion "
       "of timesteps whose final residual is within tolerance."))
    A(("note",
       "With early exit enabled, the final residual is NOT a quality ranking between "
       "configurations: each stops as soon as it crosses the tolerance, so a "
       "configuration that converges faster stops sooner and reports a larger final "
       "residual. Final residual is meaningful for configurations that do not reach "
       "the tolerance, and for the direct-prediction formulation, which never exits "
       "early because it takes no steps."))

    A(("h2", "6.4  Initial-guess residual"))
    A(("p",
       "For the formulations that record a pre-step state, the residual of that state "
       "is reported separately. It measures the quality of the starting point alone, "
       "independently of how much work the subsequent exact steps then do. It is not "
       "reported for the per-step correction formulation, which has no such state; a "
       "value there would be a post-step residual and would be compared against the "
       "others as though it were not."))

    A(("h2", "6.5  Accuracy against the reference"))
    A(("p",
       "Convergence cost says nothing about correctness, so edge mass flows are also "
       "compared against the PyDHN solution on the same timesteps. Reported are mean "
       "absolute error, root-mean-square error, maximum absolute error and the "
       "coefficient of determination over all edges, plus the proportion of edges "
       "whose flow direction agrees, with a 0.01 kg/s dead band so that near-zero "
       "flows do not dominate the direction statistic."))

    A(("h2", "6.6  Timing"))
    A(("p",
       "Wall-clock time is measured around the solve loop only, over the whole "
       "evaluated set, and divided by the number of timesteps. Model construction, "
       "checkpoint loading, sample preparation and the pressure reconstruction used "
       "for reporting are all outside the timed region. Batch size is one and the "
       "device is CPU, so these figures measure per-timestep sequential cost; they do "
       "not reflect the throughput achievable by batching independent timesteps, "
       "which Section 7.6 discusses separately."))

    A(("h2", "6.7  Parameter count"))
    A(("p",
       "The number of learned parameters is counted directly from the constructed "
       "model. The Loop-Newton configuration has none. Restricting the embedding "
       "scope does not change the parameter count, only the number of nodes and edges "
       "over which the same parameters are applied."))

    A(("h2", "6.8  Reference solver cost"))
    A(("p",
       f"The reference figure of {py['steps_mean']:.2f} hydraulic iterations per "
       f"timestep is the mean over the {py['n_steps']:,} entries of the dataset's own "
       "solver history, recorded when the ground truth was generated. Two properties "
       "of that figure must be carried with it:"))
    A(("ul", [
        "It is warm-started. The reference solver initialises each hour from the "
        "solution of the previous hour, so it must process the year in order and "
        "cannot be parallelised across timesteps.",
        "It counts iterations of a different solver on a different formulation. It "
        "establishes the order of magnitude of the classical cost; it is not a "
        "like-for-like step count.",
    ]))
    if py.get("stale"):
        A(("note",
           "The solver history does not match the dataset length. The reference "
           "figure above is therefore derived from a different run than the one "
           "scored here and should not be quoted."))

    A(("h2", "6.9  Residual of the reference solution"))
    ref_res = py.get("resid_median")
    if ref_res is not None:
        A(("p",
           "The reference solution is not run by this experiment; its stored results "
           "are read as ground truth. To place the residual column on a common "
           "footing, the operator of Section 6.3 was applied to the reference's own "
           f"edge flows on the same timesteps. Its median value is {ref_res:.1f} Pa, "
           f"with a 95th percentile of {py.get('resid_p95', float('nan')):.1f} Pa."))
        A(("note",
           f"This has a direct consequence for how Table 6 should be read. At "
           f"{ref_res:.1f} Pa the reference solution sits at the "
           f"{tol:g} Pa acceptance tolerance itself, so residuals at or below that "
           "level are at the noise floor of the comparison and should not be used to "
           "rank configurations against one another. Two contributions are folded "
           "into the figure and cannot be separated from these data alone: the "
           "reference solver stopped at its own convergence threshold, so its "
           "solution is not exactly loop-consistent; and the operator used here "
           "evaluates fluid properties at a single fixed temperature whereas the "
           "reference was produced by a thermohydraulic simulation in which each pipe "
           "carries its own temperature (Section 8). What Table 6 supports is the "
           "distinction between configurations that reach the tolerance and those "
           "that do not, and the number of steps they take to do so."))

    # ------------------------------------------------------------------ 7
    A(("h1", "7  Results"))

    A(("h2", "7.1  Summary"))
    main = []
    for key in ["N", "U1", "U2", "I1", "I2", "I3", "I4", "I5", "P1", "P2", "P3"]:
        r = R.get(key)
        if r is None:
            continue
        main.append([
            key, r["label"],
            _n(r["steps_mean"], ".2f"),
            f"{r['reached_tol_pct']:.1f} %",
            f"{r['resid_median']:.1f}",
            f"{r['under_tol_pct']:.1f} %",
            _n(r.get("guess_resid_median"), ".1f", "n/a"),
            f"{r['ms_per_ts']:.1f}",
        ])
    ref_res = py.get("resid_median")
    main.append(["R", "PyDHN reference (warm-started)",
                 f"{py['steps_mean']:.2f}", "100.0 %",
                 f"{ref_res:.1f}" if ref_res is not None else "--",
                 "100.0 %", "n/a", "--"])
    A(("table",
       "Table 6. Primary comparison. 'Steps' is the mean number of solver steps to "
       "reach the tolerance, over the timesteps that reached it (Section 6.2); "
       "'reached' is the proportion that did. 'Residual' is the median final residual "
       "in Pa and is subject to the early-exit caveat of Section 6.3. 'Guess' is the "
       "residual of the starting point (Section 6.4). The residual quoted for the "
       "reference is not its own solver's convergence measure but the same operator "
       "applied to its solution, and is discussed in Section 6.9.",
       ["ID", "Configuration", "Steps", "Reached", "Residual (Pa)",
        "Within tol.", "Guess (Pa)", "ms / timestep"], main))

    A(("h2", "7.2  Convergence behaviour"))
    A(("fig", "fig1", "Figure 1. Median loop residual against the number of solver "
                      "steps taken. The horizontal line marks the convergence "
                      "tolerance. The per-step correction curves begin at one step "
                      "because that formulation records no pre-step state "
                      "(Section 6.2)."))

    A(("h2", "7.3  Effect of the Jacobian in the base step"))
    u1, u2 = R.get("U1"), R.get("U2")
    if u1 and u2:
        A(("p",
           f"Configurations U1 and U2 differ only in newton_mode. With the exact loop "
           f"Jacobian, U1 reaches the tolerance on {u1['reached_tol_pct']:.1f} % of "
           f"timesteps in {_n(u1['steps_mean'])} steps. With the diagonal "
           f"approximation, U2 reaches it on {u2['reached_tol_pct']:.1f} % of "
           f"timesteps, with a median final residual of {u2['resid_median']:.1f} Pa "
           f"against a {tol:g} Pa tolerance."))
        A(("note",
           f"U2's step figure is the mean over the "
           f"{u2['reached_tol_pct']*d['n_timesteps']/100:.0f} of "
           f"{d['n_timesteps']} timesteps that reached the tolerance at all, and is "
           "therefore not a cost comparable with the other rows; the column to read "
           "for this configuration is the proportion reaching tolerance."))
        A(("p",
           "The diagonal approximation discards the off-diagonal coupling between "
           "loops. The resulting direction is only approximately correct, so the "
           "iteration converges linearly rather than quadratically and requires "
           "damping; within the step budget used here it does not reach the "
           "tolerance. The exact Jacobian is inexpensive at this problem size "
           f"({d['n_loops']} x {d['n_loops']}), which is why the two configurations "
           "differ so much more in accuracy than in cost per step."))
    n_row = R.get("N")
    if u1 and n_row and u1.get("best_epoch") == 0:
        A(("note",
           "U1 retained its initial weights: no training epoch improved on the "
           "monitored residual. Because the output heads are zero-initialised "
           "(Section 4.2), that initialisation is exactly the Loop-Newton solver, and "
           "U1's figures in Table 6 consequently coincide with configuration N. The "
           "learned per-step correction contributes nothing measurable here."))

    A(("h2", "7.4  Effect of the embedding scope"))
    A(("fig", "fig2", "Figure 2. Solver steps and initial-guess residual against "
                      "the number of node rings included beyond the loop-incident "
                      "set, for the learned-initialisation configurations. The dashed "
                      "line is configuration I1, which embeds every node. Note the "
                      "narrow range of the left-hand axis."))
    cand = [R[k] for k in ["I1", "I2", "I3", "I4", "I5"]
            if k in R and R[k]["steps_mean"] is not None]
    if cand:
        lo = min(cand, key=lambda r: r["steps_mean"])
        hi = max(cand, key=lambda r: r["steps_mean"])
        spread = hi["steps_mean"] - lo["steps_mean"]
        A(("p",
           f"All five configurations lie between {lo['steps_mean']:.2f} and "
           f"{hi['steps_mean']:.2f} steps, a spread of {spread:.2f} steps, and all "
           f"reach the tolerance on {min(r['reached_tol_pct'] for r in cand):.1f} % "
           f"to {max(r['reached_tol_pct'] for r in cand):.1f} % of timesteps. The "
           "differences between them are small relative to that range."))
        A(("note",
           "An earlier sweep of the same five configurations, trained on 100 rather "
           "than 200 timesteps, produced a spread of 0.79 steps with an interior "
           "minimum at one ring. That ordering does not reproduce here, where the "
           "spread is "
           f"{spread:.2f} steps and the ordering differs. With a single seed per "
           "configuration, the separation between these five is not large enough to "
           "support a claim that any particular scope is preferable on convergence "
           "grounds. The comparison that does survive both sweeps is between the "
           "family as a whole and the configurations in Sections 7.3 and 7.5."))
        sc_lo = lo.get("scope") or {}
        sc_all = (R.get("I1") or {}).get("scope") or {}
        if sc_lo and sc_all:
            A(("p",
               "The practical consequence is therefore one of cost rather than "
               f"accuracy. Configuration {rid(lo)} applies the same "
               f"{lo['n_params']:,} parameters over {sc_lo['attn_edges']:,} directed "
               f"attention edges instead of the {sc_all['attn_edges']:,} used by the "
               "all-node configuration, that is "
               f"{100*sc_lo['attn_edges']/sc_all['attn_edges']:.0f} % of them, without "
               "a measurable penalty in solver steps. Restricting the embedding to the "
               "neighbourhood of the loops is available at no observed cost; it is not "
               "demonstrated to be an improvement."))

    A(("h2", "7.5  Effect of the training objective without exact steps"))
    p1, p2, p3 = R.get("P1"), R.get("P2"), R.get("P3")
    ps = [p for p in (p1, p2, p3) if p]
    if p1 and p3:
        A(("p",
           "Configurations P1 and P3 differ only in the training objective. Measured "
           "by the proportion of timesteps whose prediction satisfies the tolerance, "
           f"the residual objective reaches {p1['under_tol_pct']:.1f} % and the "
           f"a_star regression {p3['under_tol_pct']:.1f} %; the median residuals are "
           f"{p1['resid_median']:.1f} Pa and {p3['resid_median']:.1f} Pa "
           "respectively. The regression objective minimises an error in kg/s while "
           "the acceptance criterion is stated in Pa, and the two are not "
           "monotonically related, so a configuration can reduce its training loss "
           "while its residual increases."))
    if p2 and p1:
        A(("p",
           "Restricting the embedding scope does not help this formulation: P2 "
           f"leaves {p2['resid_median']:.1f} Pa against P1's "
           f"{p1['resid_median']:.1f} Pa under the same objective, the opposite of "
           "the direction seen in an earlier sweep at half the training set size. As "
           "in Section 7.4, the scope setting is not separable from run-to-run "
           "variation at a single seed."))
    if ps:
        best_p = min(ps, key=lambda r: r["resid_median"])
        A(("p",
           f"The strongest configuration of this family, {rid(best_p)}, leaves a "
           f"median residual of {best_p['resid_median']:.1f} Pa against a {tol:g} Pa "
           f"tolerance and satisfies it on {best_p['under_tol_pct']:.1f} % of "
           f"timesteps, short by a factor of {best_p['resid_median']/tol:.1f}. Near "
           "the solution the loop residual is approximately linear in the loop-flow "
           "error, so closing that gap requires a proportionate improvement in the "
           "prediction itself. Within the configurations and training budget tested, "
           "prediction without subsequent exact steps does not meet the criterion, "
           "while the same network followed by exact steps (Section 7.4) meets it on "
           f"{max(r['under_tol_pct'] for r in cand) if cand else 0:.0f} % of "
           "timesteps."))

    A(("h2", "7.6  Cost"))
    A(("fig", "fig3", "Figure 3. Mean solver steps to reach the tolerance. The dashed "
                      "line marks the reference solver's iteration count, which is "
                      "warm-started and therefore sequential (Section 6.8)."))
    A(("p",
       "The per-timestep times in Table 6 are sequential, batch-size-one measurements "
       "(Section 6.6). They understate the difference available in practice between "
       "the formulations, because the learned-initialisation and direct-prediction "
       "configurations require no information from the preceding timestep and can "
       "therefore be evaluated concurrently across the whole year, whereas the "
       "warm-started reference cannot."))

    A(("h2", "7.7  Accuracy against the reference solution"))
    acc = []
    for key in ["N", "U1", "U2", "I1", "I2", "I3", "I4", "I5", "P1", "P2", "P3"]:
        r = R.get(key)
        if r is None or not r.get("accuracy"):
            continue
        a = r["accuracy"]
        acc.append([key, r["label"],
                    f"{a['flow_mae']:.3e}", f"{a['flow_rmse']:.3e}",
                    f"{a['flow_max']:.3e}", f"{a['flow_r2']:.6f}",
                    f"{a['dir_acc']:.2f} %"])
    if acc:
        A(("table",
           "Table 7. Edge mass-flow error against the PyDHN solution on the same "
           "held-out timesteps, in kg/s (Section 6.5).",
           ["ID", "Configuration", "MAE", "RMSE", "Max error", "R2", "Direction"],
           acc))

    # ------------------------------------------------------------------ 8
    A(("h1", "8  Limitations"))
    A(("ul", [
        f"The network has {d['n_loops']} independent loops out of {d['n_edges']} "
        "edges, so it is close to radial. The classical solver is already inexpensive "
        "on a problem this small, which bounds what any method can demonstrate here. "
        "These results do not extrapolate to large meshed networks without retesting.",
        "Each configuration was trained once, with a single seed, and no variance "
        "estimate is available. This is not a hypothetical concern here: an earlier "
        "sweep of the same eleven configurations at half the training-set size "
        "produced a different ordering within the learned-initialisation family, with "
        "a spread of 0.79 steps against 0.20 steps in the present run. Differences "
        "within a family of the order of a few tenths of a step are therefore not "
        "resolvable by this experiment, and Sections 7.4 and 7.5 are written "
        "accordingly. Comparisons BETWEEN families, which differ by whole steps or by "
        "whether the tolerance is reached at all, are unaffected.",
        "The training budget was fixed in advance rather than tuned per "
        "configuration, and several runs retained the weights of their final epoch, "
        "indicating that they were still improving when the budget ended. Those "
        "configurations are compared at equal budget, not at convergence; the epoch "
        "retained for each run is given in Table 4.",
        "The residual of the reference solution under the operator used here is "
        "comparable to the acceptance tolerance itself (Section 6.9). Differences in "
        "final residual below that level are therefore not meaningful, and the "
        "configurations that converge are separated in this report by step count and "
        "by the proportion of timesteps reaching tolerance, not by residual.",
        "The dataset was produced by a thermohydraulic simulation in which each pipe "
        "carries its own temperature, whereas the pressure-drop model used here "
        "evaluates density and viscosity at a single fixed temperature. Mass-flow "
        "results, which are what Tables 6 and 7 report, are unaffected in the sense "
        "that all configurations share the same model; comparisons of pressure drop "
        "against the reference carry a systematic offset from this mismatch.",
        "Timings are single-process CPU measurements at batch size one and should be "
        "read as relative, not absolute.",
        "The reference iteration count is warm-started and comes from a different "
        "solver formulation; it establishes scale rather than a like-for-like "
        "comparison (Section 6.8).",
    ]))

    # ------------------------------------------------------------------ 9
    A(("h1", "9  Reproduction"))
    A(("p",
       "The sweep is defined as data in dhn_gnn/reporting/sweep.py, and the commands "
       "below regenerate every number and figure in this report. Each run writes its "
       "resolved configuration next to its weights, so an individual run can also be "
       "repeated from its own output directory."))
    A(("pre",
       "python -m dhn_gnn.reporting.sweep --train --report\n"
       "python -m dhn_gnn.reporting.sweep_doc\n"
       "\n"
       "# a single configuration, reproduced from its stored recipe\n"
       "python -m dhn_gnn.cli train --arch initializer --run cycle1 \\\n"
       "       --config results/initializer/cycle1/config.yaml"))
    A(("p",
       "Outputs: results/sweep/sweep.json holds every measurement; "
       "results/sweep/predictions/ holds per-configuration edge flows on the "
       "evaluated timesteps together with the reference; results/sweep/document/ "
       "holds this report and its figures."))

    A(("h1", "Appendix A  Launch commands"))
    A(("p",
       "The exact command issued for each configuration. Arguments common to all runs "
       "are listed once in Section 3.3 and are omitted here."))
    cmds = []
    for arch, run, over, _lab in SWEEP_REF:
        key = RUN_ID.get((arch, run), f"{arch}/{run}")
        extra = " ".join(over).replace("--set ", "")
        cmds.append([key,
                     f"python -m dhn_gnn.cli train --arch {arch} --run {run}",
                     extra or "--"])
    A(("table", "Table A1. Launch commands.",
       ["ID", "Command", "Additional arguments"], cmds))
    return B


# --------------------------------------------------------------------- rendering
TITLE = "Comparison of Learned and Classical Solvers for District Heating Hydraulics"

CSS = """
@page { size: A4; margin: 22mm 18mm; }
:root { --ink:#111; --mut:#555; --rule:#333; --hair:#bbb; --accent:#1f4e79; }
* { box-sizing: border-box; }
body { margin:0; background:#fff; color:var(--ink);
  font-family: Georgia,"Times New Roman",serif; font-size:10.5pt; line-height:1.5; }
main { max-width: 178mm; margin: 0 auto; padding: 14mm 8mm 20mm; }
h1,h2,h3 { font-weight:600; line-height:1.25; }
h1 { font-size:14pt; margin:26px 0 8px; padding-bottom:4px;
     border-bottom:1px solid var(--rule); }
h2 { font-size:11.5pt; margin:18px 0 5px; }
p { margin:0 0 9px; text-align:justify; hyphens:auto; }
ul { margin:0 0 10px; padding-left:18px; } li { margin:3px 0; text-align:justify; }
pre { font-family:"DejaVu Sans Mono",Consolas,monospace; font-size:9pt;
  background:#f6f6f6; border:1px solid #e2e2e2; padding:8px 10px; margin:0 0 10px;
  white-space:pre-wrap; line-height:1.4; }
table { border-collapse:collapse; width:100%; margin:4px 0 6px; font-size:8.6pt;
  font-family:"Helvetica Neue",Arial,sans-serif; }
thead th { border-top:1.1px solid var(--rule); border-bottom:0.6px solid var(--rule);
  padding:5px 6px; text-align:left; font-weight:600; }
tbody td { border-bottom:0.4px solid var(--hair); padding:4px 6px; vertical-align:top; }
tbody tr:last-child td { border-bottom:1.1px solid var(--rule); }
td.num, th.num { text-align:right; white-space:nowrap; }
caption { caption-side:bottom; text-align:left; font-size:8.6pt; color:var(--mut);
  padding:6px 0 0; line-height:1.4; }
figure { margin:14px 0 16px; text-align:center; page-break-inside:avoid; }
figure svg, figure img { max-width:100%; height:auto; }
figcaption { font-size:8.6pt; color:var(--mut); text-align:left; margin-top:6px;
  line-height:1.4; }
.note { border-left:2px solid var(--accent); background:#f7f9fb; padding:7px 11px;
  margin:0 0 11px; font-size:9.6pt; }
.titleblock { border-bottom:1.6px solid var(--rule); padding-bottom:12px;
  margin-bottom:6px; }
.titleblock h1 { border:0; font-size:17pt; margin:0 0 6px; padding:0; }
.meta { color:var(--mut); font-size:9pt; line-height:1.6; }
.toc { font-size:9.4pt; margin:10px 0 4px; }
.toc a { color:var(--ink); text-decoration:none; }
.toc div { padding:1.5px 0; }
.toc .lvl2 { padding-left:16px; color:var(--mut); }
@media print { h1 { page-break-after:avoid; } table,figure { page-break-inside:avoid; }
  a { color:inherit; text-decoration:none; } }
"""


def _esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _is_num(v):
    s = str(v).replace(",", "").replace("%", "").replace("e", "E").strip()
    for token in ("-", "+", ".", " "):
        s = s.replace(token, "")
    return s.replace("E", "").isdigit() and str(v).strip() not in ("", "--")


def render_html(d, figs):
    blocks = content(d)
    env = d.get("environment", {})
    out, toc = [], []
    h1n = 0
    for kind, *rest in blocks:
        if kind == "h1":
            h1n += 1
            toc.append(f'<div><a href="#s{h1n}">{_esc(rest[0])}</a></div>')
        elif kind == "h2":
            toc.append(f'<div class="lvl2"><a href="#s{h1n}">{_esc(rest[0])}</a></div>')

    h1n = 0
    fign = {"fig1": 1, "fig2": 2, "fig3": 3, "fig4": 4}
    for kind, *rest in blocks:
        if kind == "h1":
            h1n += 1
            out.append(f'<h1 id="s{h1n}">{_esc(rest[0])}</h1>')
        elif kind == "h2":
            out.append(f"<h2>{_esc(rest[0])}</h2>")
        elif kind == "p":
            out.append(f"<p>{_esc(rest[0])}</p>")
        elif kind == "note":
            out.append(f'<div class="note">{_esc(rest[0])}</div>')
        elif kind == "pre":
            out.append(f"<pre>{_esc(rest[0])}</pre>")
        elif kind == "ul":
            items = "".join(f"<li>{_esc(i)}</li>" for i in rest[0])
            out.append(f"<ul>{items}</ul>")
        elif kind == "table":
            cap, headers, rows = rest
            th = "".join(
                f'<th class="{"num" if i else ""}">{_esc(h)}</th>'
                for i, h in enumerate(headers))
            body = []
            for row in rows:
                tds = "".join(
                    f'<td class="{"num" if _is_num(c) else ""}">{_esc(c)}</td>'
                    for c in row)
                body.append(f"<tr>{tds}</tr>")
            out.append(f"<table><caption>{_esc(cap)}</caption>"
                       f"<thead><tr>{th}</tr></thead>"
                       f"<tbody>{''.join(body)}</tbody></table>")
        elif kind == "fig":
            key, cap = rest
            path = figs.get(key)
            svg = ""
            if path and Path(path).exists():
                raw = Path(path).read_text(encoding="utf-8")
                # drop the XML prolog and DOCTYPE so the SVG can be inlined
                i = raw.find("<svg")
                svg = raw[i:] if i >= 0 else ""
            out.append(f"<figure>{svg}<figcaption>{_esc(cap)}</figcaption></figure>")

    meta = (f"Dataset {d['dataset']} &middot; {d['n_timesteps']} held-out timesteps "
            f"&middot; tolerance {d['tol']:g} Pa<br>"
            f"{d['n_nodes']:,} nodes, {d['n_edges']:,} edges, {d['n_loops']} "
            f"independent loops<br>Generated {d.get('generated','')} &middot; "
            f"Python {env.get('python','')}, PyTorch {env.get('torch','')}")
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{_esc(TITLE)}</title><style>{CSS}</style></head><body><main>"
            f'<div class="titleblock"><h1>{_esc(TITLE)}</h1>'
            f'<div class="meta">{meta}</div></div>'
            f'<div class="toc">{"".join(toc)}</div>'
            + "".join(out) + "</main></body></html>")


def render_docx(d, figs, path):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    blocks = content(d)
    env = d.get("environment", {})
    doc = Document()

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = "Georgia"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    for name, size in (("Heading 1", 14), ("Heading 2", 11.5)):
        st = doc.styles[name]
        st.font.name = "Georgia"; st.font.size = Pt(size); st.font.bold = True
        st.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
        st.paragraph_format.space_before = Pt(14)
        st.paragraph_format.space_after = Pt(4)

    t = doc.add_paragraph(); r = t.add_run(TITLE)
    r.font.size = Pt(17); r.bold = True
    m = doc.add_paragraph()
    r = m.add_run(f"Dataset {d['dataset']}  |  {d['n_timesteps']} held-out timesteps  "
                  f"|  tolerance {d['tol']:g} Pa\n"
                  f"{d['n_nodes']:,} nodes, {d['n_edges']:,} edges, "
                  f"{d['n_loops']} independent loops\n"
                  f"Generated {d.get('generated','')}  |  "
                  f"Python {env.get('python','')}, PyTorch {env.get('torch','')}")
    r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    for kind, *rest in blocks:
        if kind == "h1":
            doc.add_heading(rest[0], level=1)
        elif kind == "h2":
            doc.add_heading(rest[0], level=2)
        elif kind == "p":
            p = doc.add_paragraph(rest[0])
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "note":
            p = doc.add_paragraph()
            r = p.add_run(rest[0]); r.italic = True; r.font.size = Pt(9.8)
            p.paragraph_format.left_indent = Cm(0.6)
        elif kind == "pre":
            p = doc.add_paragraph()
            r = p.add_run(rest[0])
            r.font.name = "Consolas"; r.font.size = Pt(9)
        elif kind == "ul":
            for item in rest[0]:
                p = doc.add_paragraph(item, style="List Bullet")
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.space_after = Pt(3)
        elif kind == "table":
            cap, headers, rows = rest
            tb = doc.add_table(rows=1, cols=len(headers))
            tb.style = "Table Grid"
            for i, h in enumerate(headers):
                c = tb.rows[0].cells[i]; c.text = ""
                r = c.paragraphs[0].add_run(str(h))
                r.bold = True; r.font.size = Pt(8.2); r.font.name = "Arial"
            for row in rows:
                cells = tb.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = ""
                    par = cells[i].paragraphs[0]
                    r = par.add_run(str(v))
                    r.font.size = Pt(8.2); r.font.name = "Arial"
                    if i and _is_num(v):
                        par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p = doc.add_paragraph()
            r = p.add_run(cap); r.font.size = Pt(8.4); r.italic = True
            r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        elif kind == "fig":
            key, cap = rest
            png = Path(str(figs.get(key, ""))).with_suffix(".png")
            if png.exists():
                doc.add_picture(str(png), width=Cm(15.6))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p = doc.add_paragraph()
            r = p.add_run(cap); r.font.size = Pt(8.4); r.italic = True
            r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC,
                    help="sweep.json produced by dhn_gnn.reporting.sweep --report")
    ap.add_argument("--no-docx", action="store_true",
                    help="skip the .docx (useful when python-docx is unavailable)")
    a = ap.parse_args()

    if not a.src.exists():
        raise SystemExit(f"{a.src} not found. Run:\n"
                         "    python -m dhn_gnn.reporting.sweep --report")
    d = json.loads(a.src.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    figs = make_figures(d)
    print(f"figures -> {FIG}")

    html_path = OUT / "solver-comparison.html"
    html_path.write_text(render_html(d, figs), encoding="utf-8")
    print(f"HTML    -> {html_path}  ({html_path.stat().st_size//1024} KB)")

    if not a.no_docx:
        target = OUT / "solver-comparison.docx"
        try:
            p = render_docx(d, figs, target)
            print(f"DOCX    -> {p}  ({p.stat().st_size//1024} KB)")
        except ImportError:
            print("DOCX    -> skipped (python-docx not installed)")
        except PermissionError:
            # Almost always Word holding the file open. Write beside it rather
            # than discarding the render or forcing the handle.
            n = 2
            while (OUT / f"solver-comparison-v{n}.docx").exists():
                n += 1
            alt = OUT / f"solver-comparison-v{n}.docx"
            p = render_docx(d, figs, alt)
            print(f"DOCX    -> {p}  ({p.stat().st_size//1024} KB)")
            print(f"          NOTE: {target.name} was locked (open in Word), so the "
                  f"current version was written as {alt.name}.")


if __name__ == "__main__":
    main()
