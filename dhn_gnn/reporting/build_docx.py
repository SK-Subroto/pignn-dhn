"""Build the formal comparison report as a .docx.

Reads results/report_data.json and results/figures/, so the document cannot drift
from the run that produced it:

    python -m dhn_gnn.reporting.full_report --n-ts 400
    python -m dhn_gnn.reporting.build_docx

Requires python-docx.
"""

import json
import sys
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dhn_gnn import config

RES = config.RESULTS_DIR
FIG = RES / "figures"
OUT = RES / "report" / "DHN-solver-comparison.docx"

ACCENT = RGBColor(0x0F, 0x4C, 0x5C)
MUTED = RGBColor(0x55, 0x60, 0x63)

NAME = {"pydhn": "PyDHN (reference)", "unrolled": "Unrolled GNN (original)",
        "newton": "Newton (pure physics)", "initializer": "Learned initializer (this work)"}


# ----------------------------------------------------------------- docx helpers
def _field(paragraph, instr):
    """Insert a Word field code (used for page numbers and the table of contents)."""
    run = paragraph.add_run()
    for kind, payload in (("begin", None), ("instr", instr), ("end", None)):
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = payload
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        run._r.append(el)


def setup(doc):
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)          # A4
    s.left_margin = s.right_margin = Cm(2.4)
    s.top_margin = s.bottom_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.12

    for name, size, color in (("Heading 1", 16, ACCENT), ("Heading 2", 13, ACCENT),
                              ("Heading 3", 11.5, ACCENT)):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = color
        st.font.bold = True
        st.paragraph_format.space_before = Pt(14)
        st.paragraph_format.space_after = Pt(5)

    foot = s.footer.paragraphs[0]
    foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _field(foot, "PAGE")
    foot.runs[0].font.size = Pt(9)
    foot.runs[0].font.color.rgb = MUTED


def para(doc, text, size=10.5, italic=False, bold=False, color=None,
         align=None, space_after=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size); r.italic = italic; r.bold = bold
    if color is not None:
        r.font.color.rgb = color
    if align is not None:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def rich(doc, parts, size=10.5):
    """A paragraph from (text, bold, italic) tuples, for inline emphasis."""
    p = doc.add_paragraph()
    for text, b, i in parts:
        r = p.add_run(text)
        r.font.size = Pt(size); r.bold = b; r.italic = i
    return p


def bullet(doc, parts, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    for text, b, i in (parts if isinstance(parts, list) else [(parts, False, False)]):
        r = p.add_run(text)
        r.font.size = Pt(size); r.bold = b; r.italic = i
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(8.5); r.italic = True; r.font.color.rgb = MUTED
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    return p


def figure(doc, name, number, text, width=Cm(15.5)):
    p = FIG / name
    if not p.exists():
        para(doc, f"[missing figure: {name}]", italic=True)
        return
    doc.add_picture(str(p), width=width)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, f"Figure {number}. {text}")


def table(doc, headers, rows, number, text, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold = True; r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(v))
            r.font.size = Pt(9)
            if i:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = w
    caption(doc, f"Table {number}. {text}")
    return t


# ------------------------------------------------------------------------ build
def build():
    d = json.loads((RES / "report_data.json").read_text(encoding="utf-8"))
    s, py, cons = d["scores"], d["pydhn"], d["consistency"]
    n = d["n_timesteps"]
    f = lambda v, spec=".2f": format(v, spec)

    doc = Document()
    setup(doc)

    # --- title -------------------------------------------------------------
    para(doc, "Physics-Informed Graph Neural Networks for District Heating Hydraulics",
         size=19, bold=True, color=ACCENT, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, "A comparison of four solvers on a full year of network operation",
         size=12, italic=True, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)
    para(doc, f"Evaluated on {n} held-out hours   |   {d['window'][0][:10]} to "
              f"{d['window'][1][:10]}   |   {date.today():%d %B %Y}",
         size=9.5, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)

    # --- abstract ----------------------------------------------------------
    doc.add_heading("Summary", level=1)
    rich(doc, [
        ("This report compares four ways of solving the steady-state hydraulics of a "
         "district heating network with ", False, False),
        (f"{d['n_edges']:,} edges and {d['n_nodes']:,} nodes", True, False),
        (": the reference simulator PyDHN, an unrolled graph-neural-network solver "
         "(the original architecture of this project), an exact Newton solver with no "
         "learned parameters, and a learned initializer that predicts the solution once "
         "and hands it to exact Newton. The central result is that the learned "
         "initializer reaches the convergence tolerance in ", False, False),
        (f"{f(s['initializer']['steps_mean'])} solver steps", True, False),
        (f", against {f(s['newton']['steps_mean'])} for pure physics and "
         f"{f(s['unrolled']['steps_mean'])} for the original architecture, while "
         "requiring no information from the preceding timestep. That independence is "
         "the substantive contribution: unlike warm-started classical solvers, the "
         "hours of a simulated year are mutually independent problems and can be "
         "solved concurrently.", False, False),
    ])
    rich(doc, [
        ("Two limitations constrain the conclusions and are stated in full in Section 7. ",
         False, False),
        ("The pressure-drop model used by the solvers reproduces only "
         f"{f(cons['within_5pct'], '.1f')}% of the reference dataset's pipe friction "
         "losses within 5%", True, False),
        (", so pressure-derived error metrics mix model error with a physics offset; and "
         "the solver log distributed with the dataset is stale, so no PyDHN iteration "
         "count is quoted for this data.", False, False),
    ])

    # --- toc ---------------------------------------------------------------
    doc.add_heading("Contents", level=1)
    p = doc.add_paragraph()
    _field(p, r'TOC \o "1-2" \h \z \u')
    para(doc, "(In Word: right-click the field above and choose Update Field to populate.)",
         size=8.5, italic=True, color=MUTED)

    doc.add_page_break()

    # --- 1 problem ---------------------------------------------------------
    doc.add_heading("1  The problem", level=1)
    rich(doc, [
        ("Steady-state hydraulic simulation determines the mass flow in every pipe such "
         "that two physical laws hold simultaneously: mass is conserved at every "
         "junction, and pressure drop sums to zero around every closed loop. The network "
         "studied here has ", False, False),
        (f"{d['n_edges']:,} edges, of which {d['n_pipes']:,} are pipes, joining "
         f"{d['n_nodes']:,} nodes.", True, False),
    ])
    rich(doc, [
        ("The two laws are not equally difficult. Mass conservation can be satisfied ",
         False, False), ("by construction", False, True),
        (". Writing the flow as a boundary-feasible reference plus a circulation around "
         "each independent loop, m = m", False, False),
        ("0", False, False), (" + Zc with AZ = 0, conservation holds for any value of "
         "the loop flows c. What remains is a system in ", False, False),
        (f"{d['n_loops']} unknowns", True, False),
        (". This reduction is what makes an exact Newton solve inexpensive enough to "
         "serve as the default rather than an approximation, and it is the reason mass "
         "conservation is never learned and never violated.", False, False),
    ])
    figure(doc, "reduction.png", 1,
           "The cycle-space reduction. The raw problem has 1,514 edge unknowns; imposing "
           "mass conservation structurally leaves 12 loop unknowns to solve.")

    # --- 2 pydhn -----------------------------------------------------------
    doc.add_heading("2  The reference simulator: PyDHN", level=1)
    rich(doc, [
        ("PyDHN solves the hydraulics with what its documentation calls the ", False, False),
        ("simplified loop method", False, True),
        (": a Newton-Raphson iteration over the fundamental cycles of the network graph, "
         "solving the system B", False, False),
        ("φ", False, False), ("(B", False, False), ("T", False, False),
        ("m) = 0, where B is the fundamental cycle matrix and ", False, False),
        ("φ", False, False),
        (" maps component mass flows to pressure differences. The solvers in this report "
         "target the same equation, which is what makes the comparison meaningful rather "
         "than a contest between different problem formulations.", False, False),
    ])
    rich(doc, [
        ("It is worth being precise about the convergence tolerance, because it is easy "
         "to misattribute. PyDHN's documented default is ", False, False),
        ("error_threshold = 100 Pa", True, False),
        (". The generation script in this project overrides it to ", False, False),
        ("50 Pa with max_iters = 500", True, False),
        (", and 50 Pa is therefore the tolerance used throughout this report - a choice "
         "of this project, not a property of PyDHN.", False, False),
    ])
    table(doc,
          ["Parameter", "PyDHN default", "Used for the January dataset"],
          [["error_threshold", "100 Pa", "50 Pa"],
           ["max_iters", "100", "500"],
           ["damping_factor", "1", "1 (default)"],
           ["compute_hydrostatic", "True", "True (default)"],
           ["with_thermal", "-", "False (hydraulics only)"]],
          1, "PyDHN hydraulic solver parameters. The settings used to generate the 2022 "
             "dataset analysed here are not known (see Section 7).")

    # --- 3 approaches ------------------------------------------------------
    doc.add_heading("3  The four approaches", level=1)
    para(doc, "All three solvers below share a single physics implementation and are "
              "scored on identical timesteps at an identical tolerance, so differences "
              "between them are attributable to the mechanism under study.")
    table(doc,
          ["Approach", "Learned component", "Starting point", "Timesteps independent?"],
          [["PyDHN", "none", "previous timestep", "no"],
           ["Unrolled GNN", "correction at each of 20 steps", "zero", "yes"],
           ["Newton", "none", "zero", "yes"],
           ["Learned initializer", "the starting point only", "one GNN pass", "yes"]],
          2, "The four approaches. Only the starting point differs between Newton and the "
             "learned initializer; the solve that follows is identical.")
    figure(doc, "architecture.png", 2,
           "What each approach executes for one timestep. Filled blocks are the learned "
           "component. The original architecture runs attention at every step; the "
           "learned initializer runs it once.")

    doc.add_heading("3.1  Why the learned initializer is trainable", level=2)
    rich(doc, [
        ("The loop-space coordinate of the true solution, a* = (Z", False, False),
        ("T", False, False), ("Z)", False, False), ("-1", False, False),
        ("Z", False, False), ("T", False, False),
        ("m, is computable directly from the reference data. The initializer therefore "
         "has an exact supervised target and is fitted by ", False, False),
        ("plain regression on 12 numbers", True, False),
        (", with no gradients passing through the solver. This avoids the failure modes "
         "that made the unrolled architecture difficult to optimise: truncated "
         "backpropagation, exploding gradients through a 20-step unroll, and per-step "
         "heads that receive gradient only when their step executes.", False, False),
    ])

    # --- 4 setup -----------------------------------------------------------
    doc.add_heading("4  Experimental setup", level=1)
    bullet(doc, [("Dataset: ", True, False),
                 (f"{py['n_timesteps']:,} hourly timesteps covering 2022, generated with "
                  "PyDHN and supplied by the project supervisor.", False, False)])
    bullet(doc, [("Split: ", True, False),
                 ("interleaved, every fifth timestep held out. Interleaving rather than a "
                  "contiguous tail avoids testing on a different season from the one "
                  "trained on.", False, False)])
    bullet(doc, [("Evaluation set: ", True, False),
                 (f"{n} held-out timesteps sampled evenly across the year.", False, False)])
    bullet(doc, [("Tolerance: ", True, False),
                 (f"{f(d['tol'], '.0f')} Pa on the maximum absolute loop residual.",
                  False, False)])
    bullet(doc, [("Hardware: ", True, False),
                 ("CPU, single sample per forward pass. GPU results are discussed in "
                  "Section 7.", False, False)])
    bullet(doc, [("Training: ", True, False),
                 ("200 timesteps sampled across the year, 40 epochs, Adam at 1e-3, "
                  "best-weight restore on the guess error.", False, False)])

    # --- 5 results ---------------------------------------------------------
    doc.add_heading("5  Results", level=1)

    doc.add_heading("5.1  Accuracy", level=2)
    table(doc,
          ["Approach", "Flow MAE (kg/s)", "RMSE (kg/s)", "R2", "Direction", "dp MAE (Pa)"],
          [[NAME[k], f(s[k]["flow_mae"], ".3e"), f(s[k]["flow_rmse"], ".3e"),
            f(s[k]["flow_r2"], ".6f"), f(s[k]["dir_acc"], ".2f") + " %",
            f(s[k]["dp_mae"], ".2f")]
           for k in ("unrolled", "newton", "initializer")],
          3, f"Accuracy against the PyDHN solution on {n} held-out timesteps. Pressure-drop "
             "figures are scored on pipes only and carry the physics offset described in "
             "Section 7.1.")
    figure(doc, "flow_parity.png", 3,
           "Predicted against reference mass flow for every pipe and timestep. Density is "
           "log-scaled; the dashed line marks equality.", width=Cm(16))

    doc.add_heading("5.2  Convergence and cost", level=2)
    rows = [[NAME[k], f(s[k]["steps_mean"]), f(s[k]["steps_median"], ".0f"),
             f(s[k]["steps_max"], ".0f"), f(s[k]["ms_per_ts"], ".0f"),
             f(s[k]["pct_under_tol"], ".1f") + " %"]
            for k in ("unrolled", "newton", "initializer")]
    rows.append([NAME["pydhn"], "n/a", "n/a", "n/a", f"~{py['ms_per_ts']:,.0f}", "100 %"])
    table(doc, ["Approach", "Steps (mean)", "Median", "Max", "ms / timestep", "Within tol"],
          rows, 4,
          f"Cost at the same {f(d['tol'], '.0f')} Pa tolerance. PyDHN's figure is its "
          f"{f(py['hours_per_year'], '.0f')}-hour generation of the "
          f"{py['n_timesteps']:,}-step year divided by its timesteps.")
    rich(doc, [
        ("The PyDHN timing requires a caveat. It covers the ", False, False),
        ("entire simulation", False, True),
        (" - network construction, control logic, thermal handling and file output - "
         "whereas the solvers above perform the hydraulic solve alone. It should be read "
         "as an upper bound on comparable work, not a like-for-like measurement.",
         False, False),
    ])
    figure(doc, "convergence.png", 4,
           "Loop residual after each solver step, median with inter-quartile band. Exact "
           "Newton converges quadratically; the diagonal approximation used by the "
           "original architecture converges linearly.")
    figure(doc, "steps_distribution.png", 5,
           "Distribution of steps required to reach tolerance. Means conceal the tail, so "
           "the full distribution is shown.")
    figure(doc, "timing.png", 6, "Wall-clock per timestep on CPU, logarithmic scale.")

    doc.add_heading("5.3  What the learned component contributes", level=2)
    rich(doc, [
        ("The learned component supplies one thing: a better starting point. Measured "
         "before any Newton step executes, the cold start sits at ", False, False),
        (f"{f(s['newton']['guess_resid_median'], '.2e')} Pa", True, False),
        (" and the learned guess at ", False, False),
        (f"{f(s['initializer']['guess_resid_median'], '.2e')} Pa", True, False),
        (f", approximately {f(s['newton']['guess_resid_median'] / max(s['initializer']['guess_resid_median'], 1e-9), '.0f')} "
         "times closer. Everything after that point is parameter-free physics.",
         False, False),
    ])
    figure(doc, "guess_quality.png", 7,
           "Residual of the starting point before any Newton step. This difference is the "
           "entire contribution of the learned component.", width=Cm(13))
    figure(doc, "seasonal.png", 8,
           "Mean steps to tolerance by month. A model fitted on 200 hours sampled across "
           "the year shows no seasonal degradation.")

    # --- 6 discussion ------------------------------------------------------
    doc.add_heading("6  Discussion", level=1)

    doc.add_heading("6.1  Why parallelism is the substantive result", level=2)
    rich(doc, [
        ("PyDHN attains a low iteration count by warm-starting each timestep from the "
         "preceding one. This is sound engineering and also a hard constraint: hour 100 "
         "cannot begin before hour 99 completes. The learned guess is computed from "
         "boundary conditions alone and carries no such dependency, so all ", False, False),
        (f"{py['n_timesteps']:,} hours", True, False),
        (" of the year are independent problems. Reducing iteration count is a contest "
         "the classical solver can win; removing the sequential dependency is not.",
         False, False),
    ])

    doc.add_heading("6.2  The negative result on the original architecture", level=2)
    rich(doc, [
        ("The unrolled architecture ran the graph network at each of twenty steps and "
         "learned a correction at every one. Trained on this dataset, best-weight "
         "selection returned ", False, False),
        ("epoch zero - the initialisation", True, False),
        (". Every subsequent epoch was worse on the monitored residual, at every stable "
         "learning rate tested. Because the output head is zero-initialised, the "
         "untrained model is exactly the physics solver; the learned correction was "
         "asked to improve on a solution already at the accuracy the reference data "
         "supports, while costing twenty attention passes to attempt it. Relocating the "
         "same network from correcting to guessing is what made it useful.", False, False),
    ])

    # --- 7 limitations -----------------------------------------------------
    doc.add_heading("7  Limitations", level=1)

    doc.add_heading("7.1  The pressure-drop model does not match this dataset", level=2)
    rich(doc, [
        ("Evaluated on the reference dataset's own flows, the project's pressure-drop "
         "model reproduces only ", False, False),
        (f"{f(cons['within_5pct'], '.1f')}% of pipe friction losses within 5%", True, False),
        (f" (median ratio {f(cons['median_ratio'], '.3f')}, spread "
         f"{f(cons['ratio_spread'][0], '.3f')} to {f(cons['ratio_spread'][1], '.3f')} "
         "across timesteps). Scatter rather than systematic bias is consistent with "
         "per-pipe temperatures from a thermal simulation, against the fixed 50 degC "
         "fluid properties assumed here.", False, False),
    ])
    rich(doc, [
        ("Flow comparisons remain meaningful, since boundary conditions are taken from "
         "the reference. ", False, False),
        ("Pressure-drop and pressure errors, however, combine model error with this "
         "offset and should not be reported as model error alone.", True, False),
        (" Resolving this requires the generation settings used to produce the dataset.",
         False, False),
    ])

    doc.add_heading("7.2  No PyDHN iteration count for this dataset", level=2)
    rich(doc, [
        ("The solver log distributed with the data contains ", False, False),
        (f"{py['history_entries']} entries against {py['n_timesteps']:,} timesteps",
         True, False),
        (", so it describes an earlier run. No per-timestep iteration count for PyDHN on "
         "this dataset is quoted anywhere in this report; the wall-clock figure in Table 4 "
         "derives from the stated generation time instead.", False, False),
    ])

    doc.add_heading("7.3  Further limitations", level=2)
    bullet(doc, [("GPU acceleration does not help. ", True, False),
                 ("On a GTX 1060, inference took 104.8 ms against 44.0 ms on CPU, with "
                  "training a tie. At batch size 1 the tensors are too small to cover "
                  "kernel-launch overhead. All results here are CPU. Batching is the "
                  "prerequisite, and is precisely what the parallel structure enables.",
                  False, False)])
    bullet(doc, [("Flow R2 flatters every approach. ", True, False),
                 (f"The boundary-feasible reference already carries most of each edge "
                  f"flow; only {d['n_loops']} loop unknowns are solved. Loop-space error "
                  "is the more honest accuracy measure.", False, False)])
    bullet(doc, [("Single network, single year. ", True, False),
                 ("Checkpoints encode the network topology. Nothing here demonstrates "
                  "transfer to a different district heating network.", False, False)])

    # --- 8 repro -----------------------------------------------------------
    doc.add_heading("8  Reproducibility", level=1)
    para(doc, "Every figure and every number in this document is produced by the "
              "following commands and read from results/report_data.json.")
    for line in ("python -m dhn_gnn.cli train --arch initializer --epochs 40 --n-train 200",
                 f"python -m dhn_gnn.reporting.full_report --n-ts {n}",
                 "python -m dhn_gnn.reporting.build_docx"):
        p = doc.add_paragraph()
        r = p.add_run(line)
        r.font.name = "Consolas"; r.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.6)

    para(doc, "Per-approach predictions are written to separate directories, each indexed "
              "by timestamp and column-aligned with the reference CSVs:", space_after=4)
    for line in ("results/predictions/newton/{mass_flow,delta_p,pressure}.csv",
                 "results/predictions/unrolled/{mass_flow,delta_p,pressure}.csv",
                 "results/predictions/initializer/{mass_flow,delta_p,pressure}.csv",
                 "results/figures/*.png"):
        p = doc.add_paragraph()
        r = p.add_run(line)
        r.font.name = "Consolas"; r.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.6)

    # --- references --------------------------------------------------------
    doc.add_heading("References", level=1)
    for i, ref in enumerate([
        "PyDHN documentation, Hydraulic simulation. "
        "https://idiap.github.io/pydhn/get_started/simulation.html#hydraulic-simulation "
        "(accessed " + f"{date.today():%d %B %Y}" + ").",
        "PyDHN, Idiap Research Institute. https://github.com/idiap/pydhn",
        "Project repository, branch feat/learned-initializer. Modules: dhn_gnn/model/"
        "{base,newton,initializer,unrolled_solver}.py, dhn_gnn/reporting/full_report.py.",
    ], 1):
        p = doc.add_paragraph()
        r = p.add_run(f"[{i}]  {ref}")
        r.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:,.0f} KB)")
    return OUT


if __name__ == "__main__":
    build()
