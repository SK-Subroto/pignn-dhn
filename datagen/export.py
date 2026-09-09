#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Publish a finished simulation into the flat layout dhn_gnn reads.

A run from `datagen.run` is organised for inspection -- schedules/, timeseries/,
full_network_state/, plots/. dhn_gnn wants one flat directory of solved CSVs plus
a history.json. This bridges the two, and is the ONLY supported way to produce a
`data/solved_<version>/`, so the step can never again be done by hand and
half-finished.

    python -m datagen.export data/raw_runs/steady
    python -m datagen.export <a pydhnClaude Results/ dir>

That second form matters: the dataset this project already ships was published by
hand from such a run, and only 4 of the 11 CSVs were copied. nodes-pressure.csv
was missed -- which silently broke `cli evaluate` and the full report -- and the
history.json left in place still described the older 745-step notebook dataset
while the CSVs held 8760 rows. Everything below exists to make that class of
mistake impossible: required files are checked, and history.json is rebuilt from
the run that actually produced the data.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Must match dhn_gnn.config.GEN_DATA_VERSION's default, so a plain run lands in
# the directory the project already reads without anyone editing config.py.
DEFAULT_VERSION = "steady"

# What dhn_gnn actually opens. Paths in config.py: MASS_FLOW_CSV, DELTA_P_CSV,
# DELTA_P_FRICTION_CSV, NODE_PRESSURE_CSV.
REQUIRED = [
    "edges-mass_flow.csv",
    "edges-delta_p.csv",
    "edges-delta_p_friction.csv",
    "nodes-pressure.csv",
]

# Not read by the solver today, but carried across because they are what a
# per-pipe rho/mu fix would need (see the Gate 5 note in datagen/__init__.py).
OPTIONAL = [
    "edges-delta_p_hydrostatic.csv",
    "edges-temperature.csv",
    "nodes-temperature.csv",
]


def build_history(summary_csv: Path) -> dict:
    """
    Rebuild history.json from the run's own network_summary.csv.

    dhn_gnn quotes `hydraulics iterations` as the reference solver cost it is
    measured against, so a history from a DIFFERENT run is worse than none: it
    silently benchmarks against a baseline that never saw this data.
    """
    s = pd.read_csv(summary_csv)
    return {
        "hydraulics converged": s["hydraulics_converged"].astype(bool).tolist(),
        "hydraulics iterations": s["hydraulics_iterations"].astype(int).tolist(),
    }


class PublishError(RuntimeError):
    """Publishing failed. The simulation output itself is untouched and intact."""


def publish(run_dir, version="steady", data_root=None, move=False, force=False):
    """
    Copy a finished run's solved CSVs into data/solved_<version>/ + history.json.

    Importable as well as CLI-callable, because `datagen.run` finishes by calling
    this: a ten-hour simulation that lands in a directory nobody remembers to
    publish is the same failure as not running it. Raises PublishError rather
    than exiting, so that caller can report the problem without discarding the
    run it just spent all night producing.
    """
    run_dir = Path(run_dir)
    data_root = Path(data_root) if data_root is not None else PROJECT_ROOT / "data"

    full = run_dir / "full_network_state"
    summary = run_dir / "timeseries" / "network_summary.csv"
    if not full.is_dir():
        raise PublishError(
            f"{full} not found. Either {run_dir} is not a simulation output "
            "directory, or the run used --no-full-state, which omits exactly the "
            "per-edge quantities dhn_gnn trains on."
        )
    missing = [n for n in REQUIRED if not (full / n).exists()]
    if missing:
        raise PublishError(f"{full} is missing required file(s): {missing}")
    if not summary.exists():
        raise PublishError(f"{summary} not found; cannot rebuild history.json")

    out = data_root / f"solved_{version}"
    if out.exists() and not force:
        raise PublishError(
            f"{out} already exists. Pass --force to overwrite, or pick another "
            "--version. Refusing by default: overwriting the dataset a checkpoint "
            "was trained on makes its recorded scores unreproducible."
        )
    out.mkdir(parents=True, exist_ok=True)

    op = shutil.move if move else shutil.copy2
    verb = "moved" if move else "copied"
    for name in REQUIRED + OPTIONAL:
        src = full / name
        if not src.exists():
            print(f"  (skip, absent) {name}")
            continue
        op(str(src), str(out / name))
        print(f"  {verb} {name}  ({(out / name).stat().st_size / 1e6:.0f} MB)")

    hist = build_history(summary)
    (out / "history.json").write_text(json.dumps(hist), encoding="utf-8")
    n = len(hist["hydraulics iterations"])
    mean_it = sum(hist["hydraulics iterations"]) / n
    n_conv = sum(hist["hydraulics converged"])
    print(f"  wrote history.json  ({n} steps, mean {mean_it:.2f} hydraulic "
          f"iterations/timestep, {n_conv}/{n} converged)")

    # The row counts must agree, or every downstream join is silently misaligned.
    rows = len(pd.read_csv(out / REQUIRED[0], index_col=0, usecols=[0]))
    if rows != n:
        print(f"\nWARNING: {REQUIRED[0]} has {rows} rows but history.json has {n} "
              "steps. These should match; check the run for an aborted tail.",
              file=sys.stderr)
    return out


def report_published(out, version):
    print(f"\npublished -> {out}")
    if version == DEFAULT_VERSION:
        print("dhn_gnn reads this by default (config.GEN_DATA_VERSION).")
    else:
        print("point dhn_gnn at it by setting in dhn_gnn/config.py:")
        print(f'    GEN_DATA_VERSION = "{version}"')
        print(f'or per-shell:  set DHN_DATA_VERSION={version}')


def add_publish_args(ap):
    """Shared by this CLI and `datagen.run`, so the two cannot drift apart."""
    ap.add_argument("--version", default=DEFAULT_VERSION,
                    help="published as data/solved_<version>/ (default: "
                         "%(default)s -> data/solved_steady, which is what "
                         "config.GEN_DATA_VERSION reads). Pass another name to "
                         "keep a second dataset side by side instead of "
                         "overwriting this one.")
    ap.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--move", action="store_true",
                    help="move the CSVs instead of copying (these files are "
                         "hundreds of MB each; use when disk is tight). The run "
                         "directory keeps its schedules, timeseries and metadata.")
    return ap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path,
                    help="a completed datagen.run output directory")
    add_publish_args(ap)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing data/solved_<version>/")
    args = ap.parse_args()

    try:
        out = publish(args.run_dir, version=args.version, data_root=args.data_root,
                      move=args.move, force=args.force)
    except PublishError as exc:
        raise SystemExit(str(exc))
    report_published(out, args.version)


if __name__ == "__main__":
    main()
