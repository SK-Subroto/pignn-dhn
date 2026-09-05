#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI entrypoint for the Verbier DHN thermohydraulic steady-state simulation.

Reads the network topology and the CASE1 measurement snapshot that already ship
with this project (data/network/, data/measurements/) -- they are byte-identical
to the standalone project's opendhn-data/, so there is only ever one copy of the
topology and it cannot drift.

Simulating and publishing are ONE command. The simulation writes its own rich
output (schedules/, timeseries/, full_network_state/, plots/) to --output-dir,
then the solved CSVs and a freshly built history.json are published into
data/solved_<version>/, which is what dhn_gnn actually reads.

Examples
--------
Quick 3-day hourly demo (the default -- deliberately small, so an accidental
bare invocation cannot start a ten-hour job):

    python -m datagen.run --output-dir data/raw_runs/demo_3day --version demo

The canonical full-year hourly dataset. THIS TAKES ABOUT 10 HOURS
(the reference 8760-step run logged 36385 s) and writes several GB:

    python -m datagen.run --full-year --output-dir data/raw_runs/steady

Full year at 6-hour resolution -- much faster, still resolves the seasonal cycle:

    python -m datagen.run --start 2022-01-01 --end 2022-12-31 --freq 6h \\
        --output-dir data/raw_runs/steady_6h --version steady_6h

KEEP THE RUN DIRECTORY. Publishing takes only the CSVs dhn_gnn reads today;
the run also holds the per-pipe temperatures, schedules and weather cache, and
regenerating those is another ten hours. If disk is tight use --move, which
relocates the large CSVs but leaves the small metadata in place.
"""

import argparse
import sys
from pathlib import Path

from .export import PublishError, add_publish_args, publish, report_published
from .simulate import SimulationConfig, run_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The project's own copies, rather than a second checkout of opendhn-data.
DEFAULT_NETWORK_DIR = PROJECT_ROOT / "data" / "network"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "measurements"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_runs" / "run"

FULL_YEAR = ("2022-01-01", "2022-12-31 23:00", "1h")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Verbier DHN thermohydraulic steady-state simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--network-dir", default=str(DEFAULT_NETWORK_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", default="2022-01-03", help="Simulation start (inclusive)")
    parser.add_argument("--end", default="2022-01-06", help="Simulation end (inclusive)")
    parser.add_argument(
        "--freq",
        default="1h",
        help="Pandas frequency string for the timestep, e.g. 15min, 1h, 6h, 1D",
    )
    parser.add_argument(
        "--full-year",
        action="store_true",
        help="Shorthand for the canonical dataset: 2022-01-01..2022-12-31 hourly "
             "(8760 steps). Overrides --start/--end/--freq. Expect ~10 hours.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for demand noise")
    parser.add_argument(
        "--no-full-state",
        action="store_true",
        help="Skip saving the full per-node/per-edge network state each step "
        "(only substation/heating-station/summary timeseries are saved). "
        "Saves disk space, but dhn_gnn CANNOT use such a run -- the per-edge "
        "mass flow and pressure drop are exactly what it trains on.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the live weather fetch and use the deterministic synthetic "
        "climate fallback (useful without internet access).",
    )
    parser.add_argument(
        "--main-pressure-lift-pa",
        type=float,
        default=None,
        help="Override the main producer's pressure lift setpoint (Pa, negative).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a progress line every N steps.",
    )

    # Publishing happens automatically at the end -- a ten-hour simulation left
    # sitting in a directory nobody remembers to publish is the same failure as
    # not running it at all.
    add_publish_args(parser)
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Stop after simulating; leave the run unpublished. Publish it later "
             "with `python -m datagen.export <output-dir>`.",
    )
    parser.add_argument(
        "--force-publish",
        action="store_true",
        help="Allow the automatic publish to overwrite an existing "
             "data/solved_<version>/. Decide this UP FRONT: without it a run "
             "whose target already exists still finishes and is kept, but stops "
             "short of publishing.",
    )
    args = parser.parse_args()

    start, end, freq = args.start, args.end, args.freq
    if args.full_year:
        start, end, freq = FULL_YEAR

    overrides = {}
    if args.main_pressure_lift_pa is not None:
        overrides["main_pressure_lift_pa"] = args.main_pressure_lift_pa

    config = SimulationConfig(
        network_dir=args.network_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start=start,
        end=end,
        freq=freq,
        seed=args.seed,
        save_full_network_state=not args.no_full_state,
        allow_live_weather=not args.offline,
        progress_every=args.progress_every,
        **overrides,
    )
    run_simulation(config)
    print(f"\nsimulation written to {args.output_dir}")

    if args.no_publish:
        print("--no-publish given; publish it later with:")
        print(f"    python -m datagen.export {args.output_dir}")
        return

    print(f"\npublishing into data/solved_{args.version}/ ...")
    try:
        out = publish(args.output_dir, version=args.version,
                      data_root=args.data_root, move=args.move,
                      force=args.force_publish)
    except PublishError as exc:
        # Never lose the simulation over a publishing problem: the run is on
        # disk and complete, so report what happened and how to finish it.
        print(f"\npublish skipped: {exc}", file=sys.stderr)
        print(f"\nThe simulation itself is COMPLETE and safe at {args.output_dir}.\n"
              f"Finish it with either of:\n"
              f"    python -m datagen.export {args.output_dir} --force\n"
              f"    python -m datagen.export {args.output_dir} --version <other-name>",
              file=sys.stderr)
        raise SystemExit(1)
    report_published(out, args.version)


if __name__ == "__main__":
    main()
