#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main driver for the Verbier DHN thermohydraulic steady-state, time-marching
simulation.

At each timestep, the network is assumed to have already reached a
hydraulic + thermal steady state under that timestep's boundary conditions
(weather-driven supply temperature, weather + occupancy driven consumer
demand) -- i.e. pipe transport delay and thermal inertia are neglected
within a step, consistent with an hourly-or-coarser "thermohydraulic
steady-state" simulation (as opposed to a fully dynamic/Lagrangian one).
Each step's solve is therefore independent of the others for correctness;
the previous step's state is reused only as a warm-start initial guess for
the Newton-Raphson solvers, which speeds up convergence but does not affect
the result.

Usage (see also run.py for a CLI):

    from simulation.simulate import SimulationConfig, run_simulation
    cfg = SimulationConfig(
        network_dir="opendhn-data/network",
        data_dir="opendhn-data/data",
        output_dir="results/demo_run",
        start="2022-01-03", end="2022-01-10", freq="1h",
    )
    run_simulation(cfg)
"""

import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pydhn.classes import Results
from pydhn.default_values import CP_FLUID, MASS_FLOW_MIN_CONS
from pydhn.fluids import Water
from pydhn.soils import KusudaSoil
from pydhn.solving import SimpleStep

from .network_builder import (
    BURIED_PIPE_DEPTH_M,
    CONSUMER_T_OUT_MIN_C,
    MAIN_PRESSURE_LIFT_PA,
    PIPE_DISCRETIZATION_M,
    SOIL_K_W_PER_MK,
    build_network,
)
from .schedules import SUPPLY_CURVE_SLOPE, T_DESIGN_C, T_NO_HEAT_C, build_schedules
from .weather import prepare_weather

DEFAULT_HYDRAULIC_KWARGS = dict(
    error_threshold=50.0,
    max_iters=200,
    damping_factor=0.3,
    adaptive=False,
    decreasing=False,
    affine_fd=True,
    verbose=0,
)

DEFAULT_THERMAL_KWARGS = dict(
    error_threshold=1e-4,
    max_iters=200,
    verbose=0,
)

#: Relative shortfall (vs. requested heat_demand) below which a consumer is
#: considered to have received "full" demand -- accounts for solver
#: tolerance, not a deliberate allowance for under-delivery.
DEMAND_CORRECTION_TOL = 0.001

#: Multiplicative headroom applied to the mass flow computed to just-hit
#: t_out_min_hx, so the correction usually settles in one extra solve
#: instead of asymptotically approaching the target over many iterations.
DEMAND_CORRECTION_SAFETY_FACTOR = 1.02

#: Max extra hydraulic+thermal solves per timestep spent correcting
#: consumers whose outlet temperature would otherwise breach t_out_min_hx.
#: Steps where no consumer is anywhere near the floor cost exactly one
#: solve, same as before -- this only adds cost where it's needed.
DEMAND_CORRECTION_MAX_ITERS = 4

#: Below this local supply-side margin (inlet temperature minus
#: t_out_min_hx, degC), no amount of consumer-side flow can help (any
#: positive extraction would already breach the floor), so the shortfall is
#: reported as genuinely unmet rather than chased with near-infinite flow.
DEMAND_CORRECTION_MIN_MARGIN_C = 0.1

#: Hard ceiling on the corrected mass flow, expressed as a multiple of the
#: substation's own design-condition mass flow, so a pathological margin
#: can't ask the hydraulic solver for an unbounded flow.
DEMAND_CORRECTION_MAX_FLOW_MULTIPLE = 50.0


@dataclass
class SimulationConfig:
    network_dir: str
    data_dir: str
    output_dir: str
    start: str
    end: str
    freq: str = "1h"
    seed: int = 0
    main_pressure_lift_pa: float = MAIN_PRESSURE_LIFT_PA
    buried_pipe_depth_m: float = BURIED_PIPE_DEPTH_M
    soil_k_w_per_mk: float = SOIL_K_W_PER_MK
    pipe_discretization_m: float = PIPE_DISCRETIZATION_M
    save_full_network_state: bool = True
    allow_live_weather: bool = True
    progress_every: int = 10
    hydraulic_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_HYDRAULIC_KWARGS))
    thermal_kwargs: dict = field(default_factory=lambda: dict(DEFAULT_THERMAL_KWARGS))
    consumer_t_out_min_c: float = CONSUMER_T_OUT_MIN_C
    demand_correction_tol: float = DEMAND_CORRECTION_TOL
    demand_correction_safety_factor: float = DEMAND_CORRECTION_SAFETY_FACTOR
    demand_correction_max_iters: int = DEMAND_CORRECTION_MAX_ITERS
    demand_correction_min_margin_c: float = DEMAND_CORRECTION_MIN_MARGIN_C
    demand_correction_max_flow_multiple: float = DEMAND_CORRECTION_MAX_FLOW_MULTIPLE


def _solve_with_demand_correction(
    loop, net, fluid, soil, ts_id, ref, substation_edges, heat_demand_row, dt_hours, config
):
    """
    Runs the steady-state solve, then checks whether any consumer's outlet
    temperature hit `t_out_min_hx` and therefore delivered less than its
    requested `heat_demand`. If so, raises that consumer's `mass_flow_min`
    just enough to hit the demand at exactly the floor temperature (trading
    delta-T for flow, the way a real control valve would respond) and
    re-solves. Repeats up to `config.demand_correction_max_iters` times.

    Substations whose local supply temperature leaves essentially no margin
    above the floor (`inlet_temperature - t_out_min_hx <=
    demand_correction_min_margin_c`) aren't chased with near-infinite flow.
    pydhn's `t_out_min_hx` clip is unconditional -- if inlet temperature is
    already at or below the floor, clipping `t_out` up to it would make the
    "consumer" inject heat into the return line (t_out > t_in), which is
    just as unphysical as the sub-floor temperature the clip is meant to
    prevent. For those substations we instead zero their demand and disable
    their floor for this step only (both reset next step), so they honestly
    report zero extraction (t_out == t_in) instead of heating the network --
    a real supply-side constraint no amount of consumer-side flow can fix.

    Returns (res, n_solves_used, unmet: dict[sub_id -> shortfall_w]).
    """
    net.set_edge_attributes(
        {substation_edges[s]: MASS_FLOW_MIN_CONS for s in ref.substation_ids},
        "mass_flow_min",
    )
    net.set_edge_attributes(
        {substation_edges[s]: config.consumer_t_out_min_c for s in ref.substation_ids},
        "t_out_min_hx",
    )

    starved = set()
    max_iters = max(1, config.demand_correction_max_iters)
    shortfalls = {}
    for correction_iter in range(1, max_iters + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = loop.execute(net=net, fluid=fluid, soil=soil, ts_id=ts_id)

        shortfalls = {}
        correctable = {}
        newly_starved = {}
        for s in ref.substation_ids:
            edge = substation_edges[s]
            demand_w = float(heat_demand_row[s])
            if demand_w <= 0:
                continue

            if s in starved:
                shortfalls[s] = demand_w
                continue

            achieved_w = -net[edge]["delta_q"] / dt_hours
            if achieved_w >= demand_w * (1 - config.demand_correction_tol):
                continue

            shortfalls[s] = demand_w - achieved_w

            t_in = net[edge]["inlet_temperature"]
            margin = t_in - config.consumer_t_out_min_c
            if margin <= config.demand_correction_min_margin_c:
                newly_starved[s] = edge
                continue

            mass_flow_needed = demand_w / (CP_FLUID * margin)
            nominal_mass_flow = demand_w / (CP_FLUID * abs(ref.design_delta_t[s]))
            cap = nominal_mass_flow * config.demand_correction_max_flow_multiple
            correctable[edge] = min(
                mass_flow_needed * config.demand_correction_safety_factor, cap
            )

        if not shortfalls:
            return res, correction_iter, {}
        if correction_iter == max_iters:
            return res, correction_iter, shortfalls

        if newly_starved:
            starved |= set(newly_starved)
            net.set_edge_attributes(dict.fromkeys(newly_starved.values(), 0.0), "heat_demand")
            net.set_edge_attributes(dict.fromkeys(newly_starved.values(), 0.0), "setpoint_value_hx")
            net.set_edge_attributes(dict.fromkeys(newly_starved.values(), np.nan), "t_out_min_hx")
        if correctable:
            net.set_edge_attributes(correctable, "mass_flow_min")
        if not newly_starved and not correctable:
            return res, correction_iter, shortfalls

    return res, max_iters, shortfalls


def _write_run_metadata(config: SimulationConfig, ref, out_dir: Path):
    meta = {
        "config": asdict(config),
        "assumptions": {
            "design_reference_outdoor_temp_C": T_DESIGN_C,
            "no_heat_outdoor_threshold_C": T_NO_HEAT_C,
            "supply_curve_slope_C_per_C": SUPPLY_CURVE_SLOPE,
            "consumer_t_out_min_C": config.consumer_t_out_min_c,
            "main_producer": ref.main_producer_id,
            "secondary_producers": ref.secondary_producer_ids,
            "note": (
                "The CASE1 measurement snapshot has no timestamp, so it is "
                "treated as the network's design/reference operating point, "
                "assumed to occur at outdoor temperature "
                f"{T_DESIGN_C} degC. Demand and supply temperature scale "
                "away from that point via standard heating-curve / "
                "degree-day relationships (see schedules.py). Consumers use "
                "an energy-based thermal setpoint (setpoint_type_hx="
                "'delta_q') with a "
                f"{config.consumer_t_out_min_c} degC outlet-temperature "
                "floor; if a consumer's local supply temperature is too "
                "low to hit full demand without breaching the floor, its "
                "mass_flow_min is raised each step (up to "
                f"{config.demand_correction_max_flow_multiple}x its design "
                "flow) to trade delta-T for flow until demand is met or the "
                "shortfall is reported as genuinely supply-side limited "
                "(see network_summary.csv: n_substations_demand_unmet)."
            ),
        },
        "network": {
            "n_substations": len(ref.substation_ids),
            "n_heating_stations": len(ref.heating_station_ids),
            "n_aerial_pipes": len(ref.aerial_pipe_ids),
            "n_buried_pipes": len(ref.buried_pipe_ids),
        },
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def run_simulation(config: SimulationConfig):
    out_dir = Path(config.output_dir)
    for sub in ("schedules", "timeseries", "plots", "weather_cache"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"[sim] building network from {config.network_dir} ...")
    net, ref = build_network(
        config.network_dir,
        config.data_dir,
        buried_depth=config.buried_pipe_depth_m,
        discretization=config.pipe_discretization_m,
        main_pressure_lift_pa=config.main_pressure_lift_pa,
        consumer_t_out_min_c=config.consumer_t_out_min_c,
    )
    net.run_sanity_checks(verbose=0)
    print(
        f"[sim] network ready: {len(ref.substation_ids)} substations, "
        f"{len(ref.heating_station_ids)} heating stations, "
        f"{len(ref.buried_pipe_ids)} buried + {len(ref.aerial_pipe_ids)} aerial pipes"
    )

    _write_run_metadata(config, ref, out_dir)

    print(f"[sim] preparing weather {config.start} -> {config.end} @ {config.freq} ...")
    t_air_sim, t_air_hourly_full, year_start = prepare_weather(
        config.start,
        config.end,
        config.freq,
        cache_dir=out_dir / "weather_cache",
        allow_live=config.allow_live_weather,
    )

    print("[sim] building demand / supply-temperature / mass-flow schedules ...")
    sched = build_schedules(ref, t_air_sim, seed=config.seed)

    sched.heat_demand.to_csv(out_dir / "schedules" / "heat_demand_w.csv")
    sched.supply_temperature.to_csv(out_dir / "schedules" / "supply_temperature_c.csv")
    sched.secondary_mass_flow.to_csv(
        out_dir / "schedules" / "secondary_producer_mass_flow_kgs.csv"
    )
    sched.t_air.to_frame("t_air_c").to_csv(
        out_dir / "schedules" / "outdoor_air_temperature_c.csv"
    )
    sched.consumer_types.to_frame("consumer_type").to_csv(
        out_dir / "schedules" / "consumer_types.csv"
    )

    fluid = Water()
    soil = KusudaSoil(t_air_hourly_full, k=config.soil_k_w_per_mk)

    index = sched.heat_demand.index
    dt_seconds = (
        (index[1] - index[0]).total_seconds() if len(index) > 1 else 3600.0
    )
    dt_hours = dt_seconds / 3600.0
    net.set_edge_attribute(value=dt_seconds, name="stepsize")
    print(f"[sim] timestep = {dt_seconds:.0f} s, {len(index)} steps")

    edges, names = net.edges("name")
    name_to_edge = dict(zip(names, (tuple(e) for e in edges)))

    substation_edges = {s: name_to_edge[s] for s in ref.substation_ids}
    hs_edges = {h: name_to_edge[h] for h in ref.heating_station_ids}

    loop = SimpleStep(
        hydraulic_sim_kwargs=config.hydraulic_kwargs,
        thermal_sim_kwargs=config.thermal_kwargs,
        with_thermal=True,
    )

    sub_series = {
        "mass_flow_kgs": [],
        "delta_q_w": [],
        "inlet_temperature_c": [],
        "outlet_temperature_c": [],
    }
    hs_series = {
        "mass_flow_kgs": [],
        "outlet_temperature_c": [],
        "inlet_temperature_c": [],
        "delta_q_w": [],
    }
    summary_rows = []

    full_state = Results() if config.save_full_network_state else None

    n_steps = len(index)
    progress_every = max(1, config.progress_every)
    t_wall_start = time.time()

    for i, ts in enumerate(index):
        heat_demand_row = sched.heat_demand.iloc[i]

        demand_updates = {
            substation_edges[s]: float(heat_demand_row[s]) for s in ref.substation_ids
        }
        net.set_edge_attributes(demand_updates, "heat_demand")

        # setpoint_type_hx="delta_q" ties the thermal setpoint directly to
        # the energy demand (Wh over the current step), independent of mass
        # flow -- see _solve_with_demand_correction and network_builder.py.
        hx_updates = {
            substation_edges[s]: -float(heat_demand_row[s]) * dt_hours
            for s in ref.substation_ids
        }
        net.set_edge_attributes(hx_updates, "setpoint_value_hx")

        supply_updates = {
            hs_edges[h]: float(sched.supply_temperature.iloc[i][h])
            for h in ref.heating_station_ids
        }
        net.set_edge_attributes(supply_updates, "setpoint_value_hx")

        if ref.secondary_producer_ids:
            secondary_updates = {
                hs_edges[h]: float(sched.secondary_mass_flow.iloc[i][h])
                for h in ref.secondary_producer_ids
            }
            net.set_edge_attributes(secondary_updates, "setpoint_value_hyd")

        t_air_value = float(sched.t_air.iloc[i])

        ts_id = int(round((ts - year_start).total_seconds() / 3600.0))

        t0 = time.time()
        res, correction_solves, unmet = _solve_with_demand_correction(
            loop, net, fluid, soil, ts_id, ref, substation_edges,
            heat_demand_row, dt_hours, config,
        )
        solve_time = time.time() - t0

        hyd_conv = bool(res["history"]["hydraulics converged"])
        hyd_iters = int(res["history"]["hydraulics iterations"])
        th_conv = bool(res["history"]["thermal converged"])
        th_iters = int(res["history"]["thermal iterations"])
        if not hyd_conv:
            warnings.warn(f"step {i} ({ts}): hydraulics did not converge")
        if not th_conv:
            warnings.warn(f"step {i} ({ts}): thermal did not converge")
        if unmet:
            worst = max(unmet.items(), key=lambda kv: kv[1])
            warnings.warn(
                f"step {i} ({ts}): {len(unmet)} substation(s) below full "
                f"demand after {correction_solves} solve(s) -- local supply "
                f"temperature too close to the {config.consumer_t_out_min_c} "
                f"degC floor; worst is {worst[0]} short by {worst[1]:.0f} W"
            )

        _, pressures = net.nodes("pressure")

        sub_mdot = {s: net[substation_edges[s]]["mass_flow"] for s in ref.substation_ids}
        sub_dq = {s: net[substation_edges[s]]["delta_q"] for s in ref.substation_ids}
        sub_tin = {
            s: net[substation_edges[s]]["inlet_temperature"] for s in ref.substation_ids
        }
        sub_tout = {
            s: net[substation_edges[s]]["outlet_temperature"] for s in ref.substation_ids
        }
        sub_series["mass_flow_kgs"].append(sub_mdot)
        sub_series["delta_q_w"].append(sub_dq)
        sub_series["inlet_temperature_c"].append(sub_tin)
        sub_series["outlet_temperature_c"].append(sub_tout)

        hs_mdot = {h: net[hs_edges[h]]["mass_flow"] for h in ref.heating_station_ids}
        hs_tout = {h: net[hs_edges[h]]["outlet_temperature"] for h in ref.heating_station_ids}
        hs_tin = {h: net[hs_edges[h]]["inlet_temperature"] for h in ref.heating_station_ids}
        hs_dq = {h: net[hs_edges[h]]["delta_q"] for h in ref.heating_station_ids}
        hs_series["mass_flow_kgs"].append(hs_mdot)
        hs_series["outlet_temperature_c"].append(hs_tout)
        hs_series["inlet_temperature_c"].append(hs_tin)
        hs_series["delta_q_w"].append(hs_dq)

        total_demand_w = float(sched.heat_demand.iloc[i].sum())
        total_extraction_w = -float(sum(sub_dq.values()))
        total_production_w = float(sum(hs_dq.values()))

        summary_rows.append(
            {
                "timestamp": ts,
                "t_air_c": t_air_value,
                "total_demand_w": total_demand_w,
                "total_consumer_extraction_w": total_extraction_w,
                "total_production_w": total_production_w,
                "network_loss_w": total_production_w - total_extraction_w,
                "min_pressure_bar": float(np.nanmin(pressures)) / 1e5,
                "max_pressure_bar": float(np.nanmax(pressures)) / 1e5,
                "hydraulics_converged": hyd_conv,
                "hydraulics_iterations": hyd_iters,
                "thermal_converged": th_conv,
                "thermal_iterations": th_iters,
                "solve_time_s": solve_time,
                "demand_correction_solves": correction_solves,
                "n_substations_demand_unmet": len(unmet),
                "unmet_demand_w": float(sum(unmet.values())),
            }
        )

        if full_state is not None:
            full_state.append(res)

        if (i + 1) % progress_every == 0 or i == n_steps - 1:
            elapsed = time.time() - t_wall_start
            eta = elapsed / (i + 1) * (n_steps - i - 1)
            print(
                f"[sim] step {i + 1}/{n_steps} ({ts}) "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s "
                f"solve={solve_time:.2f}s hyd_ok={hyd_conv} th_ok={th_conv}"
            )

    print(f"[sim] simulation loop finished in {time.time() - t_wall_start:.1f}s, saving results ...")

    _save_results(out_dir, index, sub_series, hs_series, summary_rows, full_state)
    _save_summary_plots(out_dir, sched, pd.DataFrame(summary_rows).set_index("timestamp"))

    print(f"[sim] done. Results saved to {out_dir.resolve()}")
    return out_dir


def _save_results(out_dir, index, sub_series, hs_series, summary_rows, full_state):
    ts_dir = out_dir / "timeseries"

    for key, records in sub_series.items():
        pd.DataFrame(records, index=index).to_csv(ts_dir / f"substations_{key}.csv")

    for key, records in hs_series.items():
        pd.DataFrame(records, index=index).to_csv(ts_dir / f"heating_stations_{key}.csv")

    summary_df = pd.DataFrame(summary_rows).set_index("timestamp")
    summary_df.to_csv(ts_dir / "network_summary.csv")

    if full_state is not None:
        dfs = full_state.to_dataframes()
        full_dir = out_dir / "full_network_state"
        full_dir.mkdir(parents=True, exist_ok=True)
        for kind in ("edges", "nodes"):
            for key, df in dfs[kind].items():
                df.index = index
                df.to_csv(full_dir / f"{kind}-{key}.csv")


def _save_summary_plots(out_dir, sched, summary_df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(summary_df.index, summary_df["t_air_c"], color="tab:blue", label="Outdoor air temp")
    ax1.set_ylabel("Outdoor air temperature (degC)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(
        summary_df.index,
        summary_df["total_demand_w"] / 1e6,
        color="tab:red",
        label="Total demand",
    )
    ax2.set_ylabel("Total network heat demand (MW)", color="tab:red")
    fig.suptitle("Weather-driven total demand")
    fig.tight_layout()
    fig.savefig(plots_dir / "weather_and_demand.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(summary_df.index, summary_df["total_production_w"] / 1e6, label="Produced (HS0+HS1)")
    ax.plot(summary_df.index, summary_df["total_consumer_extraction_w"] / 1e6, label="Consumed (substations)")
    ax.plot(summary_df.index, summary_df["network_loss_w"] / 1e6, label="Network loss")
    ax.set_ylabel("Power (MW)")
    ax.legend()
    ax.set_title("Network energy balance")
    fig.tight_layout()
    fig.savefig(plots_dir / "energy_balance.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(summary_df.index, summary_df["min_pressure_bar"], label="min node pressure")
    ax.plot(summary_df.index, summary_df["max_pressure_bar"], label="max node pressure")
    ax.set_ylabel("Pressure (bar)")
    ax.legend()
    ax.set_title("Node pressure range")
    fig.tight_layout()
    fig.savefig(plots_dir / "pressure_range.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for hs_id in sched.supply_temperature.columns:
        ax.plot(sched.supply_temperature.index, sched.supply_temperature[hs_id], label=f"{hs_id} supply setpoint")
    ax.set_ylabel("Temperature (degC)")
    ax.legend()
    ax.set_title("Producer supply temperature schedule")
    fig.tight_layout()
    fig.savefig(plots_dir / "supply_temperature_schedule.png", dpi=120)
    plt.close(fig)
