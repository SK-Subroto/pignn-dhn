#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a pydhn `Network` from the opendhn-data CSV topology (Verbier DHN),
and derive the per-component design/reference parameters needed to drive a
physically consistent time-marching simulation from the single-timestamp
measurement snapshot ("CASE1").

Design choices (see project notes for the reasoning):

* Pipes flagged `is_aerial` use the standard `Pipe` component with
  `depth=0`. In pydhn's buried-pipe formula the soil-conduction term is
  mathematically singular at `depth=0` (`log(0)`), but pydhn's `safe_divide`
  helper silently clamps that singularity to a resistance of exactly 0 --
  so in practice this models an aerial pipe as having *no* extra resistance
  beyond its own pipe-wall/insulation/casing layers, referenced to the soil
  model's depth-0 ("surface") temperature. That's a simplification (no
  explicit convective film resistance to open air, and it inherits the
  soil model's smoothed seasonal surface temperature rather than raw hourly
  air temperature), and it runs somewhat hotter on heat loss than the
  buried case, but it is numerically well-behaved -- confirmed directly
  against `compute_pipe_temp`, not just inferred from reading the formula.
* Each substation gets its own fixed design temperature drop
  ``design_delta_t = return_temp_snapshot - supply_temp_snapshot`` (i.e. the
  actual operating point observed in the snapshot), used to size the
  hydraulic control setpoint (`control_type="energy"`) under normal
  conditions.
* The thermal setpoint is `setpoint_type_hx="delta_q"` (energy, not a fixed
  delta-T), with `setpoint_value_hx` tied directly to `-heat_demand` each
  step (see `simulate.py`). Algebraically, `delta_q` comes out to exactly
  `-heat_demand` for *any* mass flow, as long as the resulting outlet
  temperature doesn't hit `t_out_min_hx` -- so under normal conditions this
  reproduces the same delta-T (and the same consistency-by-construction
  between hydraulics and thermal) as pinning `setpoint_type_hx="delta_t"`
  would. It only starts to differ, deliberately, when a substation's local
  supply temperature is too low to hit the design delta-T without the
  outlet temperature dropping below the floor -- see `simulate.py`'s
  per-step demand-correction loop, which raises that substation's
  `mass_flow_min` (trading delta-T for flow, same as a real control valve)
  until full demand is met without breaching the floor, or flags it as
  genuinely unmet if the local supply temperature itself is too low.
* HS0 is the network's single pressure-referenced ("main") producer; HS1 is
  a secondary producer with a prescribed mass-flow setpoint, updated at
  runtime to track total demand (see `schedules.py`).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pydhn import Network

#: Burial depth (m) used for non-aerial pipes. 0.8 m is the value quoted for
#: this network in Boghetti & Kaempf (2023).
BURIED_PIPE_DEPTH_M = 0.8

#: Depth (m) used for aerial pipes -- see module docstring for what this
#: actually means in pydhn's pipe thermal model.
AERIAL_PIPE_DEPTH_M = 0.0

#: Soil thermal conductivity (W/(m*K)) quoted for this network in the paper.
SOIL_K_W_PER_MK = 2.4

#: Pipe thermal-loss discretization length (m) for buried pipes.
PIPE_DISCRETIZATION_M = 10.0

#: Minimum allowed outlet (return) temperature at any consumer (degC). Below
#: this, the substation's control valve is assumed to have opened fully --
#: it can no longer trade delta-T for flow, so demand genuinely goes unmet.
#: pydhn's plain `Pipe` component can never itself produce a fluid
#: temperature below the local soil/ambient reference (it's clipped inside
#: `compute_pipe_temp`), so this floor at consumers is the one place in the
#: model where an unphysical sub-floor temperature could otherwise appear.
CONSUMER_T_OUT_MIN_C = 10.0

#: Pressure lift (Pa, negative = rise from return to supply) imposed at the
#: main producer. This is a free modeling parameter: since every other leaf
#: in the network has a prescribed mass flow (consumers via energy demand,
#: secondary producers via their schedule), the main producer's mass flow is
#: fully determined by overall mass conservation and does *not* depend on
#: this value -- it doesn't affect hydraulic convergence either (it never
#: enters the solved Newton-Raphson system; see below), only the absolute
#: pressure level pydhn reports.
#:
#: This network's actual round-trip friction loss (main producer outlet,
#: through the whole network, back to its inlet) is only ~25 Pa -- tiny
#: compared to any reasonable pump lift. pydhn's node-pressure bookkeeping
#: (`_assign_node_pressure`) doesn't reconcile that mismatch: the main
#: producer's own loop is deliberately left unconstrained by the solver
#: (constraining it would make the lift dependent on network friction
#: rather than a free input), so the whole return line's reported pressure
#: ends up offset from the supply line by exactly this lift value, not by
#: the network's real friction. A 1 bar lift keeps that offset modest
#: relative to the previous 4.5 bar default, without needing to match the
#: network's real ~25 Pa friction to make the reporting look sane.
MAIN_PRESSURE_LIFT_PA = -100_000.0


@dataclass
class NetworkReference:
    """Reference values derived from the network topology and the CASE1
    measurement snapshot, needed to drive time-varying schedules."""

    substation_ids: list
    heating_station_ids: list
    main_producer_id: str
    secondary_producer_ids: list

    # Per-substation design (snapshot) values
    design_delta_t: pd.Series  # K, negative (outlet - inlet)
    snapshot_power_w: pd.Series  # W, positive
    snapshot_mass_flow: pd.Series  # kg/s

    # Per-heating-station snapshot values
    hs_snapshot_supply_temp: pd.Series  # deg C
    hs_snapshot_return_temp: pd.Series  # deg C
    hs_snapshot_mass_flow: pd.Series  # kg/s
    hs_snapshot_power_w: pd.Series  # W

    # Fraction of total producer mass flow historically supplied by each
    # secondary producer (used to keep the same HS0/HS1 split as demand
    # varies over time).
    secondary_mass_flow_share: pd.Series

    aerial_pipe_ids: list = field(default_factory=list)
    buried_pipe_ids: list = field(default_factory=list)


def _read_topology(network_dir: Path):
    nodes = pd.read_csv(network_dir / "nodes.csv")
    pipes = pd.read_csv(network_dir / "pipes.csv")
    substations = pd.read_csv(network_dir / "substations.csv")
    heating_stations = pd.read_csv(network_dir / "heating_stations.csv")
    return nodes, pipes, substations, heating_stations


def _read_snapshot(data_dir: Path):
    mass_flow = pd.read_csv(data_dir / "mass_flow.csv", index_col=0).iloc[0]
    power = pd.read_csv(data_dir / "power.csv", index_col=0).iloc[0]
    supply_temp = pd.read_csv(data_dir / "supply_temperature.csv", index_col=0).iloc[0]
    return_temp = pd.read_csv(data_dir / "return_temperature.csv", index_col=0).iloc[0]
    return (
        mass_flow.astype(float),
        power.astype(float),
        supply_temp.astype(float),
        return_temp.astype(float),
    )


def build_network(
    network_dir,
    data_dir,
    buried_depth=BURIED_PIPE_DEPTH_M,
    discretization=PIPE_DISCRETIZATION_M,
    main_producer_id=None,
    main_pressure_lift_pa=MAIN_PRESSURE_LIFT_PA,
    consumer_t_out_min_c=CONSUMER_T_OUT_MIN_C,
):
    """
    Build the pydhn Network and derive reference/design values from the
    opendhn-data CSVs.

    Parameters
    ----------
    network_dir : Path-like
        Path to the `network/` folder (nodes.csv, pipes.csv, substations.csv,
        heating_stations.csv).
    data_dir : Path-like
        Path to the `data/` folder with the CASE1 measurement snapshot.
    buried_depth : float
        Burial depth (m) assigned to non-aerial pipes.
    discretization : float
        Target segment length (m) for the buried-pipe thermal-loss model.
    main_producer_id : str, optional
        Heating station to use as the network's pressure-referenced ("main")
        producer. Defaults to the first row of heating_stations.csv (HS0).
    consumer_t_out_min_c : float
        Minimum allowed consumer outlet (return) temperature (degC).

    Returns
    -------
    net : pydhn.Network
    ref : NetworkReference
    """
    network_dir = Path(network_dir)
    data_dir = Path(data_dir)

    nodes, pipes, substations, heating_stations = _read_topology(network_dir)
    mass_flow, power, supply_temp, return_temp = _read_snapshot(data_dir)

    net = Network()

    for _, node in nodes.iterrows():
        net.add_node(name=node["node_id"], x=node["x"], y=node["y"], z=node["z"])

    aerial_pipe_ids, buried_pipe_ids = [], []
    for _, pipe in pipes.iterrows():
        line = "supply" if pipe["is_supply"] else "return"
        is_aerial = bool(pipe["is_aerial"])
        (aerial_pipe_ids if is_aerial else buried_pipe_ids).append(pipe["pipe_id"])
        net.add_pipe(
            name=pipe["pipe_id"],
            start_node=pipe["inlet_node"],
            end_node=pipe["outlet_node"],
            length=pipe["length"],
            diameter=pipe["d_int"],
            roughness=pipe["roughness"],
            internal_pipe_thickness=pipe["t_int"],
            insulation_thickness=pipe["t_ins"],
            casing_thickness=pipe["t_ext"],
            k_insulation=pipe["lambda_ins"],
            line=line,
            depth=AERIAL_PIPE_DEPTH_M if is_aerial else buried_depth,
            discretization=discretization,
        )

    hs_ids = list(heating_stations["hs_id"])
    if main_producer_id is None:
        main_producer_id = hs_ids[0]
    secondary_ids = [h for h in hs_ids if h != main_producer_id]

    for _, hs in heating_stations.iterrows():
        is_main = hs["hs_id"] == main_producer_id
        net.add_producer(
            name=hs["hs_id"],
            start_node=hs["inlet_node"],
            end_node=hs["outlet_node"],
            setpoint_type_hx="t_out",
            setpoint_value_hx=float(supply_temp[hs["hs_id"]]),
            setpoint_type_hyd="pressure" if is_main else "mass_flow",
            static_pressure=1.8e6,
            setpoint_value_hyd=(
                main_pressure_lift_pa
                if is_main
                else float(mass_flow[hs["hs_id"]])
            ),
        )

    design_delta_t = {}
    for _, sub in substations.iterrows():
        sub_id = sub["sub_id"]
        dt = float(return_temp[sub_id]) - float(supply_temp[sub_id])
        design_delta_t[sub_id] = dt
        heat_demand_w = float(power[sub_id])
        net.add_consumer(
            name=sub_id,
            start_node=sub["inlet_node"],
            end_node=sub["outlet_node"],
            control_type="energy",
            heat_demand=heat_demand_w,
            design_delta_t=dt,
            setpoint_type_hx="delta_q",
            # Initial value assuming the default 3600 s stepsize; both this
            # and heat_demand are overwritten every step in simulate.py.
            setpoint_value_hx=-heat_demand_w,
            t_out_min_hx=consumer_t_out_min_c,
        )
    design_delta_t = pd.Series(design_delta_t)

    hs_mdot = mass_flow[hs_ids]
    total_hs_mdot = hs_mdot.sum()
    secondary_share = (
        hs_mdot[secondary_ids] / total_hs_mdot
        if total_hs_mdot
        else pd.Series(0.0, index=secondary_ids)
    )

    ref = NetworkReference(
        substation_ids=list(substations["sub_id"]),
        heating_station_ids=hs_ids,
        main_producer_id=main_producer_id,
        secondary_producer_ids=secondary_ids,
        design_delta_t=design_delta_t,
        snapshot_power_w=power[substations["sub_id"]],
        snapshot_mass_flow=mass_flow[substations["sub_id"]],
        hs_snapshot_supply_temp=supply_temp[hs_ids],
        hs_snapshot_return_temp=return_temp[hs_ids],
        hs_snapshot_mass_flow=hs_mdot,
        hs_snapshot_power_w=power[hs_ids],
        secondary_mass_flow_share=secondary_share,
        aerial_pipe_ids=aerial_pipe_ids,
        buried_pipe_ids=buried_pipe_ids,
    )

    return net, ref
