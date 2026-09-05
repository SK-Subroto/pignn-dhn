#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build time-varying schedules (consumer heat demand, producer supply
temperature, secondary-producer mass flow split) from the CASE1 snapshot and
an outdoor air temperature series.

With only a single measurement snapshot (one row per CSV) there is no way to
*fit* a weather response from data -- there is only one point. Instead, the
snapshot is treated as a **design/reference operating condition**, assumed
to occur at a chosen reference outdoor temperature `T_DESIGN` (a plausible
cold day for this climate/altitude). Away from that reference, demand and
supply temperature follow standard, explicitly-parameterized degree-day /
heating-curve relationships anchored so that they reproduce the snapshot
exactly at `T_DESIGN`. All assumptions are named constants below, not
hidden magic numbers, and are surfaced in the run's saved config.

For a full-year run, two effects matter that a single snapshot can't
possibly inform, so they're asserted explicitly rather than left out:
weekday/weekend occupancy patterns (`WEEKEND_DEMAND_FACTOR`, dominant for
commercial/industrial) and a persistent domestic-hot-water/process-load
baseline that doesn't vanish in summer (`dhw_share`, per consumer type).
Everything weather-driven (`f_demand`, the outdoor temperature itself) uses
real fetched data for the actual year simulated, not a synthetic proxy,
so day-to-day and season-to-season weather variability is real.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pydhn.default_values import CP_FLUID

#: Outdoor temperature (degC) at which the CASE1 snapshot is assumed to have
#: been recorded, i.e. the network's winter design condition.
T_DESIGN_C = -8.0

#: Outdoor temperature (degC) above which space-heating demand is assumed to
#: vanish (a standard heating-degree-day threshold).
T_NO_HEAT_C = 16.0

#: Heating-curve slope (degC supply decrease per degC outdoor increase),
#: anchored at the snapshot supply temperature at T_DESIGN_C.
SUPPLY_CURVE_SLOPE = 1.3
SUPPLY_TEMP_MIN_C = 60.0
SUPPLY_TEMP_MAX_C = 90.0
SUPPLY_TEMP_SMOOTHING_HOURS = 6

#: Consumer classification thresholds (percentile of snapshot power) and
#: per-type modeling parameters: `dhw_share` is the fraction of design power
#: assumed to be weather-independent (domestic hot water / process load that
#: persists even with no space-heating demand); `phase_shift_h` jitters the
#: daily profile's peak time per substation to avoid every building in a
#: class peaking in perfect lockstep.
CONSUMER_TYPES = {
    "residential": {"percentile": (0, 70), "dhw_share": 0.15},
    "commercial": {"percentile": (70, 90), "dhw_share": 0.10},
    "industrial": {"percentile": (90, 100), "dhw_share": 0.30},
}

#: Weekday/weekend demand multiplier by consumer type -- one of the most
#: well-established features of real building/DH load data that a
#: single-snapshot dataset can't tell us anything about, so it's asserted
#: explicitly here rather than left out. Residential demand is mildly
#: higher on weekends (more time at home); commercial drops sharply (most
#: shops/offices closed or on a reduced schedule); industrial drops
#: moderately (fewer weekend shifts, though some processes run
#: continuously). Applied to the full demand (space heating + DHW/process
#: baseline), not just the weather-driven part -- a real unoccupied
#: building still gets setback heating, not zero.
WEEKEND_DEMAND_FACTOR = {
    "residential": 1.05,
    "commercial": 0.45,
    "industrial": 0.75,
}

NOISE_STD = 0.06
NOISE_SMOOTHING_HOURS = 3


def _demand_factor(t_out):
    return np.clip((T_NO_HEAT_C - t_out) / (T_NO_HEAT_C - T_DESIGN_C), 0.0, 1.0)


def _normalized_daily_profile(consumer_type, hour_of_day):
    """Shape of demand across the day, normalized to a 24h mean of 1."""
    h = np.asarray(hour_of_day, dtype=float)

    def raw(hh):
        if consumer_type == "residential":
            morning = np.exp(-0.5 * ((hh - 7.0) / 2.0) ** 2)
            evening = np.exp(-0.5 * ((hh - 19.0) / 2.5) ** 2)
            return 1.0 + 1.4 * (morning + evening)
        elif consumer_type == "commercial":
            return 1.0 + 1.2 * np.exp(-0.5 * ((hh - 13.0) / 3.5) ** 2)
        else:  # industrial
            return 1.0 + 0.2 * np.exp(-0.5 * ((hh - 12.0) / 6.0) ** 2)

    mean_shape = np.mean(raw(np.linspace(0, 24, 288, endpoint=False)))
    return raw(h % 24.0) / mean_shape


def classify_consumers(snapshot_power):
    """Buckets substations into types by percentile of snapshot power."""
    types = pd.Series(index=snapshot_power.index, dtype=object)
    for name, cfg in CONSUMER_TYPES.items():
        lo, hi = np.percentile(snapshot_power.values, cfg["percentile"])
        if cfg["percentile"][1] == 100:
            mask = snapshot_power >= lo
        else:
            mask = (snapshot_power >= lo) & (snapshot_power < hi)
        types[mask] = name
    types = types.fillna("residential")
    return types


@dataclass
class Schedules:
    t_air: pd.Series  # degC, index = simulation timestamps
    heat_demand: pd.DataFrame  # W, columns = substation ids
    supply_temperature: pd.DataFrame  # degC, columns = heating station ids
    secondary_mass_flow: pd.DataFrame  # kg/s, columns = secondary producer ids
    consumer_types: pd.Series


def build_schedules(ref, t_air, seed=0):
    """
    Build all time-varying schedules from the network reference values and
    an outdoor air temperature series `t_air` (indexed by the simulation's
    DatetimeIndex).

    Parameters
    ----------
    ref : network_builder.NetworkReference
    t_air : pd.Series
    seed : int
        Seed for the per-substation demand noise (reproducibility).
    """
    index = t_air.index
    hours = index.hour.values + index.minute.values / 60.0
    is_weekend = index.dayofweek.values >= 5  # Saturday=5, Sunday=6

    f_demand = pd.Series(_demand_factor(t_air.values), index=index)

    consumer_types = classify_consumers(ref.snapshot_power_w)

    rng = np.random.default_rng(seed)
    demand = {}
    for sub_id in ref.substation_ids:
        ctype = consumer_types[sub_id]
        dhw_share = CONSUMER_TYPES[ctype]["dhw_share"]
        q_design = ref.snapshot_power_w[sub_id]

        phase_shift = rng.uniform(-1.5, 1.5)
        daily = _normalized_daily_profile(ctype, hours + phase_shift)

        raw_noise = rng.normal(0.0, NOISE_STD, size=len(index))
        noise = 1.0 + pd.Series(raw_noise).ewm(span=NOISE_SMOOTHING_HOURS).mean().values

        weekend_mult = np.where(is_weekend, WEEKEND_DEMAND_FACTOR[ctype], 1.0)

        weather_term = dhw_share + (1.0 - dhw_share) * f_demand.values
        demand[sub_id] = q_design * weather_term * daily * noise * weekend_mult

    heat_demand = pd.DataFrame(demand, index=index).clip(lower=0.0)

    supply_temp = {}
    for hs_id in ref.heating_station_ids:
        t_design = ref.hs_snapshot_supply_temp[hs_id]
        raw = t_design - SUPPLY_CURVE_SLOPE * (t_air.values - T_DESIGN_C)
        raw = np.clip(raw, SUPPLY_TEMP_MIN_C, SUPPLY_TEMP_MAX_C)
        smoothed = (
            pd.Series(raw, index=index)
            .rolling(SUPPLY_TEMP_SMOOTHING_HOURS, min_periods=1)
            .mean()
        )
        supply_temp[hs_id] = smoothed

    supply_temperature = pd.DataFrame(supply_temp, index=index)

    # Total consumer mass flow implied by the demand schedule and each
    # substation's fixed design delta-T (same relation pydhn's Consumer uses
    # internally: mdot = heat_demand / (-design_delta_t * cp)).
    dt = ref.design_delta_t.reindex(ref.substation_ids)
    per_sub_mdot = heat_demand[ref.substation_ids].div(-dt * CP_FLUID, axis=1)
    total_mdot = per_sub_mdot.sum(axis=1)

    secondary_mass_flow = pd.DataFrame(
        {
            hs_id: total_mdot * ref.secondary_mass_flow_share[hs_id]
            for hs_id in ref.secondary_producer_ids
        },
        index=index,
    )

    return Schedules(
        t_air=t_air,
        heat_demand=heat_demand,
        supply_temperature=supply_temperature,
        secondary_mass_flow=secondary_mass_flow,
        consumer_types=consumer_types,
    )
