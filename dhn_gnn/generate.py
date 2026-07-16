"""Synthetic steady-state dataset generator (port of synthetic_data_steady_custom_v4.ipynb).

Reproduces the v4 pipeline entirely inside this project:

    outdoor weather  ->  hourly heat-demand profiles  ->  mass-flow schedules
                     ->  PyDHN hydraulics-only simulation (745 steps)  ->  solved CSVs

Differences from the original notebook, for self-containment + reproducibility:
  * weather is read from a cached CSV (data/weather/...), not fetched live from
    meteostat. Re-cache with `fetch_weather()` if you need a different window.
  * the RNG is SEEDED, so a given (seed, weather) reproduces the same dataset.
Because the original notebook used live weather + an unseeded RNG, output here is
statistically equivalent to the bundled `data/solved_steady_v4`, not byte-identical.

Usage:
    python -m dhn_gnn.generate                 # full 745-step run -> data/solved_<ver>
    python -m dhn_gnn.generate --steps 24      # quick smoke run
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pydhn.fluids import Water
from pydhn.networks import Network
from pydhn.soils import Soil
from pydhn.solving import Scheduler, SimpleStep

from dhn_gnn import config

CP_WATER = 4182.0  # J/(kg*K), PyDHN's default specific heat of water
WEATHER_CSV = config.DATA_ROOT / "weather" / "outdoor_temp_verbier_2022-01.csv"


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def fetch_weather(start="2022-01-01", end="2022-02-01", freq="1h", save=True):
    """Fetch Verbier hourly outdoor temperature from meteostat and (optionally) cache it."""
    import meteostat as ms
    point = ms.Point(46.1, 7.23, 1500)
    st = ms.stations.nearby(point, limit=4)
    w = ms.interpolate(ms.hourly(st, pd.Timestamp(start), pd.Timestamp(end)), point).fetch()
    T_out = w["temp"]
    if freq != "1h":
        T_out = w.resample(freq).ffill()["temp"]
    if save:
        WEATHER_CSV.parent.mkdir(parents=True, exist_ok=True)
        T_out.rename_axis("time").to_frame("temp").to_csv(WEATHER_CSV)
    return T_out


def load_weather():
    """Load the cached outdoor-temperature series (fetch first if missing)."""
    if not WEATHER_CSV.exists():
        return fetch_weather()
    s = pd.read_csv(WEATHER_CSV, index_col=0, parse_dates=True)["temp"]
    return s


# --------------------------------------------------------------------------- #
# Demand / temperature / mass-flow schedule generation (verbatim logic)
# --------------------------------------------------------------------------- #
def weather_factor(T, T_ref=-5, T_base=18):
    """Heating-demand scaling in [0, 1]: 0 above T_base, 1 at/below T_ref."""
    return np.clip((T_base - T) / (T_base - T_ref), 0, 1)


def assign_consumer_types(ref_power, consumers):
    """Bucket substations into residential/commercial/industrial by snapshot power."""
    Q_vals = [ref_power[sub].iloc[0] for sub in consumers["sub_id"]]
    q1, q2 = np.percentile(Q_vals, 70), np.percentile(Q_vals, 90)
    types = {}
    for sub in consumers["sub_id"]:
        Q0 = ref_power[sub].iloc[0]
        types[sub] = "residential" if Q0 <= q1 else ("commercial" if Q0 <= q2 else "industrial")
    return types


def daily_factor(t, consumer_type, shift):
    """Per-type daily load shape, randomly shifted in time by `shift` hours."""
    hour = (t.hour + t.minute / 60 + shift) % 24
    if consumer_type == "residential":
        morning = np.exp(-0.5 * ((hour - 7) / 2) ** 2)
        evening = np.exp(-0.5 * ((hour - 19) / 3) ** 2)
        return 0.5 + 0.5 * (morning + evening)
    if consumer_type == "commercial":
        return np.exp(-0.5 * ((hour - 13) / 3) ** 2)
    return 0.8 + 0.2 * np.exp(-0.5 * ((hour - 12) / 6) ** 2)  # industrial


MIN_LOAD = {"residential": 0.10, "commercial": 0.15, "industrial": 0.60}


def compute_peak_power(power, ref_temp=0):
    """Back out each entity's nominal/peak power from the single snapshot."""
    f_w0, f_d0 = weather_factor(ref_temp), 0.9
    peak = power.copy()
    cols = peak.columns[1:]
    peak[cols] = peak[cols].astype(float)
    peak.loc[0, cols] /= f_w0 * f_d0
    return peak


def generate_demand(ref_power, consumers, weather_data, time_index, rng):
    """Expand each substation's snapshot power into an hourly demand profile (W)."""
    Q_profiles, ctypes = {}, assign_consumer_types(ref_power, consumers)
    for sub_id in consumers["sub_id"]:
        Q0, ctype = ref_power[sub_id].iloc[0], ctypes[sub_id]
        shift = rng.uniform(-1.5, 1.5)
        noise = pd.Series(rng.normal(0, 0.05, len(time_index))).ewm(alpha=0.1).mean().values
        Q_t = []
        for i, t in enumerate(time_index):
            T = weather_data.loc[t]
            Q = Q0 * weather_factor(T) * daily_factor(t, ctype, shift) * (1 + noise[i])
            Q = max(Q, Q0 * MIN_LOAD[ctype])
            Q = Q0 * np.tanh(Q / Q0)  # soft saturation
            Q_t.append(Q)
        Q_profiles[sub_id] = pd.Series(Q_t).ewm(alpha=0.2).mean().values  # thermal inertia
    return pd.DataFrame(Q_profiles, index=time_index), ctypes


def generate_supply_temperature(weather_data, time_index, T_min=70, T_max=95):
    """Linear, weather-coupled supply temperature for HS0/HS1, fit to the snapshot."""
    rows = [(np.clip(81.11 - 1.5 * weather_data.loc[t], T_min, T_max),
             np.clip(81.81 - 1.5 * weather_data.loc[t], T_min, T_max)) for t in time_index]
    df = pd.DataFrame(rows, index=time_index, columns=["HS0", "HS1"])
    return df.rolling(8, min_periods=1).mean()


def compute_producer_massflow(power_ref, m_ref, power_profiles, supply_temp_hs1, dt_ref=37.74):
    """Scale HS1 mass flow via the energy balance Q = m*cp*dT, relative to the snapshot."""
    power_total = power_profiles.sum(axis=1)
    dt_current = supply_temp_hs1 - 42
    return pd.DataFrame(m_ref * (power_total / power_ref) * (dt_ref / dt_current), columns=["HS1"])


# --------------------------------------------------------------------------- #
# Network (same construction as network_operators.build_network)
# --------------------------------------------------------------------------- #
def build_network(nodes, pipes, subs, hs, mass_flow):
    net = Network()
    for _, n in nodes.iterrows():
        net.add_node(name=n["node_id"], x=n["x"], y=n["y"], z=n["z"],
                     line="supply" if n["is_supply"] else "return")
    for _, p in pipes.iterrows():
        net.add_pipe(name=p["pipe_id"], start_node=p["inlet_node"], end_node=p["outlet_node"],
                     length=p["length"], diameter=p["d_int"], roughness=p["roughness"],
                     internal_pipe_thickness=p["t_int"], insulation_thickness=p["t_ins"],
                     casing_thickness=p["t_ext"], k_insulation=p["lambda_ins"],
                     line="supply" if p["is_supply"] else "return",
                     depth=0 if p["is_aerial"] else 0.8, stepsize=config.STEPSIZE, discretization=1)
    for idx, h in hs.iterrows():
        net.add_producer(
            name=h["hs_id"], start_node=h["inlet_node"], end_node=h["outlet_node"],
            setpoint_type_hx="t_out",
            setpoint_type_hyd="pressure" if idx == 0 else "mass_flow",
            setpoint_value_hyd=-100000 if idx == 0 else mass_flow["HS1"].iloc[0],
            stepsize=config.STEPSIZE)
    for _, s in subs.iterrows():
        net.add_consumer(name=s["sub_id"], start_node=s["inlet_node"], end_node=s["outlet_node"],
                         setpoint_type_hx="t_out", setpoint_value_hx=45,
                         setpoint_type_hyd="mass_flow", control_type="mass_flow",
                         stepsize=config.STEPSIZE)
    return net


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def generate(version="steady_v4_regen", steps=None, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    nd = config.NETWORK_DIR
    md = config.MEASUREMENTS_DIR
    nodes = pd.read_csv(nd / "nodes.csv"); pipes = pd.read_csv(nd / "pipes.csv")
    subs = pd.read_csv(nd / "substations.csv"); hs = pd.read_csv(nd / "heating_stations.csv")
    mass_flow = pd.read_csv(md / "mass_flow.csv"); power = pd.read_csv(md / "power.csv")
    supply_temp = pd.read_csv(md / "supply_temperature.csv")
    return_temp = pd.read_csv(md / "return_temperature.csv")

    T_out = load_weather()
    time_index = T_out.index if steps is None else T_out.index[:steps]
    T_out = T_out.loc[time_index]

    # 1. schedules
    peak_power = compute_peak_power(power)
    Q_profiles, ctypes = generate_demand(peak_power, subs, T_out, time_index, rng)
    supply_temp_df = generate_supply_temperature(T_out, time_index)

    design_dt = supply_temp[subs["sub_id"]].iloc[0] - return_temp[subs["sub_id"]].iloc[0]
    consumer_mf = Q_profiles[subs["sub_id"]].div(CP_WATER * design_dt, axis=1).clip(lower=1e-6)

    m_ref = mass_flow["HS1"].iloc[0]
    total_power_ref = power.drop(columns=["Unnamed: 0", "HS0", "HS1"]).sum(axis=1).iloc[0]
    producer_mf = compute_producer_massflow(total_power_ref, m_ref, Q_profiles, supply_temp_df["HS1"])
    hydraulic_schedule = pd.concat([producer_mf, consumer_mf], axis=1)

    # 2. simulate (hydraulics only, exactly as v4)
    net = build_network(nodes, pipes, subs, hs, mass_flow)
    hydro_loop = SimpleStep(with_thermal=False,
                            hydraulic_sim_kwargs={"error_threshold": 50, "verbose": False, "max_iters": 500})
    scheduler = Scheduler(base_loop=hydro_loop, with_thermal=False,
                          schedules={"setpoint_value_hyd": hydraulic_schedule}, steps=len(time_index))
    t0 = time.perf_counter()
    results = scheduler.execute(net=net, fluid=Water(), soil=Soil())
    dt = time.perf_counter() - t0

    out_dir = config.DATA_ROOT / f"solved_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir)
    conv = results["history"]["hydraulics converged"]
    if verbose:
        print(f"Generated {len(time_index)} steps in {dt:.1f}s "
              f"({dt/len(time_index):.3f}s/step); converged {sum(conv)}/{len(conv)}")
        print(f"Saved -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="steady_v4_regen")
    ap.add_argument("--steps", type=int, default=None, help="limit steps (default: all 745)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refetch-weather", action="store_true", help="re-fetch weather from meteostat")
    args = ap.parse_args()
    if args.refetch_weather:
        fetch_weather()
    generate(version=args.version, steps=args.steps, seed=args.seed)
