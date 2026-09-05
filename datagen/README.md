> **Ported into pignn-dhn.** This is the standalone `pydhnClaude/simulation`
> package, adapted so it reads the topology and CASE1 snapshot already in this
> repo (`data/network/`, `data/measurements/` — byte-identical to the
> standalone project's `opendhn-data/`). Paths written as `opendhn-data/...`
> below are the defaults of the standalone version; here they default to the
> project's own copies. Entry points are `python -m datagen.run` and, to publish
> a finished run into the layout `dhn_gnn` reads, `python -m datagen.export`.
> See the root README section "Dataset generation".

# Verbier DHN thermohydraulic steady-state simulation

A configurable, time-marching thermohydraulic **steady-state** simulation of
the Verbier district heating network (`opendhn-data/`), built on top of the
[`pydhn`](https://github.com/idiap/pydhn) library. At each timestep the
network is solved to full hydraulic + thermal steady state under that
timestep's boundary conditions (weather-driven producer supply temperature,
weather- and occupancy-driven consumer demand); pipe transport delay and
thermal inertia are intentionally neglected, consistent with an
hourly-or-coarser "steady-state" simulation (as opposed to a fully dynamic
Lagrangian one, which `pydhn` also supports but which this project does not
use).

## Why this isn't just the reference notebook

`synthetic_data steady custom.ipynb` explores similar territory but has a
few issues this implementation fixes:

* `compute_producer_massflow` (meant to scale HS1's mass flow with total
  demand) is an unfinished stub (`pass`) and is called with the wrong number
  of arguments -- HS1's mass flow was actually left at a single constant
  value, unrelated to demand. Here it's implemented properly: `HS1`'s mass
  flow tracks total demand while preserving the historical HS0/HS1 split
  ratio (`schedules.py::build_schedules`).
* All substations were pinned to one fixed outlet temperature (45 degC),
  independent of the real per-substation return temperature the CASE1
  snapshot actually shows (ranging ~18-55 K delta-T across substations).
  Reconciling a fixed mass-flow-from-energy demand with a fixed *outlet
  temperature* target requires an outer fixed-point loop (present in the
  notebook: up to 10 extra iterations per step) because the two setpoints
  aren't consistent by construction. Here, each substation instead keeps
  its own **design delta-T**, fitted from its own snapshot supply/return
  temperatures, used to size the nominal mass flow, with an
  energy-based thermal setpoint (`setpoint_type_hx="delta_q"`) tied
  directly to demand -- see "Consumer demand delivery" below for how this
  avoids the notebook's iteration loop in the normal case, and what happens
  at the edges where it can't be avoided.
* The demand-generation "peak power" back-calculation assumed two different,
  mutually inconsistent reference outdoor temperatures in two different
  places. Here there's a single, explicit reference condition (see below).

## Modeling approach

**The CASE1 snapshot has no timestamp.** With only one measurement per
sensor, there's no way to fit a real weather response (that needs at least
two points). Instead of guessing backwards to an unknown historical
condition, the snapshot is treated as the network's **design/reference
operating point**, assumed to occur at a stated reference outdoor
temperature `T_DESIGN_C = -8 degC` (`schedules.py`). Away from that point:

* **Consumer demand** scales via a standard linear degree-day factor
  between `T_NO_HEAT_C = 16 degC` (no space heating above this) and
  `T_DESIGN_C` (full design load), plus a persistent domestic-hot-water/
  process-load baseline (`dhw_share`, type-dependent) that doesn't vanish in
  warm weather, plus a per-substation-type normalized daily occupancy
  profile (residential/commercial/industrial, classified by snapshot power
  percentile) with a random phase shift, plus a weekday/weekend multiplier
  per type (`WEEKEND_DEMAND_FACTOR` -- commercial drops to ~45% on
  weekends, industrial ~75%, residential rises slightly to ~105%; this
  matters over a full year and a single snapshot can't inform it, so it's
  asserted explicitly rather than omitted), plus smoothed per-substation
  noise. The outdoor temperature driving all of this is real fetched
  weather for the actual calendar year simulated, not synthetic.
* **Producer supply temperature** follows a linear heating curve anchored at
  the snapshot value at `T_DESIGN_C`, with slope `SUPPLY_CURVE_SLOPE`,
  clipped to `[60, 90] degC` and smoothed with a 6h rolling mean (real
  control systems don't chase instantaneous temperature swings).
* **HS1's mass-flow setpoint** (the network's secondary producer) is scaled
  each step to track total implied consumer mass flow while preserving the
  historical HS0/HS1 split fraction from the snapshot. HS0 (the main,
  pressure-referenced producer) automatically absorbs the rest via mass
  conservation in `pydhn`'s solver -- its own pressure-lift setpoint value
  does *not* affect the resulting flow split, only the absolute pressure
  level (see `network_builder.MAIN_PRESSURE_LIFT_PA`).
* **Aerial pipes** (`is_aerial=True`, ~8% of pipes) use the standard `Pipe`
  component with `depth=0`, rather than a separate above-ground component.
  In pydhn's buried-pipe formula the soil-conduction term is mathematically
  singular at `depth=0` (`log(0)`), but pydhn's `safe_divide` helper clamps
  that to a resistance of exactly 0 -- confirmed directly against
  `compute_pipe_temp`, not just inferred from the formula. In practice this
  models an aerial pipe as having no resistance beyond its own pipe-wall/
  insulation/casing layers, referenced to the soil model's depth-0
  ("surface") temperature -- a workable simplification, but it skips any
  explicit convective film resistance to open air and inherits the soil
  model's smoothed seasonal surface temperature rather than raw hourly air
  temperature, so it runs somewhat hotter on heat loss than the buried case
  (roughly +8% in a spot check with typical pipe dimensions).

All of these constants are named at the top of `schedules.py` /
`network_builder.py` and are dumped into each run's `run_metadata.json` for
traceability. They are reasonable, explicit engineering assumptions, not a
fit to real weather -- flagged clearly so they can be swapped for better
data (e.g. a real SCADA time series) if it becomes available.

**Validation against the snapshot**: replaying the CASE1 snapshot itself
through this model (demand, design delta-T, and pressure lift all taken
directly from the snapshot) reproduces the measured heating-station return
temperatures to within ~0.4 degC and node pressures in a physically sane
5.5-10 bar range -- i.e. the pipe network, mixing, and substation model
reconstruct the real aggregate behavior well before any weather-driven
extrapolation is applied.

## Consumer demand delivery and the outlet-temperature floor

Each consumer has a `t_out_min_hx` floor (`network_builder.CONSUMER_T_OUT_MIN_C
= 10 degC` by default) below which its return temperature isn't allowed to
drop. pydhn's `Consumer` uses `setpoint_type_hx="delta_q"` rather than a
fixed delta-T: algebraically, `delta_q` comes out to exactly `-heat_demand`
for *any* mass flow, as long as `t_out` doesn't hit that floor -- so mass
flow is a free lever `simulate._solve_with_demand_correction` can raise
(trading delta-T for flow, like a real control valve opening further) to
keep meeting full demand even when a substation's local supply temperature
sags. This only costs extra solves on the (usually small) subset of
steps/substations that actually approach the floor; a normal step still
costs exactly one hydraulic+thermal solve.

pydhn's `t_out_min_hx` clip is unconditional, though: if a substation's
*inlet* temperature is already at or below the floor, clipping `t_out` up
to it would make the consumer inject heat into the return line
(`t_out > t_in`) -- exactly as unphysical as the sub-floor temperature the
floor is meant to prevent, since a passive heat exchanger cannot produce an
outlet warmer than its own inlet. When that happens, the substation's
demand is zeroed and its own floor is disabled for that step only (both
reset next step), so it honestly reports zero delivery and `t_out == t_in`
instead of manufacturing energy. That shortfall is recorded per step in
`network_summary.csv` (`n_substations_demand_unmet`, `unmet_demand_w`).

In practice this only affects the network's single smallest substation
(`S132`, 299 W design load, on a low-flow branch that can't stay warm at
near-zero draw) for a handful of hours in cold spells, and the energy
involved is negligible in aggregate (~0.002% of total demand in a 3-day
test). Whether that counts as "acceptable" is a judgment call: forcing a
hard floor everywhere would require fixing the pipe/hydraulic side (e.g. a
minimum bypass flow on that branch -- pydhn has a `BypassPipe` component
for exactly this) rather than the consumer, which this implementation
doesn't do. A stagnant low-flow branch settling to near the local ground
temperature in winter is a real phenomenon, not a modeling artifact, so
letting it show up in the results (clearly flagged) was judged preferable
to forcing an artificial floor that would hide it.

## Package layout

| File | Purpose |
|---|---|
| `network_builder.py` | Builds the `pydhn.Network` from the opendhn-data CSVs (aerial pipes at `depth=0`, buried pipes at `depth=0.8`); derives per-substation design delta-T and other reference/design values from the CASE1 snapshot. |
| `weather.py` | Outdoor air temperature time series (live via `meteostat`, with a deterministic synthetic Alpine-climate fallback if offline). Always fetches whole calendar year(s) of hourly data, since `KusudaSoil`'s seasonal model requires it, decoupled from the simulation's own (possibly coarser) timestep. |
| `schedules.py` | Builds the time-varying consumer demand, producer supply-temperature, and secondary-producer mass-flow schedules from the network reference values and the weather series. |
| `simulate.py` | `SimulationConfig` + `run_simulation()`: the main per-timestep loop and results/plot saving. |
| `run.py` | CLI entrypoint. |

## Usage

```bash
# Quick 3-day hourly demo
python -m simulation.run --start 2022-01-03 --end 2022-01-06 \
    --output-dir results/demo_3day

# Full year at 6-hour resolution (resolves the seasonal cycle, much faster
# than hourly)
python -m simulation.run --start 2022-01-01 --end 2022-12-31 --freq 6h \
    --output-dir results/full_year_6h

# Full year hourly -- see "Solver performance" below before running this
python -m simulation.run --start 2022-01-01 --end 2022-12-31 --freq 1h \
    --output-dir results/full_year_1h
```

Key options: `--freq` (any pandas frequency string: `15min`, `1h`, `6h`,
`1D`, ...) and `--start`/`--end` control the timestep and horizon per the
requirements; `--no-full-state` skips saving the full 1352-node/1514-edge
state each step (keeps only the substation/heating-station/summary
timeseries, recommended for long horizons); `--offline` skips the live
weather fetch; `--main-pressure-lift-pa` overrides the main producer's
pressure setpoint.

Programmatic use: `simulation.simulate.SimulationConfig` +
`run_simulation(config)`.

## Output structure

```
<output_dir>/
  run_metadata.json              # full config + modeling assumptions used
  schedules/                      # the time-varying inputs actually applied
    heat_demand_w.csv                    # per substation
    supply_temperature_c.csv             # per heating station
    secondary_producer_mass_flow_kgs.csv
    outdoor_air_temperature_c.csv
    consumer_types.csv
  timeseries/
    substations_mass_flow_kgs.csv
    substations_delta_q_w.csv
    substations_inlet_temperature_c.csv
    substations_outlet_temperature_c.csv
    heating_stations_*.csv               # same metrics, per heating station
    network_summary.csv                  # totals, energy balance, pressure
                                          # range, solver convergence/timing,
                                          # demand-correction activity
  full_network_state/             # optional (--no-full-state to skip):
    nodes-temperature.csv, nodes-pressure.csv
    edges-mass_flow.csv, edges-delta_q.csv, edges-inlet_temperature.csv, ...
                                          # all 1352 nodes / 1514 edges
  plots/
    weather_and_demand.png, energy_balance.png,
    pressure_range.png, supply_temperature_schedule.png
  weather_cache/                  # cached weather fetch, reused on rerun
```

## Solver performance (important)

`pydhn`'s hydraulic/thermal solvers re-read edge attributes through
per-edge Python/NetworkX calls on every Newton iteration rather than
maintaining a persistent vectorized cache; on this network (1352 nodes,
1514 edges) that makes a single steady-state step take on the order of a
few seconds, occasionally more at certain flow regimes. This is a
characteristic of the library on a network this size, not of this
simulation's code. Concretely:

* A single step: ~2-8 s typically, occasionally ~15-20 s.
* A full year, hourly (8760 steps): expect several hours. Use a coarser
  `--freq` (e.g. `6h`) or a shorter horizon for quick iteration, and reserve
  hourly full-year runs for a long/background run.

Two things were needed to make the solver reliably *converge* (not just be
slow) across the demand range this network will see over a year:

* `affine_fd=True` in `solve_hydraulics`, smoothing the laminar/turbulent
  friction-factor transition.
* A fixed, non-adaptive Newton damping factor of `0.3` -- the library's
  default adaptive damping was observed to settle into a stable 2-cycle
  oscillation (never converging) at some demand levels on this specific
  meshed topology. See `simulate.DEFAULT_HYDRAULIC_KWARGS`.

Each step's convergence flags, iteration counts, and wall-clock time are
recorded in `network_summary.csv` so a long run can be audited for any
step that didn't converge (the simulation logs a warning and continues
rather than aborting the whole run).

## Known simplifications

* Quasi-steady assumption: no pipe transport delay or thermal inertia
  within/between steps (use `pydhn`'s `LagrangianPipe` + dynamic simulation
  loop instead if that matters for your use case).
* No capacity limits (`power_max_hx`) are imposed on consumers/producers --
  demand and supply temperature can in principle exceed the snapshot's
  values (by design, to allow colder-than-snapshot conditions), unbounded.
  Consumers do have a `t_out_min_hx` floor -- see "Consumer demand delivery
  and the outlet-temperature floor" above.
* Single calendar year of weather is fetched per run; a horizon spanning a
  year boundary will use a soil seasonal model referenced to the first
  year's coldest day, which is a minor approximation right at the boundary.
