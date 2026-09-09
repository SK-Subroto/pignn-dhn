# pignn-dhn — Physics-Informed GNN Solver for District Heating Hydraulics

A physics-informed, unrolled Graph Neural Network that solves the **steady-state
hydraulics** of a district heating network (DHN) — i.e. the mass flow in every pipe
given the consumer/producer setpoints. It ports the methodology of Kim et al.'s
edge-attention + unrolled-correction GNN for **AC power flow**
([`PIGNN-Attn-LS`](../PIGNN-Attn-LS)) to DHN hydraulics, replacing PyDHN's
Newton–Raphson loop solver with a learned unrolled corrector.

Training is **self-supervised on the governing equation** `B·φ(Bᵀ·m̃) = 0` — no
Newton–Raphson labels enter the gradient (same philosophy as the paper's PINN mode).

> **Status — read `docs/model_layer_status.md` before relying on this.** The physics
> core and the unrolled solver are built and verified. On the bundled Verbier network
> the **physics (Newton) base already solves the problem to tolerance**, and the
> *learned* refinement adds no measurable value — that network is nearly radial (only
> 12 independent loops), so there is little for learning to improve. The learned method
> is expected to pay off on **large, meshed** networks, which this benchmark isn't.

---

## Governing equation & mapping

| AC-PF (paper) | DHN (here) | Code |
|---|---|---|
| state `[V, θ]` per node | loop mass flows `m̃` (dim = E−N+1) | `approaches/unrolled/model.py` |
| residual `[ΔP; ΔQ]` | loop residual `B·φ(Bᵀ·m̃)` | `physics/darcy.py`, `physics/operators.py` |
| admittance `Y` | Darcy–Weisbach resistance `φ` | `physics/darcy.py` |
| edge admittance bias `β_ij` | linearized resistance `∂φ/∂ṁ` (recomputed per step) | `solvers/attention.py` |

Full derivation and design decisions: [`docs/architecture_mapping.md`](docs/architecture_mapping.md).

## How it works (one paragraph)

The state is corrected in the network's **cycle space** (`mdot = mdot0 + Z·c`, where
`Z = Bᵀ` restricted to independent loops), so **mass conservation holds by construction**
(`A·Z = 0`). Each unrolled step applies a **damped diagonal-Newton step** from the known
operator **plus a bounded, learned GNN refinement** (edge-biased multi-head attention over
the pipe graph). Training uses **truncated backprop** (state detached between steps — needed
because backprop through the quadratic operator otherwise explodes) with a **discounted
per-step physics-residual loss**.

---

## Repository layout

One rule explains the whole tree:

> **Shared by every approach → top level. Specific to one approach → inside its
> folder under `approaches/`.**

Every name is used once and says what it is: `physics/` holds equations, the root
`data/` holds numbers; `solvers/` is shared machinery, `approaches/*/model.py` is
one network; `config.py` is project settings, `approaches/*/hparams.py` is knobs.

```
dhn_gnn/
  cli.py                    single entry point (train / evaluate / benchmark / compare / config)
  config.py                 constants (50 °C fluid props), data paths, results/<arch>/<run>/ layout
  losses.py                 discounted physics loss
  datasets.py               sample construction, timestep split, operator cache
  checkpoints.py            save/load weights together with their model kwargs

  physics/                  THE EQUATIONS (the root data/ folder holds the numbers)
    darcy.py                differentiable PyTorch port of PyDHN's Darcy–Weisbach φ, ∂φ/∂ṁ
    operators.py            builds cycle matrix B, incidence A, internal-loop mask (once)
    pressure.py             least-squares pressure reconstruction from Δp

  solvers/                  SHARED solver machinery
    physics_base.py         DHNPhysicsBase: edge_flow, pipe_dp, dp_der, residual
    newton.py               NewtonSolver — pure physics, the control condition
    attention.py            edge-biased multi-head self-attention (paper's block)
    stability.py            velocity cap + Armijo (eval-time stabilizers)

  approaches/               ONE PACKAGE PER LEARNED APPROACH
    hyperparams.py          knob machinery (dataclasses <-> YAML <-> --set)
    __init__.py             registry; resolves model/trainer lazily
    unrolled/               GNN correction at every one of K steps (original)
      hparams.py            UnrolledConfig: model + train knobs
      model.py              DHNUnrolledSolver
      train.py              deep-supervised training loop + fit()
    initializer/            GNN once, then exact Newton (current)
      hparams.py            InitializerConfig
      model.py              DHNInitializerSolver
      train.py              regression on a_star + fit()
    predictor/              GNN once, NO Newton (the ablation)
      hparams.py            PredictorConfig
      model.py              DHNPredictorSolver
      train.py              physics-residual training + fit()

  reporting/                evaluate, compare (per-run) + benchmark, full_report (cross-approach)
configs/                    editable YAML recipes, one per approach
datagen/                    the PyDHN simulator that MAKES the ground truth
data/
  network/                  topology + pipe geometry (Verbier, 1352 nodes / 1514 edges)
  measurements/             real boundary snapshot
  weather/                  cached outdoor temperature (for reproducible generation)
  solved_<version>/         published ground truth, from datagen (git-ignored, ~1.5 GB)
  raw_runs/                 full datagen output dirs (git-ignored)
docs/                       architecture mapping, spike plan, status/findings
tests/                      verification gates (physics core + model invariants)
results/
  <arch>/<run>/             config.yaml, model.pt, metrics.txt, pred-*.csv, model_eval.png
  benchmark.txt             cross-approach, so it belongs to no single one
  report_data.json          "
```

## Setup

```bash
pip install -r requirements.txt        # torch, numpy, pandas, matplotlib, pydhn, meteostat
```
Python 3.12. `torch` is CPU here; for GPU install the matching CUDA build first.

## Usage

```bash
# 1. verify the physics core (differentiable φ/∂φ/∂ṁ + cycle matrix B) — 13 gates
python tests/run_gates.py

# 2. verify the model layer invariants (mass conservation, gradients, ...) — 11 checks
python tests/run_model_smoke.py

# 3. train an approach -> results/<arch>/<run>/
python -m dhn_gnn.cli train --arch initializer --run default   # GNN + Newton polish
python -m dhn_gnn.cli train --arch unrolled    --run default   # GNN every step
python -m dhn_gnn.cli train --arch predictor   --run default   # GNN only, no Newton

# 4. score it -> results/<arch>/<run>/{metrics.txt, pred-*.csv, model_eval.png}
python -m dhn_gnn.cli evaluate --arch initializer --run default
python -m dhn_gnn.cli compare  --arch initializer --run default

# 5. all approaches vs PyDHN on identical timesteps -> results/benchmark.txt
python -m dhn_gnn.cli benchmark

# 6. (optional) regenerate the ground-truth dataset — simulates AND publishes
python -m datagen.run --full-year --output-dir data/raw_runs/steady   # ~10 HOURS
```

### Dataset generation

`datagen/` is the simulator that produces the PyDHN ground truth the models are
trained against. It is deliberately **outside** `dhn_gnn/`: it shares nothing with
the model code but the topology on disk, and pulls generation-only dependencies
(meteostat, pydhn's thermal solver) that the training path must not import.

**One command.** It simulates, then publishes automatically:

```bash
python -m datagen.run --full-year --output-dir data/raw_runs/steady   # ~10 hours
```

That leaves you with two directories, and both matter:

| | what | keep? |
|---|---|---|
| `data/raw_runs/steady/` | the full simulation — `full_network_state/`, `timeseries/`, `schedules/`, `weather_cache/`, `run_metadata.json` | **yes** — holds the per-pipe temperatures and metadata; regenerating is another 10 h |
| `data/solved_steady/` | the flat CSVs + `history.json` that `dhn_gnn` reads | yes — this is `config.GEN_DATA_DIR` |

`--version steady` is the default, matching `config.GEN_DATA_VERSION`, so nothing
needs editing after a normal run. Useful variations:

```bash
python -m datagen.run --output-dir data/raw_runs/demo --version demo  # 3-day default
python -m datagen.run --full-year --move                # relocate CSVs, save ~1.5 GB
python -m datagen.run --full-year --no-publish          # simulate only
python -m datagen.export data/raw_runs/steady --force   # (re)publish an existing run
```

If `data/solved_steady/` already exists, the automatic publish **stops rather than
overwriting** — the simulation is still complete and safe on disk, and the message
tells you to finish with `--force` or a different `--version`. Pass `--force-publish`
up front if you know you want to replace it.

Publishing is a separate, checked step because doing it by hand is how the shipped
dataset ended up broken: only 4 of 11 CSVs were copied (`nodes-pressure.csv` was
missed, silently breaking `evaluate` and the full report), and the `history.json`
left in place still described an older 745-step dataset while the CSVs held 8760
rows. `publish` checks the required files and rebuilds `history.json` from the run
that actually produced the data.

> **This simulation is thermohydraulic** — each pipe carries its own temperature,
> so ρ and μ vary across the network, while `dhn_gnn` currently assumes a fixed
> 50 °C. That mismatch is why `tests/run_gates.py` Gate 5 fails on this data. The
> run's `edges-temperature.csv` is what a per-pipe fix would need.

### Hyperparameters

Every knob is declared in its approach's config dataclass
(`dhn_gnn/approaches/<arch>/hparams.py`) and resolved in this order:

```
dataclass defaults  ->  --config file.yaml  ->  named flags  ->  --set key=value
```

`configs/unrolled.yaml` and `configs/initializer.yaml` are commented, editable
copies of the defaults — the place to look when you want to see every knob at
once. They are starting points, not the source of truth: leaving them out
changes nothing, because the dataclasses hold the same values.

Nothing needs a source edit:

```bash
# see the resolved recipe without training
python -m dhn_gnn.cli config --arch unrolled --set K=25 --set model.newton_mode=diagonal

# sweep, each into its own directory
python -m dhn_gnn.cli train --arch initializer --run d128 --set model.d_model=128
python -m dhn_gnn.cli train --arch initializer --run lr4  --set lr=1e-4

# reproduce a finished run from its own output
python -m dhn_gnn.cli train --arch initializer --run rerun \
    --config results/initializer/d128/config.yaml
```

A bare key works when it is unambiguous (`--set K=25`); qualify it otherwise
(`--set model.d_model=128`). An unknown key is an **error** listing the valid
ones — a typo can never silently train the default and report it as your run.

The fully resolved `config.yaml` is written next to the weights it produced, so
every run in `results/` is reproducible from its own directory.

## Results (Newton-base solver vs PyDHN, 373 timesteps)

| Metric | Value |
|---|---|
| Edge mass-flow R² | 0.99999 |
| Flow-direction accuracy | 99.6 % |
| Physics residual ≤ 50 Pa (PyDHN's threshold) | 95 % of timesteps |

Figure: [`results/model_eval.png`](results/model_eval.png).
**Caveat:** these reflect the *physics* base solving an easy network — not evidence that
the *learned* component helps here. See findings below.

---

## Data

- **Inputs** (`data/network`, `data/measurements`, `data/weather`) are small and tracked in git.
- **Solved outputs** (`data/solved_*`) are ~82 MB each and **git-ignored** — regenerate with
  `python -m datagen.run` + `python -m datagen.export`, or restore from backup / Git LFS.
- The canonical dataset is **`steady`**: a hydraulics-only run (`with_thermal=False`),
  all substations mass-flow-controlled, friction evaluated isothermally at **50 °C**
  (hence the exact fluid constants in `config.py`). Units are SI throughout (flow kg/s,
  pressure Pa, length m, roughness mm).

## Provenance & dependencies

- Network + measurement data originate from the OpenDHN Verbier snapshot; the solved data
  was generated by the v4 notebook in the sibling `pydhn-simulations` repo. `generate.py`
  reproduces that pipeline here (seeded RNG + cached weather → statistically equivalent,
  not byte-identical, to the original).
- **`pydhn`** is a runtime dependency: it builds the `Network` object (for topology/`B`) and
  is the physics oracle for the verification gates. The *data* is self-contained; the
  *code* still imports the `pydhn` library.

## Key findings (see `docs/model_layer_status.md`)

1. The physics core is exact (spike gates 1–5) and the objective is well-conditioned:
   solving the 12 loop unknowns directly drives the residual to **0 Pa**.
2. The untrained **Newton-base** unrolled solver already meets PyDHN's tolerance on 95 % of
   timesteps.
3. Training the learned GNN refinement did **not** improve on that (and could destabilize) —
   because this near-radial network is too easy for a learned solver to add value. To
   demonstrate the method's benefit, benchmark on a **meshed / large** DHN where NR is the
   bottleneck.
