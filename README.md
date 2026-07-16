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
| state `[V, θ]` per node | loop mass flows `m̃` (dim = E−N+1) | `model/unrolled_solver.py` |
| residual `[ΔP; ΔQ]` | loop residual `B·φ(Bᵀ·m̃)` | `data/physics.py`, `data/network_operators.py` |
| admittance `Y` | Darcy–Weisbach resistance `φ` | `data/physics.py` |
| edge admittance bias `β_ij` | linearized resistance `∂φ/∂ṁ` (recomputed per step) | `model/attention.py` |

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

```
dhn_gnn/
  config.py                 constants (50 °C fluid props) + data paths
  generate.py               synthetic dataset generator (ports the v4 notebook)
  evaluate.py               accuracy metrics + figure generation
  train.py                  deep-supervised / physics training loop
  losses.py                 discounted physics loss
  data/
    physics.py              differentiable PyTorch port of PyDHN's Darcy–Weisbach φ, ∂φ/∂ṁ
    network_operators.py    builds cycle matrix B, incidence A, internal-loop mask (once)
  model/
    attention.py            edge-biased multi-head self-attention (paper's block)
    unrolled_solver.py      the K-step Newton-base + learned-refinement solver
    stability.py            velocity cap + Armijo (eval-time stabilizers)
data/
  network/                  topology + pipe geometry (Verbier, 1352 nodes / 1514 edges)
  measurements/             real boundary snapshot
  weather/                  cached outdoor temperature (for reproducible generation)
  solved_steady_v4/         PyDHN ground-truth outputs (git-ignored, ~82 MB)
docs/                       architecture mapping, spike plan, status/findings
tests/                      verification gates (physics core + model invariants)
results/                    evaluation figure + metrics
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

# 3. evaluate the (untrained, Newton-base) solver vs PyDHN -> results/
python -m dhn_gnn.evaluate

# 4. (optional) regenerate the synthetic dataset — reproducible, seeded, offline
python -m dhn_gnn.generate --version steady_v4_regen        # full 745-step run
python -m dhn_gnn.generate --steps 24 --version smoke       # quick check
```

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
  `python -m dhn_gnn.generate`, or restore from backup / Git LFS.
- The canonical dataset is **`steady_v4`**: a hydraulics-only run (`with_thermal=False`),
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
