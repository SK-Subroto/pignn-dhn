# How the PIGNN-DHN solver works

A complete walkthrough of the unrolled hydraulic solver: the physics it encodes, the
shapes that flow through it, how training works, how it compares to PyDHN, and where
it currently falls short.

All figures in this document are verified against `data/solved_steady/`.
The worked example uses timestep 100.

**Contents**

1. [Background — what problem is being solved](#1-background--what-problem-is-being-solved)
2. [The network, in real numbers](#2-the-network-in-real-numbers)
3. [The key idea: 1,514 unknowns become 12](#3-the-key-idea-1514-unknowns-become-12)
4. [What the model actually is](#4-what-the-model-actually-is)
5. [The forward pass, with shapes](#5-the-forward-pass-with-shapes)
6. [A worked example — timestep 100](#6-a-worked-example--timestep-100)
7. [How training works](#7-how-training-works)
8. [Executing the code, step by step](#8-executing-the-code-step-by-step)
9. [Difference from PyDHN](#9-difference-from-pydhn)
10. [Inputs and outputs](#10-inputs-and-outputs)
11. [Drawbacks and open problems](#11-drawbacks-and-open-problems)
12. [Summary in one page](#12-summary-in-one-page)

---

## 1. Background — what problem is being solved

A district heating network pumps hot water from a central plant through buried pipes to
buildings, which extract heat and return cooler water. Verbier's network is the case
study here.

The engineering question is deceptively simple: **given how much heat each building
needs right now, how much water flows through each pipe?**

It is hard because the network is **meshed** — there are loops, so water can reach a
building by several routes. At every junction the flow splits, and how it splits depends
on pressure. But pressure drop depends on flow, and flow is what you are solving for.
The system is circular and nonlinear.

Two physical laws pin it down, the direct analogue of Kirchhoff's circuit laws:

- **Mass conservation** — water in equals water out at every junction.
- **Pressure consistency** — walk around any closed loop and the pressure drops must sum
  to zero. You cannot return to where you started at a different pressure.

Pressure drop through a pipe follows a quadratic law, `Δp ∝ |ṁ|·ṁ`, which is what makes
the system nonlinear and forces an iterative solution.

**PyDHN** is the established Python library that solves this with Newton–Raphson. It
produced every reference file in `data/solved_steady/`, and is treated here as ground
truth. The goal of this project is a *differentiable* solver that reaches the same answer.

---

## 2. The network, in real numbers

Counted from the dataset, not estimated.

| Quantity | Value | Detail |
|---|---:|---|
| Edges | **1,514** | 1,362 pipes + 150 substations + 2 producers |
| Nodes | **1,352** | 676 supply, 676 return |
| Independent cycles | **163** | `1514 − 1352 + 1` |
| Internal loops | **12** | the actual unknowns |
| Timesteps | **745** | hourly, January 2022 |
| Trainable parameters | ~110k | |

The network is a closed circuit: a supply side carries hot water out, a return side
brings it back, and the two are joined at each substation and at the plant. That is why
nodes come in perfectly matched pairs, 676 and 676.

---

## 3. The key idea: 1,514 unknowns become 12

This is the single most important thing to understand about the design. The model does
not predict 1,514 pipe flows. It predicts **12 numbers**, and the rest follows by fixed
arithmetic.

```
   1,514 edge flows
        │
        │  − 1,351   mass balance
        │            (water in = water out at every junction)
        ▼
     163 independent cycles
        │
        │  − 151     known setpoints
        │            (150 substation flows + 1 producer flow are GIVEN)
        ▼
      12 internal loops  ←  the model's entire output
```

Everything removed is a constraint you already know — no information is lost. Solve the
12, and all 1,514 follow exactly. This is implemented in
`network_operators.py::_internal_loop_rows`.

Every flow the model produces is built from those 12 numbers by one line:

```python
mdot = mdot0 + Z @ c
#      (1514,)  (1514,12) @ (12,)
```

- `mdot0` — a starting flow that already satisfies every building's demand
- `Z` — fixed network topology, `B[internal].T`
- `c` — the 12 unknowns

The crucial property: **adding circulation around a closed loop cannot change any
building's supply.** Water going around a ring arrives at each node and leaves it in
equal amount, so the net injection is zero.

```
        +c
   ●───────────▶●
   ▲            │
+c │            │ +c        at every node:  +c arrives, +c departs  →  net 0
   │            ▼
   ●◀───────────●
        +c
```

Mass conservation therefore holds automatically, for any value of `c` whatsoever. The
optimiser can never break it, and never has to learn it.

---

## 4. What the model actually is

It is **not** a neural network that predicts flows. It is a **learned correction to
Newton's method**, run for a fixed number of steps.

Each of the K steps applies this update to the 12 loop flows
(`unrolled_solver.py:154`):

```
dc = −0.5 · (r / J_diag)   +   0.5 · tanh(head(…))
     └── damped Newton ──┘      └─ learned refinement ─┘
```

- `r` — the loop residual, how badly pressure fails to balance, in Pascals
- `J_diag` — how stiffly each loop resists a change in its own flow

Their ratio is the classic Newton step. The learned term is deliberately constrained:

- **Bounded** by `tanh`, so it can never exceed ±0.5 kg/s regardless of network output.
- **Zero-initialised** — the head weights start at exactly zero, so an untrained model
  *is* plain damped Newton.

That design choice has a consequence worth stating plainly: **every number in
`results/model_eval.png` was produced with the neural network contributing exactly
nothing.** R² = 0.999986 is the physics baseline alone. The figure title says
`untrained`.

### Why the GNN is there at all

`J_diag` is only the *diagonal* of the true Jacobian. It treats each loop as isolated,
but loops share pipes — changing one loop's flow disturbs its neighbours. That coupling
is exactly what a graph neural network can see, because "which loops share pipes" is a
statement about network structure. The GNN's job is to learn the off-diagonal correction
that the cheap Newton step ignores.

---

## 5. The forward pass, with shapes

One call to `forward(mdot0)` runs K = 20 steps. Below is what happens inside a single
step, with every tensor shape.

Constants: `E = 1514`, `N = 1352`, `L_int = 12`, `d_model = 64`.

### Lane 1 — physics (the known operator)

```
mdot            (1514,)
  │ Darcy friction law
  ▼
dp = φ(mdot)    (1514,)      pressure drop per edge, pipes only
  │ d/dṁ
  ▼
dp_der          (1514,)      local hydraulic resistance
  │ sum per loop:  B_int @ dp
  ▼
r               (12,)        the loop residual, in Pa
```

### Lane 2 — the graph network

```
node features   (1352, 4)    [is_boundary, injection, Σdp, Σ|ṁ|]
  │ node_proj:  Linear 4 → 64
  ▼
x               (1352, 64)
  │ 2 × EdgeBiasedAttention, 4 heads
  │ ← dp_der (1514,) biases the attention logits
  ▼
x               (1352, 64)
  │ edge_readout: Linear(2·64 + 3 → 64) over [h_src, h_dst, slog(mdot), slog(dp), slog(dp_der)]
  ▼
edge_emb        (1514, 64)
```

### Lane 3 — collapse to loop space

```
edge_emb        (1514, 64)
  │ |Z|ᵀ @ edge_emb          aggregate edges into the loops they belong to
  ▼
loop_emb        (12, 64)
  │ concat with slog(r), slog(J_diag), newton_step
  ▼
head_in         (12, 67)
  │ heads[k]: Linear 67 → 1, zero-init
  ▼
learned term    (12,)
  │ dc = −damping · newton  +  step_scale · tanh(learned)
  ▼
dc              (12,)
```

### Lane 4 — update and repeat

```
c ← c + dc                   (12,)
mdot = mdot0 + Z @ c         (1514,)
store residual_k             (12,)   ← one per step, K total

repeat K = 20 times
c is DETACHED at the start of every step (truncated backprop)
```

Physics runs first and produces everything the network sees. The graph network never
touches raw physics; it consumes compressed features and outputs a bounded nudge to 12
numbers. Note where the funnel narrows: 1,514 edge embeddings collapse to 12 loop
embeddings via `|Z|ᵀ`, and everything downstream is 12-dimensional.

### Why `_slog` is everywhere

Flows span 0.001 to 10 kg/s; pressure drops span single Pascals to 100,000. Feeding that
raw into a network is hopeless. Every feature passes through
`_slog(x) = sign(x)·log1p(|x|)`, which compresses the range while preserving sign and
the value zero.

Critically, this happens **only** on the path into the network. The physics itself stays
in SI units, because `physics.py` is a validated port of PyDHN (Gate 1 in
`tests/run_gates.py`) and rescaling it would break that agreement.

### Where the edge bias enters

Standard attention decides how much node A should listen to node B from their embeddings
alone. Here each edge additionally contributes `dp_der` — its local hydraulic resistance
— as an additive bias on the attention logits (`attention.py:66`). A high-resistance pipe
is a weak hydraulic connection, and the model is told so directly rather than having to
infer it. This is recomputed *every step*, because resistance changes as flow changes.

---

## 6. A worked example — timestep 100

Real values from `data/solved_steady/`, hour 100 of the January 2022 simulation.

### What is given (the boundary conditions)

| Quantity | Value | Meaning |
|---|---:|---|
| 150 substation flows | 12.8221 kg/s total | what every building is drawing |
| S0 | 0.0125 kg/s | a small residential load |
| S4 | 0.3622 kg/s | a large load, ~29× S0 |
| HS1 producer | 3.0283 kg/s | mass-flow controlled — given |
| HS0 producer | −100,000 Pa | pressure controlled — the reference |

HS0 is the slack producer: its flow is *not* specified, it supplies whatever the network
needs. And it does — PyDHN's answer has HS0 delivering **9.7937 kg/s**, which closes the
balance exactly:

```
   9.7937   (HS0)
+  3.0283   (HS1)
──────────
  12.8220   =  12.8221 total substation draw   ✓
```

### What has to be worked out

Knowing the totals does not tell you the routes. 1,362 pipes carry flows from 0.001054
up to 9.7937 kg/s — a factor of 9,000 between the quietest and busiest pipe. Node
pressures run from 9 to 10 bar across the network.

### How the solve proceeds

| Stage | Loop residual | What it means |
|---|---:|---|
| initial guess `c = 0` | ~20,000 Pa | demands met, but pressure badly inconsistent |
| after 5 steps | ~2,000 Pa | circulation forming |
| after 15 steps | ~60 Pa | approaching PyDHN's own tolerance |
| after 20–25 steps | ~12 Pa | converged, below the 50 Pa bar |

Those figures are read off the convergence panel in `results/model_eval.png`. Across all
373 evaluated timesteps the final residual averages **15.4 Pa**, with 95.4% under 50 Pa.

The whole solve moved 12 numbers. Every one of the 1,514 edge flows came out of
`mdot0 + Z @ c`.

---

## 7. How training works

Training here is unusual in a way worth being precise about: **there are no labels in the
loss.** The model is never shown PyDHN's answer and told to match it.

### The loss is physics

Each of the K steps produces a residual. The loss is their discounted sum
(`train.py:31-41`):

```
L = Σ_k  γ^(K−1−k) · mean( (r_k / 1e5)² )        γ = 0.9
```

Every step is scored, not just the last — this is **deep supervision**. Later steps carry
more weight, since `γ^(K−1−k)` grows toward k = K−1. Dividing by `p_ref = 1e5` keeps the
numbers in a sane range; residuals of 20,000 Pa squared would otherwise be 4e8.

A residual of zero means the physics is satisfied, and the physics has exactly one
solution. So driving the residual down *is* finding the right answer — no ground truth
needed.

### Truncated backpropagation

```
 step 0  ──detach──▶  step 1  ──detach──▶  step 2  ──▶ ··· ──▶  step 19
   │                    │                   │                     │
   ▼                    ▼                   ▼                     ▼
   r₀                   r₁                  r₂                   r₁₉
  γ¹⁹                  γ¹⁸                 γ¹⁷                   γ⁰
   └────────────────────┴───────────────────┴─────────────────────┘
                                │
                                ▼
             L = Σ γ^(K−1−k) · mean((r_k / 1e5)²)
             (no ground truth anywhere in this expression)
```

Each `r_k` gradient reaches only `head[k]` — the detach walls stop it going further back.

The detach at the start of every step (`unrolled_solver.py:126`) is not an optimisation
detail — it is what makes training possible at all. Backpropagating through all 20 steps
of the quadratic operator produced gradients around 1e14 and stalled every earlier
attempt. Cutting the graph means each step's head learns to reduce its *own* step's
residual, like a learned optimiser.

### The training loop itself

```python
for ep in range(epochs):            # 400
    for i in randperm(len(samples)):
        m0, _, _ = samples[i]
        opt.zero_grad()
        _, terms, _ = model(m0)     # K residuals
        loss = deep_physics_loss(terms, gamma)
        loss.backward()
        clip_grad_norm_(params, 1.0)  # hard safety rail
        opt.step()
```

Adam, lr = 1e-3, gradient clipping at norm 1.0. One sample at a time — there is no
batching.

> **Training has not actually been run.**
> The `__main__` block trains on a **single timestep** (`make_samples(ops, [100])`) as an
> overfit sanity check — proof that the unroll *can* reach solver-level residual. It is
> not a real training run. And nothing is saved: `train()` returns the model, the process
> exits, the weights are gone. There is no `torch.save` anywhere in the project.

---

## 8. Executing the code, step by step

### Building the network operators

```python
ops = netops.build_operators()
```

1. Reads `nodes.csv`, `pipes.csv`, `substations.csv`, `heating_stations.csv`.
2. Reconstructs the PyDHN `Network` object — same topology, same edge ordering as the
   solver that produced the targets.
3. Extracts `B` (163×1514 cycle matrix) and `A` (1352×1514 incidence).
4. `_internal_loop_rows()` drops the 151 setpoint-carrying loops, leaving the 12
   internal rows.
5. Pulls per-edge geometry: diameter, length, roughness, and the pipe mask.

This requires PyDHN to be importable — it is **not** in `venv/`, it lives in the conda
environment from `environment.yml`.

### Training

```bash
python -m dhn_gnn.train
```

1. Build operators, seed to 0.
2. `make_samples(ops, [100])` — read the solved mass flows, project out circulation to
   get `mdot0`.
3. Construct `DHNUnrolledSolver(K=20, d_model=64, step_scale=0.5, newton_damping=0.5)`.
4. Report the untrained residual (pure Newton baseline).
5. Run 400 epochs, printing every 50.
6. Return the model — **and discard it**.

### Evaluation

```bash
python -m dhn_gnn.evaluate
```

1. Build a **fresh, untrained** model with K = 25. It never loads weights.
2. Walk timesteps 0, 2, 4, … (stride 2) → 373 of the 745.
3. For each: build `mdot0`, run the forward pass, record predicted flow, loop flow,
   convergence curve, final residual.
4. Compute R², MAE, RMSE, direction accuracy, residual statistics.
5. Write `results/metrics.txt` and the six-panel `results/model_eval.png`.

### Regenerating the dataset (PyDHN, not the model)

```bash
python -m datagen.run  # then: python -m datagen.export
```

Weather → per-building demand profiles → mass-flow schedules → PyDHN simulation →
solved CSVs. This is the reference pipeline; the model is not involved.

> **Two scripts, disconnected.**
> `train.py` and `evaluate.py` do not talk to each other. Training produces weights that
> vanish; evaluation builds a fresh untrained model. Every published metric is the
> physics baseline. Wiring a checkpoint between them is the smallest change with the
> largest effect on what you can claim.

---

## 9. Difference from PyDHN

Both solve the same equation — `B·φ(Bᵀṁ) = 0` — and both use Newton–Raphson. The
difference is *which* Newton, and what surrounds it.

```
        PyDHN                                This model
        NumPy · not differentiable           PyTorch · differentiable

  ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │ assemble FULL Jacobian 12×12 │    │ DIAGONAL only (12 numbers)   │
  └──────────────┬───────────────┘    └──────────────┬───────────────┘
                 ▼                                   ▼
  ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │ solve  J · dc = −r           │    │ GNN predicts missing coupling│
  └──────────────┬───────────────┘    └──────────────┬───────────────┘
                 ▼                                   ▼
  ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │ full step, damping = 1.0     │    │ half step, damping = 0.5     │
  └──────────────┬───────────────┘    └──────────────┬───────────────┘
                 ▼                                   ▼
  ┌──────────────────────────────┐    ┌──────────────────────────────┐
  │ converged in 2–3 iterations  │    │ fixed 20–25 steps, no exit   │
  │ 745/745 timesteps under 50Pa │    │ 95.4 % under 50 Pa           │
  └──────────────────────────────┘    └──────────────────────────────┘
```

PyDHN can take a full step because it knows the full Jacobian. Using only the diagonal
makes each step cheaper but wrong in a specific way — it ignores that loops share pipes —
so the step must be halved for stability, and many more are needed. The GNN exists to
recover what the diagonal discards.

| | PyDHN | This model |
|---|---|---|
| Equation solved | `B·φ(Bᵀṁ) = 0` | identical |
| Jacobian | full | diagonal + learned |
| Damping | 1.0 | 0.5 |
| Iterations | 2–3, adaptive | 20–25, fixed |
| Differentiable | no | **yes** |
| Batchable on GPU | no | possible, not implemented |
| Convergence guarantee | checked, can report failure | none — returns after K regardless |
| Outputs | flow, Δp, node pressure | flow only |
| Hydrostatic pressure | included | excluded |
| Non-pipe edge Δp | solved | masked to zero |

---

## 10. Inputs and outputs

### What goes in

```python
model(mdot0)     # mdot0 : (1514,) float32
```

A starting flow that already satisfies every building's demand and circulates nothing.
Conceptually it answers "everyone gets what they asked for, by the most direct routing."

### What comes out

```python
mdot_final, residual_terms, c_hist = model(mdot0)

mdot_final      # (1514,)     the solved flow — the real output
residual_terms  # K × (12,)   per-step residuals, for the loss
c_hist          # K × (12,)   per-step loop corrections, diagnostic
```

### What the scripts write to disk

| File | Written by | Contents |
|---|---|---|
| `results/metrics.txt` | `evaluate.py` | R², MAE, RMSE, direction accuracy, residual stats |
| `results/model_eval.png` | `evaluate.py` | six-panel diagnostic figure |
| `data/solved_<ver>/` | `datagen` | five CSVs — **from PyDHN, not the model** |

The model itself writes nothing. It produces no CSV, no checkpoint, no saved state.

> **The gap versus the reference dataset.**
> `data/solved_steady/` contains five files. The model can currently produce **one**
> of them — mass flow. `edges-delta_p` needs the hydrostatic term (excluded by design,
> `physics.py:116-117`) and Δp on the 152 non-pipe edges (masked to zero).
> `nodes-pressure` does not exist in any form: there is no pressure state anywhere in
> the model.

---

## 11. Drawbacks and open problems

Stated plainly, worst first.

### 1. The physics baseline already solves the problem

R² = 0.999986 with the network contributing nothing. Damped diagonal Newton alone is
essentially exact on this network. There is almost no headroom for a learned component to
demonstrate value, and any honest evaluation must compare against *this* baseline — not
against "no model."

### 2. It is slower than PyDHN

20–25 expensive steps against 2–3 cheap ones. On a 12-unknown system there is nothing for
a GPU to exploit. Any speed claim has to come from batching many timesteps at once, or
from networks large enough that the O(n³) linear solve starts to hurt — neither of which
the current benchmark shows.

### 3. It is less accurate than PyDHN

PyDHN converged on 745/745 timesteps under 50 Pa (`history.json`). The model reaches
95.4%, worst case 166.8 Pa. Since PyDHN is the ground truth, parity is the ceiling.

### 4. The input is derived from the answer

`mdot0` is built by projecting PyDHN's *solved* flows (`train.py:21-27`,
`evaluate.py:50-52`). The nodal injections it encodes are legitimately boundary
conditions, so this is defensible for benchmarking a solver — but it is circular for
generating data, and it means the reported accuracy benefits from a starting point
computed from the target.

### 5. No trained model exists

Training runs on one timestep and saves nothing. There are no checkpoints, no learning
curves, no trained-versus-untrained comparison.

### 6. No convergence guarantee

K steps run, then it stops. PyDHN reports `hydraulics converged: False` when it fails;
this model returns a plausible-looking answer either way. The 4.6% of timesteps above
50 Pa are silent.

### 7. Single sample, no batching

`forward()` takes one `mdot0`, not a batch — which forecloses the strongest available
speed argument.

### 8. Pressure recovery will expose accumulated error

Node pressure is obtained by integrating Δp along paths, so errors accumulate with
distance from the plant. Worse, where the loop residual is non-zero the integration is
genuinely path-dependent — two valid routes give two different pressures. Those timesteps
cannot be made to look good.

---

## 12. Summary in one page

**What it is.** A differentiable hydraulic solver for district heating. It reformulates
the problem so only 12 numbers are unknown, then runs 20 damped-Newton steps with a
bounded graph-network correction on each.

**What it does.** Takes one timestep's demands, returns the mass flow in all 1,514 edges,
at R² = 0.999986 against PyDHN.

**Why it is built this way.** Loop-space corrections make mass conservation structural
rather than learned. Zero-initialised heads make the untrained model a working solver, so
training can only improve on an already-correct baseline. Truncated backpropagation makes
the 20-step unroll trainable at all.

**What it cannot do yet.** Produce pressure or Δp. Run from boundary conditions rather
than solved data. Save or load weights. Batch. Beat PyDHN on speed or accuracy.

**Where the value actually lies.** Not in solving faster — PyDHN wins that, and it is the
ground truth so it wins accuracy by definition. The value is **differentiability**: you
can backpropagate through the solve. That makes inverse problems tractable — optimal pump
scheduling, leak and fouling detection, controllers trained against the network — none of
which PyDHN can support at any speed.

**The defensible claim:**

> A differentiable, physics-informed hydraulic solver that reaches PyDHN-equivalent
> accuracy, enabling gradient-based optimisation of network operation.

Accuracy parity is the success condition, not the contribution. Claiming "faster and
better than PyDHN" would be contradicted by the project's own numbers.
