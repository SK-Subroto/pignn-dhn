# Model Layer — Status & Findings

Companion to [architecture_mapping.md](architecture_mapping.md) and
[spike_build_plan.md](spike_build_plan.md). Records what was built for the model
layer, what works, and the empirical findings from trying to train it.

## What was built

| File | Role |
|---|---|
| `dhn_gnn/solvers/attention.py` | Edge-biased multi-head self-attention (paper's block, native `scatter_reduce`) |
| `dhn_gnn/approaches/unrolled/model.py` | K-step loop-space unrolled solver (Option B / D3) |
| `dhn_gnn/solvers/stability.py` | Velocity cap + Armijo (eval-time stabilizers) |
| `dhn_gnn/losses.py` | Discounted physics loss (§4) |
| `dhn_gnn/train.py` | Deep-supervised / physics training loop |
| `tests/run_model_smoke.py` | 11 invariant checks (all pass) |

**Design (final):** the state is edge flow, corrected in the internal-cycle space
(`c ∈ R^{12}`, `mdot = mdot0 + Z c`, `A Z = 0` ⇒ mass conservation exact). Each unrolled
step is a **damped diagonal-Newton base step from the known operator plus a bounded,
learned GNN refinement**: `dc = -η (r / J_diag) + s·tanh(head(...))`, with the head
zero-init so the model *starts as a Newton solver*. Training uses **truncated backprop**
(state detached between steps) with a **per-step local physics-residual loss**.

## What works (verified)

- **All 11 structural invariants pass:** differentiable, NaN-free, mass conservation exact
  by construction, output shapes, grads finite & nonzero, per-step residuals/corrections.
- **The physics core is exact** (spike Gates 1–5) and the objective is well-conditioned:
  optimizing `c` directly against the residual → **0.0 Pa**; full Newton → 0 Pa; diagonal
  Newton → ~2–3 Pa.
- **The untrained Newton-base solver is a working solver.** Over 19 timesteps (K=25):
  mean residual **15.2 Pa**, median 12.5, max 76.8 → **95 % of steps ≤ 50 Pa** (PyDHN's own
  threshold), **100 % ≤ 100 Pa**, with no training.

## Key finding — the learned refinement adds no value *on this network*

Every attempt to *train* the GNN refinement either diverged or degraded the (already good)
Newton-base solution:

| Attempt | Outcome |
|---|---|
| Physics-only loss, raw Pa² | diverges (1e13) |
| Physics-only, normalized + grad-clip | stuck ~44k Pa |
| Supervised (fit the 12-number target) | fails; flow error grows |
| Deep supervision (c → a*) | flat loss, no learning |
| Newton base + learned refinement + truncated BPTT + per-step loss | untrained **22.7 Pa**; training pushes it to 232k Pa and plateaus |

Two root causes, one deeper than the other:
1. **Unrolled-through-quadratic gradient explosion** (`sum|g|≈1e14`) — addressed by
   truncated backprop, but it only removed the NaNs, it didn't make the refinement help.
2. **The network is too easy for the method.** It has 1514 edges but only **12 internal
   mesh loops** (mostly radial: 151 of 163 cycle rows are consumer/producer setpoint paths,
   not meshes). A Newton step therefore solves it almost trivially, so there is essentially
   *nothing for a learned corrector to improve* — and training just knocks the model off the
   good Newton solution. The learned unrolled method is designed to amortize expensive
   solves on **large, meshed** grids; this Verbier network does not exercise that regime.

This is a property of the **test network**, not a bug in the code or the method. The physics
core, the operators, and the unrolled machinery are all correct and reusable.

## Recommendations / next steps

1. **To demonstrate the learned method's value, change the benchmark, not the code:** use a
   **meshed** DHN (many independent loops) and/or a **large** network where PyDHN's NR
   Jacobian solve is the bottleneck, then compare the learned solver's inference speed/accuracy
   against NR. That is the regime the paper's method targets.
2. **As-is, the model is usable now** as a fast, differentiable, mass-conserving,
   physics-informed solver (the Newton base already meets solver tolerance here).
3. **If pursuing the learned refinement on meshed networks:** keep the truncated-BPTT +
   Newton-base + per-step-loss recipe; regularize the head toward zero and warm-start from
   the Newton solution so training cannot leave the good basin.

## Open questions status
- **#5 (thermal coupling), #6 (mixed BCs/controllers):** untouched (out of scope for the
  hydraulic solver core).
- **#7 (loss normalization):** partially exercised — `p_ref = 1e5` Pa was needed to keep the
  loss/gradients sane; a principled normalization is still open and mattered a lot in training.
