# Spike Build Plan — Physics Core De-Risking

**Goal:** before any learning code, prove the physics core is correct — a differentiable
PyTorch twin of PyDHN's pipe physics (`φ`, `∂φ/∂ṁ`) and the cycle matrix `B` — so that
`B·φ(Bᵀ·m̃) = 0` is trustworthy. If this spike passes, the model in
[architecture_mapping.md](architecture_mapping.md) can be built on solid ground.

**Constraint:** these two modules only. No model, no training, no dataset class yet.

**Precision:** do all validation in **float64**. (The model may later run float32; the port
is *validated* in double so tolerances are tight.)

---

## Module 1 — `dhn_gnn/data/physics.py`

A pure-PyTorch, autograd-differentiable port of PyDHN's pipe hydraulics. **Reimplement, do
not call PyDHN** (PyDHN is NumPy, non-differentiable).

### Functions (contracts, not implementations)

| Function | Contract | Port source (PyDHN) |
|---|---|---|
| `reynolds(mdot, d, mu)` | `Re = 4·|ṁ| / (π·d·μ)`, elementwise, `≥ 0` | `fluids/dimensionless_numbers.py::compute_reynolds` — **read and match exactly** |
| `friction_factor(Re, d, roughness)` | 3-branch: laminar `Re≤2320` → `64/Re`; turbulent `Re≥4000` → Haaland; transition `2320<Re<4000` → Colebrook surrogate. **Non-affine** (matches the data generator). | `base_components_hydraulics.py:112–133` (turbulent :29–42, transition :45–109) |
| `phi(mdot, d, L, fd, rho)` | `Δp = L·fd·|ṁ|·ṁ / (4π²·ρ·(d/2)⁵)` | `compute_dp_pipe`, `base_components_hydraulics.py:150–154` |
| `dphi_dmdot(mdot, d, L, fd, rho)` | `∂φ/∂ṁ = L·fd·|ṁ| / (2π²·ρ·(d/2)⁵)` (PyDHN's `dp_der`; `fd` held fixed w.r.t. `ṁ`) | same, `:157–160` |

### Notes / decisions folded in
- **`ε`-floor (from D4):** clamp `Re` (or `|ṁ|`) to a small floor inside `friction_factor`
  so intermediate unrolled iterates that momentarily hit exact zero can't divide-by-zero.
  Training targets never need it (min `Re`=0.82, at 50 °C props), but the model's inner loop can.
- **Transition branch** is the only fiddly port (cube roots, logs). It *is* differentiable.
  Port non-affine to match the oracle; keep the affine blend (`:60–68`) as a fallback only
  if the non-affine port fights you — but that trades an exact match for ~few-% error in
  ~12% of samples, so prefer non-affine.
- **`ρ, μ`** are inputs here, not computed. The canonical dataset (**v4**) runs isothermal
  at PyDHN's default **50 °C**, so these are exact known constants — use
  **`ρ = 988.00 kg/m³`, `μ = 5.466e-4 Pa·s`** (from `Water().get_rho(50)`/`get_mu(50)`).
  Thermal coupling (#5) stays out of the spike; no per-edge temperature needed.

### ✅ Gate 1 — physics.py is a faithful port (unit test vs the PyDHN *function*)
No CSV, no topology. Sweep a grid + random sample of `(ṁ, d, L, roughness, ρ, μ)` covering
all three Re regimes (include `Re` near 2320 and 4000 boundaries, and near-zero flow):
- `phi(...)` vs PyDHN `compute_dp_pipe(...)[0]` → **max relative error ≤ 1e-8**.
- `dphi_dmdot(...)` vs PyDHN `compute_dp_pipe(...)[1]` → **max relative error ≤ 1e-8**.
- `dphi_dmdot(...)` vs **finite-difference** of your own `phi` → **≤ 1e-5** (guards against
  both being wrong the same way; expect a small mismatch in the transition band where `fd`
  varies with `ṁ` — document the size, it is the `∂fd/∂ṁ` term PyDHN's `dp_der` omits).
- No `NaN`/`inf` for any input including `ṁ = 0` (after the `ε`-floor).

---

## Module 2 — `dhn_gnn/data/network_operators.py`

Build the fixed network operators **once** from the PyDHN `Network` (decision D1: topology
is constant), cache as sparse tensors.

### What it produces
| Object | Contract | Source |
|---|---|---|
| `B` (cycle matrix) | `(L, E)`, entries ∈ `{−1, 0, +1}` | `net.cycle_matrix` → `compute_cycle_matrix`, `utilities/matrices.py:168–176` |
| `A` (oriented node-edge incidence) | `(N, E)`, entries ∈ `{−1, 0, +1}` | `compute_incidence_matrix(net, oriented=True)`, `matrices.py:184–189` |
| `internal_loops` (row mask) | indices of `B`'s rows that are true mesh loops (exclude pressure-setpoint and mass-flow-setpoint rows) | replicate `p_mask`/`m_mask`/`cycles_idxs` logic, `hydraulic_simulation.py:154–168` |
| edge order, node order | canonical index maps (align CSV columns → tensor rows) | `net.edges()`, `net.nodes()` |

`Bᵀ` (loop→edge lift) and `B` (edge→loop residual) are just the matrix and its transpose.

### ✅ Gate 2 — B is a genuine cycle matrix (structural, topology-only, no physics)
- `B` entries all in `{−1, 0, +1}`; shape `(L, E)`.
- **`A @ Bᵀ = 0`** to ≤ 1e-9 — every row of `B` is divergence-free (Kirchhoff/cycle
  structure). This is the single most important structural check.
- `rank(B) = E − N + C` (C = number of connected components; expect the supply+return
  graph structure PyDHN builds). Report actual `L`, `E`, `N`, rank.

---

## Integration gates (physics core wired together, on real solved data)

Data: `opendhn-data/network/pipes.csv` (geometry) + a chosen `gen_data/gen_data_steady_v4/`
(solved `edges-mass_flow.csv`, `edges-delta_p_friction.csv`). Use ~10–20 random timesteps,
not just one.

### ✅ Gate 3 — B closes the loops on the oracle's own pressures (isolates B)
Using PyDHN's **saved** `delta_p_friction` (not your `φ`):
- `r = B @ Δp_friction`; **max |r| over `internal_loops` rows ≤ ~100 Pa** (the generator
  converged to `error_threshold = 50 Pa`, notebook cell 24 — allow 2×). Report the max.
- (Non-internal rows are *not* expected to be zero — they carry the setpoint balance,
  `hydraulic_simulation.py:237, 244`. Only check `internal_loops`.)

### ✅ Gate 4 — mass conservation holds (isolates A + solved flows)
- At every **internal junction node** (non-source, non-sink), `(A @ ṁ*) ≈ 0` to ≤ 1e-6.
  This is exact-by-construction in the loop method, so a failure means a column/row
  alignment bug — a cheap, high-value smoke test for your index maps.

### ✅ Gate 5 — the money check: your `φ` + your `B` reproduce closure end-to-end
`ρ, μ` are exact constants — v4 is isothermal at 50 °C (`ρ = 988.00`, `μ = 5.466e-4`; see
Module 1 notes). No temperature file and no v3 fallback needed. To reproduce the oracle's
`Δp_friction` exactly you must also compute `fd` from the same `Re` PyDHN used at 50 °C, so
use these same constants in `reynolds(...)`. Then:
- `r = B @ phi_torch(ṁ*, d, L, fd, ρ)`; **max |r| over `internal_loops` ≤ ~100 Pa**.
- Also verify per-edge `phi_torch(ṁ*, …)` vs PyDHN's **saved** `edges-delta_p_friction.csv`
  → **max relative error ≤ 1e-6** (now feasible because props are exact). This upgrades the
  old "within a few %" sanity check into a hard per-edge gate.
- This composes Gate 1 (`φ` correct) ∘ Gate 2/3 (`B` correct). If Gates 1+3 pass, Gate 5 is
  a formality — but run it once to confirm the wiring and the edge/column alignment.

---

## Execution order

1. `network_operators.py` → **Gate 2** (pure topology; fastest, no physics).
2. `physics.py` → **Gate 1** (pure function port; no topology).
3. Align CSV columns ↔ tensor order → **Gate 4** (catches index bugs early).
4. **Gate 3** (B on oracle pressures).
5. **Gate 5** (full compose).

Gates 2 and 1 are independent — do them first, in either order.

## Definition of done
All five gates green on ≥10 timesteps. At that point the physics core is trustworthy and
the loop-space model (`unrolled_solver.py`), loss (`losses.py`), and dataset class can be
built against it with confidence.

### ✅ RESULTS — spike PASSED (13/13 checks), all 5 gates green
Code: `dhn_gnn/data/physics.py`, `dhn_gnn/data/network_operators.py`, `dhn_gnn/config.py`;
runner `tests/run_gates.py` (run with the pydhn venv python, 15 random timesteps).
Network: N=1352 nodes, E=1514 edges, L=163 cycle rows (**12 internal mesh loops**; the other
151 rows carry the 150 mass-flow consumers + HS1 — as expected, most cycles are setpoint
paths, not meshes).

| Gate | Result |
|---|---|
| 1 — physics port vs PyDHN | Re/fd/φ/`dp_der` all match to **~1.9e-15**; FD (fd fixed) 1.9e-8; no NaN/inf incl. `ṁ=0` |
| 2 — B structural | entries ∈{−1,0,1}; **`A·Bᵀ = 0` exactly**; rank(B)=163=E−N+C (C=1) |
| 3 — B closes loops (oracle Δp) | max internal-loop residual **46.36 Pa** (< 50 Pa solver threshold) |
| 4 — mass conservation | max `|A·ṁ*|` at 1048 internal nodes = **8.9e-15** (index maps aligned) |
| 5 — φ + B end-to-end | per-edge φ vs oracle `Δp_friction` **4.2e-8**; composed residual **46.36 Pa** (identical to Gate 3 — confirms φ reproduces the oracle's friction closure) |

**Conclusion:** the differentiable physics core is faithful to PyDHN and `Bφ(Bᵀm̃)=0` holds
on real solved data to solver tolerance. Cleared to build the model layer.

## Explicitly OUT of scope for this spike
- The GNN / attention / unrolled loop, heads, training, Armijo (§3–§5 of the spec).
- Feature scaling (the `dp_der` log-scale note from D4) — that's a training concern.
- Thermal coupling (#5), boundary-condition/controller handling (#6), loss units (#7).

**Canonical dataset — DECIDED: `gen_data_steady_v4`.** It is the purpose-built hydraulic
run: `SimpleStep(with_thermal=False)`, all 150 substations `mass_flow`-controlled, friction
at a fixed 50 °C. v3 is coupled hydraulic+thermal (`control_type="energy"`, per-edge varying
temperature) and its flows differ from v4 by ~18% median — it entangles the thermal state
(#5) the surrogate does not model, so it is **not** used. v4's isothermal setup is also why
`ρ, μ` are exact constants (Gate 5).
