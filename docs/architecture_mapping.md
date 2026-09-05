# Architecture Mapping: PIGNN-Attn-LS → DHN Hydraulic Solver

Porting the methodology of Changhun Kim et al.'s edge-aware attention +
unrolled-correction GNN (applied to AC power flow) to a district-heating-network
(DHN) steady hydraulic solver.

**Reference (paper side, read-only):**
`reference/pignn-attn-ls/GNSMsg_SelfAttention_armijo.py` (model `GNSMsg_EdgeSelfAttn`).
Line numbers below refer to that file.

**Target (DHN side):** PyDHN's modified loop-method Newton–Raphson solver, invoked
in the steady-simulation notebooks via `SimpleStep` → `solve_hydraulics`. Key files:
- `pydhn/solving/hydraulic_simulation.py` — `solve_hydraulics` (the NR loop we replace)
- `pydhn/solving/pressure.py` — `compute_dp` (assembles φ over all edges)
- `pydhn/components/base_components_hydraulics.py` — `compute_dp_pipe` (Darcy–Weisbach φ + derivative)
- `pydhn/utilities/matrices.py` — cycle matrix **B** construction

> Scope note: this is a planning document. No code is written or modified here.

---

## 1. State space

- **AC-PF (paper):** the unknown is a **per-node** vector `x = [V, θ] ∈ ℝ^{2N}`
  (voltage magnitude and angle at each of the `N` buses), solved by Newton–Raphson.
  The GNN carries this state on graph *nodes* and message-passes over the physical
  bus/line graph (`v`, `th` initialized from `V0`, updated by per-node heads
  `v_head`/`theta_head`, GNSMsg:174–175, 364–365).

- **DHN (PyDHN loop method):** the solver does **not** solve for per-node quantities.
  It solves the loop equation

  ```
  B · φ(Bᵀ · m̃) = 0
  ```

  for the vector of **loop (cycle) mass flows `m̃`**, whose dimension is the cycle
  rank `E − N + 1` (number of independent loops), **not** `N`.
  (`solve_hydraulics` docstring, `hydraulic_simulation.py:88–101`; `m̃` is `mdot_loop`,
  edge flows recovered as `mdot = cycle_matrix.T @ mdot_loop`, `hydraulic_simulation.py:205, 311`.)

### ⚠️ STRUCTURAL MISMATCH — flagged explicitly

The paper's state lives on **nodes** (`N` of them, 2 dof each). PyDHN's unknown lives
on **independent loops** (`E − N + 1` of them, 1 dof each). These are different spaces
on different graphs. The paper's whole architecture — node embeddings, per-edge
admittance-biased attention over the *physical* graph — assumes the unknown is nodal.
We must decide where the GNN's state and message passing live before anything else
maps cleanly. Three concrete options, with tradeoffs:

| # | Where MP / state lives | One-line tradeoff |
|---|---|---|
| **A** | **Loop/cycle graph induced by B.** MP nodes = loops; two loops are adjacent iff they share a pipe (adjacency ≈ off-diagonals of `B Bᵀ`). State = `m̃`. | Matches the true unknown space *exactly* (Jacobian is naturally `B·diag·Bᵀ`), but the loop graph is abstract/non-geometric and the cycle basis is **non-unique** — topology changes with the basis choice, which hurts transfer across networks. |
| **B** | **Per-edge (per-pipe) MP on the physical graph; loop residuals injected.** MP nodes = pipes (line graph), state = edge flow `ṁ`; the loop-closure residual `Bφ(Bᵀm̃)` is fed in as conditioning and the correction is applied in loop space (`Δm̃`, lifted by `Bᵀ`). | Keeps the physical geometry and lets per-pipe Darcy resistance bias attention exactly like line admittance; mass continuity is guaranteed by parameterizing the update in loop space. Cost: edge flows are over-parameterized (`E` vs `E−N+1`), so the update must be projected/lifted to stay on the feasible manifold. |
| **C** | **Per-node MP on the physical graph, pressure `p` as state, nodal mass balance as residual** (i.e. reformulate as the *nodal/pressure method* rather than the loop method). | Cleanest structural match to AC-PF (per-node dof, per-node residual, physical node graph, conductance-biased attention). But it is **not** the method PyDHN uses: it requires inverting φ (flow as a function of Δp) and a re-derived Jacobian, and — critically — its natural physics residual is *nodal continuity*, so the loop residual `Bφ(Bᵀm̃)` becomes identically ~0 and unusable as a loss (see §4, §6). |

**DECISION (locked — see Decisions Log D3):** Option **B**. It is the only one that
simultaneously (i) preserves the physical graph so the paper's node/edge attention
transfers directly, (ii) keeps the exact governing operator `Bφ(Bᵀm̃)` the physics loss
in §4 needs, and (iii) guarantees mass conservation at every unrolled step by applying
corrections in loop space. §2–§5 are written against this choice; it is no longer an
open recommendation.

---

## 2. Residual and known-operator mapping

Every DHN-side entry cites a function + file; paper-side entries cite the paper's
code (`GNSMsg_SelfAttention_armijo.py`).

| AC-PF concept | Paper's notation | DHN equivalent | PyDHN notation / code location |
|---|---|---|---|
| **Line admittance / network operator** `Y` | `Y`, series `Ys` and shunt `Yc`; edge feature `[Ysr, Ysi, Yc]` (GNSMsg:164, 272–274) | **Pipe hydraulic resistance** `K` in `Δp = K·\|ṁ\|·ṁ`, with `K = L·f_d / (4π²·ρ·(d/2)⁵)`. Note: **not constant** — `f_d` (Darcy friction factor) depends on Reynolds, hence on `ṁ`, so φ is genuinely nonlinear (unlike the constant-`Y` AC case; see §6). | `compute_dp_pipe`, `base_components_hydraulics.py:136–163`; `f_d` from `compute_friction_factor`, same file `:112–133` |
| **Nonlinear balance operator** (injection) `S = V ⊙ (Y V)*` | `Vc = v·e^{jθ}; Ic = Y·Vc; Sc = Vc·Ic.conj()` (GNSMsg:337–339) | **Pressure–flow map** φ (edge flow → edge Δp), assembled over all edges | `compute_dp`, `pressure.py:150–220`; per-component dispatch to `compute_dp_pipe` |
| **Residual** `r(x) = [ΔP; ΔQ]` | `DP = P_set − Sc.real; DQ = Q_set − Sc.imag` (GNSMsg:340–341) | **Loop residual** `r(m̃) = B·φ(Bᵀ·m̃)` (directed pressure sum around each cycle; a nonzero value is that loop's error) | `residuals = cycle_matrix @ dp` and `errors = cycle_matrix @ dp`, `hydraulic_simulation.py:243, 248` |
| **Jacobian** `J = ∂(P,Q)/∂(V,θ)` | Not explicitly formed — the paper is Jacobian-*free*; the learned heads replace the NR solve of `J·Δx = −r` | **NR loop Jacobian** `J_loop = B · diag(∂φ/∂ṁ) · Bᵀ` (block-diagonal per-edge derivative sandwiched by the cycle matrix) | `jac = np.diag(dp_der); jac_loop = cycle_matrix[loops] @ jac @ cycle_matrix[loops].T`, `hydraulic_simulation.py:278–279`; per-edge `dp_der = L·f_d·\|ṁ\| / (2π²·ρ·(d/2)⁵)` from `compute_dp_pipe:157–160` |
| **Per-edge admittance bias in attention** `β_ij` | `edge_bias = MLP([Ysr, Ysi, Yc]) → per-head scalar`, added to attention logits (`EdgeSelfAttnBlock.edge_bias`, GNSMsg:82–86; applied :120–121) | **Chosen:** the **linearized local resistance `∂φ/∂ṁ` (the `dp_der` term), recomputed from the current flow estimate at each unrolled step `k`** — *not* a frozen geometric conductance. DHN's effective resistance is flow-dependent (via `f_d` and the `\|ṁ\|` factor), so freezing it would discard the operating-point information; recomputing mirrors the paper's per-step re-evaluation of its physics operator (see Decisions Log). | `dp_der`, `base_components_hydraulics.py:157–160`, returned alongside `dp` by `compute_dp` (so the per-step recompute is free); geometry inputs `diameter/length/roughness` set at `net.add_pipe`, notebook `synthetic_data steady custom.ipynb` cell 17 |
| **Boundary buses** (slack / PV masks) | `slack_mask = (bus_type==1)`, `pv_mask = (bus_type==2)`; mismatches/updates zeroed there (GNSMsg:329–330, 342–343, 372–373) | **Setpoint edges** (pressure-controlled reference edge; mass-flow-controlled edges). Loops through these are handled specially rather than being free unknowns. | `pressure_setpoints_mask`, `mass_flow_setpoints_mask`; `p_mask`/`m_mask` and `dp[p_mask]=setpoint`, `hydraulic_simulation.py:156–166, 237, 244` |
| **State initialization** `V0` (flat start `V=1, θ=0`) | forward arg `V0`; `v, th = V0[...,0], V0[...,1]` (GNSMsg:210–211) | **Initial loop-flow guess** (warm-start cache, else `1e-4`) | `mdot_loop` init + `CYC_SOL_CACHE`, `hydraulic_simulation.py:191–202` |

**No clean analog (called out, not forced):**
- **Voltage *angle* `θ`.** AC-PF has 2 dof per node (`V`, `θ`); a DHN pipe carries a
  single scalar dof (flow, or equivalently a pressure difference). There is no phase
  analog — the state is effectively halved.
- **Complex / reactive part** (`Ysi`, reactive power `Q`, shunt `Yc`). DHN hydraulics
  is purely real and dissipative; there is no reactive counterpart. The 3-dim edge
  feature `[Ysr, Ysi, Yc]` collapses to a real 1-D (or geometry-derived) feature.

---

## 3. Unrolled correction loop

Using the locked state-space resolution (**Option B** from §1 — per-edge MP on
the physical graph, corrections applied in loop space; see Decisions Log D3). This is
our analog of the paper's per-iteration block
(GNSMsg:335–430, the `for k in range(K)` loop that is the code form of the paper's
eq. 1).

For unrolled step `k = 0 … K−1`, with current loop flows `m̃_k`:

1. **Lift to edges (known operator):** `ṁ_k = Bᵀ · m̃_k`.
   *(Paper analog: reconstruct `Vc` from `v, θ`.)*
2. **Evaluate physics / mismatch (known operator φ):**
   - per-edge pressure drop `Δp_k = φ(ṁ_k)` (Darcy–Weisbach, `compute_dp_pipe`),
   - loop residual `r_k = B · Δp_k` (one value per independent loop).
   *(Paper analog: `Sc`, then `DP, DQ` — GNSMsg:339–341.)*
3. **Assemble features:** per-edge features `[ṁ_k, Δp_k, ∂φ/∂ṁ_k]` (flow, drop, and the
   local resistance recomputed at this step) plus the loop residual `r_k` scattered
   back to the edges that compose each loop as conditioning.
   *(Paper analog: `bus_feat = [v, θ, ΔP, ΔQ]` concatenated with memory `m` — GNSMsg:345–346.)*
4. **Message passing (edge-biased attention):** run the attention block(s) on the
   physical graph, with attention logits biased by the linearized local resistance
   `∂φ/∂ṁ_k` recomputed at this step `k` (per D2 and the §2 attention-bias row — no
   frozen conductance term).
   *(Paper analog: `EdgeSelfAttnBlock` with `edge_bias` — GNSMsg:99–135.)*
5. **Propose update:** per-loop head outputs a loop-flow correction
   `Δm̃_k ∈ ℝ^{E−N+1}`. Applying the correction in loop space and lifting via `Bᵀ`
   keeps `ṁ` on the mass-conservation manifold automatically.
   *(Paper analog: `theta_head`/`v_head`/`m_head` per iteration — GNSMsg:364–366.)*
6. **Damped / line-searched update:** `m̃_{k+1} = m̃_k + α_k · Δm̃_k`, with `α_k`
   chosen by the stability mechanism of §5 (caps + backtracking on the merit `‖r_k‖`).
   *(Paper analog: clamp + Armijo, GNSMsg:375–425.)*

The learned heads replace PyDHN's explicit NR solve
`Δm̃ = −J_loop⁻¹ · r` (`hydraulic_simulation.py:305–310`); the operators `B` and φ
remain *known* and are evaluated exactly at every step (the "known-operator" /
physics-informed structure).

---

## 4. Physics loss

Paper's `L_phys` is the discounted sum of squared mismatches over the `K` unrolled
steps (GNSMsg:428–430):

```python
term = (gamma ** (K - 1 - k)) * ((DP**2 + DQ**2).mean())   # per step k
phys_loss = sum(term over k)
```

Rewritten in our governing equation `B·φ(Bᵀ·m̃) = 0`:

```
L_phys = Σ_{k=0}^{K-1}  γ^{(K-1-k)} · (1/L) · ‖ B · φ(Bᵀ · m̃_k) ‖²₂
```

where
- `m̃_k` is the loop-flow state at unrolled step `k`,
- `B·φ(Bᵀ·m̃_k)` is the per-loop pressure-closure residual (Pa), i.e. PyDHN's `errors`
  (`hydraulic_simulation.py:248`) — squared and mean-reduced over the `L = E − N + 1`
  loops (paper's `.mean()`),
- `γ` is the same discount as the paper (`self.gamma`, default 0.9, GNSMsg:429), which
  weights later steps more heavily.

Notes:
- Units are Pa² (vs. AC-PF's per-unit power²). PyDHN's own convergence check uses the
  **∞-norm** of the same residual (`max_abs_error`, `hydraulic_simulation.py:250`) with
  a threshold in Pa; the training loss follows the paper's squared-mean form instead.
- **Formulation dependency:** this loss is only meaningful under a loop/edge-flow
  state (Options A/B). Under Option C (nodal pressure) the loop residual is identically
  ~0 by construction, and the correct physics loss would instead be nodal mass
  imbalance `‖A·ṁ − demand‖²`. See §6.

---

## 5. Stability mechanisms

The paper stabilizes each unrolled Newton-like step with (a) hard per-iteration
update caps and (b) an Armijo backtracking line search on a mismatch merit function.
Mapping each to physically meaningful DHN quantities:

| Paper mechanism | Paper detail (GNSMsg) | DHN analog |
|---|---|---|
| **Angle step cap** `|Δθ| ≤ 0.30` | `dtheta_max = 0.30`, `dth = clamp(...)` :376–379 | **No angular analog.** Repurpose as a cap on the per-step loop-flow increment `|Δm̃_k| ≤ Δm̃_max`, i.e. exactly PyDHN's `damping_factor` / adaptive damping role (`hydraulic_simulation.py:130–135, 310, 319–329`). |
| **Voltage-magnitude step cap** `|ΔV| ≤ 0.10·V` | `dvm_frac = 0.10` :377, 380 | **Max flow-velocity constraint.** Convert flow to velocity `u = ṁ / (ρ·A)`, `A = π(d/2)²`; cap the step so `|u| ≤ u_max` (typical DHN design limit ~2–3 m/s). Velocity is the natural physical bound the flow update should respect. |
| **Voltage-magnitude box** `V ∈ [0.75, 1.2]` | `v_min, v_max = 0.75, 1.2`; `v = clamp(...)` :384, 394 | **Nodal pressure bounds.** After lifting flows→Δp→nodal pressures, clamp to physical envelope: absolute pressure ≥ 0 (no cavitation / non-negativity), and within pump head / static-pressure limits. |
| **Merit function** `F0 = ‖mismatch‖_∞` | `_batched_mismatch_inf_norm(...)` :38–61, 388 | **Loop-residual ∞-norm** `‖B·φ(Bᵀ·m̃)‖_∞` — identical to PyDHN's `max_abs_error` (`hydraulic_simulation.py:250`). Backtracking accepts a step only if it reduces this merit. |
| **Armijo backtracking** step ladder + sufficient-decrease test | `alphas = [1, .5, .25, .125, .0625]`, `c1 = 1e-4`, `F_all ≤ (1 − c1·α)·F0` :390–420 | Same ladder over the loop-flow step `Δm̃_k`. **PyDHN already has a crude analog:** adaptive/decreasing damping that shrinks `damp` on a residual plateau (`hydraulic_simulation.py:319–329`). Armijo is the principled replacement. |
| — (params fixed in AC-PF) | n/a | **Non-negative pipe diameter / geometry constraints** are *parameters*, not state, in the DHN forward problem, so they are not step constraints here. They only become active constraints if diameter is a design variable (inverse problem) — out of scope for the surrogate solver. |

---

## 6. Open questions

Uncertainties and places where PyDHN's code / the notebooks did not give enough to
map confidently. Listed rather than guessed.

1. **Residual duality (loop vs nodal) — state-space part SETTLED; item now essentially
   closed.** The state-space half is decided: §1 is locked to **Option B** (loop/edge-flow
   formulation, Decisions Log D3), so the physics loss is the *loop* residual `Bφ(Bᵀm̃)`
   and the Option-C nodal alternative (where that residual is trivially ~0 and the loss
   would instead be nodal mass imbalance `A·ṁ = demand`) is off the table. The §4 loss
   form is therefore fixed and already written to match. Nothing about the state space or
   loss *form* remains open here; the only residual work is the loss *scaling/units*
   detail, which is tracked separately as #7. In effect #1 is now resolved and hands its
   last loose end to #7.

2. **~~Non-uniqueness of the cycle matrix `B`.~~ — RESOLVED.** The dataset topology is
   **fixed** (network built once, `synthetic_data steady custom.ipynb` cell 17; only
   setpoint/demand *values* vary per timestep — see Decisions Log D1). Therefore `B` is
   invariant and is **computed once and reused** for every sample. The basis-choice
   sub-question collapses to a single, one-time selection with no generalization
   consequence: any fixed construction (`_compute_cycle_matrix_nx` or
   `compute_network_cycle_matrix`, `matrices.py:67–176`) is used identically across the
   whole dataset. Default to `_compute_cycle_matrix_nx` unless a reason emerges.

3. **~~φ is nonlinear in the unknown; `Y` is not.~~ — RESOLVED.** Decision: bias attention
   with the **linearized local resistance `∂φ/∂ṁ` (`dp_der`), recomputed each unrolled
   step `k`** from the current flow estimate (not a frozen geometric conductance). The
   `Y ↔ K` mapping being exact only for a frozen `f_d` is precisely why we do *not*
   freeze it — the paper re-evaluates its physics operator every step; we mirror that.
   See Decisions Log D2 and the updated §2 attention-bias row.

4. **~~Friction-regime discontinuities / near-zero-flow singularity.~~ — RESOLVED (empirical).**
   The original worry — `dp_der → 0` and a singular Jacobian near `Re → 0` — was checked
   against the generated data and does **not** hold; the premise was analytically wrong for
   the laminar branch. See Decisions Log **D4**. The derivative never collapses (self-
   compensating: `fd = 64/Re ∝ 1/|ṁ|` cancels the vanishing `|ṁ|` in `dp_der ∝ fd·|ṁ|`,
   so it plateaus at the Poiseuille resistance). Residual work is minor and folded into
   the build: (i) **log-scale/standardize** the `dp_der` attention-bias feature (~5.5
   orders of dynamic range); (ii) keep a cheap **`ε`-floor on flow/Re** as defensive
   hygiene, because intermediate unrolled iterates `m̃_k` (unlike the converged targets)
   could momentarily pass through exact zero.

5. **Thermal–hydraulic coupling.** The steady simulation runs `SimpleStep` with
   *both* hydraulic and thermal solves (notebook cell 26; `loops.py:143–177`), and φ
   depends on fluid `ρ, μ`, which depend on temperature (`compute_dp_pipe_net:191–199`).
   The paper's AC-PF is single-physics. Do we (a) freeze temperature/fluid props and
   learn hydraulics only, or (b) fold temperature into node features? The notebook fixes
   a design `ΔT` per substation (cell 50), which suggests (a) is viable — but this
   should be confirmed.

6. **Mixed boundary conditions and controllers.** PyDHN handles pressure setpoints,
   mass-flow setpoints, and optional controllers with their own residual/Jacobian rows
   (`hydraulic_simulation.py:170–177, 252–299`). The paper's slack/PV masking is
   simpler. How controller states and mixed setpoints enter the GNN's masking scheme is
   not yet specified.

7. **Target units / convergence metric.** PyDHN converges on ∞-norm error in **Pa**
   with a threshold of 50–100 Pa (notebook cell 26; `hydraulic_simulation.py:263`),
   while the training loss uses squared-mean. Need to pin down the normalization
   (per-unit? by nominal pump head?) so the loss is comparable across networks of
   different scale.

8. **~~Batching variable-topology graphs.~~ — RESOLVED.** Topology is fixed across the
   dataset (Decisions Log D1), so there is a **single, shared `B`** — no variable-topology
   batching is needed. `B` (and the `Bᵀ` lift / `B` scatter operators) are computed once
   and broadcast across the batch; every sample shares the same `E − N + 1` loop
   structure. The paper's `n_nodes_per_graph` block-diagonal machinery (GNSMsg:250–299)
   is unnecessary here. *(Re-opens only if a later dataset introduces topology variation
   — e.g. pipe failures or valve-closure-as-edge-removal, which the current generator
   does **not** do.)*

---

## Decisions Log

**D1 — Fixed-topology assumption: CONFIRMED (Task 1).**
Across all samples/timesteps in the generated dataset, the `Network` object — and hence
the cycle matrix `B` — is constructed **once and reused**; only demand/setpoint *values*
vary per hour.
- *Evidence:* Every `Network()`, `add_node`, `add_pipe`, `add_producer`, `add_consumer`
  call is in `synthetic_data steady custom.ipynb` **cell 17 only**. The per-timestep loop
  (`custom_steady_simulation`, cell 24) mutates the net exclusively via
  `net.set_edge_attributes(...)` for `setpoint_value_hyd`, `setpoint_value_hx`,
  `heat_demand`, and `design_delta_t` — i.e. values on existing edges — then calls
  `base_loop.execute(net=net, …)` on the same object. Both invocations (init run cell 26,
  main run cell 32) pass the identical `net`.
- *Refutation of alternatives:* a full-notebook scan found **no** `remove_edge`,
  `remove_node`, `del net`, `drop_edge`, valve-closure, or per-sample `Network()` rebuild.
  (The `failures` list in cells 53/56 is a validation-check helper, unrelated to pipes.)
- *Consequence:* Open questions **#2** and **#8** RESOLVED. `B` is a compile-time constant
  of the problem; compute once, reuse/broadcast. No topology variation (pipe failures,
  valve closures) is modeled anywhere in the current pipeline.

**D2 — Attention-bias quantity: linearized local resistance, recomputed per step (Task 2).**
The per-edge attention bias uses `∂φ/∂ṁ` (PyDHN's `dp_der`), **recomputed from the current
flow estimate at each unrolled step `k`**, rather than a frozen static geometric conductance.
- *Reasoning:* (1) The paper does not "freeze" its bias operator `Y` because `Y` is
  genuinely constant in a linear AC network — it still *re-evaluates the physics* (`Sc`,
  `ΔP`, `ΔQ`) from the current state at every unrolled step (GNSMsg:337–341). In DHN the
  effective resistance is **not** constant (it depends on `f_d(Re)` and the `|ṁ|` factor),
  so the faithful analog of "evaluate the true operator at the current state each step" is
  to recompute the resistance each step, not freeze it. (2) `∂φ/∂ṁ` is exactly the local
  sensitivity PyDHN's own NR Jacobian uses (`dp_der`, `hydraulic_simulation.py:278`), so it
  carries the most decision-relevant information for the correction. (3) It is essentially
  free: `compute_dp` already returns `dp_der` alongside `dp`, so it is a byproduct of the
  mismatch evaluation the step performs anyway.
- *Consequence:* Open question **#3** RESOLVED; §2 attention-bias row updated to state the
  single chosen approach.

**D3 — State-space resolution: LOCKED to Option B (§1).**
The GNN operates as per-edge (per-pipe) message passing on the *physical* graph, with the
correction applied in loop space (`Δm̃`, lifted by `Bᵀ`); state = edge flow `ṁ`, with the
loop-closure residual `Bφ(Bᵀm̃)` injected as conditioning. Previously framed as a
recommendation; now a locked decision.
- *Reasoning (as already argued in §1):* Option B is the only resolution that
  simultaneously (i) **preserves the physical graph**, so the paper's node/edge
  admittance-biased attention transfers directly; (ii) **keeps the exact governing
  operator `Bφ(Bᵀm̃)`** that the §4 physics loss is written against; and (iii)
  **guarantees mass conservation** at every unrolled step by parameterizing the update in
  loop space. Options A (abstract loop graph, basis-dependent, poor transfer) and C (nodal
  reformulation — not PyDHN's method, and makes the loop residual identically ~0) were
  rejected in §1.
- *Consequence:* Closes the **state-space half of open question #1** — the loop/edge-flow
  formulation is fixed, so the §4 loss form (loop residual, not nodal imbalance) is fixed
  too. This does **not** by itself resolve the loss *scaling/units* question, which is
  tracked as #7. §2–§5 were already written assuming Option B; this entry makes that
  assumption official.

**D4 — Near-zero-flow / friction-regime: NOT a blocker (empirically verified).**
Open question #4 asked whether `∂φ/∂ṁ` collapses (singular Jacobian) at low flow. Checked
directly against generated data (`solved_steady/edges-mass_flow.csv`, 745 timesteps ×
1362 pipes = 1,014,690 entries; Reynolds via `Re = 4|ṁ|/(π·d·μ)` with pipe diameters from
`opendhn-data/network/pipes.csv`; `dp_der` via PyDHN's own `compute_friction_factor` +
`compute_dp_pipe`). **Props are the exact v4 isothermal constants** — `Water()` at 50 °C,
`ρ = 988.00 kg/m³`, `μ = 5.466e-4 Pa·s` (this rerun supersedes an earlier estimate that
assumed `μ≈4e-4`).
- *Findings:* 0.000% exact-zero flow; min `|ṁ|`=5.7e-5 kg/s, min `Re`=0.82 (Re<1 is 0.001%
  of entries, ~10 of 1M). `dp_der` has **no** NaN/inf/zero anywhere; min = 2.2e-2; laminar
  median (16.5) is within ~3.4× of the turbulent median (55.8). Regime split: **~20% laminar
  (Re<2320), ~15% transition (2320–4000), ~65% turbulent** — i.e. ~35% of entries sit below
  Re=4000. 180 of 1362 pipes are laminar 100% of the time (248 more than half the time).
- *Why the original premise was wrong:* in the laminar branch `fd = 64/Re ∝ 1/|ṁ|`, and
  since `dp_der ∝ fd·|ṁ|` the `|ṁ|` cancels — the derivative plateaus at the Poiseuille
  resistance rather than vanishing. The self-compensation makes it well-behaved.
- *Consequence:* #4 RESOLVED. No special singular-Jacobian handling is needed for the
  training targets. Two minor items fold into the build (not new decisions): log-scale the
  wide-range (max/min ≈ `3.8e5`, `~10^5.6`) `dp_der` bias feature, and keep a cheap
  `ε`-floor on flow/Re as defensive hygiene for intermediate unrolled iterates (which —
  unlike the converged targets — could momentarily hit exact zero).
- *Note on the corrected props:* with the true (higher) `μ`, the low-Re share rose vs the
  first pass (laminar 13%→20%, sub-4000 ~26%→35%), which **strengthens** the case for
  handling the transition band carefully (feature smoothness) but changes nothing
  structural — the `dp_der` non-collapse is analytic and independent of `μ`.

**Still open after this pass:** #5 (thermal–hydraulic coupling — freeze fluid props vs.
condition on temperature), #6 (mixed BCs / controllers), #7 (loss normalization & units).
Open question **#1** is effectively resolved (state space locked by D3; loss form fixed),
loose end folded into #7. **#4** is now resolved (D4). All remaining open items are
"decide-in-file" build choices, not structural blockers.

---

*Pass note (D3/D2 revision):* That edit **only fixed internal consistency** — §3 steps 3–4
were brought in line with decision **D2** (removed the stale static-conductance `g_ij`
term; the attention bias is solely `∂φ/∂ṁ_k` recomputed per step) — and **locked a
previously-open recommendation** (§1 → Option B, recorded as **D3**). No new technical
decisions were introduced.

*Pass note (D4 revision):* Resolved open question **#4** by measuring the flow/Reynolds and
`∂φ/∂ṁ` distributions across the generated dataset (see D4). This was an **empirical
verification**, not a design change: it retired a suspected blocker and recorded two minor
build-time notes (feature log-scaling, `ε`-floor). No architectural decision was altered.

