"""Spike verification gates 1-5 (see docs/spike_build_plan.md).

Run with the pydhn venv python (has torch + pydhn), from the pignn-dhn root:
    .../.venv/Scripts/python.exe tests/run_gates.py

Each gate isolates one source of error. All validation is in float64.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# make the package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhn_gnn import config
from dhn_gnn.data import physics
from dhn_gnn.data import network_operators as netops

from pydhn.components.base_components_hydraulics import (
    compute_dp_pipe,
    compute_friction_factor,
)

torch.set_default_dtype(torch.float64)
RHO, MU = config.RHO_50, config.MU_50
_results = []


def _check(name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    _results.append((name, ok))


# ---------------------------------------------------------------------------
def gate1_physics_port():
    print("\n=== Gate 1 - physics.py port vs PyDHN function (float64) ===")
    rng = np.random.default_rng(0)
    # geometry grid spanning the real network + boundary Re values
    d = rng.uniform(0.028, 0.263, 4000)
    L = rng.uniform(0.5, 200.0, 4000)
    rough = np.full(4000, 0.045)
    # mdot chosen to sweep all regimes incl. transition boundaries and near-zero
    mdot = np.concatenate([
        rng.uniform(-15, 15, 3600),
        rng.uniform(-1e-3, 1e-3, 300),          # near-zero
        np.array([0.0] * 100),                   # exact zero
    ])
    d, L, rough, mdot = (a[:4000] for a in (d, L, rough, mdot))

    # PyDHN reference
    Re_np = 4 * np.abs(mdot) / (np.pi * d * MU)
    fd_np = compute_friction_factor(Re_np, d, rough, affine=False)
    dp_np, dpder_np = compute_dp_pipe(mdot=mdot, fd=fd_np, diameter=d, length=L, rho_fluid=RHO)

    # our port
    t = lambda a: torch.as_tensor(a)
    Re = physics.reynolds(t(mdot), t(d), MU)
    fd = physics.friction_factor(Re, t(d), t(rough))
    dp = physics.phi(t(mdot), t(d), t(L), fd, RHO)
    dpder = physics.dphi_dmdot(t(mdot), t(d), t(L), fd, RHO)

    def relerr(a, b):
        a, b = np.asarray(a), b.numpy()
        m = np.abs(b) > 1e-12
        return np.max(np.abs(a[m] - b[m]) / np.abs(b[m])) if m.any() else 0.0

    _check("Re matches", relerr(Re_np, Re) <= 1e-10, f"max rel err {relerr(Re_np, Re):.2e}")
    _check("friction_factor matches", relerr(fd_np, fd) <= 1e-8, f"max rel err {relerr(fd_np, fd):.2e}")
    _check("phi matches compute_dp_pipe", relerr(dp_np, dp) <= 1e-8, f"max rel err {relerr(dp_np, dp):.2e}")
    _check("dphi_dmdot matches dp_der", relerr(dpder_np, dpder) <= 1e-8, f"max rel err {relerr(dpder_np, dpder):.2e}")

    # finite-difference cross-check: analytic dphi_dmdot is the derivative of phi
    # with fd held FIXED, so the FD must also hold fd fixed to match (relative check,
    # skip near-zero flow where |dp_der| is tiny and the ratio is ill-conditioned).
    h = 1e-7
    dp_p = physics.phi(t(mdot + h), t(d), t(L), fd, RHO)
    dp_m = physics.phi(t(mdot - h), t(d), t(L), fd, RHO)
    fd_num = (dp_p - dp_m) / (2 * h)
    big = np.abs(mdot) > 1e-2
    fd_rel = float(((fd_num - dpder).abs() / dpder.abs().clamp_min(1e-9))[big].max())
    _check("dphi_dmdot vs finite-diff (fd fixed)", fd_rel <= 1e-5, f"max rel err {fd_rel:.2e}")

    nan = bool(torch.isnan(dp).any() or torch.isinf(dp).any()
               or torch.isnan(dpder).any() or torch.isinf(dpder).any())
    _check("no NaN/inf incl. mdot=0", not nan, "clean" if not nan else "found NaN/inf")


# ---------------------------------------------------------------------------
def gate2_operators_structural(ops):
    print("\n=== Gate 2 - B is a genuine cycle matrix (topology only) ===")
    B, A = ops.B, ops.A
    vals = torch.unique(B)
    in_set = bool(torch.all((B == -1) | (B == 0) | (B == 1)))
    _check("B entries in {-1,0,1}", in_set, f"unique values {vals.tolist()[:6]}")

    ABt = A @ B.t()
    max_abt = float(ABt.abs().max())
    _check("A @ B^T == 0 (cycles divergence-free)", max_abt <= 1e-9, f"max |A B^T| = {max_abt:.2e}")

    N, E = A.shape
    L = B.shape[0]
    rankB = int(torch.linalg.matrix_rank(B))
    # connected components of the undirected graph, via rank(A) = N - C
    rankA = int(torch.linalg.matrix_rank(A))
    C = N - rankA
    expected = E - N + C
    _check("rank(B) == E - N + C", rankB == expected,
           f"rank(B)={rankB}, E-N+C={E}-{N}+{C}={expected}, L(rows)={L}")


# ---------------------------------------------------------------------------
def _load_solved(ops, csv_path, n_ts=15, seed=1):
    df = pd.read_csv(csv_path, index_col=0)
    df = df[ops.edge_names]  # reorder columns to canonical edge order
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(df), size=min(n_ts, len(df)), replace=False)
    return torch.as_tensor(df.iloc[rows].to_numpy(dtype=np.float64))  # (n_ts, E)


def gate3_B_closes_loops(ops):
    print("\n=== Gate 3 - B closes loops on the oracle's own pressures ===")
    dp = _load_solved(ops, config.DELTA_P_CSV)              # (T, E) total delta_p
    r = dp @ ops.B.t()                                      # (T, L)
    r_int = r[:, ops.internal_loops]
    max_r = float(r_int.abs().max())
    _check("max |B @ dp| over internal loops <= 100 Pa", max_r <= 100.0,
           f"max {max_r:.3f} Pa over {ops.internal_loops.numel()} internal loops x {dp.shape[0]} ts")


def gate4_mass_conservation(ops):
    print("\n=== Gate 4 - mass conservation at internal junctions ===")
    mdot = _load_solved(ops, config.MASS_FLOW_CSV)          # (T, E)
    imbalance = mdot @ ops.A.t()                            # (T, N)
    # internal nodes = not incident to any non-pipe (leaf) edge
    leaf = ~ops.pipe_mask
    incident_leaf = (ops.A[:, leaf].abs().sum(dim=1) > 0)   # (N,)
    internal_nodes = ~incident_leaf
    max_imb = float(imbalance[:, internal_nodes].abs().max())
    _check("max |A @ mdot| at internal nodes ~ 0", max_imb <= 1e-6,
           f"max {max_imb:.2e} kg/s over {int(internal_nodes.sum())} internal nodes")


def gate5_full_compose(ops):
    print("\n=== Gate 5 - our phi + our B reproduce closure end-to-end ===")
    mdot = _load_solved(ops, config.MASS_FLOW_CSV)          # (T, E)
    dpf_true = _load_solved(ops, config.DELTA_P_FRICTION_CSV)

    d = ops.diameter.clamp_min(1e-9)                        # avoid /0 on non-pipe cols
    dp_mine = physics.pipe_dp(mdot, d, ops.length, ops.roughness, RHO, MU)
    dp_mine = dp_mine * ops.pipe_mask                       # friction only defined on pipes

    # per-edge match vs oracle friction dp (pipes only)
    pm = ops.pipe_mask
    a, b = dp_mine[:, pm], dpf_true[:, pm]
    sig = b.abs() > 1.0  # ignore sub-Pa noise floor
    rel = float(((a - b).abs() / b.abs().clamp_min(1e-9))[sig].max()) if sig.any() else 0.0
    _check("per-edge phi vs oracle delta_p_friction <= 1e-6", rel <= 1e-6, f"max rel err {rel:.2e}")

    # composed loop residual with our phi
    r_int = (dp_mine @ ops.B.t())[:, ops.internal_loops]
    max_r = float(r_int.abs().max())
    _check("max |B @ phi(mdot*)| over internal loops <= 100 Pa", max_r <= 100.0, f"max {max_r:.3f} Pa")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gate1_physics_port()
    print("\nBuilding network operators (once)...")
    ops = netops.build_operators()
    print(f"  N={len(ops.node_names)} nodes, E={ops.n_edges} edges, L={ops.n_loops} loops, "
          f"{int(ops.pipe_mask.sum())} pipes, {ops.internal_loops.numel()} internal loops")
    gate2_operators_structural(ops)
    gate3_B_closes_loops(ops)
    gate4_mass_conservation(ops)
    gate5_full_compose(ops)

    print("\n" + "=" * 60)
    passed = sum(ok for _, ok in _results)
    total = len(_results)
    for name, ok in _results:
        if not ok:
            print(f"  FAILED: {name}")
    print(f"SPIKE GATES: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
