"""Stability mechanisms for the unrolled solver (spec §5).

Maps the paper's per-iteration update caps + Armijo line search to physically
meaningful DHN quantities:
  - voltage-magnitude step cap  -> max pipe flow-velocity change (`velocity_cap`)
  - Armijo merit = ||mismatch||_inf -> ||B_internal . phi(mdot)||_inf (`armijo_line_search`)

Corrections live in internal-cycle space (`c`), so a *global* scaling of the
correction keeps the flow exactly on the mass-conservation manifold (A . Z = 0).
"""

import torch


def velocity_cap(delta_mdot_edges, area, rho: float, v_max: float = 3.0):
    """
    Return a single scale in (0, 1] so the induced pipe velocity change stays within
    +/- v_max [m/s]. Scaling the whole update by one scalar preserves cycle-space
    feasibility. `area` = pipe cross-section (m^2); non-pipe entries should be masked
    upstream (area huge / delta 0) so they never bind.
    """
    dv = (delta_mdot_edges.abs() / (rho * area)).max()
    if dv <= v_max or dv == 0:
        return delta_mdot_edges.new_tensor(1.0)
    return v_max / dv


def armijo_line_search(c, dc, merit_fn, alphas=(1.0, 0.5, 0.25, 0.125, 0.0625), c1=1e-4):
    """
    Backtracking line search on the correction `c` along direction `dc`.
    merit_fn(c) -> scalar residual norm (evaluated WITHOUT grad, decision only).
    Returns the accepted step size alpha (float); keeps grad flowing through the
    caller's `c + alpha*dc`. Falls back to the smallest alpha only if it reduces
    the merit, else 0 (reject).
    """
    with torch.no_grad():
        F0 = merit_fn(c)
        for a in alphas:
            if merit_fn(c + a * dc) <= (1.0 - c1 * a) * F0:
                return a
        a = alphas[-1]
        if merit_fn(c + a * dc) < F0:
            return a
    return 0.0
