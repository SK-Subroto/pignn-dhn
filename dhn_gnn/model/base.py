"""Shared physics core for every solver variant.

All three approaches -- pure Newton, the unrolled GNN, and the learned
initializer -- operate on the same reduced problem and were duplicating the same
five physics methods. They live here once, so a change to the pressure-drop model
cannot silently apply to one solver and not another, and so the comparison
between them is guaranteed to be like-for-like.

The reduced problem, in one line:

    mdot = mdot0 + Z @ c        Z = B_internal^T,  A @ Z = 0

`mdot0` is any flow satisfying the boundary conditions, and `c` is the flow
circulating around each independent loop. Because A @ Z = 0, ANY choice of `c`
conserves mass exactly -- that constraint is structural, never learned and never
violated. It also collapses 1514 edge unknowns to 12 loop unknowns, which is what
makes an exact Newton solve cheap enough to be the default.
"""

import torch
import torch.nn as nn

from dhn_gnn import config
from dhn_gnn.data import physics


def slog(x):
    """Signed log1p -- compresses the wide dynamic range of flow/pressure features."""
    return torch.sign(x) * torch.log1p(x.abs())


class DHNPhysicsBase(nn.Module):
    """Fixed network operators + the pressure-flow physics, with no solve policy."""

    def __init__(self, ops, rho: float = config.RHO_50, mu: float = config.MU_50):
        super().__init__()
        self.rho, self.mu = rho, mu

        pm = ops.pipe_mask
        internal = ops.internal_loops
        self.n_free = internal.numel()

        self.register_buffer("B_int", ops.B[internal].float())           # (L_int, E)
        self.register_buffer("Z", ops.B[internal].t().float())           # (E, L_int)
        self.register_buffer("A", ops.A.float())                         # (N, E)
        # |A| is used every step; materializing once instead of per step saves a
        # 2M-element allocation per call (~11% of the unrolled forward pass).
        self.register_buffer("A_abs", ops.A.abs().float())
        self.register_buffer("pipe_mask", pm)
        self.register_buffer("diameter", ops.diameter.float())
        self.register_buffer("length", ops.length.float())
        self.register_buffer("roughness", ops.roughness.float())

        absZ = ops.B[internal].abs().t().float()                         # (E, L_int)
        self.register_buffer("absZ", absZ)
        self.register_buffer("loop_size", absZ.sum(0).clamp_min(1.0))    # (L_int,)

        is_boundary = (ops.A[:, ~pm].abs().sum(1) > 0).float()
        self.register_buffer("is_boundary", is_boundary)                 # (N,)

        src = ops.A.t().argmin(1)     # incidence -1 = start node
        dst = ops.A.t().argmax(1)     # incidence +1 = end node
        edge_index = torch.stack([src, dst])
        self.register_buffer("edge_index", torch.cat([edge_index, edge_index.flip(0)], 1))

    # --- flow / pressure -----------------------------------------------------
    def edge_flow(self, mdot0, c):
        return mdot0 + (self.Z @ c)

    def pipe_dp(self, mdot):
        d = self.diameter.clamp_min(1e-9)
        return physics.pipe_dp(mdot, d, self.length, self.roughness,
                               self.rho, self.mu, re_floor=1e-6) * self.pipe_mask

    def dp_der(self, mdot):
        """dphi/dmdot -- the local hydraulic resistance, i.e. the Jacobian entries."""
        d = self.diameter.clamp_min(1e-9)
        Re = physics.reynolds(mdot, d, self.mu)
        fd = physics.friction_factor(Re, d, self.roughness, re_floor=1e-6)
        return physics.dphi_dmdot(mdot, d, self.length, fd, self.rho) * self.pipe_mask

    def residual(self, mdot):
        """Loop-law violation in Pa: zero exactly when the flow is the solution."""
        return self.B_int @ self.pipe_dp(mdot)

    # --- Newton --------------------------------------------------------------
    def newton_step(self, r, dp_der, mode: str = "full"):
        """
        Newton correction in loop space, solving J dc = r.

        "full"     uses the exact Jacobian J = B_int diag(dphi/dm) B_int^T. It is
                   only L_int x L_int (12x12 here), symmetric positive semi-definite,
                   so a small ridge covers the case where a loop carries near-zero
                   flow and its dphi/dm collapses. Converges quadratically.
        "diagonal" keeps only diag(J), i.e. pretends the loops do not interact.
                   The direction is only roughly right, so it needs damping and
                   converges linearly -- this is the original behaviour, kept so the
                   old approach stays measurable.
        """
        if mode == "diagonal":
            jac_diag = (self.B_int ** 2) @ dp_der
            return r / jac_diag.clamp_min(1e-9)
        J = self.B_int @ (dp_der.unsqueeze(-1) * self.B_int.t())
        J = J + 1e-9 * torch.eye(J.shape[0], dtype=J.dtype, device=J.device)
        return torch.linalg.solve(J, r)
