"""Learned initializer + exact Newton polish (the "GNN once" architecture).

Motivation, from measurements on this network:
  * PyDHN needs ~1.52 hydraulic iterations per timestep, because it WARM-STARTS
    from the previous timestep. That warm start is also what makes it strictly
    sequential: it cannot solve hour 100 before hour 99.
  * DHNUnrolledSolver runs the attention stack once per unrolled step (K times),
    which dominates the runtime, and the learned per-step correction was measured
    to contribute nothing over the physics Newton step.

So invert the design. The graph network runs ONCE to predict where the solution
is, and exact Newton -- which is nearly free here, the loop system being only
L_int x L_int -- takes it the rest of the way:

    c0 = GNN(boundary conditions)          <- one attention pass, learned
    repeat n_newton times:  c <- c - J^-1 r  <- exact, no parameters

The learned guess replaces the *previous timestep* as the source of a good
starting point. That is the whole point: it needs no history, so timesteps stay
independent and the entire year can be solved in parallel, which PyDHN's
warm-start chain structurally cannot do.

The target for c0 is exact and available: `a_star = (Z^T Z)^-1 Z^T mdot_true` is
the loop-space coordinate of the true flow, i.e. the c that would make the answer
correct. So the initializer trains by plain regression on L_int numbers, rather
than by backpropagating through an unrolled solver -- which is what made the
original architecture so hard to optimize.
"""

import torch
import torch.nn as nn

from dhn_gnn import config
from dhn_gnn.data import physics
from dhn_gnn.model.attention import EdgeBiasedAttention
from dhn_gnn.model.unrolled_solver import _slog


class DHNInitializerSolver(nn.Module):
    """One learned attention pass for c0, then parameter-free full-Newton steps."""

    def __init__(self, ops, n_newton: int = 4, d_model: int = 64, n_heads: int = 4,
                 num_attn_layers: int = 2, rho: float = config.RHO_50,
                 mu: float = config.MU_50):
        super().__init__()
        self.n_newton = n_newton
        self.rho, self.mu = rho, mu

        pm = ops.pipe_mask
        internal = ops.internal_loops
        self.register_buffer("B_int", ops.B[internal].float())
        self.register_buffer("Z", ops.B[internal].t().float())
        self.register_buffer("A", ops.A.float())
        self.register_buffer("A_abs", ops.A.abs().float())
        self.register_buffer("pipe_mask", pm)
        self.register_buffer("diameter", ops.diameter.float())
        self.register_buffer("length", ops.length.float())
        self.register_buffer("roughness", ops.roughness.float())
        absZ = ops.B[internal].abs().t().float()                 # (E, L_int)
        self.register_buffer("absZ", absZ)
        # MEAN-aggregate edges into loops, not sum: a loop spans hundreds of edges,
        # so a plain sum hands the head features ~1e2 larger than anything the rest
        # of the network is scaled for.
        self.register_buffer("loop_size", absZ.sum(0).clamp_min(1.0))   # (L_int,)
        is_boundary = (ops.A[:, ~pm].abs().sum(1) > 0).float()
        self.register_buffer("is_boundary", is_boundary)

        self.n_free = internal.numel()
        src = ops.A.t().argmin(1)
        dst = ops.A.t().argmax(1)
        edge_index = torch.stack([src, dst])
        self.register_buffer("edge_index", torch.cat([edge_index, edge_index.flip(0)], 1))

        self.node_proj = nn.Linear(4, d_model)
        self.blocks = nn.ModuleList([
            EdgeBiasedAttention(d_model, n_heads, edge_bias_dim=1) for _ in range(num_attn_layers)
        ])
        self.edge_readout = nn.Sequential(nn.Linear(2 * d_model + 3, d_model), nn.GELU())
        self.loop_norm = nn.LayerNorm(d_model)
        # Single head, used once. The output layer is ZERO-init so the model starts
        # at c0 = 0 -- exactly the cold start the solver uses today. Training can
        # then only improve on that reference point, and a random init cannot throw
        # the first Newton step into a region where the Jacobian is meaningless.
        self.head = nn.Sequential(
            nn.Linear(d_model + 2, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    # --- physics (shared with the unrolled solver) ---------------------------
    def _edge_flow(self, mdot0, c):
        return mdot0 + (self.Z @ c)

    def _pipe_dp(self, mdot):
        d = self.diameter.clamp_min(1e-9)
        return physics.pipe_dp(mdot, d, self.length, self.roughness,
                               self.rho, self.mu, re_floor=1e-6) * self.pipe_mask

    def _dp_der(self, mdot):
        d = self.diameter.clamp_min(1e-9)
        Re = physics.reynolds(mdot, d, self.mu)
        fd = physics.friction_factor(Re, d, self.roughness, re_floor=1e-6)
        return physics.dphi_dmdot(mdot, d, self.length, fd, self.rho) * self.pipe_mask

    def _residual(self, mdot):
        return self.B_int @ self._pipe_dp(mdot)

    def _newton(self, r, dp_der):
        J = self.B_int @ (dp_der.unsqueeze(-1) * self.B_int.t())
        J = J + 1e-9 * torch.eye(J.shape[0], dtype=J.dtype, device=J.device)
        return torch.linalg.solve(J, r)

    # --- the one learned pass ------------------------------------------------
    def predict_c0(self, mdot0):
        """Predict the loop-space solution coordinate from the boundary state."""
        dp = self._pipe_dp(mdot0)
        dp_der = self._dp_der(mdot0)
        injection = self.A @ mdot0

        node_feat = torch.stack([
            self.is_boundary, injection, self.A @ dp, self.A_abs @ mdot0.abs(),
        ], dim=-1)
        x = self.node_proj(node_feat)

        edge_bias = _slog(dp_der).unsqueeze(-1)
        edge_bias_dir = torch.cat([edge_bias, edge_bias], 0)
        for blk in self.blocks:
            x = blk(x, self.edge_index, edge_bias_dir)

        E = dp.numel()
        s, d_ = self.edge_index[0, :E], self.edge_index[1, :E]
        edge_emb = self.edge_readout(torch.cat(
            [x[s], x[d_], _slog(mdot0).unsqueeze(-1), _slog(dp).unsqueeze(-1),
             _slog(dp_der).unsqueeze(-1)], dim=-1))
        loop_emb = (self.absZ.t() @ edge_emb) / self.loop_size.unsqueeze(-1)
        loop_emb = self.loop_norm(loop_emb)                      # (L_int, d)

        r0 = self.B_int @ dp
        jac0 = (self.B_int ** 2) @ dp_der
        head_in = torch.cat([loop_emb, _slog(r0).unsqueeze(-1),
                             _slog(jac0).unsqueeze(-1)], dim=-1)
        return self.head(head_in).squeeze(-1)                    # (L_int,)

    def forward(self, mdot0, tol=None, n_newton=None, c_init=None):
        """
        Returns (mdot, residual_terms, c_hist) -- same contract as
        DHNUnrolledSolver, so evaluate.py works unchanged. residual_terms[0] is
        the residual of the raw learned guess, before any Newton step, which is
        what shows how much the initializer is actually worth.

        `c_init` overrides the learned guess with a supplied starting point -- used
        for the WARM-START ablation, where the previous timestep's solution is fed
        in instead. That ablation is the direct comparison: warm start buys the same
        head start by looking backwards in time, at the cost of making the solve
        sequential, whereas the learned guess needs no history and stays parallel.
        """
        n = self.n_newton if n_newton is None else n_newton
        c = self.predict_c0(mdot0) if c_init is None else c_init
        residual_terms = [self._residual(self._edge_flow(mdot0, c))]
        c_hist = [c]

        for _ in range(n):
            if tol is not None and residual_terms[-1].abs().max() < tol:
                break
            mdot = self._edge_flow(mdot0, c)
            c = c - self._newton(self._residual(mdot), self._dp_der(mdot))
            c_hist.append(c)
            residual_terms.append(self._residual(self._edge_flow(mdot0, c)))

        return self._edge_flow(mdot0, c), residual_terms, c_hist
