"""DHN unrolled correction solver (spec §3, Option B / Decision D3).

State = edge mass flow, but corrections are applied in the internal-cycle space
(`c` in R^{L_int}) so mass conservation holds by construction: the flow is always

    mdot = mdot0 + Z @ c ,   Z = B[internal]^T ,   with   A @ Z = 0 .

Each unrolled step k re-evaluates the known operator phi (Darcy) at the current
flow, message-passes with edge-biased attention (bias = d phi/d mdot recomputed
this step, D2), proposes a per-loop correction, caps it (velocity) and accepts it
via Armijo backtracking on the loop-residual merit.

`mdot0` (a boundary-feasible reference flow) is supplied per sample; constructing
it from boundary conditions is a dataset concern and out of scope here.
"""

import torch
import torch.nn as nn

from dhn_gnn import config
from dhn_gnn.data import physics
from dhn_gnn.model.attention import EdgeBiasedAttention
from dhn_gnn.model import stability


def _slog(x):
    """Signed log1p — compresses the wide dynamic range of flow/pressure features."""
    return torch.sign(x) * torch.log1p(x.abs())


class DHNUnrolledSolver(nn.Module):
    def __init__(self, ops, K: int = 20, d_model: int = 64, n_heads: int = 4,
                 num_attn_layers: int = 2, gamma: float = 0.9, step_scale: float = 0.5,
                 newton_damping: float = 0.5, newton_mode: str = "diagonal",
                 rho: float = config.RHO_50, mu: float = config.MU_50):
        # Each step is a Newton base step (from the known operator) plus a bounded
        # learned GNN refinement:
        #     dc = -newton_damping * newton_step  +  step_scale * tanh(head(...))
        # With the head zero-init the model STARTS as pure Newton and learns to
        # refine it. This is a stronger, physics-informed inductive bias than
        # learning the whole step.
        #
        # `newton_mode` selects the base step:
        #   "diagonal" - keep only diag(J), i.e. pretend the loops are independent.
        #                Cheap but the direction is only roughly right, so it needs
        #                damping (0.5) and converges linearly: ~19 steps to 45 Pa.
        #   "full"     - solve the actual L_int x L_int loop system J dc = r with
        #                J = B_int diag(dphi/dm) B_int^T. The cycle-space reduction
        #                already shrank 1514 unknowns to 12, so this matrix is 12x12:
        #                forming it is ~218k FLOPs and solving it is negligible.
        #                Correct direction => no damping needed and quadratic
        #                convergence, which is what PyDHN gets its ~1.5 iterations
        #                per timestep from.
        super().__init__()
        if newton_mode not in ("diagonal", "full"):
            raise ValueError(f"newton_mode must be 'diagonal' or 'full', got {newton_mode!r}")
        self.K, self.gamma = K, gamma
        self.step_scale, self.newton_damping = step_scale, newton_damping
        self.newton_mode = newton_mode
        self.rho, self.mu = rho, mu

        # --- fixed operators as buffers (float32 for the net; physics done in fp32) ---
        pm = ops.pipe_mask
        internal = ops.internal_loops
        self.register_buffer("B_int", ops.B[internal].float())          # (L_int, E)
        self.register_buffer("Z", ops.B[internal].t().float())          # (E, L_int)
        self.register_buffer("A", ops.A.float())                        # (N, E)
        # |A| is used every unrolled step; materializing it once instead of per
        # step saves a 2M-element allocation x K (~11% of the forward pass).
        self.register_buffer("A_abs", ops.A.abs().float())               # (N, E)
        self.register_buffer("pipe_mask", pm)
        self.register_buffer("diameter", ops.diameter.float())
        self.register_buffer("length", ops.length.float())
        self.register_buffer("roughness", ops.roughness.float())
        # cross-section area (pipes only; large elsewhere so velocity_cap never binds)
        area = torch.where(pm, (ops.diameter / 2.0) ** 2 * torch.pi, torch.full_like(ops.diameter, 1e6))
        self.register_buffer("area", area.float())
        # |Z| loop-incidence for aggregating edge -> loop, and boundary node flag
        self.register_buffer("absZ", ops.B[internal].abs().t().float())  # (E, L_int)
        is_boundary = (ops.A[:, ~pm].abs().sum(1) > 0).float()
        self.register_buffer("is_boundary", is_boundary)                 # (N,)

        E = ops.B.shape[1]
        self.n_free = internal.numel()
        src = ops.A.t().argmin(1)   # incidence -1 = src
        dst = ops.A.t().argmax(1)   # incidence +1 = dst
        edge_index = torch.stack([src, dst])                            # (2, E)
        self.register_buffer("edge_index", torch.cat([edge_index, edge_index.flip(0)], 1))  # bidir

        node_in = 4    # [is_boundary, injection, agg signed dp, agg |flow|]
        self.node_proj = nn.Linear(node_in, d_model)
        self.blocks = nn.ModuleList([
            EdgeBiasedAttention(d_model, n_heads, edge_bias_dim=1) for _ in range(num_attn_layers)
        ])
        # edge readout: [h_src, h_dst, 3 edge scalars] -> hidden
        self.edge_readout = nn.Sequential(nn.Linear(2 * d_model + 3, d_model), nn.GELU())
        # per-step loop-correction heads, zero-init (identity start, like the paper).
        # Head input = [loop_emb, slog(residual), slog(loop-Jacobian diag), newton step]:
        # the residual to zero, the local resistance, and the diagonal-Newton step, so the
        # head learns a refinement on top of the physics Newton base.
        self.heads = nn.ModuleList([nn.Linear(d_model + 3, 1) for _ in range(K)])
        for hd in self.heads:
            nn.init.zeros_(hd.weight); nn.init.zeros_(hd.bias)

    # --- physics helpers -----------------------------------------------------
    def _edge_flow(self, mdot0, c):
        return mdot0 + (self.Z @ c)

    def _pipe_dp(self, mdot):
        d = self.diameter.clamp_min(1e-9)
        dp = physics.pipe_dp(mdot, d, self.length, self.roughness, self.rho, self.mu, re_floor=1e-6)
        return dp * self.pipe_mask

    def _dp_der(self, mdot):
        d = self.diameter.clamp_min(1e-9)
        Re = physics.reynolds(mdot, d, self.mu)
        fd = physics.friction_factor(Re, d, self.roughness, re_floor=1e-6)
        return physics.dphi_dmdot(mdot, d, self.length, fd, self.rho) * self.pipe_mask

    def _residual(self, mdot):
        return self.B_int @ self._pipe_dp(mdot)          # (L_int,)

    def _newton_step(self, r, dp_der, jac_diag):
        """
        Solve for the Newton correction in loop space.

        Full mode builds J = B_int diag(dphi/dmdot) B_int^T, the exact Jacobian of
        the loop residual w.r.t. the loop correction c. It is symmetric positive
        semi-definite (dphi/dmdot >= 0 for friction), so a small ridge is enough to
        keep it invertible when a loop carries near-zero flow and its dphi/dmdot
        collapses -- without it, `solve` blows up exactly on the easy timesteps.
        """
        if self.newton_mode == "diagonal":
            return r / jac_diag.clamp_min(1e-9)
        J = self.B_int @ (dp_der.unsqueeze(-1) * self.B_int.t())    # (L_int, L_int)
        J = J + 1e-9 * torch.eye(J.shape[0], dtype=J.dtype, device=J.device)
        return torch.linalg.solve(J, r)

    # --- forward -------------------------------------------------------------
    def forward(self, mdot0, tol=None):
        """
        mdot0: (E,) boundary-feasible reference flow.
        tol:   optional early-exit tolerance in Pa. When set AND the module is in
               eval mode, unrolling stops as soon as max|residual| < tol, so an
               easy timestep costs far fewer than K steps. Deliberately inert in
               training mode: each step owns a separate head (`self.heads[k]`),
               so exiting early would starve the later heads of gradient and
               leave them at their zero-init exactly on the hard samples that
               need them. Training always runs the full fixed K.
        Returns (mdot_final, residual_terms, c_hist):
          residual_terms: internal-loop residual after each executed step
                          (K of them, or fewer if the early exit fired)
          c_hist:         per-step loop corrections c_k (for deep supervision)

        Update is a *smooth, damped, bounded* step  dc = step_scale * tanh(head(...)):
        no hard velocity-cap max() (non-smooth gradient) and no compounding Armijo in
        the training path. Those were the two instability sources; Armijo/velocity_cap
        remain available as eval-time stabilizers via `eval_stabilize`.
        """
        c = mdot0.new_zeros(self.n_free)
        injection = self.A @ mdot0                       # (N,), constant across steps
        residual_terms, c_hist = [], []

        for k in range(self.K):
            # Truncated backprop: detach the accumulated state so gradients flow only
            # within one unrolled step (learned-optimizer style). Backpropagating
            # through all K steps of the quadratic operator phi explodes the gradient
            # (observed sum|g| ~ 1e14) and is what stalled every earlier variant.
            c = c.detach()
            mdot = self._edge_flow(mdot0, c)
            dp = self._pipe_dp(mdot)
            dp_der = self._dp_der(mdot)
            r = self.B_int @ dp

            node_feat = torch.stack([
                self.is_boundary, injection,
                self.A @ dp, self.A_abs @ mdot.abs(),
            ], dim=-1)
            x = self.node_proj(node_feat)

            edge_bias = _slog(dp_der).unsqueeze(-1)                       # (E,1) pipes only
            edge_bias_dir = torch.cat([edge_bias, edge_bias], 0)
            for blk in self.blocks:
                x = blk(x, self.edge_index, edge_bias_dir)

            s, d_ = self.edge_index[0, :dp.numel()], self.edge_index[1, :dp.numel()]
            edge_emb = self.edge_readout(torch.cat(
                [x[s], x[d_], _slog(mdot).unsqueeze(-1), _slog(dp).unsqueeze(-1),
                 _slog(dp_der).unsqueeze(-1)], dim=-1))                   # (E, d)
            loop_emb = self.absZ.t() @ edge_emb                          # (L_int, d)
            jac_diag = (self.B_int ** 2) @ dp_der                        # (L_int,)
            newton = self._newton_step(r, dp_der, jac_diag)              # Newton step
            head_in = torch.cat([loop_emb, _slog(r).unsqueeze(-1),
                                 _slog(jac_diag).unsqueeze(-1), newton.unsqueeze(-1)], -1)

            # physics Newton base step + bounded learned refinement (zero-init head -> pure Newton)
            dc = -self.newton_damping * newton + self.step_scale * torch.tanh(self.heads[k](head_in).squeeze(-1))
            c = c + dc
            c_hist.append(c)
            # post-update residual: depends on dc, so its loss trains head_k locally
            residual_terms.append(self._residual(self._edge_flow(mdot0, c)))

            if tol is not None and not self.training:
                if residual_terms[-1].abs().max() < tol:
                    break

        return self._edge_flow(mdot0, c), residual_terms, c_hist
