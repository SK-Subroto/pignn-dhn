"""Learned initializer: one GNN pass predicts where the solution is, Newton finishes.

Motivation, from measurements on this network:
  * PyDHN needs ~1.52 hydraulic iterations per timestep, because it WARM-STARTS
    from the previous timestep. That warm start is also what makes it strictly
    sequential: it cannot solve hour 100 before hour 99.
  * The unrolled architecture ran the attention stack once per unrolled step (K
    times), dominating runtime, and its learned per-step correction was measured
    to contribute nothing over the physics Newton step.

So invert the design. The graph network runs ONCE to predict where the solution
is, and exact Newton -- nearly free here, the loop system being only 12x12 --
takes it the rest of the way. The learned guess replaces the *previous timestep*
as the source of a good starting point, which is the whole point: it needs no
history, so timesteps stay independent and the entire year can be solved in
parallel, which PyDHN's warm-start chain structurally cannot do.

The target is exact and available: `a_star = (Z^T Z)^-1 Z^T mdot_true` is the
loop-space coordinate of the true flow, i.e. the c that makes the answer correct.
So this trains by plain regression on L_int numbers, with no gradients through the
solver -- sidestepping the truncated-backprop and head-starvation problems that
made the unrolled architecture so hard to optimize.

Measured: 3.69 -> 1.66 Newton steps to reach 50 Pa on held-out 2022 hours.
"""

import torch
import torch.nn as nn

from dhn_gnn import config
from dhn_gnn.solvers.attention import EdgeBiasedAttention
from dhn_gnn.solvers.physics_base import slog
from dhn_gnn.solvers.newton import NewtonSolver


class DHNInitializerSolver(NewtonSolver):
    """NewtonSolver whose starting point is predicted instead of zero."""

    def __init__(self, ops, n_newton: int = 4, d_model: int = 64, n_heads: int = 4,
                 num_attn_layers: int = 2, newton_mode: str = "full",
                 node_scope: str = "all", cycle_hops: int = 1,
                 rho: float = config.RHO_50, mu: float = config.MU_50):
        super().__init__(ops, n_newton=n_newton, newton_mode=newton_mode, rho=rho, mu=mu)

        if node_scope not in ("all", "cycle"):
            raise ValueError(f"node_scope must be 'all' or 'cycle', got {node_scope!r}")
        self.node_scope, self.cycle_hops = node_scope, int(cycle_hops)

        # 'cycle' restricts attention to the cycle subgraph grown by cycle_hops.
        # The mask is a BUFFER, not recomputed per forward: topology is constant
        # across the dataset, and it must travel with .to(device) like every other
        # derived operator. 'all' stores an all-true mask so the forward path has
        # a single branch and the two modes stay measurably comparable.
        node_mask, edge_sel = (self.cycle_scope(self.cycle_hops)
                               if node_scope == "cycle"
                               else (self.is_boundary.new_ones(self.A.shape[0]).bool(),
                                     self.edge_index.new_ones(self.edge_index.shape[1]).bool()))
        self.register_buffer("node_mask", node_mask)
        self.register_buffer("attn_edge_index", self.edge_index[:, edge_sel])
        self.register_buffer("attn_edge_sel", edge_sel)

        self.node_proj = nn.Linear(4, d_model)
        self.blocks = nn.ModuleList([
            EdgeBiasedAttention(d_model, n_heads, edge_bias_dim=1)
            for _ in range(num_attn_layers)
        ])
        self.edge_readout = nn.Sequential(nn.Linear(2 * d_model + 3, d_model), nn.GELU())
        self.loop_norm = nn.LayerNorm(d_model)
        # The output layer is ZERO-init, so an untrained model predicts c0 = 0 and
        # degrades exactly to NewtonSolver's cold start. Training can then only
        # improve on that reference, and a random init cannot throw the first
        # Newton step into a region where the Jacobian is meaningless.
        self.head = nn.Sequential(
            nn.Linear(d_model + 2, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def initial_guess(self, mdot0):
        """Predict the loop-space solution from the boundary state. One pass."""
        dp = self.pipe_dp(mdot0)
        der = self.dp_der(mdot0)

        node_feat = torch.stack([
            self.is_boundary, self.A @ mdot0, self.A @ dp, self.A_abs @ mdot0.abs(),
        ], dim=-1)
        x = self.node_proj(node_feat)

        edge_bias = slog(der).unsqueeze(-1)
        edge_bias_dir = torch.cat([edge_bias, edge_bias], 0)[self.attn_edge_sel]
        for blk in self.blocks:
            x = blk(x, self.attn_edge_index, edge_bias_dir)
            # Nodes outside the scope receive no messages, so their embedding is
            # just node_proj's output carried forward. Zeroing them keeps that
            # stale value from leaking into edge_readout for edges that straddle
            # the boundary of the scope.
            x = x * self.node_mask.unsqueeze(-1)

        E = dp.numel()
        s, d_ = self.edge_index[0, :E], self.edge_index[1, :E]
        edge_emb = self.edge_readout(torch.cat(
            [x[s], x[d_], slog(mdot0).unsqueeze(-1), slog(dp).unsqueeze(-1),
             slog(der).unsqueeze(-1)], dim=-1))

        # MEAN-aggregate edges into loops, not sum: a loop spans hundreds of edges,
        # so a plain sum hands the head features ~1e2 larger than the rest of the
        # network is scaled for, and the predictions come out wildly off-scale.
        loop_emb = (self.absZ.t() @ edge_emb) / self.loop_size.unsqueeze(-1)
        loop_emb = self.loop_norm(loop_emb)

        head_in = torch.cat([
            loop_emb,
            slog(self.B_int @ dp).unsqueeze(-1),
            slog((self.B_int ** 2) @ der).unsqueeze(-1),
        ], dim=-1)
        return self.head(head_in).squeeze(-1)                    # (L_int,)

    # kept as an alias: the training loop and reports read better calling this by
    # name, and it is the one method that distinguishes this class from its parent
    predict_c0 = initial_guess
