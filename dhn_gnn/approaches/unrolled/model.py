"""DHN unrolled correction solver -- the ORIGINAL approach, kept for comparison.

Each of K unrolled steps re-evaluates the physics, runs the attention stack, and
proposes a per-loop correction on top of a Newton base step:

    dc = -newton_damping * newton_step  +  step_scale * tanh(head_k(...))

with one head PER step, zero-init so the untrained model is pure Newton.

Retained because it is the baseline the current architecture replaced, and the
comparison has to stay runnable. Two measured findings are why it was replaced:

  * the learned per-step correction contributes nothing. Trained at every stable
    learning rate found, held-out residual never beat the untrained (pure Newton)
    starting point -- see the sweep in dhn_gnn/approaches/unrolled/train.py.
  * running attention K times per timestep dominates runtime for no benefit;
    the same network run ONCE as an initial guess does measurably better
    (dhn_gnn/approaches/initializer/model.py).

State is edge mass flow, but corrections live in internal-cycle space so mass
conservation holds by construction -- see dhn_gnn.solvers.physics_base.
"""

import torch
import torch.nn as nn

from dhn_gnn import config
from dhn_gnn.solvers.attention import EdgeBiasedAttention
from dhn_gnn.solvers.physics_base import DHNPhysicsBase, slog

_slog = slog   # back-compat for anything importing the private name


class DHNUnrolledSolver(DHNPhysicsBase):
    def __init__(self, ops, K: int = 20, d_model: int = 64, n_heads: int = 4,
                 num_attn_layers: int = 2, gamma: float = 0.9, step_scale: float = 0.5,
                 newton_damping: float = 0.5, newton_mode: str = "diagonal",
                 rho: float = config.RHO_50, mu: float = config.MU_50):
        super().__init__(ops, rho=rho, mu=mu)
        if newton_mode not in ("diagonal", "full"):
            raise ValueError(f"newton_mode must be 'diagonal' or 'full', got {newton_mode!r}")
        self.K, self.gamma = K, gamma
        self.step_scale, self.newton_damping = step_scale, newton_damping
        self.newton_mode = newton_mode

        self.node_proj = nn.Linear(4, d_model)   # [is_boundary, injection, agg dp, agg |flow|]
        self.blocks = nn.ModuleList([
            EdgeBiasedAttention(d_model, n_heads, edge_bias_dim=1)
            for _ in range(num_attn_layers)
        ])
        self.edge_readout = nn.Sequential(nn.Linear(2 * d_model + 3, d_model), nn.GELU())
        # per-step heads, zero-init => the model starts as pure Newton
        self.heads = nn.ModuleList([nn.Linear(d_model + 3, 1) for _ in range(K)])
        for hd in self.heads:
            nn.init.zeros_(hd.weight); nn.init.zeros_(hd.bias)

    # legacy private aliases: earlier code and tests reach for these names
    _edge_flow = DHNPhysicsBase.edge_flow
    _pipe_dp = DHNPhysicsBase.pipe_dp
    _dp_der = DHNPhysicsBase.dp_der
    _residual = DHNPhysicsBase.residual

    def forward(self, mdot0, tol=None):
        """
        mdot0: (E,) boundary-feasible reference flow.
        tol:   optional early-exit tolerance in Pa, honoured only in eval mode.
               Deliberately inert while training: each step owns a separate head,
               so exiting early would starve the later heads of gradient and leave
               them at zero-init exactly on the hard samples that need them.
        Returns (mdot_final, residual_terms, c_hist), one entry per executed step.
        """
        c = mdot0.new_zeros(self.n_free)
        injection = self.A @ mdot0                       # constant across steps
        residual_terms, c_hist = [], []

        for k in range(self.K):
            # Truncated backprop: detach the accumulated state so gradients flow
            # only within one unrolled step. Backpropagating through all K steps of
            # the quadratic operator explodes the gradient (observed sum|g| ~ 1e14)
            # and is what stalled every earlier variant.
            c = c.detach()
            mdot = self.edge_flow(mdot0, c)
            dp = self.pipe_dp(mdot)
            der = self.dp_der(mdot)
            r = self.B_int @ dp

            node_feat = torch.stack([
                self.is_boundary, injection, self.A @ dp, self.A_abs @ mdot.abs(),
            ], dim=-1)
            x = self.node_proj(node_feat)

            edge_bias = slog(der).unsqueeze(-1)
            edge_bias_dir = torch.cat([edge_bias, edge_bias], 0)
            for blk in self.blocks:
                x = blk(x, self.edge_index, edge_bias_dir)

            E = dp.numel()
            s, d_ = self.edge_index[0, :E], self.edge_index[1, :E]
            edge_emb = self.edge_readout(torch.cat(
                [x[s], x[d_], slog(mdot).unsqueeze(-1), slog(dp).unsqueeze(-1),
                 slog(der).unsqueeze(-1)], dim=-1))
            loop_emb = self.absZ.t() @ edge_emb
            jac_diag = (self.B_int ** 2) @ der
            newton = self.newton_step(r, der, self.newton_mode)
            head_in = torch.cat([loop_emb, slog(r).unsqueeze(-1),
                                 slog(jac_diag).unsqueeze(-1), newton.unsqueeze(-1)], -1)

            dc = (-self.newton_damping * newton
                  + self.step_scale * torch.tanh(self.heads[k](head_in).squeeze(-1)))
            c = c + dc
            c_hist.append(c)
            residual_terms.append(self.residual(self.edge_flow(mdot0, c)))

            if tol is not None and not self.training:
                if residual_terms[-1].abs().max() < tol:
                    break

        return self.edge_flow(mdot0, c), residual_terms, c_hist
