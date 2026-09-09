"""Edge-biased multi-head self-attention on the physical graph.

Port of the paper's `EdgeSelfAttnBlock` (GNSMsg_SelfAttention_armijo.py:65-135),
adapted to DHN and implemented without `torch_scatter` (uses native
`scatter_reduce`). Nodes attend to neighbours over directed physical edges; each
edge contributes a per-head additive bias derived from a per-edge physical scalar
(for us: the linearized local resistance ``dphi/dmdot``, recomputed each unrolled
step — Decision D2).
"""

import math

import torch
import torch.nn as nn


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of ``scores`` (M, H) over edges grouped by destination ``index`` (M,)."""
    M, H = scores.shape
    # per-(dst,head) max for numerical stability
    maxes = scores.new_full((num_nodes, H), float("-inf"))
    maxes = maxes.scatter_reduce(0, index[:, None].expand(M, H), scores, reduce="amax",
                                 include_self=True)
    scores = scores - maxes[index]
    exp = scores.exp()
    denom = torch.zeros(num_nodes, H, dtype=exp.dtype, device=exp.device)
    denom = denom.scatter_reduce(0, index[:, None].expand(M, H), exp, reduce="sum",
                                 include_self=True)
    return exp / (denom[index] + 1e-12)


class EdgeBiasedAttention(nn.Module):
    """One pre-norm attention + FFN block with an additive per-edge bias."""

    def __init__(self, d_model: int, n_heads: int = 4, edge_bias_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.h, self.dh = d_model, n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_bias_dim, max(8, 2 * edge_bias_dim)), nn.LeakyReLU(0.1),
            nn.Linear(max(8, 2 * edge_bias_dim), n_heads),
        )
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_feat):
        """
        x:          (N, d_model) node embeddings
        edge_index: (2, M) directed edges (row 0 = src, row 1 = dst)
        edge_feat:  (M, edge_bias_dim) per-directed-edge bias input
        """
        N = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        y = self.ln1(x)
        Q = self.q(y).view(N, self.h, self.dh)
        K = self.k(y).view(N, self.h, self.dh)
        V = self.v(y).view(N, self.h, self.dh)

        logits = (Q[dst] * K[src]).sum(-1) / math.sqrt(self.dh)   # (M, H)
        logits = logits + self.edge_bias(edge_feat)               # (M, H)
        alpha = segment_softmax(logits, dst, N)                   # (M, H)
        msg = alpha.unsqueeze(-1) * V[src]                        # (M, H, dh)

        agg = torch.zeros(N, self.h, self.dh, dtype=x.dtype, device=x.device)
        agg.index_add_(0, dst, msg)
        x = x + self.drop(self.out(agg.reshape(N, self.d_model)))
        return x + self.drop(self.ffn(self.ln2(x)))
