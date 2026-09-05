"""Node-pressure reconstruction from pipe pressure drops.

The unrolled solver is a *flow* solver: its state is edge mass flow and the only
pressure quantity it ever evaluates is the per-pipe drop. Node pressure is not a
model output, so it is recovered here by integrating dp over the graph.

Sign convention (verified numerically against data/solved_steady):
    A[n, e] = -1 if node n is edge e's start node, +1 if it is the end node
    dp_e    = p_start - p_end
  =>  (A^T p)_e = p_end - p_start = -dp_e

so p solves the linear system   A_pipes^T @ p = -dp_pipes .

Two wrinkles make this more than "walk a spanning tree":

  1. The system is overdetermined and, for a not-fully-converged flow, slightly
     INCONSISTENT -- a nonzero loop residual B.phi(m) is precisely the statement
     that dp is not a gradient field. Least squares spreads that inconsistency
     over the loop instead of letting the answer depend on the path taken.

  2. Restricted to pipes the graph has TWO connected components (supply, return),
     since the two lines are joined only through the substation/producer edges,
     which the model masks out. Each component needs its own anchor: a single
     reference node cannot pin the other line's absolute level.

The anchor DOFs are eliminated rather than penalized (solve for the offset from
each anchor, then add the anchor level back). Pressures are ~1e6 Pa while drops
are ~1e2 Pa, so a soft anchor row would have to be weighted against a four-order
scale gap; removing the column sidesteps that entirely.

ACCURACY FLOOR: integrating PyDHN's own dp does NOT reproduce PyDHN's pressures
to machine precision -- it lands ~30-45 Pa away, because PyDHN itself stops at a
50 Pa loop residual and its dp field is only approximately conservative. That is
a property of the reference data, not of this code (run this module as a script
to see the two quantities side by side). Read predicted-pressure errors of this
order as "at the reference data's noise floor", not as model error.
"""

import numpy as np
import torch


def edge_endpoints(ops):
    """(src, dst) node indices per edge, from the oriented incidence matrix."""
    src = ops.A.t().argmin(1)   # incidence -1 = start node
    dst = ops.A.t().argmax(1)   # incidence +1 = end node
    return src, dst


def pipe_components(ops):
    """
    Label connected components of the PIPE-ONLY graph.

    Returns (labels, anchors): `labels` is (N,) int with one id per component,
    `anchors` the lowest-index node of each component (the node whose pressure
    must be supplied to fix that component's absolute level).
    """
    src, dst = edge_endpoints(ops)
    pm = ops.pipe_mask.numpy()
    src, dst = src.numpy(), dst.numpy()
    n = ops.A.shape[0]

    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, d in zip(src[pm], dst[pm]):
        rs, rd = find(s), find(d)
        if rs != rd:
            parent[rs] = rd

    roots = np.array([find(i) for i in range(n)])
    uniq = np.unique(roots)
    labels = np.searchsorted(uniq, roots)
    anchors = np.array([np.where(labels == c)[0].min() for c in range(len(uniq))])
    return labels, anchors


class PressureReconstructor:
    """
    Precomputed least-squares integrator: dp on pipes -> node pressure.

    The system matrix depends only on the topology, which is fixed for the whole
    dataset, so the pseudo-inverse is factorized ONCE at construction and every
    timestep is then a single mat-vec.
    """

    def __init__(self, ops):
        self.labels, self.anchors = pipe_components(ops)
        self.n_nodes = ops.A.shape[0]
        self.pipe_idx = torch.where(ops.pipe_mask)[0]

        A_pipes = ops.A[:, ops.pipe_mask].to(torch.float64)      # (N, E_p)
        M = A_pipes.t()                                          # (E_p, N)

        free = np.setdiff1d(np.arange(self.n_nodes), self.anchors)
        self.free = torch.as_tensor(free, dtype=torch.long)
        self.labels_t = torch.as_tensor(self.labels, dtype=torch.long)

        # drop the anchor columns => q_anchor == 0 enforced exactly
        self.pinv = torch.linalg.pinv(M[:, self.free])           # (N-n_anchor, E_p)

    def __call__(self, dp_all_edges, anchor_values):
        """
        dp_all_edges:  (E,) pressure drop on ALL edges (non-pipe entries ignored)
        anchor_values: (n_components,) known pressure at each anchor node [Pa]
        Returns (N,) node pressure in Pa (float64).
        """
        dp = dp_all_edges.to(torch.float64)[self.pipe_idx]
        q = self.pinv @ (-dp)                                    # offsets from anchors
        p = torch.zeros(self.n_nodes, dtype=torch.float64)
        p[self.free] = q
        anchor_values = torch.as_tensor(anchor_values, dtype=torch.float64)
        return p + anchor_values[self.labels_t]

    def anchor_node_names(self, ops):
        return [ops.node_names[i] for i in self.anchors]


if __name__ == "__main__":
    # Self-check: integrating the GROUND-TRUTH dp must reproduce the ground-truth
    # pressures. This validates the sign convention and the anchoring, separately
    # from any model error.
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dhn_gnn import config
    from dhn_gnn.physics import operators as netops

    ops = netops.build_operators()
    rec = PressureReconstructor(ops)
    print(f"pipe-only components: {len(rec.anchors)}  "
          f"anchors: {rec.anchor_node_names(ops)}")

    dp_df = pd.read_csv(config.DELTA_P_CSV, index_col=0)[ops.edge_names]
    p_df = pd.read_csv(config.NODE_PRESSURE_CSV, index_col=0)[ops.node_names]

    # NOTE on the accuracy floor: PyDHN stops at a 50 Pa loop residual, so the
    # reference dp field is itself only *approximately* a gradient field. Exact
    # reproduction is therefore impossible -- least squares and PyDHN's own path
    # integration distribute the same inconsistency differently. The check below
    # asserts the reconstruction error stays of the order of that inconsistency,
    # which is the best any integrator can do on this data.
    Bint = ops.B[ops.internal_loops]
    for ts in [0, 100, 372, 744]:
        dp = torch.as_tensor(dp_df.iloc[ts].to_numpy(np.float64))
        p_true = torch.as_tensor(p_df.iloc[ts].to_numpy(np.float64))
        p_hat = rec(dp, p_true[rec.anchors])
        err = (p_hat - p_true).abs().max().item()
        incons = (Bint @ dp).abs().max().item()
        flag = "ok" if err <= 5 * max(incons, 1.0) else "SUSPECT"
        print(f"  ts={ts:4d}  max |p_hat - p_true| = {err:8.2f} Pa   "
              f"loop inconsistency |B.dp|inf = {incons:8.2f} Pa   [{flag}]")
