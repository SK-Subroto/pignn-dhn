"""Sample construction and the train/test timestep split.

Shared by every training entry point and by the reports, so a split can never
drift between the model that was fitted and the numbers that get published.
"""

import numpy as np
import pandas as pd
import torch

from dhn_gnn import config
from dhn_gnn.data import network_operators as netops

_OPS = None


def ops_cache():
    """Network operators are fixed for the whole dataset -- build once, reuse."""
    global _OPS
    if _OPS is None:
        _OPS = netops.build_operators()
    return _OPS


def make_samples(ops, timesteps, device=None):
    """
    Return (mdot0, a_star, mdot_true) tensors (float32) for the given timesteps.

    `a_star` is the loop-space coordinate of the TRUE flow -- the c that makes the
    answer exactly right. That is what makes the learned initializer trainable by
    plain regression, with no gradients through the solver.

    The projector is built in float64 on CPU and only the float32 results move to
    `device`: the inverse is the one numerically delicate step, and doing it once
    in double precision costs nothing.
    """
    Z = ops.B[ops.internal_loops].t()                      # (E, L_int) float64
    G = torch.linalg.inv(Z.t() @ Z) @ Z.t()                # (L_int, E) projector
    df = pd.read_csv(config.MASS_FLOW_CSV, index_col=0)[ops.edge_names]
    out = []
    for ts in timesteps:
        mt = torch.as_tensor(df.iloc[ts].to_numpy(np.float64))
        a = G @ mt
        m0 = mt - Z @ a                                    # boundary-feasible reference
        trio = (m0.float(), a.float(), mt.float())
        out.append(tuple(t.to(device) for t in trio) if device is not None else trio)
    return out


def timestep_split(split_every=5, n_train=0):
    """
    (n_timesteps, train_ts, test_ts) for the current dataset.

    `n_train` subsamples the training half with an EVEN stride rather than taking
    a prefix, so a reduced training set still spans the whole weather window
    instead of one season.
    """
    n_ts = len(pd.read_csv(config.MASS_FLOW_CSV, index_col=0))
    train_ts, test_ts = config.split_timesteps(n_ts, split_every)
    if n_train:
        idx = np.linspace(0, len(train_ts) - 1, n_train).round().astype(int)
        train_ts = train_ts[np.unique(idx)]
    return n_ts, train_ts, test_ts
