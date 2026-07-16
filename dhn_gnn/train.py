"""Deep-supervised training for the unrolled solver (Approach A).

Rationale (from the failed physics-only / final-supervised attempts):
  - the K-step unroll is hard to optimize when only the *final* output is supervised;
  - deep supervision (a target at EVERY step) gives each step a clean gradient and
    tames the compounding instability.

Targets come from PyDHN's solved flows: the true internal-loop correction is
    a_star = (Z^T Z)^{-1} Z^T mdot_true ,   with   mdot_true = mdot0 + Z a_star .
We supervise every step's c_k -> a_star (discounted, later steps weighted more), and
add the physics residual as a light auxiliary so the model also *satisfies* the
governing equation, not just imitates the reference.
"""

import numpy as np
import pandas as pd
import torch

from dhn_gnn import config
from dhn_gnn.data import network_operators as netops


def make_samples(ops, timesteps):
    """Return (mdot0, a_star, mdot_true) tensors (float32) for the given timesteps."""
    Z = ops.B[ops.internal_loops].t()                      # (E, L_int) float64
    G = torch.linalg.inv(Z.t() @ Z) @ Z.t()                # (L_int, E) projector
    df = pd.read_csv(config.MASS_FLOW_CSV, index_col=0)[ops.edge_names]
    out = []
    for ts in timesteps:
        mt = torch.as_tensor(df.iloc[ts].to_numpy(np.float64))
        a = G @ mt                                         # (L_int,)
        m0 = mt - Z @ a                                    # boundary-feasible reference
        out.append((m0.float(), a.float(), mt.float()))
    return out


def deep_physics_loss(residual_terms, gamma=0.9, p_ref=1e5):
    """
    Discounted per-step physics residual (Pa^2, normalized by p_ref). With truncated
    backprop each residual_terms[k] carries a LOCAL gradient to head_k, so this deep
    supervision trains every unrolled step to reduce its own step's residual.
    """
    K = len(residual_terms)
    loss = residual_terms[0].new_zeros(())
    for k, r in enumerate(residual_terms):
        loss = loss + gamma ** (K - 1 - k) * ((r / p_ref) ** 2).mean()
    return loss


def train(model, samples, epochs=400, lr=1e-3, clip=1.0, gamma=0.9,
          log_every=50, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(samples))
        tot = 0.0
        for i in perm:
            m0, _, _ = samples[i]
            opt.zero_grad()
            _, terms, _ = model(m0)
            loss = deep_physics_loss(terms, gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += loss.item()
        if verbose and (ep % log_every == log_every - 1 or ep == 0):
            with torch.no_grad():
                res = [model(m0)[1][-1].abs().max().item() for m0, _, _ in samples]
            print(f"ep{ep+1:4d}: loss={tot/len(samples):.3e}  "
                  f"final residual mean={np.mean(res):8.1f} max={np.max(res):8.1f} Pa")
    return model


if __name__ == "__main__":
    from dhn_gnn.model.unrolled_solver import DHNUnrolledSolver
    torch.manual_seed(0)
    ops = netops.build_operators()

    # --- overfit proof: a single sample must reach solver-level residual ---
    samples = make_samples(ops, [100])
    model = DHNUnrolledSolver(ops, K=20, d_model=64, step_scale=0.5, newton_damping=0.5).float()
    with torch.no_grad():
        r0 = model(samples[0][0])[1][-1].abs().max().item()
    print(f"OVERFIT one sample (target residual ~15 Pa). Untrained final residual: {r0:.1f} Pa")
    train(model, samples, epochs=400, lr=1e-3, log_every=50)
