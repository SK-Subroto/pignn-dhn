"""Model-layer smoke test: invariants of the unrolled solver on the real network.

Validates wiring (not learned accuracy):
  1. physics loss on the TRUE solved flow is small (residual path correct)
  2. a boundary-feasible-but-loop-violating reference has a LARGE residual (there is
     something to solve)
  3. forward pass: finite, correct shapes, no NaN
  4. mass conservation preserved by construction (A.mdot unchanged by the model)
  5. zero-init heads => identity start (mdot_final == mdot0)
  6. backward() produces finite gradients

Run with the pydhn venv python from the pignn-dhn root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhn_gnn import config
from dhn_gnn.physics import operators as netops
from dhn_gnn.losses import discounted_physics_loss
from dhn_gnn.approaches.unrolled.model import DHNUnrolledSolver

torch.manual_seed(0)
_results = []


def _check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    _results.append((name, ok))


def main():
    ops = netops.build_operators()  # float64
    Z64 = ops.B[ops.internal_loops].t()  # (E, L_int)

    # one real solved timestep
    df = pd.read_csv(config.MASS_FLOW_CSV, index_col=0)[ops.edge_names]
    mdot_true = torch.as_tensor(df.iloc[100].to_numpy(np.float64))  # (E,)

    # boundary-feasible reference: remove the internal-cycle component of the true flow
    # (satisfies the same nodal injections, but breaks loop closure -> nonzero residual)
    a_star = torch.linalg.solve(Z64.t() @ Z64, Z64.t() @ mdot_true)
    mdot0 = (mdot_true - Z64 @ a_star)

    A64 = ops.A
    _check("reference preserves injections (A.mdot0 == A.mdot_true)",
           float((A64 @ mdot0 - A64 @ mdot_true).abs().max()) <= 1e-9,
           f"max diff {float((A64 @ mdot0 - A64 @ mdot_true).abs().max()):.2e}")

    model = DHNUnrolledSolver(ops, K=15, d_model=64, n_heads=4, num_attn_layers=2).float()

    Bint = ops.B[ops.internal_loops].float()

    def residual_of(mdot):  # via model physics (float32)
        return model._residual(mdot.float())

    r_true = residual_of(mdot_true).abs().max().item()
    r_ref = residual_of(mdot0).abs().max().item()
    _check("residual(true solved flow) <= 100 Pa", r_true <= 100.0, f"max {r_true:.2f} Pa")
    _check("residual(reference) >> residual(true)", r_ref > 10 * max(r_true, 1e-6),
           f"ref {r_ref:.1f} Pa  vs true {r_true:.2f} Pa")

    # forward pass
    mdot0_f = mdot0.float()
    mdot_out, terms, c_hist = model(mdot0_f)
    _check("forward finite, no NaN", torch.isfinite(mdot_out).all().item() and
           all(torch.isfinite(t).all() for t in terms), "clean")
    _check("residual_terms length == K (post-update per step)", len(terms) == 15, f"got {len(terms)}")
    _check("output shape == (E,)", tuple(mdot_out.shape) == (ops.n_edges,), str(tuple(mdot_out.shape)))

    # mass conservation preserved by construction
    dmc = float((model.A @ mdot_out - (model.A @ mdot0_f)).abs().max())
    _check("mass conservation preserved (A.mdot_out == A.mdot0)", dmc <= 1e-3, f"max diff {dmc:.2e} kg/s")

    # untrained model (zero-init heads => pure Newton base) must REDUCE the residual
    r_in = residual_of(mdot0).abs().max().item()
    r_out = residual_of(mdot_out).abs().max().item()
    _check("untrained Newton-base reduces residual", r_out < 0.1 * r_in,
           f"residual {r_in:.0f} -> {r_out:.1f} Pa (heads zero-init => pure Newton)")

    # loss + backward (training config: use_armijo=False so grads flow at zero-init)
    loss = discounted_physics_loss(terms, gamma=0.9)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_grad = sum(g.abs().sum().item() for g in grads)
    finite = all(torch.isfinite(g).all() for g in grads)
    _check("physics loss finite", np.isfinite(loss.item()), f"L_phys = {loss.item():.3e} Pa^2")
    _check("backward: grads finite AND nonzero (trainable at init)",
           finite and total_grad > 0, f"{len(grads)} grad tensors, sum|g|={total_grad:.3e}")
    _check("c_hist length == K (per-step corrections for deep supervision)",
           len(c_hist) == 15, f"got {len(c_hist)}")

    print("\n" + "=" * 60)
    passed = sum(ok for _, ok in _results)
    for n, ok in _results:
        if not ok:
            print(f"  FAILED: {n}")
    print(f"MODEL SMOKE: {passed}/{len(_results)} passed")
    sys.exit(0 if passed == len(_results) else 1)


if __name__ == "__main__":
    main()
