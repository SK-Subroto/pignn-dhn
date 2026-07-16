"""Evaluate the (Newton-base) unrolled solver against PyDHN and generate figures.

This is a REGRESSION / solver task, so "accuracy" means: does the predicted flow
reproduce PyDHN's solution and satisfy the physics (B.phi(B^T m) = 0)? A standard
confusion matrix does not apply; the meaningful analogue is a per-pipe flow-DIRECTION
confusion matrix (sign of flow: -, ~0, +).

Outputs:
  results/model_eval.png     6-panel figure
  results/metrics.txt        numeric accuracy summary
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dhn_gnn import config
from dhn_gnn.data import network_operators as netops
from dhn_gnn.data import physics
from dhn_gnn.model.unrolled_solver import DHNUnrolledSolver

RES = Path(__file__).resolve().parents[1] / "results"


def evaluate(stride=2, K=25):
    torch.manual_seed(0)
    ops = netops.build_operators()
    Z = ops.B[ops.internal_loops].t().float()
    Bint = ops.B[ops.internal_loops].float()
    Gproj = torch.linalg.inv(Z.t() @ Z) @ Z.t()
    pm = ops.pipe_mask
    model = DHNUnrolledSolver(ops, K=K, d_model=64, step_scale=0.5, newton_damping=0.6).float()
    model.eval()

    df = pd.read_csv(config.MASS_FLOW_CSV, index_col=0)[ops.edge_names]
    ts_list = list(range(0, len(df), stride))

    pred_flow, true_flow = [], []
    pred_c, true_c = [], []
    conv_curves, final_res = [], []
    with torch.no_grad():
        for ts in ts_list:
            mt = torch.as_tensor(df.iloc[ts].to_numpy(np.float64)).float()
            a_star = Gproj @ mt
            mdot0 = mt - Z @ a_star                        # boundary-feasible, zero-circulation start
            mdot_out, terms, c_hist = model(mdot0)
            pred_flow.append(mdot_out.numpy()); true_flow.append(mt.numpy())
            pred_c.append(c_hist[-1].numpy()); true_c.append(a_star.numpy())
            conv_curves.append([t.abs().max().item() for t in terms])
            final_res.append(terms[-1].abs().max().item())

    pred_flow = np.concatenate(pred_flow); true_flow = np.concatenate(true_flow)
    pred_c = np.concatenate(pred_c); true_c = np.concatenate(true_c)
    conv = np.array(conv_curves); final_res = np.array(final_res)
    pipe_np = pm.numpy()
    pipe_mask_full = np.tile(pipe_np, len(ts_list))
    return dict(pred_flow=pred_flow, true_flow=true_flow, pred_c=pred_c, true_c=true_c,
                conv=conv, final_res=final_res, pipe_mask=pipe_mask_full, n_ts=len(ts_list))


def metrics(d):
    pf, tf = d["pred_flow"], d["true_flow"]
    err = pf - tf
    ss_res = np.sum(err ** 2); ss_tot = np.sum((tf - tf.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(err)); rmse = np.sqrt(np.mean(err ** 2)); mx = np.abs(err).max()
    # direction (deadband 1e-2 kg/s)
    db = 1e-2
    sp = np.sign(pf) * (np.abs(pf) > db); st = np.sign(tf) * (np.abs(tf) > db)
    dir_acc = np.mean(sp == st)
    return dict(r2=r2, mae=mae, rmse=rmse, maxerr=mx, dir_acc=dir_acc,
                res_mean=d["final_res"].mean(), res_med=np.median(d["final_res"]),
                res_max=d["final_res"].max(), pct50=100 * np.mean(d["final_res"] <= 50),
                pct100=100 * np.mean(d["final_res"] <= 100))


def dir_confusion(d):
    db = 1e-2
    pf, tf = d["pred_flow"], d["true_flow"]
    def lab(x): return np.where(x > db, 2, np.where(x < -db, 0, 1))  # 0:-,1:~0,2:+
    p, t = lab(pf), lab(tf)
    M = np.zeros((3, 3), int)
    for i in range(3):
        for j in range(3):
            M[i, j] = np.sum((t == i) & (p == j))
    return M


def make_figure(d, m):
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    C0, C1, C2 = "#2563eb", "#dc2626", "#16a34a"

    # (1) flow parity
    a = ax[0, 0]
    lim = np.abs(d["true_flow"]).max() * 1.05
    a.hexbin(d["true_flow"], d["pred_flow"], gridsize=60, bins="log", cmap="Blues", mincnt=1)
    a.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    a.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="PyDHN mass flow (kg/s)",
          ylabel="Predicted mass flow (kg/s)", title=f"Edge flow parity  (R²={m['r2']:.5f})")

    # (2) loop-flow parity (the 12 actually-solved unknowns)
    a = ax[0, 1]
    lim2 = np.abs(d["true_c"]).max() * 1.1
    a.scatter(d["true_c"], d["pred_c"], s=8, alpha=0.4, color=C0)
    a.plot([-lim2, lim2], [-lim2, lim2], "k--", lw=1)
    a.set(xlabel="True loop flow a* (kg/s)", ylabel="Predicted loop flow (kg/s)",
          title="Internal loop-flow accuracy (12 unknowns)")

    # (3) Newton convergence
    a = ax[0, 2]
    steps = np.arange(1, d["conv"].shape[1] + 1)
    md = np.median(d["conv"], 0); q1 = np.percentile(d["conv"], 25, 0); q3 = np.percentile(d["conv"], 75, 0)
    a.fill_between(steps, q1, q3, alpha=0.25, color=C0)
    a.plot(steps, md, "-o", ms=3, color=C0)
    a.axhline(50, ls="--", color=C1, label="50 Pa threshold")
    a.set(yscale="log", xlabel="Unrolled step k", ylabel="max |residual| (Pa)",
          title="Convergence per unrolled step"); a.legend()

    # (4) residual distribution
    a = ax[1, 0]
    a.hist(d["final_res"], bins=40, color=C0, alpha=0.8)
    a.axvline(50, ls="--", color=C1, label="50 Pa threshold")
    a.set(xlabel="final max |residual| (Pa)", ylabel="# timesteps",
          title=f"Physics residual  ({m['pct50']:.0f}% ≤ 50 Pa)"); a.legend()

    # (5) flow-direction confusion matrix
    a = ax[1, 1]
    M = dir_confusion(d); labels = ["← (−)", "~0", "→ (+)"]
    im = a.imshow(M, cmap="Greens")
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{M[i,j]:,}", ha="center", va="center",
                   color="white" if M[i, j] > M.max() / 2 else "black", fontsize=10)
    a.set(xticks=[0, 1, 2], yticks=[0, 1, 2], xticklabels=labels, yticklabels=labels,
          xlabel="Predicted direction", ylabel="PyDHN direction",
          title=f"Flow-direction confusion (acc={m['dir_acc']*100:.2f}%)")

    # (6) residual over time
    a = ax[1, 2]
    a.plot(d["final_res"], lw=0.8, color=C0)
    a.axhline(50, ls="--", color=C1)
    a.set(xlabel="timestep (sampled)", ylabel="max |residual| (Pa)",
          title="Residual across the simulation window")

    fig.suptitle("Newton-base unrolled DHN hydraulic solver vs PyDHN  "
                 f"(untrained; {d['n_ts']} timesteps, {d['pred_flow'].size:,} edge samples)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    RES.mkdir(exist_ok=True)
    fig.savefig(RES / "model_eval.png", dpi=130, bbox_inches="tight")
    print(f"saved {RES / 'model_eval.png'}")


if __name__ == "__main__":
    d = evaluate(stride=2, K=25)
    m = metrics(d)
    lines = [
        "=== Newton-base unrolled solver vs PyDHN (untrained) ===",
        f"timesteps evaluated: {d['n_ts']}   edge samples: {d['pred_flow'].size:,}",
        "",
        "-- Edge mass-flow accuracy (predicted vs PyDHN) --",
        f"  R^2            : {m['r2']:.6f}",
        f"  MAE            : {m['mae']:.4e} kg/s",
        f"  RMSE           : {m['rmse']:.4e} kg/s",
        f"  max abs error  : {m['maxerr']:.4e} kg/s",
        f"  flow-direction accuracy : {m['dir_acc']*100:.3f} %",
        "",
        "-- Physics residual  |B.phi(B^T m)|_inf (Pa) --",
        f"  mean={m['res_mean']:.1f}  median={m['res_med']:.1f}  max={m['res_max']:.1f}",
        f"  <= 50 Pa (PyDHN threshold): {m['pct50']:.1f} %   <= 100 Pa: {m['pct100']:.1f} %",
    ]
    txt = "\n".join(lines)
    print(txt)
    (RES).mkdir(exist_ok=True)
    (RES / "metrics.txt").write_text(txt, encoding="utf-8")
    make_figure(d, m)
