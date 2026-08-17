"""Head-to-head comparison: model predictions vs the PyDHN reference.

Reads the prediction CSVs written by dhn_gnn.evaluate and scores them against the
PyDHN solved state on the SAME timestamps, for all three quantities.

The important subtlety is pressure. The model does not output node pressure; it is
reconstructed by integrating dp. So a raw "model pressure error" conflates two
different things:

    model pressure error  =  error in the predicted dp
                          +  error the integrator makes even on PERFECT dp

The second term is not zero, because PyDHN stops at a 50 Pa loop residual and its
own dp field is therefore not exactly conservative. This script measures that floor
directly -- integrate PyDHN's own dp, compare against PyDHN's own pressures -- so
the model's genuine contribution can be separated from it.

Usage:
    python -m dhn_gnn.compare
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dhn_gnn import config
from dhn_gnn.data import network_operators as netops
from dhn_gnn.data.pressure import PressureReconstructor

RES = config.RESULTS_DIR


def _stats(pred, true):
    err = pred - true
    ss_tot = np.sum((true - true.mean()) ** 2)
    return dict(
        mae=np.mean(np.abs(err)), rmse=np.sqrt(np.mean(err ** 2)),
        max=np.abs(err).max(), r2=1 - np.sum(err ** 2) / ss_tot,
        rel=100 * np.mean(np.abs(err)) / max(np.mean(np.abs(true)), 1e-12),
    )


def main():
    ops = netops.build_operators()
    pipe = ops.pipe_mask.numpy()

    flow_p = pd.read_csv(RES / "pred-mass_flow.csv", index_col=0)
    dp_p = pd.read_csv(RES / "pred-delta_p.csv", index_col=0)
    pr_p = pd.read_csv(RES / "pred-pressure.csv", index_col=0)
    ts = flow_p.index

    flow_t = pd.read_csv(config.MASS_FLOW_CSV, index_col=0).loc[ts, ops.edge_names]
    dp_t = pd.read_csv(config.DELTA_P_CSV, index_col=0).loc[ts, ops.edge_names]
    pr_t = pd.read_csv(config.NODE_PRESSURE_CSV, index_col=0).loc[ts, ops.node_names]

    rows = [
        ("mass flow  [kg/s]  all edges",
         _stats(flow_p[ops.edge_names].to_numpy().ravel(), flow_t.to_numpy().ravel())),
        ("mass flow  [kg/s]  pipes only",
         _stats(flow_p[ops.edge_names].to_numpy()[:, pipe].ravel(),
                flow_t.to_numpy()[:, pipe].ravel())),
        ("delta_p    [Pa]    pipes only",
         _stats(dp_p[ops.edge_names].to_numpy()[:, pipe].ravel(),
                dp_t.to_numpy()[:, pipe].ravel())),
        ("pressure   [Pa]    all nodes",
         _stats(pr_p[ops.node_names].to_numpy().ravel(), pr_t.to_numpy().ravel())),
    ]

    # --- how much of the pressure error is the integrator, not the model? ---
    rec = PressureReconstructor(ops)
    sub = np.linspace(0, len(ts) - 1, 60).round().astype(int)
    floor, model_err = [], []
    for i in sub:
        dp_true_i = torch.as_tensor(dp_t.iloc[i].to_numpy(np.float64))
        p_true_i = torch.as_tensor(pr_t.iloc[i].to_numpy(np.float64))
        p_from_true_dp = rec(dp_true_i, p_true_i[rec.anchors])
        floor.append(float((p_from_true_dp - p_true_i).abs().mean()))
        model_err.append(float(
            (torch.as_tensor(pr_p.iloc[i].to_numpy(np.float64)) - p_true_i).abs().mean()))
    floor, model_err = np.array(floor), np.array(model_err)

    hist = json.loads((config.GEN_DATA_DIR / "history.json").read_text())
    it = np.array(hist["hydraulics iterations"])

    w = 34
    lines = [
        "=== Model vs PyDHN reference ===", "",
        f"timesteps compared : {len(ts)}   ({ts[0]} .. {ts[-1]})",
        f"edges {len(ops.edge_names)} ({pipe.sum()} pipes)   nodes {len(ops.node_names)}",
        "",
        f"{'quantity':<{w}}{'MAE':>12}{'RMSE':>12}{'max err':>12}{'R2':>10}{'rel%':>8}",
        "-" * (w + 54),
    ]
    for name, s in rows:
        lines.append(f"{name:<{w}}{s['mae']:>12.4e}{s['rmse']:>12.4e}"
                     f"{s['max']:>12.4e}{s['r2']:>10.6f}{s['rel']:>8.2f}")

    lines += [
        "", "-- Pressure: model error vs reconstruction floor --",
        f"  reconstruction floor (integrate PyDHN's OWN dp) : {floor.mean():8.1f} Pa MAE",
        f"  model predicted pressure                        : {model_err.mean():8.1f} Pa MAE",
        f"  attributable to the model                       : {model_err.mean()-floor.mean():8.1f} Pa",
        "  The floor exists because PyDHN stops at a 50 Pa loop residual, so its dp",
        "  field is not exactly a gradient field and NO integrator can reproduce its",
        "  pressures exactly. Only the excess over the floor is the model's doing.",
        "",
        "-- Cost --",
        f"  PyDHN   : {it.mean():.2f} hydraulic iterations/timestep (median {np.median(it):.0f}), "
        f"warm-started => timesteps must be solved IN ORDER",
        f"  model   : see results/metrics.txt 'steps' (cold start, no history "
        f"=> timesteps are independent and can be solved in parallel)",
    ]

    txt = "\n".join(lines)
    print(txt)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "comparison.txt").write_text(txt, encoding="utf-8")
    print(f"\nsaved {RES / 'comparison.txt'}")


if __name__ == "__main__":
    main()
