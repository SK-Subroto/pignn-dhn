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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dhn_gnn import config
from dhn_gnn.data import network_operators as netops
from dhn_gnn.data.pressure import PressureReconstructor

RES = config.RESULTS_DIR


def dataset_consistency(ops, dp_true_df, mf_df, n_ts=40):
    """
    Does OUR pressure-drop model reproduce the reference dataset's friction losses?

    This is not a model check -- it evaluates phi on the reference's OWN flows. If
    it disagrees, the solver is minimizing a different physics than the one that
    produced the targets, and every dp/pressure comparison downstream inherits that
    offset. Worth stating up front rather than discovering in review.

    Known status: the 2022 dataset supplied by the supervisor was generated with
    settings this project cannot currently reproduce (dhn_gnn.generate predates
    it), and agreement is poor -- consistent with per-pipe temperatures from a
    thermal run, where our fixed 50 degC rho/mu is right for no pipe in particular.
    """
    from dhn_gnn.data import physics
    pm = ops.pipe_mask.numpy()
    d = ops.diameter.clamp_min(1e-9)
    idx = np.linspace(0, len(mf_df) - 1, n_ts).round().astype(int)
    within, ratios = [], []
    for i in idx:
        m = torch.as_tensor(mf_df.iloc[i].to_numpy(np.float64))
        Re = physics.reynolds(m, d, config.MU_50)
        fd = physics.friction_factor(Re, d, ops.roughness)
        ours = (physics.phi(m, d, ops.length, fd, config.RHO_50) * ops.pipe_mask).numpy()
        ref = dp_true_df.iloc[i].to_numpy(np.float64)
        sel = pm & (np.abs(ref) > 1e-6)
        r = ours[sel] / ref[sel]
        within.append(100 * np.mean(np.abs(r - 1) <= 0.05))
        ratios.append(np.median(r))
    return dict(within_5pct=float(np.mean(within)),
                median_ratio=float(np.median(ratios)),
                ratio_spread=(float(np.min(ratios)), float(np.max(ratios))))


def _stats(pred, true):
    err = pred - true
    ss_tot = np.sum((true - true.mean()) ** 2)
    return dict(
        mae=np.mean(np.abs(err)), rmse=np.sqrt(np.mean(err ** 2)),
        max=np.abs(err).max(), r2=1 - np.sum(err ** 2) / ss_tot,
        rel=100 * np.mean(np.abs(err)) / max(np.mean(np.abs(true)), 1e-12),
    )


def main(args=None):
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
    mf_all = pd.read_csv(config.MASS_FLOW_CSV, index_col=0)[ops.edge_names]
    dpf_all = pd.read_csv(config.DELTA_P_FRICTION_CSV, index_col=0)[ops.edge_names]
    cons = dataset_consistency(ops, dpf_all, mf_all)

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

    # PyDHN's own cost, but ONLY if history.json actually describes this dataset.
    # A stale history (e.g. the CSVs regenerated without it) would otherwise be
    # quoted as the reference iteration count for data it never saw.
    n_rows = len(pd.read_csv(config.MASS_FLOW_CSV, index_col=0, usecols=[0]))
    hist = json.loads((config.GEN_DATA_DIR / "history.json").read_text())
    it = np.array(hist["hydraulics iterations"])
    hist_valid = len(it) == n_rows

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
        (f"  PyDHN   : {it.mean():.2f} hydraulic iterations/timestep "
         f"(median {np.median(it):.0f}), warm-started => timesteps must be solved IN ORDER"
         if hist_valid else
         f"  PyDHN   : NOT AVAILABLE for this dataset. history.json holds {len(it)} "
         f"entries but the data has {n_rows} timesteps, so it is STALE -- it "
         f"describes an earlier run. Do NOT quote its iteration count here."),
        f"  model   : see results/metrics.txt 'steps' (cold start, no history "
        f"=> timesteps are independent and can be solved in parallel)",
        "",
        "-- Dataset consistency: does OUR physics match this reference? --",
        f"  pipes whose phi is within 5% of the reference friction : {cons['within_5pct']:.1f} %",
        f"  median ratio ours/reference                            : {cons['median_ratio']:.3f} "
        f"(range {cons['ratio_spread'][0]:.3f}-{cons['ratio_spread'][1]:.3f} across timesteps)",
    ]
    if cons["within_5pct"] < 95:
        lines += [
            "  WARNING: the solver is minimizing a DIFFERENT physics than the one that",
            "  produced these targets. Flow comparisons remain meaningful (the boundary",
            "  conditions are taken from the reference), but dp and pressure errors",
            "  carry this offset and must not be reported as pure model error.",
        ]

    txt = "\n".join(lines)
    print(txt)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "comparison.txt").write_text(txt, encoding="utf-8")
    print(f"\nsaved {RES / 'comparison.txt'}")


if __name__ == "__main__":
    main()
