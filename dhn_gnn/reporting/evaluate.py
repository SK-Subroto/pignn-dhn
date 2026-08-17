"""Evaluate the trained unrolled solver against PyDHN and generate figures/CSVs.

This is a REGRESSION / solver task, so "accuracy" means: does the predicted flow
reproduce PyDHN's solution and satisfy the physics (B.phi(B^T m) = 0)? A standard
confusion matrix does not apply; the meaningful analogue is a per-pipe flow-DIRECTION
confusion matrix (sign of flow: -, ~0, +).

Evaluated on the HELD-OUT timesteps from config.split_timesteps, with the trained
checkpoint from results/model.pt. The untrained model (zero-init heads == plain
damped diagonal Newton) is scored alongside as the baseline the learned refinement
has to beat -- without it, a good-looking residual says nothing, since the physics
base step already produces one.

Unrolling stops early once the loop residual drops below --tol, so an easy timestep
costs fewer than K steps; the mean/max steps actually used is reported.

Outputs (results/):
  model_eval.png        6-panel figure
  metrics.txt           numeric accuracy summary, test vs train vs baseline
  pred-mass_flow.csv    predicted edge mass flow   [kg/s]
  pred-delta_p.csv      predicted edge pressure drop [Pa]  (pipes; 0 elsewhere)
  pred-pressure.csv     reconstructed node pressure  [Pa]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dhn_gnn import config
from dhn_gnn.data import network_operators as netops
from dhn_gnn.data.pressure import PressureReconstructor
from dhn_gnn.model.unrolled_solver import DHNUnrolledSolver
from dhn_gnn.training import load_checkpoint

RES = config.RESULTS_DIR


def run(model, ops, timesteps, rec, p_true_df, tol=None, flow_df=None, device=None):
    """
    Roll the model over `timesteps`; return raw per-timestep predictions.

    `flow_df` is passed in rather than re-read: the full-year mass-flow CSV is
    ~260 MB, and this is called once per variant (test / train / baseline).
    """
    Z = ops.B[ops.internal_loops].t().float()
    Gproj = torch.linalg.inv(Z.t() @ Z) @ Z.t()
    if device is not None:
        Z, Gproj = Z.to(device), Gproj.to(device)
    df = flow_df if flow_df is not None else \
        pd.read_csv(config.MASS_FLOW_CSV, index_col=0)[ops.edge_names]
    # unrolled solver reports K steps; the initializer reports the guess + n_newton
    K = getattr(model, "K", None) or (model.n_newton + 1)

    flow, dp, press, cs, true_c = [], [], [], [], []
    conv, steps_used, final_res = [], [], []

    model.eval()
    with torch.no_grad():
        for ts in timesteps:
            mt = torch.as_tensor(df.iloc[ts].to_numpy(np.float64)).float()
            if device is not None:
                mt = mt.to(device)
            a_star = Gproj @ mt
            mdot0 = mt - Z @ a_star           # boundary-feasible, zero-circulation start
            mdot_out, terms, c_hist = model(mdot0, tol=tol)

            dp_pred = model.pipe_dp(mdot_out)
            anchors = torch.as_tensor(
                p_true_df.iloc[ts].to_numpy(np.float64)[rec.anchors])
            # pressure reconstruction stays on CPU in float64: it is a one-shot
            # dense solve, and consumer GPUs run fp64 at a small fraction of fp32
            p_pred = rec(dp_pred.detach().cpu().to(torch.float64), anchors)

            flow.append(mdot_out.cpu().numpy())
            dp.append(dp_pred.cpu().numpy())
            press.append(p_pred.numpy())
            cs.append(c_hist[-1].cpu().numpy()); true_c.append(a_star.cpu().numpy())

            curve = [t.abs().max().item() for t in terms]
            steps_used.append(len(curve))
            # pad to K with the converged value: early exit makes the curves ragged,
            # and the convergence panel needs a rectangular array
            conv.append(curve + [curve[-1]] * (K - len(curve)))
            final_res.append(curve[-1])

    return dict(
        timesteps=np.asarray(timesteps),
        flow=np.stack(flow), dp=np.stack(dp), pressure=np.stack(press),
        pred_c=np.concatenate(cs), true_c=np.concatenate(true_c),
        conv=np.asarray(conv), steps_used=np.asarray(steps_used),
        final_res=np.asarray(final_res), n_ts=len(timesteps),
    )


def score(d, ops, dp_true_df, p_true_df, flow_true_df):
    """Accuracy of flow / delta_p / pressure against the PyDHN reference."""
    ts = d["timesteps"]
    pipe = ops.pipe_mask.numpy()

    pf = d["flow"].ravel()
    tf = flow_true_df.iloc[ts].to_numpy(np.float64).ravel()
    err = pf - tf
    ss_res = np.sum(err ** 2); ss_tot = np.sum((tf - tf.mean()) ** 2)

    db = 1e-2   # direction deadband, kg/s
    sp = np.sign(pf) * (np.abs(pf) > db); st = np.sign(tf) * (np.abs(tf) > db)

    # delta_p and pressure: compare on PIPES only. The model masks the 151
    # substation/producer edges to zero, and on pipes the reference total dp
    # equals friction dp (hydrostatic is 0 for this flat network), so this is a
    # like-for-like comparison rather than scoring structural zeros.
    dpp = d["dp"][:, pipe].ravel()
    dpt = dp_true_df.iloc[ts].to_numpy(np.float64)[:, pipe].ravel()
    pp = d["pressure"].ravel()
    pt = p_true_df.iloc[ts].to_numpy(np.float64).ravel()

    return dict(
        r2=1 - ss_res / ss_tot,
        mae=np.mean(np.abs(err)), rmse=np.sqrt(np.mean(err ** 2)),
        maxerr=np.abs(err).max(), dir_acc=np.mean(sp == st),
        dp_mae=np.mean(np.abs(dpp - dpt)), dp_max=np.abs(dpp - dpt).max(),
        p_mae=np.mean(np.abs(pp - pt)), p_max=np.abs(pp - pt).max(),
        res_mean=d["final_res"].mean(), res_med=np.median(d["final_res"]),
        res_max=d["final_res"].max(),
        pct50=100 * np.mean(d["final_res"] <= 50),
        pct100=100 * np.mean(d["final_res"] <= 100),
        steps_mean=d["steps_used"].mean(), steps_max=d["steps_used"].max(),
        n_ts=d["n_ts"], n_edge=pf.size,
    )


def write_csvs(d, ops, index=None):
    """
    Predicted flow / delta_p / pressure, aligned 1:1 with the PyDHN CSVs.

    `index` carries the reference file's own row labels (timestamps in the full-year
    dataset) so the output joins directly against the ground truth; without it the
    rows would be labelled by integer position, which no longer identifies an hour.
    """
    RES.mkdir(parents=True, exist_ok=True)
    ts = d["timesteps"] if index is None else index
    out = {
        "pred-mass_flow.csv": pd.DataFrame(d["flow"], index=ts, columns=ops.edge_names),
        "pred-delta_p.csv": pd.DataFrame(d["dp"], index=ts, columns=ops.edge_names),
        "pred-pressure.csv": pd.DataFrame(d["pressure"], index=ts, columns=ops.node_names),
    }
    for name, frame in out.items():
        frame.to_csv(RES / name)
        print(f"saved {RES / name}  {frame.shape}")


def dir_confusion(d, flow_true_df):
    db = 1e-2
    pf = d["flow"].ravel()
    tf = flow_true_df.iloc[d["timesteps"]].to_numpy(np.float64).ravel()
    def lab(x): return np.where(x > db, 2, np.where(x < -db, 0, 1))  # 0:-,1:~0,2:+
    p, t = lab(pf), lab(tf)
    M = np.zeros((3, 3), int)
    for i in range(3):
        for j in range(3):
            M[i, j] = np.sum((t == i) & (p == j))
    return M


def make_figure(d, m, ops, flow_true_df, base=None, tol=None):
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    C0, C1, C2 = "#2563eb", "#dc2626", "#16a34a"
    tf = flow_true_df.iloc[d["timesteps"]].to_numpy(np.float64).ravel()
    pf = d["flow"].ravel()

    # (1) flow parity
    a = ax[0, 0]
    lim = np.abs(tf).max() * 1.05
    a.hexbin(tf, pf, gridsize=60, bins="log", cmap="Blues", mincnt=1)
    a.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    a.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="PyDHN mass flow (kg/s)",
          ylabel="Predicted mass flow (kg/s)", title=f"Edge flow parity  (R²={m['r2']:.5f})")

    # (2) loop-flow parity (the actually-solved unknowns)
    a = ax[0, 1]
    lim2 = np.abs(d["true_c"]).max() * 1.1
    a.scatter(d["true_c"], d["pred_c"], s=8, alpha=0.4, color=C0)
    a.plot([-lim2, lim2], [-lim2, lim2], "k--", lw=1)
    a.set(xlabel="True loop flow a* (kg/s)", ylabel="Predicted loop flow (kg/s)",
          title=f"Internal loop-flow accuracy ({d['true_c'].size // d['n_ts']} unknowns)")

    # (3) convergence, trained vs baseline
    a = ax[0, 2]
    steps = np.arange(1, d["conv"].shape[1] + 1)
    md = np.median(d["conv"], 0); q1 = np.percentile(d["conv"], 25, 0); q3 = np.percentile(d["conv"], 75, 0)
    a.fill_between(steps, q1, q3, alpha=0.25, color=C0)
    a.plot(steps, md, "-o", ms=3, color=C0, label="trained")
    if base is not None:
        a.plot(steps, np.median(base["conv"], 0), "-s", ms=3, color=C2,
               label="untrained (damped Newton)")
    if tol:
        a.axhline(tol, ls=":", color=C1, label=f"{tol:g} Pa early-exit tol")
    a.axhline(50, ls="--", color=C1, label="50 Pa PyDHN threshold")
    a.set(yscale="log", xlabel="Unrolled step k", ylabel="max |residual| (Pa)",
          title="Convergence per unrolled step"); a.legend(fontsize=8)

    # (4) residual distribution
    a = ax[1, 0]
    a.hist(d["final_res"], bins=40, color=C0, alpha=0.8, label="trained")
    if base is not None:
        a.hist(base["final_res"], bins=40, color=C2, alpha=0.5, label="untrained")
    a.axvline(50, ls="--", color=C1)
    a.set(xlabel="final max |residual| (Pa)", ylabel="# timesteps",
          title=f"Physics residual  ({m['pct50']:.0f}% ≤ 50 Pa)"); a.legend(fontsize=8)

    # (5) flow-direction confusion matrix
    a = ax[1, 1]
    M = dir_confusion(d, flow_true_df); labels = ["← (−)", "~0", "→ (+)"]
    a.imshow(M, cmap="Greens")
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{M[i,j]:,}", ha="center", va="center",
                   color="white" if M[i, j] > M.max() / 2 else "black", fontsize=10)
    a.set(xticks=[0, 1, 2], yticks=[0, 1, 2], xticklabels=labels, yticklabels=labels,
          xlabel="Predicted direction", ylabel="PyDHN direction",
          title=f"Flow-direction confusion (acc={m['dir_acc']*100:.2f}%)")

    # (6) unrolled steps actually used
    a = ax[1, 2]
    a.plot(d["steps_used"], lw=0.8, color=C0)
    a.axhline(m["steps_mean"], ls="--", color=C1, label=f"mean {m['steps_mean']:.1f}")
    a.set(xlabel="test timestep", ylabel="unrolled steps used",
          title=f"Early exit at {tol:g} Pa (K={d['conv'].shape[1]})" if tol
                else "Steps used (no early exit)")
    a.legend(fontsize=8)

    fig.suptitle("Unrolled DHN hydraulic solver vs PyDHN — held-out timesteps  "
                 f"({d['n_ts']} timesteps, {pf.size:,} edge samples)", fontsize=13, y=1.00)
    fig.tight_layout()
    RES.mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / "model_eval.png", dpi=130, bbox_inches="tight")
    print(f"saved {RES / 'model_eval.png'}")


def _block(title, m):
    return [
        f"-- {title}  ({m['n_ts']} timesteps, {m['n_edge']:,} edge samples) --",
        f"  flow   R^2={m['r2']:.6f}  MAE={m['mae']:.4e} kg/s  RMSE={m['rmse']:.4e}  "
        f"max={m['maxerr']:.4e}",
        f"  flow   direction accuracy : {m['dir_acc']*100:.3f} %",
        f"  dp     MAE={m['dp_mae']:.3f} Pa   max={m['dp_max']:.1f} Pa      (pipes only)",
        f"  press  MAE={m['p_mae']:.3f} Pa   max={m['p_max']:.1f} Pa",
        f"  resid  mean={m['res_mean']:.1f}  median={m['res_med']:.1f}  max={m['res_max']:.1f} Pa"
        f"   <=50Pa: {m['pct50']:.1f}%   <=100Pa: {m['pct100']:.1f}%",
        f"  steps  mean={m['steps_mean']:.2f}  max={m['steps_max']}",
        "",
    ]


def _standalone_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tol", type=float, default=config.EVAL_TOL_PA,
                    help="early-exit tolerance in Pa (0 disables)")
    ap.add_argument("--split-every", type=int, default=5)
    ap.add_argument("--max-test", type=int, default=0,
                    help="cap the number of test timesteps (0 = all)")
    ap.add_argument("--arch", choices=["initializer", "unrolled"], default="initializer",
                    help="'initializer' = one GNN pass + exact Newton (default); "
                         "'unrolled' = the original GNN-every-step model")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint to evaluate (default: the one for --arch)")
    return ap.parse_args()


def main(args=None):
    args = args if args is not None else _standalone_args()
    tol = args.tol if args.tol > 0 else None
    dev = config.get_device(args.device)

    ops = netops.build_operators()
    if args.arch == "initializer":
        from dhn_gnn.model.initializer import DHNInitializerSolver
        model_cls, ckpt_path = DHNInitializerSolver, config.CHECKPOINT_INIT
    else:
        model_cls, ckpt_path = DHNUnrolledSolver, config.CHECKPOINT
    ckpt_path = args.ckpt or ckpt_path
    model, ck = load_checkpoint(ops, ckpt_path, model_cls)
    model = model.to(dev)
    meta = ck.get("meta", {})
    print(f"loaded {ckpt_path}  arch={args.arch}  device={dev}  "
          f"kwargs={ck['model_kwargs']}")

    flow_true = pd.read_csv(config.MASS_FLOW_CSV, index_col=0)[ops.edge_names]
    dp_true = pd.read_csv(config.DELTA_P_CSV, index_col=0)[ops.edge_names]
    p_true = pd.read_csv(config.NODE_PRESSURE_CSV, index_col=0)[ops.node_names]

    train_ts, test_ts = config.split_timesteps(len(flow_true), args.split_every)
    if args.max_test:
        test_ts = test_ts[:args.max_test]

    # The TRAIN block must use the timesteps the checkpoint was ACTUALLY fitted on,
    # recorded at save time -- not a freshly recomputed split. Those differ whenever
    # the dataset changed after training (e.g. the 745-row January file being
    # replaced by the 8760-row full year), and recomputing would silently relabel
    # unseen data as "seen" and destroy the whole point of the comparison.
    fitted_ts = np.asarray(meta.get("train_ts", []), dtype=int)
    fitted_ts = fitted_ts[fitted_ts < len(flow_true)]
    if len(fitted_ts):
        train_eval_ts = fitted_ts[:len(test_ts)]
        train_label = "TRAIN (actually seen during fitting)"
    else:
        train_eval_ts = train_ts[::max(1, len(train_ts) // len(test_ts))][:len(test_ts)]
        train_label = "TRAIN-SPLIT (checkpoint recorded no train_ts; NOT verified as seen)"

    rec = PressureReconstructor(ops)
    print(f"pressure anchors: {rec.anchor_node_names(ops)} "
          f"({len(rec.anchors)} pipe-only components)")

    # Untrained baseline, same architecture. Both models zero-init their output
    # layer, so the untrained network contributes nothing and the baseline is pure
    # physics: damped diagonal Newton for 'unrolled', cold-start full Newton (c0=0)
    # for 'initializer'. That is the honest thing for the learned part to beat.
    torch.manual_seed(0)
    baseline = model_cls(ops, **ck["model_kwargs"]).float().to(dev)
    baseline.eval()

    print(f"evaluating {len(test_ts)} test / {len(train_eval_ts)} train timesteps"
          f"  (tol={tol})")
    d_test = run(model, ops, test_ts, rec, p_true, tol, flow_true, dev)
    d_train = run(model, ops, train_eval_ts, rec, p_true, tol, flow_true, dev)
    d_base = run(baseline, ops, test_ts, rec, p_true, tol, flow_true, dev)

    m_test = score(d_test, ops, dp_true, p_true, flow_true)
    m_train = score(d_train, ops, dp_true, p_true, flow_true)
    m_base = score(d_base, ops, dp_true, p_true, flow_true)

    write_csvs(d_test, ops, index=flow_true.index[test_ts])

    # Under early exit, STEPS is the meaningful comparison, not terminal residual:
    # both models stop as soon as they cross `tol`, so the faster one stops sooner
    # and reports a HIGHER residual precisely because it wasted fewer steps
    # overshooting. Ranking by residual would credit the slower model.
    step_gain = m_base["steps_mean"] - m_test["steps_mean"]
    if tol:
        verdict = (
            f"the learned component SAVES {step_gain:.2f} solver steps vs the "
            f"untrained physics baseline ({m_test['steps_mean']:.2f} vs "
            f"{m_base['steps_mean']:.2f}) at the same {tol:g} Pa tolerance"
            if step_gain > 0.05 else
            f"the learned component does NOT reduce solver steps "
            f"({m_test['steps_mean']:.2f} vs {m_base['steps_mean']:.2f}) -- the "
            f"physics step is doing the work"
        )
    else:
        verdict = (
            "the trained model reaches a LOWER residual than the untrained baseline"
            if m_test["res_mean"] < m_base["res_mean"] else
            "the trained model does NOT beat the untrained physics baseline"
        )
    lines = [
        "=== Unrolled DHN hydraulic solver vs PyDHN ===",
        f"checkpoint : {ckpt_path.name}  arch={args.arch}  "
        + "  ".join(f"{k}={v}" for k, v in ck["model_kwargs"].items()
                    if k in ("K", "n_newton", "step_scale", "newton_mode")),
        f"training   : {meta.get('n_train','?')} timesteps, {meta.get('epochs','?')} epochs, "
        f"lr={meta.get('lr','?')}, best epoch {meta.get('best_epoch','?')}",
        f"split      : every {args.split_every}th timestep held out",
        f"early exit : tol={tol} Pa" if tol else "early exit : disabled",
        "",
        *_block("TEST (held out)", m_test),
        *_block(train_label, m_train),
        *_block("BASELINE untrained (zero-init head = pure physics), TEST", m_base),
        "-- Verdict --",
        f"  steps   : trained {m_test['steps_mean']:.2f} vs baseline "
        f"{m_base['steps_mean']:.2f}",
        f"  flow MAE: trained {m_test['mae']:.4e} vs baseline {m_base['mae']:.4e} kg/s",
        f"  residual: trained {m_test['res_mean']:.1f} vs baseline "
        f"{m_base['res_mean']:.1f} Pa"
        + ("   (NOT a quality ranking -- see note below)" if tol else ""),
        f"  => {verdict}",
        "",
        "-- Notes --",
        "  * with early exit on, terminal residual is NOT comparable between models:",
        "    each stops the moment it crosses the tolerance, so a faster solver stops",
        "    sooner and reports a higher residual. Compare steps (and flow error).",
        "  * dp/pressure are scored on pipes only: the 151 substation/producer edges",
        "    are masked to zero by the model. On pipes the reference total dp equals",
        "    friction dp (hydrostatic is 0 in this flat network).",
        "  * pressure is not a model output; it is reconstructed by least-squares",
        "    integration of dp with one anchor per pipe-only component (supply,",
        "    return). Integrating PyDHN's OWN dp lands ~30-45 Pa from PyDHN's",
        "    pressures, because PyDHN stops at a 50 Pa loop residual -- treat",
        "    pressure errors of that order as the reference data's noise floor.",
    ]
    txt = "\n".join(lines)
    print(txt)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "metrics.txt").write_text(txt, encoding="utf-8")
    make_figure(d_test, m_test, ops, flow_true, base=d_base, tol=tol)


if __name__ == "__main__":
    main()
