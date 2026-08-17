"""Deep-supervised training for the unrolled solver (Approach A).

Rationale (from the failed physics-only / final-supervised attempts):
  - the K-step unroll is hard to optimize when only the *final* output is supervised;
  - deep supervision (a target at EVERY step) gives each step a clean gradient and
    tames the compounding instability.

Trains on the TRAIN half of an interleaved timestep split (config.split_timesteps)
and writes a checkpoint to results/model.pt that evaluate.py loads. The checkpoint
stores the model hyperparameters alongside the weights, because the solver carries
one head per unrolled step: a K=20 checkpoint is architecturally incompatible with
any other K, so evaluation must rebuild the exact same shape.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dhn_gnn import config
from dhn_gnn.data import network_operators as netops


def make_samples(ops, timesteps, device=None):
    """
    Return (mdot0, a_star, mdot_true) tensors (float32) for the given timesteps.

    The projector is built in float64 on CPU and only the float32 results are moved
    to `device`: the inverse is the one numerically delicate step here, and it costs
    nothing to do it once in double precision.
    """
    Z = ops.B[ops.internal_loops].t()                      # (E, L_int) float64
    G = torch.linalg.inv(Z.t() @ Z) @ Z.t()                # (L_int, E) projector
    df = pd.read_csv(config.MASS_FLOW_CSV, index_col=0)[ops.edge_names]
    out = []
    for ts in timesteps:
        mt = torch.as_tensor(df.iloc[ts].to_numpy(np.float64))
        a = G @ mt                                         # (L_int,)
        m0 = mt - Z @ a                                    # boundary-feasible reference
        trio = (m0.float(), a.float(), mt.float())
        out.append(tuple(t.to(device) for t in trio) if device is not None else trio)
    return out


def deep_physics_loss(residual_terms, gamma=0.9, p_ref=1e5):
    """
    Discounted per-step physics residual (Pa^2, normalized by p_ref). With truncated
    backprop each residual_terms[k] carries a LOCAL gradient to head_k, so this deep
    supervision trains every unrolled step to reduce its own step's residual.

    Averaged over steps so the value is comparable across different K (the early-exit
    path in eval can return fewer terms; training always runs the full K).
    """
    K = len(residual_terms)
    loss = residual_terms[0].new_zeros(())
    for k, r in enumerate(residual_terms):
        loss = loss + gamma ** (K - 1 - k) * ((r / p_ref) ** 2).mean()
    return loss / K


def mean_final_residual(model, samples):
    """Mean over samples of max|internal-loop residual| after the full unroll [Pa]."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        res = [model(m0)[1][-1].abs().max().item() for m0, _, _ in samples]
    if was_training:
        model.train()
    return float(np.mean(res)), float(np.max(res))


def train(model, samples, epochs=400, lr=1e-3, clip=1.0, gamma=0.9,
          log_every=50, verbose=True, monitor=None):
    """
    Deep-supervised training with best-checkpoint tracking.

    The zero-init head makes the UNTRAINED model an already-decent damped Newton
    solver (~35 Pa here), and the learned refinement is perfectly capable of making
    that worse -- at lr=1e-3 the residual blows up to ~1e5 Pa within one epoch. So
    the best-scoring weights are kept and restored at the end: training can degrade
    the monitored residual, but it can never hand back a model worse than its own
    best epoch (or worse than the initialization).
    """
    monitor = monitor if monitor is not None else samples
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()

    best_res, _ = mean_final_residual(model, monitor)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_ep = 0
    if verbose:
        print(f"ep   0: (init)                 monitored residual mean={best_res:9.1f} Pa")

    model.train()
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

        res_mean, res_max = mean_final_residual(model, monitor)
        if res_mean < best_res:
            best_res, best_ep = res_mean, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % log_every == log_every - 1 or ep == 0):
            print(f"ep{ep+1:4d}: loss={tot/len(samples):.3e}  "
                  f"monitored residual mean={res_mean:9.1f} max={res_max:9.1f} Pa  "
                  f"[{time.time()-t0:.0f}s]")

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        tag = "initialization (training never improved on it)" if best_ep == 0 else f"epoch {best_ep}"
        print(f"restored best weights from {tag}: monitored residual {best_res:.1f} Pa")
    return model, dict(best_res=best_res, best_epoch=best_ep)


def train_initializer(model, samples, epochs=30, lr=1e-3, clip=1.0,
                      log_every=5, verbose=True, monitor=None):
    """
    Fit the learned initializer by REGRESSION on the loop-space target.

    `a_star` from make_samples is exactly the c that reproduces the true flow, so
    the initializer has a supervised target and never needs gradients through the
    solver. That sidesteps the whole failure mode of the unrolled architecture:
    no truncated backprop, no exploding unroll, no per-step head starvation --
    just 12 numbers to regress, with exact Newton handling precision afterwards.
    """
    monitor = monitor if monitor is not None else samples
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()

    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_ep = -1

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(samples))
        tot = 0.0
        for i in perm:
            m0, a_star, _ = samples[i]
            opt.zero_grad()
            c0 = model.predict_c0(m0)
            loss = torch.nn.functional.mse_loss(c0, a_star)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += loss.item()

        model.eval()
        with torch.no_grad():
            gerr = [float((model.predict_c0(m0) - a).abs().max()) for m0, a, _ in monitor]
            # residual of the raw guess, and after the Newton polish
            r_guess = [float(model(m0)[1][0].abs().max()) for m0, _, _ in monitor]
            r_final = [float(model(m0)[1][-1].abs().max()) for m0, _, _ in monitor]
        score = float(np.mean(gerr))
        if score < best:
            best, best_ep = score, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % log_every == log_every - 1 or ep == 0):
            print(f"ep{ep+1:4d}: mse={tot/len(samples):.3e}  "
                  f"|c0-a*|max={score:.4f} kg/s  "
                  f"resid guess={np.mean(r_guess):9.1f} -> polished={np.mean(r_final):8.2e} Pa"
                  f"  [{time.time()-t0:.0f}s]")

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"restored best initializer from epoch {best_ep}: |c0-a*|max={best:.4f} kg/s")
    return model, dict(best_c0_err=best, best_epoch=best_ep)


def save_checkpoint(model, model_kwargs, path, meta=None):
    """Weights + the hyperparameters needed to rebuild this exact architecture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_kwargs": dict(model_kwargs),
        "meta": meta or {},
    }, path)
    print(f"saved checkpoint -> {path}")


def load_checkpoint(ops, path, model_cls):
    """Rebuild the model exactly as trained and load its weights. Raises if absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}. Train first:  python -m dhn_gnn.train\n"
            "Refusing to evaluate an untrained model -- the zero-init heads make it "
            "silently fall back to plain damped Newton, which looks like a result "
            "but measures nothing about the learned component."
        )
    ck = torch.load(path, weights_only=False)
    model = model_cls(ops, **ck["model_kwargs"]).float()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def timestep_split(split_every, n_train):
    """Shared train/test timestep selection for both architectures."""
    n_ts = len(pd.read_csv(config.MASS_FLOW_CSV, index_col=0))
    train_ts, test_ts = config.split_timesteps(n_ts, split_every)
    if n_train:
        idx = np.linspace(0, len(train_ts) - 1, n_train).round().astype(int)
        train_ts = train_ts[np.unique(idx)]
    return n_ts, train_ts, test_ts


def main_initializer(args, device=None):
    """Train the one-GNN-pass initializer (dhn_gnn.model.initializer)."""
    from dhn_gnn.model.initializer import DHNInitializerSolver

    n_ts, train_ts, test_ts = timestep_split(args.split_every, args.n_train)
    print(f"timesteps: {n_ts} total | train {len(train_ts)} (used) | test {len(test_ts)}")
    print(f"arch: initializer  n_newton={config.INIT_MODEL_KWARGS['n_newton']}  "
          f"epochs={args.epochs}  lr={args.lr}  device={device}")

    samples = make_samples(ops_cache(), train_ts, device=device)
    model = DHNInitializerSolver(ops_cache(), **config.INIT_MODEL_KWARGS).float()
    if device is not None:
        model = model.to(device)

    # reference points: a zero guess is what "no initializer" means
    with torch.no_grad():
        zero_err = float(np.mean([float(a.abs().max()) for _, a, _ in samples]))
    print(f"zero-guess |c0-a*|max (no initializer): {zero_err:.4f} kg/s")

    monitor = samples[::max(1, len(samples) // 20)]
    _, hist = train_initializer(model, samples, epochs=args.epochs, lr=args.lr,
                                log_every=max(1, args.epochs // 10), monitor=monitor)

    save_checkpoint(
        model, config.INIT_MODEL_KWARGS, args.out or config.CHECKPOINT_INIT,
        meta=dict(arch="initializer", split_every=args.split_every,
                  n_train=len(train_ts), epochs=args.epochs, lr=args.lr,
                  seed=args.seed, zero_guess_err=zero_err, **hist),
    )


_OPS = None


def ops_cache():
    global _OPS
    if _OPS is None:
        _OPS = netops.build_operators()
    return _OPS


def main():
    from dhn_gnn.model.unrolled_solver import DHNUnrolledSolver

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=15)
    # lr/step-scale defaults come from a measured sweep, not from taste. At the
    # original lr=1e-3, step_scale=0.5 the residual goes 35 Pa -> 1.7e5 Pa in a
    # single epoch; a head step of +-0.5 kg/s is enormous next to loop flows of
    # 3-5 kg/s, so the refinement must be made fine-grained before ANY learning
    # rate is stable. Measured mean residual after 6 epochs on 3 samples
    # (untrained baseline = 34.7 Pa):
    #   step_scale=0.5  lr=1e-3 -> 179811    lr=1e-5 -> 13481
    #   step_scale=0.05 lr=1e-4 ->    206    lr=1e-5 ->    52
    #   step_scale=0.01 lr=1e-4 ->    106    lr=1e-5 ->    49
    # Note that even the best cell is WORSE than doing no learning at all -- see
    # the baseline block in evaluate.py's metrics.txt.
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--step-scale", type=float, default=0.01,
                    help="bound on the learned per-step loop correction [kg/s]")
    ap.add_argument("--split-every", type=int, default=5,
                    help="every Nth timestep is held out for test")
    ap.add_argument("--n-train", type=int, default=80,
                    help="subsample this many training timesteps (0 = all). "
                         "A full pass over all ~596 train steps costs ~1.2 s each, "
                         "so the full set x many epochs is a multi-day CPU run.")
    ap.add_argument("--overfit-one", action="store_true",
                    help="original single-sample sanity check (ts=100), no checkpoint")
    ap.add_argument("--arch", choices=["unrolled", "initializer"], default="unrolled",
                    help="'unrolled' = GNN every step (original); "
                         "'initializer' = one GNN pass + exact Newton polish")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu",
                    help="defaults to cpu: at batch size 1 this model is too small "
                         "for a GPU to help (see config.get_device)")
    ap.add_argument("--out", type=Path, default=None,
                    help="checkpoint path (default: results/model.pt or model_init.pt). "
                         "Point smoke runs somewhere else -- a 1-epoch test writing to "
                         "the default path silently destroys a real trained model.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = config.get_device(args.device)
    if args.arch == "initializer":
        return main_initializer(args, dev)

    ops = ops_cache()
    model_kwargs = dict(config.MODEL_KWARGS, step_scale=args.step_scale)

    if args.overfit_one:
        samples = make_samples(ops, [100])
        model = DHNUnrolledSolver(ops, **model_kwargs).float()
        r0, _ = mean_final_residual(model, samples)
        print(f"OVERFIT one sample (target residual ~15 Pa). Untrained: {r0:.1f} Pa")
        train(model, samples, epochs=400, lr=args.lr, log_every=50)
        return

    n_ts, train_ts, test_ts = timestep_split(args.split_every, args.n_train)
    print(f"timesteps: {n_ts} total | train {len(train_ts)} (used) | test {len(test_ts)}")
    print(f"model: K={model_kwargs['K']}  step_scale={args.step_scale}  "
          f"epochs={args.epochs}  lr={args.lr}")

    samples = make_samples(ops, train_ts, device=dev)
    model = DHNUnrolledSolver(ops, **model_kwargs).float().to(dev)
    r0, _ = mean_final_residual(model, samples)
    print(f"untrained (zero-init head = damped Newton) mean final residual: {r0:.1f} Pa")

    # monitor on a fixed subset: the per-epoch residual sweep is a full extra
    # forward pass per sample, which would otherwise dominate the epoch cost
    monitor = samples[::max(1, len(samples) // 20)]
    _, hist = train(model, samples, epochs=args.epochs, lr=args.lr,
                    log_every=max(1, args.epochs // 10), monitor=monitor)

    save_checkpoint(
        model, model_kwargs, args.out or config.CHECKPOINT,
        meta=dict(split_every=args.split_every, n_train=len(train_ts),
                  epochs=args.epochs, lr=args.lr, step_scale=args.step_scale,
                  seed=args.seed, train_ts=train_ts.tolist(),
                  untrained_mean_residual=float(r0), **hist),
    )


if __name__ == "__main__":
    main()
