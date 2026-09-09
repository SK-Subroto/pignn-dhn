"""Training for the pure predictor -- no solver in the loop, and none behind it.

Two objectives, selected by `train.loss`:

  'residual'  minimize the discounted physics residual of the single predicted
              step. This is the quantity the model is scored on, so it is the
              honest objective. Gradients flow guess -> edge_flow -> phi ->
              loop residual in ONE step, so none of the unrolled architecture's
              truncated-backprop trouble applies here: there is nothing to
              backpropagate THROUGH.

  'a_star'    plain regression onto the exact loop-space target, identical to the
              initializer's objective. Kept as the reference condition: it is
              known to train, and its 277 Pa median guess residual is the bar.

Selection is on the monitored RESIDUAL under both objectives, not on the training
loss. Under 'a_star' those are different quantities, and picking the best
regression epoch would be selecting for the wrong thing -- the model is judged in
Pa, never in kg/s.
"""

import time

import numpy as np
import torch


def residual_loss(model, mdot0, p_ref=1e5):
    """Squared loop-law violation of the one predicted step, normalized."""
    _, terms, _ = model(mdot0)
    return ((terms[-1] / p_ref) ** 2).mean()


def mean_guess_residual(model, samples):
    """Mean/max over samples of max|residual| of the prediction [Pa]."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        res = [model(m0)[1][-1].abs().max().item() for m0, _, _ in samples]
    if was_training:
        model.train()
    return float(np.mean(res)), float(np.max(res))


def train_predictor(model, samples, epochs=60, lr=1e-3, clip=1.0, loss="residual",
                    p_ref=1e5, log_every=5, verbose=True, monitor=None):
    monitor = monitor if monitor is not None else samples
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()

    # The head is zero-init, so epoch 0 predicts c = 0 and this is the residual of
    # the untouched reference flow -- the number training has to beat, recorded
    # before any step is taken.
    best, _ = mean_guess_residual(model, monitor)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_ep = 0
    if verbose:
        print(f"ep   0: (init, c=0)            monitored residual mean={best:9.1f} Pa")

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(samples))
        tot = 0.0
        for i in perm:
            m0, a_star, _ = samples[i]
            opt.zero_grad()
            if loss == "residual":
                l = residual_loss(model, m0, p_ref)
            else:
                l = torch.nn.functional.mse_loss(model.predict_c0(m0), a_star)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += l.item()

        res_mean, res_max = mean_guess_residual(model, monitor)
        if res_mean < best:
            best, best_ep = res_mean, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % log_every == log_every - 1 or ep == 0):
            print(f"ep{ep+1:4d}: {loss}={tot/len(samples):.3e}  "
                  f"monitored residual mean={res_mean:9.1f} max={res_max:9.1f} Pa"
                  f"  [{time.time()-t0:.0f}s]")

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        tag = ("initialization (training never improved on it)" if best_ep == 0
               else f"epoch {best_ep}")
        print(f"restored best weights from {tag}: monitored residual {best:.1f} Pa")
    return model, dict(best_res=best, best_epoch=best_ep)


def fit(model, samples, cfg, monitor=None, verbose=True):
    """Uniform entry point for the CLI: config in, (model, history, extra) out."""
    r0, _ = mean_guess_residual(model, samples)
    if verbose:
        print(f"untrained (zero-init head, c=0) mean residual: {r0:.1f} Pa")
        print(f"objective: {cfg.train.loss}   (scored on residual either way)")

    model, hist = train_predictor(
        model, samples,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        clip=cfg.train.clip,
        loss=cfg.train.loss,
        p_ref=cfg.train.p_ref,
        log_every=max(1, cfg.train.epochs // 10),
        verbose=verbose,
        monitor=monitor,
    )
    # tol_pa is the bar this approach exists to test: can prediction alone reach
    # the tolerance that the initializer needs ~1.6 Newton steps to clear?
    from dhn_gnn import config as _cfg
    return model, hist, dict(untrained_mean_residual=float(r0),
                             loss=cfg.train.loss,
                             tol_pa=float(_cfg.EVAL_TOL_PA))
