"""Deep-supervised training for the unrolled solver (the ORIGINAL approach).

Kept runnable so the comparison stays honest, but read the measured findings
before spending time on it:

  * At the original lr=1e-3 with step_scale=0.5 the residual goes 35 Pa -> 1.8e5 Pa
    in a single epoch. A head step of +-0.5 kg/s is enormous next to loop flows of
    3-5 kg/s, so the refinement must be made fine-grained before ANY learning rate
    is stable. Mean residual after 6 epochs on 3 samples (untrained = 34.7 Pa):

        step_scale=0.5   lr=1e-3 -> 179811     lr=1e-5 -> 13481
        step_scale=0.05  lr=1e-4 ->    206     lr=1e-5 ->    52
        step_scale=0.01  lr=1e-4 ->    106     lr=1e-5 ->    49

  * Every cell is WORSE than not training at all. On a full run the best-weight
    restore selected epoch 0, i.e. the initialization: the learned per-step
    correction never beat pure Newton on held-out data.

Rationale for deep supervision, retained from the original design: the K-step
unroll is hard to optimize when only the final output is supervised, so a target
at every step gives each step a clean gradient.
"""

import time

import numpy as np
import torch


def deep_physics_loss(residual_terms, gamma=0.9, p_ref=1e5):
    """
    Discounted per-step physics residual (Pa^2, normalized by p_ref). With truncated
    backprop each residual_terms[k] carries a LOCAL gradient to head_k, so this deep
    supervision trains every unrolled step to reduce its own step's residual.

    Averaged over steps so the value is comparable across different K.

    Note this minimizes the MEAN squared residual over loops while the reported
    metric is the MAX -- which is how a run can show the loss falling while the
    quality metric worsens.
    """
    K = len(residual_terms)
    loss = residual_terms[0].new_zeros(())
    for k, r in enumerate(residual_terms):
        loss = loss + gamma ** (K - 1 - k) * ((r / p_ref) ** 2).mean()
    return loss / K


def mean_final_residual(model, samples):
    """Mean/max over samples of max|loop residual| after the full unroll [Pa]."""
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

    The zero-init head makes the UNTRAINED model an already-decent Newton solver,
    and the learned refinement is perfectly capable of making it worse. So the
    best-scoring weights are kept and restored: training can degrade the monitored
    residual, but it can never hand back a model worse than its own initialization.
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
        tag = ("initialization (training never improved on it)" if best_ep == 0
               else f"epoch {best_ep}")
        print(f"restored best weights from {tag}: monitored residual {best_res:.1f} Pa")
    return model, dict(best_res=best_res, best_epoch=best_ep)


def fit(model, samples, cfg, monitor=None, verbose=True):
    """
    Uniform entry point for the CLI: config in, (model, history, extra meta) out.

    Approach-specific setup lives here rather than in the CLI so that adding an
    approach never means editing the command layer. The pre-training measurement
    is part of the result, not decoration: the zero-init head makes the untrained
    model an exact Newton solver, so this number is the bar training has to beat,
    and it is the bar training has so far never beaten.
    """
    r0, _ = mean_final_residual(model, samples)
    if verbose:
        print(f"untrained (zero-init head = pure Newton) mean residual: {r0:.1f} Pa")

    model, hist = train(
        model, samples,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        clip=cfg.train.clip,
        gamma=cfg.train.loss_gamma,
        log_every=max(1, cfg.train.epochs // 10),
        verbose=verbose,
        monitor=monitor,
    )
    return model, hist, dict(untrained_mean_residual=float(r0),
                             step_scale=cfg.model.step_scale)
