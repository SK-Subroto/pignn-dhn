"""Training for the learned initializer -- plain regression, no solver in the loop.

`a_star` from make_samples is exactly the loop-space coordinate that reproduces
the true flow, so the initializer has a supervised target and never needs
gradients through the solver. That is what makes this trainable where the unrolled
architecture was not: no truncated backprop, no exploding unroll, no per-step head
starvation. Just L_int numbers to regress, with exact Newton supplying precision
afterwards.

Selection is on |c0 - a*|, the guess error, NOT on the polished residual: Newton
hides a mediocre guess by working harder, so scoring after the polish would be
scoring the physics rather than the thing being trained.
"""

import time

import numpy as np
import torch


def train_initializer(model, samples, epochs=30, lr=1e-3, clip=1.0,
                      log_every=5, verbose=True, monitor=None):
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
            loss = torch.nn.functional.mse_loss(model.predict_c0(m0), a_star)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += loss.item()

        model.eval()
        with torch.no_grad():
            guess_err = [float((model.predict_c0(m0) - a).abs().max())
                         for m0, a, _ in monitor]
            r_guess = [float(model(m0)[1][0].abs().max()) for m0, _, _ in monitor]
            r_final = [float(model(m0)[1][-1].abs().max()) for m0, _, _ in monitor]
        score = float(np.mean(guess_err))
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
