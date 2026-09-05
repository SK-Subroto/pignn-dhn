"""Checkpoint save/load.

Hyperparameters travel WITH the weights. Both solver families bake architecture
into their parameter shapes -- the unrolled model carries one head per step, so a
K=20 checkpoint cannot load into any other K -- and hardcoding those values at the
evaluation site is how a report ends up describing a different model than the one
it scored. `train_ts` is recorded for the same reason: without it, nothing can
prove which timesteps a checkpoint actually saw.
"""

import torch


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
            f"no checkpoint at {path}. Train first:  python -m dhn_gnn.cli train\n"
            "Refusing to evaluate an untrained model -- the zero-init head makes it "
            "silently fall back to the pure-physics solver, which looks like a "
            "result but measures nothing about the learned component."
        )
    ck = torch.load(path, weights_only=False)
    model = model_cls(ops, **ck["model_kwargs"]).float()

    # Buffers are DERIVED from `ops` -- the cycle matrix, incidence, pipe geometry --
    # so they are rebuilt correctly by the constructor and need not match the
    # checkpoint. Only learned parameters must. Loading non-strict here means a
    # checkpoint survives a change to the set of cached buffers (which is exactly
    # what the base-class refactor did); a missing *parameter* is still fatal.
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    params = dict(model.named_parameters())
    missing_params = [k for k in missing if k in params]
    if missing_params:
        raise RuntimeError(
            f"checkpoint {path.name} is missing learned parameters {missing_params}. "
            "It was produced by a different architecture -- retrain, or load it with "
            "the model_kwargs it was saved with."
        )
    stale = [k for k in list(missing) + list(unexpected) if k not in params]
    if stale:
        print(f"note: buffer set changed since this checkpoint was saved "
              f"({', '.join(sorted(stale))}); rebuilt from the network operators.")

    model.eval()
    return model, ck
