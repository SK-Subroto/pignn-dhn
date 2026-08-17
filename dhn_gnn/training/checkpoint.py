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
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
