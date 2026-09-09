"""Hyperparameters for the pure predictor (no Newton).

Architecture knobs mirror the initializer exactly, so the two can be compared at
matched capacity. What differs is `train.loss`: with no Newton polish behind it,
the physics residual is the quantity that decides success, so it is the default
objective rather than regression onto a_star.
"""

from dataclasses import dataclass, field

from dhn_gnn.approaches.hyperparams import TrainConfig


@dataclass
class PredictorModelConfig:
    """Same network as the initializer. No n_newton, no newton_mode: there is
    no Newton step for them to configure."""
    d_model: int = 64
    n_heads: int = 4
    num_attn_layers: int = 2

    # See InitializerModelConfig for what these mean. 'cycle' matters more here
    # than for the initializer: with no Newton to absorb a mediocre guess, the
    # prediction has to be right on its own, and attention spread over the ~78%
    # of nodes that touch no loop is capacity spent away from the unknown.
    node_scope: str = "all"
    cycle_hops: int = 1


@dataclass
class PredictorTrainConfig(TrainConfig):
    # 'residual' optimizes max|B.phi(mdot)| -- the 50 Pa tolerance the model is
    #   actually judged on. Correct objective, but a harder optimization: the
    #   loop operator is quadratic in flow.
    # 'a_star'  regresses the exact loop-space target, as the initializer does.
    #   Trains easily and is the known-good reference; it plateaued at a median
    #   277 Pa guess residual, which is the number this approach must beat.
    loss: str = "residual"
    p_ref: float = 1e5             # Pa, normalizes the residual loss
    epochs: int = 60               # no closed-form target, so it needs longer


@dataclass
class PredictorConfig:
    arch: str = "predictor"
    model: PredictorModelConfig = field(default_factory=PredictorModelConfig)
    train: PredictorTrainConfig = field(default_factory=PredictorTrainConfig)

    def __post_init__(self):
        if self.train.loss not in ("residual", "a_star"):
            raise ValueError(
                f"train.loss must be 'residual' or 'a_star', got {self.train.loss!r}"
            )
        if self.model.node_scope not in ("all", "cycle"):
            raise ValueError(
                f"node_scope must be 'all' or 'cycle', got {self.model.node_scope!r}"
            )
