"""Hyperparameters for the learned initializer (the CURRENT approach).

One attention pass predicts the loop-space solution, then `n_newton` exact Newton
steps polish it. n_newton is the knob that trades accuracy against cost: the
measured median is 1.56 steps to reach 50 Pa, so 4 leaves headroom for the tail.
"""

from dataclasses import dataclass, field

from dhn_gnn.approaches.hyperparams import TrainConfig


@dataclass
class InitializerModelConfig:
    """Constructor arguments -- baked into the parameter shapes."""
    n_newton: int = 4
    d_model: int = 64
    n_heads: int = 4
    num_attn_layers: int = 2
    newton_mode: str = "full"


@dataclass
class InitializerTrainConfig(TrainConfig):
    epochs: int = 30               # regression converges faster than the unroll did


@dataclass
class InitializerConfig:
    arch: str = "initializer"
    model: InitializerModelConfig = field(default_factory=InitializerModelConfig)
    train: InitializerTrainConfig = field(default_factory=InitializerTrainConfig)
