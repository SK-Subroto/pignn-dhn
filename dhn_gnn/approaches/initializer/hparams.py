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

    # Which nodes the d_model embedding and attention actually run over.
    #   'all'   every node (1352 here)
    #   'cycle' only nodes within cycle_hops of an internal loop
    # The unknown lives in cycle space, and edges outside a loop are multiplied by
    # zero when edges are aggregated into loops -- so on this network 'cycle' with
    # hops=0 keeps 306 of 1352 nodes and discards nothing the head reads directly.
    # Extra hops buy back the demand/injection context that the radial branches
    # feed into the loops; hops=1 is the compromise, 'all' is the old behaviour.
    node_scope: str = "all"
    cycle_hops: int = 1


@dataclass
class InitializerTrainConfig(TrainConfig):
    epochs: int = 30               # regression converges faster than the unroll did


@dataclass
class InitializerConfig:
    arch: str = "initializer"
    model: InitializerModelConfig = field(default_factory=InitializerModelConfig)
    train: InitializerTrainConfig = field(default_factory=InitializerTrainConfig)
