"""Hyperparameters for the unrolled solver (the ORIGINAL approach).

Defaults reproduce the run described in dhn_gnn/approaches/unrolled/train.py.
Read that module's header before sweeping: every learning rate measured so far
scored WORSE than not training at all, so a sweep here is characterising a
negative result, not searching for a win.
"""

from dataclasses import dataclass, field

from dhn_gnn.approaches.hyperparams import TrainConfig


@dataclass
class UnrolledModelConfig:
    """Constructor arguments -- these are baked into the parameter shapes.

    K especially: the solver carries one nn.Linear head PER unrolled step, so a
    checkpoint trained at one K cannot be loaded into a model built with another.
    """
    K: int = 10
    d_model: int = 64
    n_heads: int = 4
    num_attn_layers: int = 2
    gamma: float = 0.9
    step_scale: float = 0.01
    newton_damping: float = 1.0
    newton_mode: str = "full"      # 'diagonal' reproduces the ORIGINAL paper setup


@dataclass
class UnrolledTrainConfig(TrainConfig):
    loss_gamma: float = 0.9        # deep-supervision discount over the K steps


@dataclass
class UnrolledConfig:
    arch: str = "unrolled"
    model: UnrolledModelConfig = field(default_factory=UnrolledModelConfig)
    train: UnrolledTrainConfig = field(default_factory=UnrolledTrainConfig)

    def __post_init__(self):
        # The original approach pairs diagonal Newton with 0.5 damping; full Newton
        # needs none, and carrying 0.5 over would silently halve every step. This
        # used to live in the CLI, where it only fired if --newton-mode was passed.
        if self.model.newton_mode not in ("full", "diagonal"):
            raise ValueError(
                f"newton_mode must be 'full' or 'diagonal', "
                f"got {self.model.newton_mode!r}"
            )
