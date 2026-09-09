"""Pure predictor: one GNN pass, no Newton. The ablation isolating the learning."""

from dhn_gnn.approaches.predictor.hparams import (PredictorConfig,
                                                  PredictorModelConfig,
                                                  PredictorTrainConfig)

__all__ = ["PredictorConfig", "PredictorModelConfig", "PredictorTrainConfig"]
