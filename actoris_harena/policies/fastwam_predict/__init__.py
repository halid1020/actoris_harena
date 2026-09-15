"""FastWAM with its predicted future exposed, so a world model can be scored."""

from .configuration_fastwam_predict import HarenaFastwamPredictConfig
from .modeling_fastwam_predict import HarenaFastwamPredictPolicy

__all__ = ["HarenaFastwamPredictConfig", "HarenaFastwamPredictPolicy"]
