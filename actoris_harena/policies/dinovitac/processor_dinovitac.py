#!/usr/bin/env python
"""The act_crop pipeline: ACT's steps with the tactile crop in front.

Nothing differs but the config it is built from -- whose ``normalization_mapping``
leaves images in [0, 1] for the model's own ImageNet statistics.
"""

from __future__ import annotations

from typing import Any

import torch

from ..act_crop.processor_act_crop import make_harena_act_crop_pre_post_processors
from .configuration_dinovitac import HarenaDinovitacConfig


def make_harena_dinovitac_pre_post_processors(
    config: HarenaDinovitacConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    return make_harena_act_crop_pre_post_processors(config, dataset_stats)
