#!/usr/bin/env python
"""The twin's pipeline, with the per-tile tactile crop put in front of it."""

from __future__ import annotations

from typing import Any

import torch

from ..common.tactile import HarenaTactileCropProcessorStep
from ..fastwam.processor_fastwam import make_fastwam_pre_post_processors
from .configuration_fastwam_crop import HarenaFastwamCropConfig


def make_harena_fastwam_crop_pre_post_processors(
    config: HarenaFastwamCropConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """As the twin's, with the crop inserted at index 0 (see `act_crop`).

    The frames here are still the composite as staged, so the tile grid is where
    `recording.dataset_view` wrote it; the model resizes to its own input size
    afterwards, inside `forward`.
    """
    preprocessor, postprocessor = make_fastwam_pre_post_processors(
        config, dataset_stats
    )
    crop = HarenaTactileCropProcessorStep(
        fraction=config.tactile_crop,
        cameras=tuple(config.tactile_cameras),
        tiled=dict(config.tactile_tiled),
    )
    preprocessor.steps = [crop, *preprocessor.steps]
    return preprocessor, postprocessor
