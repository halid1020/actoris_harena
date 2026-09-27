#!/usr/bin/env python
"""The twin's pipeline, with the tactile crop put in front of it."""

from __future__ import annotations

from typing import Any

import torch

from ..common.tactile import CENTRED, HarenaTactileCropProcessorStep, as_fractions
from ..dreamzero.processor_dreamzero import make_harena_dreamzero_pre_post_processors
from .configuration_dreamzero_crop import HarenaDreamzeroCropConfig


def make_harena_dreamzero_crop_pre_post_processors(
    config: HarenaDreamzeroCropConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """As the twin's, with the crop inserted at index 0 (see `act_crop`)."""
    preprocessor, postprocessor = make_harena_dreamzero_pre_post_processors(
        config, dataset_stats
    )
    crop = HarenaTactileCropProcessorStep(
        fraction=config.tactile_crop,
        cameras=tuple(config.tactile_cameras),
        centre=as_fractions(getattr(config, "tactile_crop_centre", CENTRED)),
    )
    preprocessor.steps = [crop, *preprocessor.steps]
    return preprocessor, postprocessor
