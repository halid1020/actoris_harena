#!/usr/bin/env python
"""The twin's pipeline, with the tactile crop put in front of it."""

from __future__ import annotations

from typing import Any

import torch

from ..common.tactile import HarenaTactileCropProcessorStep
from ..diffusion.processor_diffusion import make_harena_diffusion_pre_post_processors
from .configuration_diffusion_crop import HarenaDiffusionCropConfig


def make_harena_diffusion_crop_pre_post_processors(
    config: HarenaDiffusionCropConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """As the twin's, with the crop inserted at index 0.

    Index 0 is load-bearing: `RenameObservationsProcessorStep` is the first step
    of every one of these pipelines, and on pi0.5 it renames the rig's cameras
    onto openpi's slot names. A crop placed after it would look for camera names
    that no longer exist and silently do nothing at all.
    """
    preprocessor, postprocessor = make_harena_diffusion_pre_post_processors(
        config, dataset_stats
    )
    crop = HarenaTactileCropProcessorStep(
        fraction=config.tactile_crop,
        cameras=tuple(config.tactile_cameras),
    )
    preprocessor.steps = [crop, *preprocessor.steps]
    return preprocessor, postprocessor


# And under the former name, for a config decoded from an older
# checkpoint that reports the legacy type.
make_so101_diffusion_crop_pre_post_processors = (
    make_harena_diffusion_crop_pre_post_processors
)
