#!/usr/bin/env python
"""the diffusion policy with a cropped tactile front end.

The model is its twin's, unchanged: the crop is a preprocessor step, so there is
nothing to override here beyond the two attributes LeRobot resolves a policy by.
Keeping the class empty is the point -- if a cropped run differs from its
baseline, the crop is the only thing it can be.
"""

from __future__ import annotations

from ..diffusion.modeling_diffusion import HarenaDiffusionPolicy
from .configuration_diffusion_crop import HarenaDiffusionCropConfig


class HarenaDiffusionCropPolicy(HarenaDiffusionPolicy):
    """HarenaDiffusionPolicy, trained and served on cropped tactile images."""

    config_class = HarenaDiffusionCropConfig
    name = "harena_diffusion_crop"
