#!/usr/bin/env python
"""ACT with a cropped tactile front end.

The model is its twin's, unchanged: the crop is a preprocessor step, so there is
nothing to override here beyond the two attributes LeRobot resolves a policy by.
Keeping the class empty is the point -- if a cropped run differs from its
baseline, the crop is the only thing it can be.
"""

from __future__ import annotations

from ..act.modeling_act import HarenaActPolicy
from .configuration_act_crop import HarenaActCropConfig


class HarenaActCropPolicy(HarenaActPolicy):
    """HarenaActPolicy, trained and served on cropped tactile images."""

    config_class = HarenaActCropConfig
    name = "harena_act_crop"
