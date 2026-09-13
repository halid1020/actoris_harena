#!/usr/bin/env python
"""pi0.5 with a cropped tactile front end.

The model is its twin's, unchanged: the crop is a preprocessor step, so there is
nothing to override here beyond the two attributes LeRobot resolves a policy by.
Keeping the class empty is the point -- if a cropped run differs from its
baseline, the crop is the only thing it can be.
"""

from __future__ import annotations

from ..pi05.modeling_pi05 import HarenaPi05Policy
from .configuration_pi05_crop import HarenaPi05CropConfig


class HarenaPi05CropPolicy(HarenaPi05Policy):
    """HarenaPi05Policy, trained and served on cropped tactile images."""

    config_class = HarenaPi05CropConfig
    name = "harena_pi05_crop"
