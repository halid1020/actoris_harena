#!/usr/bin/env python
"""FastWAM with a cropped tactile front end; the model is its twin's, unchanged."""

from __future__ import annotations

from ..fastwam.modeling_fastwam import HarenaFastwamPolicy
from .configuration_fastwam_crop import HarenaFastwamCropConfig


class HarenaFastwamCropPolicy(HarenaFastwamPolicy):
    """HarenaFastwamPolicy, trained and served on cropped tactile images."""

    config_class = HarenaFastwamCropConfig
    name = "harena_fastwam_crop"
