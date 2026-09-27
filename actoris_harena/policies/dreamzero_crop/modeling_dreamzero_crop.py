#!/usr/bin/env python
"""DreamZero with a cropped tactile front end; the model is its twin's, unchanged."""

from __future__ import annotations

from ..dreamzero.modeling_dreamzero import HarenaDreamzeroPolicy
from .configuration_dreamzero_crop import HarenaDreamzeroCropConfig


class HarenaDreamzeroCropPolicy(HarenaDreamzeroPolicy):
    """HarenaDreamzeroPolicy, trained and served on cropped tactile images."""

    config_class = HarenaDreamzeroCropConfig
    name = "harena_dreamzero_crop"
