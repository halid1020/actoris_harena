#!/usr/bin/env python
"""DreamZero, trained on tactile images cropped to the sensor centre.

The same arrangement as `act_crop`: a config subclass carrying the crop, so the
crop travels with the checkpoint and a model trained cropped is served cropped.
DreamZero reads each fingertip camera by name, so the ordinary per-camera crop
applies unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig

from ..common.tactile import CENTRED, DEFAULT_CROP, TACTILE_CAMERAS
from ..dreamzero.configuration_dreamzero import HarenaDreamzeroConfig


@PreTrainedConfig.register_subclass("harena_dreamzero_crop")
@dataclass
class HarenaDreamzeroCropConfig(HarenaDreamzeroConfig):
    """HarenaDreamzeroConfig plus a centred crop on the fingertip cameras."""

    #: Fraction of HEIGHT and of WIDTH kept, centred, then resized back. See
    #: `common/tactile.DEFAULT_CROP`; a four-edge crop is ``(0.8, 0.8)``.
    tactile_crop: tuple[float, float] = DEFAULT_CROP
    #: Centre of the kept box as a (row, column) fraction of the image;
    #: (0.5, 0.5) is centred, as every crop before the ridge crop was.
    tactile_crop_centre: tuple[float, float] = CENTRED
    tactile_cameras: tuple[str, ...] = field(default_factory=lambda: TACTILE_CAMERAS)
