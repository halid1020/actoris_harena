#!/usr/bin/env python
"""FastWAM, trained on tactile images cropped to the sensor centre.

FastWAM does not read the fingertips as four cameras: it reads `tactile_quad`, a
2x2 composite of them written at staging time. So the crop is applied per TILE
(`common.tactile.crop_tiles_and_restore`) -- a crop of the whole composite would
leave the four inner edges, where the tiles meet, untouched. The port itself is
not edited; this is a subclass, as for the other crop arms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig

from ..common.tactile import DEFAULT_CROP, TACTILE_CAMERAS
from ..fastwam.configuration_fastwam import HarenaFastwamConfig


@PreTrainedConfig.register_subclass("harena_fastwam_crop")
@dataclass
class HarenaFastwamCropConfig(HarenaFastwamConfig):
    """HarenaFastwamConfig plus a per-tile crop of the fingertip composite."""

    #: Fraction of each tile's HEIGHT and WIDTH kept, centred, then resized back
    #: into its quadrant. A four-edge crop is ``(0.8, 0.8)``.
    tactile_crop: tuple[float, float] = DEFAULT_CROP
    #: Fingertip cameras read on their own, if a run ever passes any.
    tactile_cameras: tuple[str, ...] = field(default_factory=lambda: TACTILE_CAMERAS)
    #: Composite cameras and their tile grid, as `recording.dataset_view` wrote them.
    tactile_tiled: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"tactile_quad": (2, 2)}
    )
