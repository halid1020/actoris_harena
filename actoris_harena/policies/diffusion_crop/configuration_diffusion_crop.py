#!/usr/bin/env python
"""the diffusion policy, trained on tactile images cropped to the sensor centre.

Everything about the model is its twin's: this subclasses `HarenaDiffusionConfig` and
adds two fields. It is a separate registered policy rather than a flag on the
twin because the crop must travel with the CHECKPOINT -- a run trained cropped
has to be served cropped, and LeRobot rebuilds the processor pipeline from the
policy type. A flag would let the two drift apart silently.

The port itself is untouched. `tool/port_policies.py --check` re-derives the
nine ported files from upstream byte for byte, so an edit there would stop them
being ports; a subclass in a sibling package costs that check nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig

from ..common.tactile import DEFAULT_CROP, TACTILE_CAMERAS
from ..diffusion.configuration_diffusion import HarenaDiffusionConfig


@PreTrainedConfig.register_subclass("harena_diffusion_crop")
@dataclass
class HarenaDiffusionCropConfig(HarenaDiffusionConfig):
    """HarenaDiffusionConfig plus a centred crop on the fingertip cameras."""

    #: Fraction of HEIGHT and of WIDTH kept, centred, then resized back to the
    #: source size so no downstream shape changes. (1.0, 1.0) is the uncropped
    #: baseline, which makes an ablation a single field rather than a second
    #: policy. The default crops rows only, because MEASURED on this rig the
    #: bright rim and the responsive columns are the same pixels -- see
    #: `common/tactile.DEFAULT_CROP` for the per-camera numbers.
    tactile_crop: tuple[float, float] = DEFAULT_CROP

    #: Which cameras are tactile. No camera carries a `tactile` flag anywhere
    #: in this repo, so the set is named rather than inferred.
    tactile_cameras: tuple[str, ...] = field(default_factory=lambda: TACTILE_CAMERAS)


# Registered under its former name as well, so a checkpoint written before this
# package was shared, a run matrix that names it, and a training run resuming
# from its own train_config.json all still resolve. The NEW name is registered
# first, which is what makes ``config.type`` report it -- draccus's
# ``get_choice_name`` returns the first registered name for a class. See
# LEGACY_PREFIX in _port.py.
PreTrainedConfig.register_subclass("so101_diffusion_crop", HarenaDiffusionCropConfig)
