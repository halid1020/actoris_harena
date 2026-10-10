#!/usr/bin/env python
"""ACT with a cropped tactile front end, and optionally its own tactile encoder.

The model is its twin's, unchanged: the crop is a preprocessor step, so there is
nothing to override for it beyond the two attributes LeRobot resolves a policy
by. If a cropped run differs from its baseline, the crop is the only thing it can
be.

``separate_tactile_backbone`` is the one change to the model, and it is off by
default. ACT's forward pass sends every camera, in ``image_features`` order,
through ``model.backbone``; the port must stay byte-identical to upstream, so
rather than edit that loop the backbone is replaced by a router that hands each
call to the overhead or the tactile ResNet according to which camera's turn it
is. The tactile ResNet is shared by all four fingertips.
"""

from __future__ import annotations

import copy

from torch import Tensor, nn

from ..act.modeling_act import HarenaActPolicy
from .configuration_act_crop import HarenaActCropConfig


class CameraRoutedBackbone(nn.Module):
    """Two backbones, chosen per camera in the order ACT calls them.

    ACT calls the backbone once per camera, in ``image_features`` order, inside
    one forward pass; ``reset`` is called before every pass (a pre-hook on the
    model), so an exception halfway through a pass cannot leave the turn counter
    pointing at the wrong camera for the next one.
    """

    def __init__(
        self, overhead: nn.Module, tactile: nn.Module, is_tactile: "list[bool]"
    ):
        super().__init__()
        if not is_tactile:
            raise ValueError("a routed backbone needs at least one camera")
        self.overhead = overhead
        self.tactile = tactile
        self.is_tactile = list(is_tactile)
        self._turn = 0

    def reset(self) -> None:
        self._turn = 0

    def forward(self, img: Tensor) -> "dict[str, Tensor]":
        turn = self._turn
        self._turn = (turn + 1) % len(self.is_tactile)
        return (self.tactile if self.is_tactile[turn] else self.overhead)(img)


class HarenaActCropPolicy(HarenaActPolicy):
    """HarenaActPolicy, trained and served on cropped tactile images."""

    config_class = HarenaActCropConfig
    name = "harena_act_crop"

    def __init__(self, config: HarenaActCropConfig, **kwargs):
        super().__init__(config, **kwargs)
        if (
            getattr(config, "separate_tactile_backbone", False)
            and config.image_features
        ):
            cameras = set(config.tactile_cameras)
            is_tactile = [
                key.rsplit(".", 1)[-1] in cameras for key in config.image_features
            ]
            if all(is_tactile) or not any(is_tactile):
                raise ValueError(
                    "separate_tactile_backbone needs both a tactile and a non-tactile "
                    f"camera; got {list(config.image_features)}"
                )
            shared = self.model.backbone
            self.model.backbone = CameraRoutedBackbone(
                shared, copy.deepcopy(shared), is_tactile
            )
            self.model.register_forward_pre_hook(
                lambda module, args: module.backbone.reset()
            )
