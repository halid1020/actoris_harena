#!/usr/bin/env python
"""FastWAM, in the vocabulary the world-model scorer speaks.

The point of the subclass is that `tool/eval_world_model.py` can address
FastWAM and `harena_dreamzero` with one set of calls. These check the parts of
that contract which would be WRONG rather than loud if they broke: a tiling
pair that is not an exact inverse silently scores each camera against a strip
of its neighbour, and a context count that is off by one scores the model on
reproducing the frame it was handed.

Nothing here builds the ~6B Wan core -- that needs ~20 GB of weights on no
machine in this test's reach. What is exercised is the seam, which is where the
mistakes were.
"""

from __future__ import annotations

import unittest

import torch

from actoris_harena.policies.fastwam_predict.configuration_fastwam_predict import (
    HarenaFastwamPredictConfig,
)
from actoris_harena.policies.fastwam_predict.modeling_fastwam_predict import (
    HarenaFastwamPredictPolicy,
    image_keys_of,
)


class Tiling:
    """Just enough policy to exercise the tiling pair, without the Wan core."""

    def __init__(self, config):
        self.config = config

    tile_cameras = HarenaFastwamPredictPolicy.tile_cameras
    untile_cameras = HarenaFastwamPredictPolicy.untile_cameras


class Config:
    """The two fields the tiling pair reads, and nothing else.

    `image_features` is a read-only property on the real config, derived from
    `input_features`. The tiling only ever reads its KEYS, so a stand-in
    carrying the right names exercises the same code without building a
    PreTrainedConfig the test would then have to keep valid.
    """

    def __init__(self, names, height=32, per_camera_width=16):
        self.image_features = dict.fromkeys(names)
        self.image_size = (height, per_camera_width * max(len(names), 1))


def config_with(names, height=32, per_camera_width=16):
    return Config(names, height, per_camera_width)


class ImageKeysTest(unittest.TestCase):
    def test_pad_companions_are_not_cameras(self):
        """`*_is_pad` shares the prefix and is a mask, not a frame."""
        batch = {
            "observation.images.central": torch.zeros(1),
            "observation.images.central_is_pad": torch.zeros(1),
            "observation.images.tactile_quad": torch.zeros(1),
            "observation.state": torch.zeros(1),
        }
        self.assertEqual(
            image_keys_of(batch),
            ["observation.images.central", "observation.images.tactile_quad"],
        )

    def test_the_order_is_sorted_not_insertion(self):
        batch = {
            "observation.images.zebra": torch.zeros(1),
            "observation.images.alpha": torch.zeros(1),
        }
        self.assertEqual(
            image_keys_of(batch),
            ["observation.images.alpha", "observation.images.zebra"],
        )


class TilingIsInvertibleTest(unittest.TestCase):
    """Untiling must return each camera EXACTLY, not approximately.

    A width split that is off by a column scores every camera against a strip of
    its neighbour, and the resulting PSNR is a plausible-looking number that
    means nothing.
    """

    def setUp(self):
        self.names = ["observation.images.central", "observation.images.tactile_quad"]
        self.policy = Tiling(config_with(self.names))

    def test_a_tile_splits_back_into_its_parts(self):
        each = 16
        left = torch.full((2, 5, 3, 32, each), 1.0)
        right = torch.full((2, 5, 3, 32, each), 2.0)
        tiled = torch.cat([left, right], dim=-1)
        back = self.policy.untile_cameras(tiled)
        self.assertEqual(sorted(back), self.names)
        torch.testing.assert_close(back[self.names[0]], left)
        torch.testing.assert_close(back[self.names[1]], right)

    def test_a_width_that_does_not_divide_is_refused_not_rounded(self):
        tiled = torch.zeros(1, 2, 3, 32, 33)
        with self.assertRaises(ValueError) as caught:
            self.policy.untile_cameras(tiled)
        self.assertIn("does not divide", str(caught.exception))

    def test_a_config_with_no_cameras_is_refused(self):
        policy = Tiling(config_with([]))
        with self.assertRaises(ValueError):
            policy.untile_cameras(torch.zeros(1, 2, 3, 32, 32))

    def test_tile_cameras_puts_time_on_axis_one(self):
        """The scorer slices `tiled[:, :context]`, so time must be axis 1.

        `_stack_video_from_images` returns [B, C, T, H, W] because the Wan core
        wants channels first. Swapping it here rather than teaching the scorer a
        second layout is the whole reason this method exists.
        """
        batch = {
            self.names[0]: torch.zeros(2, 5, 3, 32, 16),
            self.names[1]: torch.zeros(2, 5, 3, 32, 16),
        }
        tiled = self.policy.tile_cameras(batch)
        self.assertEqual(tiled.shape[:3], (2, 5, 3))
        self.assertEqual(tiled.shape[-1], 32)


class ContextArithmeticTest(unittest.TestCase):
    """FastWAM conditions on ONE frame, and the scorer has to know that."""

    def test_the_context_is_a_single_frame(self):
        config = HarenaFastwamPredictConfig()
        self.assertEqual(config.n_context_chunks * config.latent_frames_per_chunk, 1)

    def test_a_scored_prediction_is_pinned(self):
        """infer_joint samples; an unpinned seed measures the sampler."""
        config = HarenaFastwamPredictConfig()
        self.assertIsNotNone(config.predict_seed)
        self.assertGreater(config.predict_inference_steps, 0)


class RegistrationTest(unittest.TestCase):
    def test_the_new_name_is_what_the_config_reports(self):
        """Legacy is an alias; draccus reports the FIRST registered name."""
        self.assertEqual(HarenaFastwamPredictConfig().type, "harena_fastwam_predict")

    def test_the_legacy_name_still_resolves(self):
        from lerobot.configs import PreTrainedConfig

        self.assertIs(
            PreTrainedConfig.get_choice_class("so101_fastwam_predict"),
            HarenaFastwamPredictConfig,
        )

    def test_it_is_a_variant_and_not_a_port(self):
        """A port test would demand a byte-identical upstream file that does
        not exist; the ceiling map must still reach its twin."""
        from actoris_harena.training.matrix import PORTED_FROM, TWIN_OF

        self.assertNotIn("harena_fastwam_predict", PORTED_FROM)
        self.assertEqual(TWIN_OF["harena_fastwam_predict"], "harena_fastwam")
        self.assertEqual(TWIN_OF["harena_fastwam"], "fastwam")


if __name__ == "__main__":
    unittest.main()
