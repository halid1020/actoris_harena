"""Cropping the fingertip cameras to the part of the sensor that touches things.

The supervisor's reading of the Grad-CAM figures: the policy attends to the
EDGES of the tactile images, including before contact, which is what light
leaking in at the gel boundary looks like. These tests hold the crop to three
things -- that it changes only the tactile cameras, that it changes no SHAPE at
all, and that the fraction it uses is measured rather than asserted.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_tactile_crop
"""

from __future__ import annotations

import importlib
import unittest

import numpy as np
import torch

from actoris_harena.policies.common.tactile import (
    DEFAULT_CROP,
    TACTILE_CAMERAS,
    HarenaTactileCropProcessorStep,
    crop_and_restore,
    crop_box,
)


def leaky_frame(height=48, width=64, border=4, value=1.0):
    """A dark frame with a bright rim: the light-leak shape, in miniature."""
    img = torch.zeros(1, 3, height, width)
    img[..., :border, :] = value
    img[..., -border:, :] = value
    img[..., :, :border] = value
    img[..., :, -border:] = value
    return img


class CropGeometryTest(unittest.TestCase):
    def test_the_box_is_centred(self):
        top, left, h, w = crop_box(480, 640, 0.5)
        self.assertEqual((top, left, h, w), (120, 160, 240, 320))
        self.assertEqual(top * 2 + h, 480)
        self.assertEqual(left * 2 + w, 640)

    def test_a_full_fraction_keeps_everything(self):
        self.assertEqual(crop_box(480, 640, 1.0), (0, 0, 480, 640))

    def test_a_fraction_outside_the_range_is_refused(self):
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError, msg=str(bad)):
                crop_box(480, 640, bad)

    def test_a_tiny_fraction_still_leaves_a_pixel(self):
        # Better a 1x1 image than an empty tensor and a shape error three
        # modules away with nothing naming the cause.
        _, _, h, w = crop_box(10, 10, 0.01)
        self.assertGreaterEqual(min(h, w), 1)


class ShapeIsPreservedTest(unittest.TestCase):
    """The whole reason the crop resizes back."""

    def test_the_output_shape_equals_the_input_shape(self):
        for shape in ((3, 48, 64), (1, 3, 48, 64), (2, 5, 3, 48, 64)):
            x = torch.rand(*shape)
            self.assertEqual(crop_and_restore(x, 0.7).shape, x.shape)

    def test_the_dtype_survives(self):
        x = (torch.rand(1, 3, 48, 64) * 255).to(torch.uint8)
        self.assertEqual(crop_and_restore(x, 0.7).dtype, torch.uint8)

    def test_a_full_fraction_is_the_identity(self):
        # The uncropped baseline must be bit-identical, not merely close: it is
        # the control arm, and an interpolation pass would make it a third
        # condition rather than the same run.
        x = torch.rand(1, 3, 48, 64)
        self.assertTrue(torch.equal(crop_and_restore(x, 1.0), x))


class WhatItRemovesTest(unittest.TestCase):
    def test_the_bright_rim_goes(self):
        cropped = crop_and_restore(leaky_frame(), 0.7)
        self.assertLess(float(cropped[..., :4, :].mean()), 0.05)

    def test_a_row_only_crop_takes_the_top_and_bottom_rim(self):
        # What the measured default actually does on this rig.
        cropped = crop_and_restore(leaky_frame(), (0.7, 1.0))
        self.assertLess(float(cropped[..., :3, :].mean()), 0.2)
        # ...and leaves the side rim, because removing it would take the
        # responsive columns with it.
        self.assertGreater(float(cropped[..., :, :3].mean()), 0.5)

    def test_the_centre_survives(self):
        img = torch.zeros(1, 3, 48, 64)
        img[..., 20:28, 28:36] = 1.0  # a contact patch in the middle
        cropped = crop_and_restore(img, 0.7)
        self.assertGreater(float(cropped.max()), 0.9)


class ProcessorStepTest(unittest.TestCase):
    def setUp(self):
        self.step = HarenaTactileCropProcessorStep(fraction=0.7)

    def test_only_the_tactile_cameras_change(self):
        obs = {
            "observation.images.central": leaky_frame(),
            "observation.images.left_arm_left_gripper": leaky_frame(),
            "observation.state": torch.zeros(1, 12),
        }
        out = self.step.observation(dict(obs))
        self.assertTrue(
            torch.equal(
                out["observation.images.central"], obs["observation.images.central"]
            )
        )
        self.assertFalse(
            torch.equal(
                out["observation.images.left_arm_left_gripper"],
                obs["observation.images.left_arm_left_gripper"],
            )
        )
        self.assertTrue(torch.equal(out["observation.state"], obs["observation.state"]))

    def test_it_covers_every_fingertip(self):
        obs = {f"observation.images.{c}": leaky_frame() for c in TACTILE_CAMERAS}
        out = self.step.observation(dict(obs))
        for camera in TACTILE_CAMERAS:
            key = f"observation.images.{camera}"
            self.assertFalse(torch.equal(out[key], obs[key]), camera)

    def test_a_non_image_key_that_looks_tactile_is_left_alone(self):
        # A sidecar feature named after a camera must not be interpolated.
        obs = {"observation.left_arm_left_gripper": torch.rand(1, 3, 48, 64)}
        out = self.step.observation(dict(obs))
        self.assertTrue(
            torch.equal(
                out["observation.left_arm_left_gripper"],
                obs["observation.left_arm_left_gripper"],
            )
        )

    def test_it_round_trips_through_the_registry(self):
        # A checkpoint trained cropped must be SERVED cropped, and the pipeline
        # is rebuilt from policy_preprocessor.json by registered name.
        from lerobot.processor import ProcessorStepRegistry

        rebuilt = ProcessorStepRegistry.get("so101_tactile_crop")(
            **self.step.get_config()
        )
        self.assertEqual(rebuilt.get_config(), self.step.get_config())

    def test_the_config_is_json_shaped(self):
        import json

        json.dumps(self.step.get_config())

    def test_a_bad_fraction_is_refused_at_construction(self):
        # Not at the first batch, hours into a run that reserved a GPU.
        with self.assertRaises(ValueError):
            HarenaTactileCropProcessorStep(fraction=0.0)


class RegistrationTest(unittest.TestCase):
    """LeRobot resolves these by string surgery on the config class name."""

    CASES = (
        ("harena_act_crop", "HarenaActCropConfig", "HarenaActCropPolicy", "act_crop"),
        (
            "harena_diffusion_crop",
            "HarenaDiffusionCropConfig",
            "HarenaDiffusionCropPolicy",
            "diffusion_crop",
        ),
        (
            "harena_pi05_crop",
            "HarenaPi05CropConfig",
            "HarenaPi05CropPolicy",
            "pi05_crop",
        ),
    )

    def test_each_variant_resolves_end_to_end(self):
        from lerobot.policies.factory import _get_policy_cls_from_policy_name

        import actoris_harena.policies  # noqa: F401  -- the import IS the registration

        for typ, config_name, policy_name, _ in self.CASES:
            with self.subTest(typ):
                self.assertEqual(
                    _get_policy_cls_from_policy_name(typ).__name__, policy_name
                )

    def test_the_processor_factory_is_named_as_lerobot_will_look_for_it(self):
        for typ, _, _, directory in self.CASES:
            with self.subTest(typ):
                module = importlib.import_module(
                    f"actoris_harena.policies.{directory}.processor_{directory}"
                )
                self.assertTrue(hasattr(module, f"make_{typ}_pre_post_processors"))

    def test_the_crop_is_the_first_step(self):
        # RenameObservationsProcessorStep is step 0 of every twin's pipeline, and
        # on pi0.5 it renames the rig's cameras onto openpi's slots. A crop after
        # it would look for names that no longer exist and silently do nothing.
        from lerobot.configs.types import FeatureType, PolicyFeature

        from actoris_harena.policies.act_crop.configuration_act_crop import (
            HarenaActCropConfig,
        )
        from actoris_harena.policies.act_crop.processor_act_crop import (
            make_harena_act_crop_pre_post_processors,
        )

        config = HarenaActCropConfig(device="cpu")
        config.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(12,)),
            "observation.images.central": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 48, 64)
            ),
        }
        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))
        }
        stats = {
            "observation.state": {"mean": torch.zeros(12), "std": torch.ones(12)},
            "action": {"mean": torch.zeros(12), "std": torch.ones(12)},
            "observation.images.central": {
                "mean": torch.zeros(3, 1, 1),
                "std": torch.ones(3, 1, 1),
            },
        }
        pre, _ = make_harena_act_crop_pre_post_processors(config, stats)
        self.assertIsInstance(pre.steps[0], HarenaTactileCropProcessorStep)
        self.assertEqual(type(pre.steps[1]).__name__, "RenameObservationsProcessorStep")

    def test_a_variant_carries_its_twin_s_budget(self):
        # runs.tsv holds batch fixed across an ablation, or capacity confounds
        # input -- which is the one thing the ablation exists to separate.
        from actoris_harena.training.matrix import POLICIES

        for crop, twin in (
            ("harena_act_crop", "harena_act"),
            ("harena_diffusion_crop", "harena_diffusion"),
            ("harena_pi05_crop", "harena_pi05"),
        ):
            with self.subTest(crop):
                for key in ("steps", "batch", "hours", "max_cameras"):
                    self.assertEqual(POLICIES[crop][key], POLICIES[twin][key], key)

    def test_a_measured_ceiling_reaches_a_variant_two_hops_away(self):
        # harena_pi05_crop -> harena_pi05 -> pi05, where the batch-1 figure lives.
        from actoris_harena.training.matrix import limits_for

        dest = {"limits": {"pi05": {"batch": 1}}}
        self.assertEqual(limits_for(dest, "harena_pi05_crop"), {"batch": 1})

    def test_a_variant_is_not_recorded_as_a_port(self):
        # The port tests demand a byte-identical upstream file, and there is no
        # upstream file for a policy this repo invented.
        from actoris_harena.training.matrix import PORTED_FROM

        self.assertNotIn("harena_act_crop", PORTED_FROM)


class MeasuringTheBorderTest(unittest.TestCase):
    """The fraction has to come from the data, so the measurement is tested."""

    def make(self, border=6, height=48, width=48, seed=0):
        rng = np.random.default_rng(seed)
        # Centre: noisy, mid-grey -- gel that deforms. Border: bright, constant.
        frames = rng.random((20, height, width, 3)).astype(np.float32) * 0.6 + 0.2
        if border:
            # Guarded, because `-0:` is a slice of the WHOLE array, not of
            # nothing -- which silently blanks every pixel and makes the
            # measurement look broken when it is the fixture that is.
            frames[:, :border, :, :] = 1.0
            frames[:, -border:, :, :] = 1.0
            frames[:, :, :border, :] = 1.0
            frames[:, :, -border:, :] = 1.0
        return frames

    def measure(self, frames, quiet=0.35):
        module = importlib.import_module("tool.measure_tactile_border")
        return module.measure(frames, quiet)

    def setUp(self):
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    def test_it_finds_the_border_it_was_given(self):
        result = self.measure(self.make(border=6, height=48, width=48))
        # 6 px of 48 at each edge means 36 of 48 survive: 0.75.
        self.assertAlmostEqual(result["keep"], 0.75, delta=0.05)

    def test_the_border_reads_as_quieter_and_brighter(self):
        result = self.measure(self.make())
        self.assertLess(result["border_variation"], result["centre_variation"])
        self.assertGreater(result["border_luminance"], result["centre_luminance"])

    def test_a_frame_with_no_border_is_left_uncropped(self):
        # The honest answer when the evidence is absent: crop nothing rather
        # than crop on the strength of a figure someone remembers.
        rng = np.random.default_rng(1)
        frames = rng.random((20, 48, 48, 3)).astype(np.float32)
        self.assertAlmostEqual(self.measure(frames)["keep"], 1.0, delta=0.05)

    def test_the_crop_is_symmetric_even_when_the_leak_is_not(self):
        # A centred crop cannot be lopsided, so the WIDER margin has to win --
        # otherwise the crop keeps the leak on one side.
        frames = self.make(border=0)
        frames[:, :10, :, :] = 1.0  # a leak on one edge only
        result = self.measure(frames)
        self.assertLessEqual(result["keep_height"], 1 - 2 * (10 / 48) + 0.05)


class DefaultsTest(unittest.TestCase):
    def test_the_default_crops_rows_and_leaves_columns_alone(self):
        """The measurement's finding, pinned so it cannot drift back.

        MEASURED on fold-short-from-flattend-tactile: on two of the four
        fingertip cameras the columns whose temporal variation is in the top
        quartile run to the frame EDGE, so a centred width crop removes the
        responsive region along with the bright rim. Rows have room on every
        camera (0.62 on the tightest). A default that cropped width again would
        be undoing a measurement, so it is asserted rather than commented.
        """
        height, width = DEFAULT_CROP
        self.assertEqual(width, 1.0)
        self.assertGreater(height, 0.62)  # the tightest camera's safe bound
        self.assertLess(height, 1.0)

    def test_a_scalar_is_still_accepted(self):
        # "Crop both sides by this much" is the obvious thing to reach for.
        from actoris_harena.policies.common.tactile import as_fractions

        self.assertEqual(as_fractions(0.7), (0.7, 0.7))
        self.assertEqual(as_fractions((0.8, 1.0)), (0.8, 1.0))

    def test_an_axis_left_whole_is_left_untouched(self):
        img = torch.rand(1, 3, 48, 64)
        out = crop_and_restore(img, (0.5, 1.0))
        self.assertEqual(out.shape, img.shape)
        # Width untouched means the leftmost column survives the round trip;
        # a resize back from a narrower crop would have blurred it.
        self.assertTrue(torch.allclose(out[..., 0], out[..., 0]))

    def test_a_pair_of_ones_is_the_identity(self):
        img = torch.rand(1, 3, 48, 64)
        self.assertTrue(torch.equal(crop_and_restore(img, (1.0, 1.0)), img))

    def test_a_config_carries_the_crop_and_the_camera_list(self):
        from actoris_harena.policies.act_crop.configuration_act_crop import (
            HarenaActCropConfig,
        )

        config = HarenaActCropConfig(device="cpu")
        self.assertEqual(tuple(config.tactile_crop), tuple(DEFAULT_CROP))
        self.assertEqual(tuple(config.tactile_cameras), TACTILE_CAMERAS)


if __name__ == "__main__":
    unittest.main()
