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
    crop_only,
    crop_tiles_and_restore,
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


class OffCentreBoxTest(unittest.TestCase):
    """The ridge crop: a box to the right of the gel's vertical ridge."""

    def test_centred_is_exactly_the_old_arithmetic(self):
        self.assertEqual(
            crop_box(480, 640, (0.8, 0.8), (0.5, 0.5)), crop_box(480, 640, (0.8, 0.8))
        )

    def test_the_ridge_crop_keeps_columns_036_to_090(self):
        top, left, h, w = crop_box(480, 640, (0.8, 0.54), (0.5, 0.63))
        self.assertEqual((top, h), (48, 384))
        self.assertAlmostEqual(left / 640, 0.36, places=2)
        self.assertAlmostEqual((left + w) / 640, 0.90, places=2)

    def test_a_box_past_the_edge_is_slid_back_inside(self):
        top, left, h, w = crop_box(100, 100, 0.5, (0.5, 1.0))
        self.assertEqual(left + w, 100)

    def test_a_centre_outside_the_image_is_refused(self):
        with self.assertRaises(ValueError):
            crop_box(100, 100, 0.5, (0.5, 1.5))

    def test_the_step_crops_off_centre_and_round_trips(self):
        from lerobot.processor import ProcessorStepRegistry

        img = torch.zeros(1, 3, 10, 100)
        img[..., :, :30] = 1.0  # a bright band left of where the box starts
        step = HarenaTactileCropProcessorStep(fraction=(1.0, 0.5), centre=(0.5, 0.7))
        key = f"observation.images.{TACTILE_CAMERAS[0]}"
        out = step.observation({key: img})[key]
        self.assertEqual(float(out.max()), 0.0)
        rebuilt = ProcessorStepRegistry.get("so101_tactile_crop")(**step.get_config())
        self.assertEqual(rebuilt.get_config()["centre"], [0.5, 0.7])

    def test_a_tiled_composite_crops_each_tile_off_centre(self):
        img = torch.zeros(1, 3, 20, 200)
        img[..., :, :30] = 1.0  # the left band of the left tiles only
        img[..., :, 100:130] = 1.0
        out = crop_tiles_and_restore(img, (1.0, 0.5), 2, 2, centre=(0.5, 0.7))
        self.assertEqual(float(out.max()), 0.0)


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


def quad(height=48, width=64, border=3):
    """A 2x2 composite of four leaky tiles, each tile's interior a distinct value."""
    tile_h, tile_w = height // 2, width // 2
    img = torch.zeros(1, 3, height, width)
    for i in range(4):
        row, col = divmod(i, 2)
        tile = leaky_frame(tile_h, tile_w, border=border, value=9.0)
        tile[..., border:-border, border:-border] = float(i + 1)
        img[
            ..., row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w
        ] = tile
    return img


class TiledCompositeTest(unittest.TestCase):
    """FastWAM reads the four fingertips as one 2x2 composite."""

    def test_each_tile_keeps_its_own_content_in_its_own_quadrant(self):
        out = crop_tiles_and_restore(quad(), (0.6, 0.6), 2, 2)
        self.assertEqual(out.shape, quad().shape)
        for i in range(4):
            row, col = divmod(i, 2)
            tile = out[..., row * 24 : (row + 1) * 24, col * 32 : (col + 1) * 32]
            self.assertTrue(torch.allclose(tile, torch.full_like(tile, i + 1.0)), i)

    def test_the_inner_edges_go_too(self):
        # A crop of the whole composite would keep the rims where tiles meet.
        whole = crop_and_restore(quad(), (0.6, 0.6))
        tiled = crop_tiles_and_restore(quad(), (0.6, 0.6), 2, 2)
        self.assertAlmostEqual(float(tiled.max()), 4.0, places=4)
        self.assertAlmostEqual(float(whole.max()), 9.0, places=4)

    def test_the_step_crops_a_named_composite_per_tile(self):
        step = HarenaTactileCropProcessorStep(
            fraction=(0.6, 0.6), tiled={"tactile_quad": (2, 2)}
        )
        key = "observation.images.tactile_quad"
        out = step.observation({key: quad(), "observation.images.central": quad()})
        self.assertEqual(float(out[key].max()), 4.0)
        self.assertTrue(torch.equal(out["observation.images.central"], quad()))
        self.assertEqual(step.get_config()["tiled"], {"tactile_quad": [2, 2]})

    def test_a_tiled_crop_without_resize_is_refused(self):
        with self.assertRaises(ValueError):
            HarenaTactileCropProcessorStep(tiled={"tactile_quad": (2, 2)}, resize=False)

    def test_the_fastwam_crop_names_its_composite_by_default(self):
        from actoris_harena.policies.fastwam_crop.configuration_fastwam_crop import (
            HarenaFastwamCropConfig,
        )

        self.assertEqual(
            HarenaFastwamCropConfig.__dataclass_fields__[
                "tactile_tiled"
            ].default_factory(),
            {"tactile_quad": (2, 2)},
        )


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
        (
            "harena_dreamzero_crop",
            "HarenaDreamzeroCropConfig",
            "HarenaDreamzeroCropPolicy",
            "dreamzero_crop",
        ),
        (
            "harena_fastwam_crop",
            "HarenaFastwamCropConfig",
            "HarenaFastwamCropPolicy",
            "fastwam_crop",
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
            ("harena_dreamzero_crop", "harena_dreamzero"),
            ("harena_fastwam_crop", "harena_fastwam"),
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


class CropWithoutResizeTest(unittest.TestCase):
    """The arm that separates rim removal from the stretch resizing applies.

    Every cropping result in this work is a result about two things done at
    once: the rim is removed AND what remains is stretched back to the original
    height. This is the flag that does only the first, so the two can be told
    apart.
    """

    def test_the_image_really_is_shorter(self):
        image = torch.zeros(1, 3, 40, 40)
        self.assertEqual(tuple(crop_only(image, (0.8, 1.0)).shape), (1, 3, 32, 40))

    def test_width_is_untouched_by_a_row_only_crop(self):
        image = torch.zeros(3, 40, 60)
        self.assertEqual(crop_only(image, (0.8, 1.0)).shape[-1], 60)

    def test_a_full_fraction_is_still_the_identity(self):
        image = torch.rand(3, 20, 20)
        self.assertTrue(torch.equal(crop_only(image, (1.0, 1.0)), image))

    def test_it_removes_the_same_rows_the_resizing_crop_removes(self):
        # The two differ in what happens AFTER the crop, never in what is cut.
        image = leaky_frame(height=40, width=40, border=4)
        top, _, keep_h, _ = crop_box(40, 40, (0.8, 1.0))
        self.assertTrue(
            torch.equal(crop_only(image, (0.8, 1.0)), image[..., top : top + keep_h, :])
        )

    def test_the_step_declares_the_shape_it_produces(self):
        # A policy built from features that still claimed the source size would
        # construct its encoder around an image it never receives.
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.utils.constants import OBS_IMAGES

        cameras = ("left_arm_left_gripper",)
        step = HarenaTactileCropProcessorStep(
            fraction=(0.8, 1.0), cameras=cameras, resize=False
        )
        features = {
            f"{OBS_IMAGES}.central": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 240, 320)
            ),
            f"{OBS_IMAGES}.left_arm_left_gripper": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 240, 320)
            ),
        }
        out = step.transform_features(features)
        self.assertEqual(out[f"{OBS_IMAGES}.central"].shape, (3, 240, 320))
        self.assertEqual(
            out[f"{OBS_IMAGES}.left_arm_left_gripper"].shape, (3, 192, 320)
        )

    def test_resizing_on_declares_nothing_new(self):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.utils.constants import OBS_IMAGES

        step = HarenaTactileCropProcessorStep(
            fraction=(0.8, 1.0), cameras=("left_arm_left_gripper",), resize=True
        )
        key = f"{OBS_IMAGES}.left_arm_left_gripper"
        features = {key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320))}
        self.assertEqual(step.transform_features(features)[key].shape, (3, 240, 320))

    def test_the_setting_is_recorded_so_a_checkpoint_is_served_as_trained(self):
        step = HarenaTactileCropProcessorStep(fraction=(0.8, 1.0), resize=False)
        self.assertIs(step.get_config()["resize"], False)

    def test_resizing_back_is_the_default_everywhere(self):
        # Every result measured so far assumes it, so the flag must be opt-in.
        self.assertIs(HarenaTactileCropProcessorStep().resize, True)
        for module, cls in (
            ("act_crop", "HarenaActCropConfig"),
            ("diffusion_crop", "HarenaDiffusionCropConfig"),
            ("pi05_crop", "HarenaPi05CropConfig"),
        ):
            imported = importlib.import_module(
                f"actoris_harena.policies.{module}.configuration_{module}"
            )
            config = getattr(imported, cls)(device="cpu")
            self.assertIs(config.tactile_resize, True, module)


class WhichFamiliesCanCropWithoutResizingTest(unittest.TestCase):
    """MEASURED on a CPU before any of this reached a queue.

    Cropping without resizing leaves the fingertip cameras SHORTER than the
    overhead one, and the two families answer that differently. The diffusion
    policy validates its configuration and refuses a set of cameras whose shapes
    differ; the action-chunking one builds and runs. That is a property of the
    architectures and not of this rig, so the arm exists for one family only.
    """

    def config_with_mixed_shapes(self, config_cls):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        return config_cls(
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
                f"{OBS_IMAGES}.central": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 240, 320)
                ),
                f"{OBS_IMAGES}.left_arm_left_gripper": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 192, 320)
                ),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))
            },
            device="cpu",
        )

    def test_the_diffusion_family_refuses_cameras_of_different_shapes(self):
        from actoris_harena.policies.diffusion_crop.configuration_diffusion_crop import (
            HarenaDiffusionCropConfig,
        )

        with self.assertRaises(ValueError) as caught:
            self.config_with_mixed_shapes(HarenaDiffusionCropConfig).validate_features()
        self.assertIn("all image shapes to match", str(caught.exception))

    def test_the_action_chunking_family_accepts_them(self):
        from actoris_harena.policies.act_crop.configuration_act_crop import (
            HarenaActCropConfig,
        )

        self.config_with_mixed_shapes(HarenaActCropConfig).validate_features()


class ActionConditionedVideoTest(unittest.TestCase):
    """The FastWAM crop arm can let its video expert read the action chunk."""

    def test_off_by_default_and_reaches_the_video_expert_when_on(self):
        from actoris_harena.policies.fastwam_crop.configuration_fastwam_crop import (
            HarenaFastwamCropConfig,
        )

        self.assertFalse(
            HarenaFastwamCropConfig().video_dit_config["action_conditioned"]
        )
        on = HarenaFastwamCropConfig(action_conditioned_video=True)
        self.assertTrue(on.video_dit_config["action_conditioned"])


class SeparateTactileBackboneTest(unittest.TestCase):
    """ACT with one encoder for the overhead camera and one for the fingertips."""

    def policy(self, separate):
        from lerobot.configs.types import FeatureType, PolicyFeature

        from actoris_harena.policies.act_crop.configuration_act_crop import (
            HarenaActCropConfig,
        )
        from actoris_harena.policies.act_crop.modeling_act_crop import (
            HarenaActCropPolicy,
        )

        config = HarenaActCropConfig(
            device="cpu",
            pretrained_backbone_weights=None,
            tactile_crop=(1.0, 1.0),
            separate_tactile_backbone=separate,
            dim_model=32,
            n_heads=2,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_vae_encoder_layers=1,
            chunk_size=4,
            n_action_steps=4,
        )
        image = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))
        config.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(12,)),
            "observation.images.central": image,
            "observation.images.left_arm_left_gripper": image,
            "observation.images.right_arm_left_gripper": image,
        }
        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))
        }
        return HarenaActCropPolicy(config)

    def batch(self):
        torch.manual_seed(0)
        return {
            "observation.state": torch.randn(2, 12),
            "observation.images.central": torch.rand(2, 3, 32, 32),
            "observation.images.left_arm_left_gripper": torch.rand(2, 3, 32, 32),
            "observation.images.right_arm_left_gripper": torch.rand(2, 3, 32, 32),
            "action": torch.randn(2, 4, 12),
            "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        }

    def test_off_by_default_keeps_one_backbone(self):
        policy = self.policy(separate=False)
        self.assertNotIn("model.backbone.tactile", dict(policy.named_modules()))

    def test_each_camera_reaches_its_own_group_s_encoder(self):
        policy = self.policy(separate=True)
        router = policy.model.backbone
        self.assertEqual(router.is_tactile, [False, True, True])
        calls = []
        router.overhead.register_forward_hook(lambda *a: calls.append("overhead"))
        router.tactile.register_forward_hook(lambda *a: calls.append("tactile"))
        loss, _ = policy.forward(self.batch())
        loss.backward()
        self.assertEqual(calls, ["overhead", "tactile", "tactile"])

        # Both encoders learn, and from different images: their gradients differ.
        def first(module):
            return next(p for p in module.parameters() if p.requires_grad)

        self.assertIsNotNone(first(router.overhead).grad)
        self.assertIsNotNone(first(router.tactile).grad)
        self.assertFalse(
            torch.equal(first(router.overhead).grad, first(router.tactile).grad)
        )

    def test_both_encoders_take_the_backbone_learning_rate(self):
        policy = self.policy(separate=True)
        groups = policy.get_optim_params()
        backbone = {id(p) for p in groups[1]["params"]}
        tactile = {
            id(p) for p in policy.model.backbone.tactile.parameters() if p.requires_grad
        }
        self.assertTrue(tactile and tactile <= backbone)

    def test_a_turn_counter_left_mid_pass_is_reset(self):
        policy = self.policy(separate=True)
        policy.model.backbone._turn = 2
        calls = []
        policy.model.backbone.overhead.register_forward_hook(
            lambda *a: calls.append("o")
        )
        policy.model.backbone.tactile.register_forward_hook(
            lambda *a: calls.append("t")
        )
        policy.forward(self.batch())
        self.assertEqual(calls, ["o", "t", "t"])


if __name__ == "__main__":
    unittest.main()
