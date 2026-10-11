"""The layout every attribution rests on: which part of the vector is which input.

An off-by-one here does not raise. It attributes the tail of one camera to the
head of the next, and the resulting figure looks entirely plausible -- so the
layout is checked against both policies' real shapes, and against the invariant
that it must TILE the vector with no gap and no overlap.

The shapes below are the five-camera ACT checkpoint's own (verified against the
loaded weights: 300 tokens a camera, 1 502 in total) and a diffusion config of
the kind hpc/runs.tsv trains.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_analysis_streams
"""

import unittest

from actoris_harena.analysis.streams import (
    act_layout,
    act_token_total,
    camera_names,
    check_layout,
    diffusion_layout,
    feature_map_hw,
    streams_of,
    tokens_per_camera,
)

CAMERAS = [
    "central",
    "left_arm_left_gripper",
    "left_arm_right_gripper",
    "right_arm_left_gripper",
    "right_arm_right_gripper",
]


class Feature:
    def __init__(self, shape):
        self.shape = shape


class FakeConfig:
    """Only what the layout reads. Deliberately not a real PreTrainedConfig."""

    def __init__(
        self, cameras=CAMERAS, hw=(480, 640), state=12, n_obs_steps=1, dilation=False
    ):
        self.image_features = {
            f"observation.images.{c}": Feature((3, *hw)) for c in cameras
        }
        self.robot_state_feature = Feature((state,)) if state else None
        self.env_state_feature = None
        self.n_obs_steps = n_obs_steps
        self.replace_final_stride_with_dilation = dilation


class TestTheResNetArithmetic(unittest.TestCase):
    def test_the_rig_resolution_gives_three_hundred_tokens(self):
        # MEASURED against the real backbone: (1, 512, 15, 20).
        self.assertEqual(feature_map_hw(480, 640), (15, 20))
        self.assertEqual(tokens_per_camera(FakeConfig()), 300)

    def test_the_diffusion_resize_gives_forty_eight(self):
        # long_vla_real.sh resizes diffusion's cameras to 180x240.
        self.assertEqual(feature_map_hw(180, 240), (6, 8))

    def test_odd_sizes_round_up_the_way_the_network_does(self):
        self.assertEqual(feature_map_hw(1, 1), (1, 1))
        self.assertEqual(feature_map_hw(33, 65), (2, 3))

    def test_dilation_quadruples_the_token_count(self):
        # replace_final_stride_with_dilation drops the last stride, and nothing
        # in the declared image shape says so -- which is why the real count is
        # measured wherever weights are at hand.
        plain = tokens_per_camera(FakeConfig())
        dilated = tokens_per_camera(FakeConfig(dilation=True))
        self.assertEqual(dilated, plain * 4)


class TestTheActLayout(unittest.TestCase):
    def setUp(self):
        self.cfg = FakeConfig()

    def test_camera_order_is_the_configs_and_not_alphabetical(self):
        shuffled = ["wrist", "central", "aaa"]
        self.assertEqual(camera_names(FakeConfig(cameras=shuffled)), shuffled)

    def test_the_state_comes_before_the_cameras_and_after_the_latent(self):
        spans = act_layout(self.cfg, 300)
        self.assertEqual(spans[0].stream.name, "state")
        self.assertEqual(spans[0].ranges, [(1, 2)])  # 0 is the VAE latent
        self.assertEqual(spans[1].stream.name, "central")
        self.assertEqual(spans[1].ranges, [(2, 302)])

    def test_the_real_checkpoint_totals_fifteen_hundred_and_two(self):
        self.assertEqual(act_token_total(self.cfg, 300), 1502)

    def test_the_layout_tiles_the_vector_with_no_gap_and_no_overlap(self):
        spans = act_layout(self.cfg, 300)
        self.assertEqual(
            check_layout(spans, act_token_total(self.cfg, 300), latent=1), []
        )

    def test_a_wrong_token_count_is_caught_rather_than_mislabelling_a_camera(self):
        spans = act_layout(self.cfg, 299)
        self.assertTrue(check_layout(spans, 1502, latent=1))

    def test_every_camera_owns_the_same_width(self):
        widths = {s.width for s in act_layout(self.cfg, 300) if s.stream.is_camera}
        self.assertEqual(widths, {300})


class TestTheDiffusionLayout(unittest.TestCase):
    def test_each_stream_repeats_once_per_observation_step(self):
        cfg = FakeConfig(n_obs_steps=2)
        spans = {s.stream.name: s for s in diffusion_layout(cfg, camera_feature_dim=64)}
        self.assertEqual(len(spans["central"].ranges), 2)
        self.assertEqual(spans["central"].width, 128)

    def test_a_step_block_is_the_state_then_the_cameras_in_order(self):
        cfg = FakeConfig(cameras=["a", "b"], state=12, n_obs_steps=2)
        spans = {s.stream.name: s for s in diffusion_layout(cfg, 64)}
        self.assertEqual(spans["state"].ranges[0], (0, 12))
        self.assertEqual(spans["a"].ranges[0], (12, 76))
        self.assertEqual(spans["b"].ranges[0], (76, 140))
        # ... and the second step picks up straight after the first.
        self.assertEqual(spans["state"].ranges[1], (140, 152))

    def test_the_layout_tiles_the_flattened_vector(self):
        cfg = FakeConfig(n_obs_steps=2)
        spans = diffusion_layout(cfg, 64)
        total = sum(s.width for s in spans)
        self.assertEqual(check_layout(spans, total), [])
        self.assertEqual(total, 2 * (12 + 5 * 64))


class TestWhatCountsAsAStream(unittest.TestCase):
    def test_a_policy_with_no_state_has_no_state_stream(self):
        names = [s.stream.name for s in act_layout(FakeConfig(state=0), 300)]
        self.assertNotIn("state", names)
        self.assertEqual(names[0], "central")

    def test_the_latent_is_not_a_stream(self):
        # It is not an input; attributing anything to it would be a category
        # error, so index 0 belongs to nobody.
        self.assertNotIn("latent", [s.name for s in streams_of(FakeConfig())])

    def test_a_policy_with_no_cameras_still_lays_out_its_state(self):
        spans = act_layout(FakeConfig(cameras=[]), 300)
        self.assertEqual([s.stream.name for s in spans], ["state"])


if __name__ == "__main__":
    unittest.main()


class TestRenamedSlots(unittest.TestCase):
    """A pi0.5 finetune names openpi's slots; the dataset names the rig's cameras."""

    SLOTS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    P = "observation.images."

    def test_every_slot_filled_maps_back_in_slot_order(self):
        from actoris_harena.analysis.inference import source_cameras

        rename = {
            self.P + "central": self.P + "base_0_rgb",
            self.P + "left_arm_left_gripper": self.P + "left_wrist_0_rgb",
            self.P + "right_arm_left_gripper": self.P + "right_wrist_0_rgb",
        }
        self.assertEqual(
            source_cameras(self.SLOTS, rename),
            ["central", "left_arm_left_gripper", "right_arm_left_gripper"],
        )

    def test_a_slot_no_camera_was_renamed_onto_is_not_asked_for(self):
        from actoris_harena.analysis.inference import source_cameras

        rename = {self.P + "central": self.P + "base_0_rgb"}
        self.assertEqual(source_cameras(self.SLOTS, rename), ["central"])

    def test_no_map_keeps_the_policys_own_names(self):
        from actoris_harena.analysis.inference import source_cameras

        self.assertEqual(source_cameras(CAMERAS, {}), CAMERAS)

    def test_streams_are_renamed_by_key_and_a_padded_slot_keeps_its_name(self):
        from actoris_harena.analysis.inference import renamed_streams
        from actoris_harena.analysis.streams import Stream

        streams = [Stream(s, "camera", self.P + s) for s in self.SLOTS]
        rename = {self.P + "central": self.P + "base_0_rgb"}
        names = [s.name for s in renamed_streams(streams, rename)]
        self.assertEqual(names, ["central", "left_wrist_0_rgb", "right_wrist_0_rgb"])
