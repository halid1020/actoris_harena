"""The three things that made this package ACT-only, each measured.

`act` has an attribution study and `diffusion` and `pi05` do not, and the reason
was never that nobody ran the command. Three parts of the package assumed ACT's
shape, and one of them produced numbers rather than an error:

* **occlusion compared unpinned plans.** The gradient path was pinned when the
  sampler was found to be 83 times the signal; `perturb.py` was not, so a
  diffusion occlusion table measured the sampler.
* **Grad-CAM hooked the wrong object for diffusion.** `rgb_encoder` is an
  `nn.ModuleList` under this rig's config and is never called itself, so the
  hook collected nothing; and the encoder returns a pooled vector anyway, with
  no spatial dimensions left to draw.
* **pi0.5 had a second `@torch.no_grad()`.** Unwrapping only
  `predict_action_chunk` returns a tensor with no graph, and autograd then
  complains about the input rather than about the decorator.

Run:  PYTHONPATH=.:src python -m unittest test.unit.test_analysis_multipolicy
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch

from actoris_harena.action_layout import set_gripper_columns
from actoris_harena.analysis import gradients as grads
from actoris_harena.analysis import paths, perturb
from actoris_harena.analysis.streams import STATE_KEY, Stream

# The bench these fixtures describe: five body joints then a gripper, per arm,
# so the two gripper channels of the 12-D action are 5 and 11. Declared here
# rather than defaulted in the package -- a stale default would make `perturb`
# occlude a joint on a seven-wide action and report a number nobody could tell
# was wrong.
set_gripper_columns((5, 11))


class DrawingInference:
    """A policy whose plan depends on the input AND on the global RNG.

    Both halves matter. Without the draw, a pinning test passes whether or not
    anything is pinned; without the dependence on the input, occlusion has
    nothing to find and every share is zero.
    """

    def __init__(self, jitter: float = 1.0):
        self.torch = torch
        self.jitter = jitter
        self.streams = [
            Stream("state", "state", STATE_KEY),
            Stream("central", "camera", "observation.images.central"),
            Stream("tip", "camera", "observation.images.tip"),
        ]

    def batch(self, state, images):
        return {"state": np.asarray(state), "images": {k: v for k, v in images.items()}}

    def chunk_from(self, batch):
        # Squares, not sums: the `mean` baseline preserves a frame's sum exactly,
        # so a linear stand-in would report every effect as zero and the pinning
        # test would pass against a policy that ignores its cameras.
        signal = float((batch["state"] ** 2).sum()) + sum(
            float((v**2).sum()) for v in batch["images"].values()
        )
        return (signal + self.jitter * torch.randn(4, 2).numpy()).astype(np.float32)


def varied_observation():
    """Frames that are not constant, so the `mean` baseline actually changes them."""
    state = np.arange(3, dtype=np.float32)
    images = {
        "central": np.arange(48, dtype=np.float32).reshape(4, 4, 3) / 48.0,
        "tip": np.arange(48, dtype=np.float32).reshape(4, 4, 3) / 96.0,
    }
    return state, images


def observation():
    state = np.ones(3, dtype=np.float32)
    images = {
        "central": np.full((4, 4, 3), 0.5, dtype=np.float32),
        "tip": np.full((4, 4, 3), 0.25, dtype=np.float32),
    }
    return state, images


class OcclusionIsPinnedTest(unittest.TestCase):
    """Two runs of the same occlusion must agree exactly, sampler or no sampler."""

    def test_two_runs_agree(self):
        inference = DrawingInference(jitter=1.0)
        state, images = observation()
        first = perturb.occlusion(inference, state, images)
        second = perturb.occlusion(inference, state, images)
        for name in ("state", "central", "tip"):
            self.assertAlmostEqual(
                first["streams"][name]["l2"],
                second["streams"][name]["l2"],
                places=6,
                msg=f"{name} moved between two identical occlusion runs",
            )

    def test_it_would_not_agree_unpinned(self):
        # The control. Without this the test above passes on a policy that never
        # draws, and would have passed before the fix as well.
        inference = DrawingInference(jitter=1.0)
        state, images = observation()
        first = inference.chunk_from(inference.batch(state, images))
        second = inference.chunk_from(inference.batch(state, images))
        self.assertFalse(np.allclose(first, second))

    def test_the_answer_does_not_depend_on_which_seed(self):
        """The point of the pin, stated as the property it buys.

        Both plans in a comparison are drawn under the SAME seed, so the sampler
        cancels out of the difference entirely -- and the effect that survives is
        the input's, whichever seed was chosen. A number that moved with the seed
        would still be partly the sampler.
        """
        inference = DrawingInference(jitter=1.0)
        state, images = varied_observation()
        one = perturb.occlusion(inference, state, images, seed=0)
        two = perturb.occlusion(inference, state, images, seed=7)
        for name in ("state", "central", "tip"):
            self.assertAlmostEqual(
                one["streams"][name]["l2"],
                two["streams"][name]["l2"],
                places=5,
                msg=f"{name} depends on the seed, so it is still partly the sampler",
            )
        self.assertGreater(one["streams"]["central"]["l2"], 0.0)

    def test_the_seed_is_recorded_beside_the_baseline(self):
        # A figure names the baseline it used; it must be able to name the seed
        # too, or a reader cannot tell a pinned table from an unpinned one.
        inference = DrawingInference(jitter=0.0)
        result = perturb.occlusion(inference, *observation())
        self.assertIn("seed", result)

    def test_the_effect_still_tracks_the_input(self):
        # Pinning must not flatten the measurement it exists to make honest.
        inference = DrawingInference(jitter=0.0)
        state, images = observation()
        result = perturb.occlusion(inference, state, images, baseline="zeros")
        self.assertGreater(result["streams"]["central"]["l2"], 0.0)


# -- Grad-CAM's trunk, on all three architectures ------------------------------


class Trunk(torch.nn.Module):
    """Stands in for a ResNet: a spatial map in, a spatial map out."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Encoder(torch.nn.Module):
    """A DiffusionRgbEncoder in the one respect that matters: it pools."""

    def __init__(self):
        super().__init__()
        self.backbone = Trunk()

    def forward(self, x):
        return self.backbone(x).flatten(start_dim=1)


class TrunkInference:
    def __init__(self, policy, cameras, n_obs_steps=2):
        self.torch = torch
        self.policy = policy
        self.type = "stand-in"
        self.n_obs_steps = n_obs_steps
        self._cameras = cameras

    def image_keys(self):
        return [f"observation.images.{c}" for c in self._cameras]


class CamTrunkTest(unittest.TestCase):
    def test_act_is_one_module_called_once_per_camera(self):
        policy = torch.nn.Module()
        policy.model = torch.nn.Module()
        policy.model.backbone = Trunk()
        trunks, mode = grads.cam_trunks(TrunkInference(policy, ["a", "b"]))
        self.assertEqual(mode, "per_call")
        self.assertEqual(len(trunks), 1)

    def test_a_module_list_gives_one_trunk_per_camera(self):
        # The case this rig actually trains, and the one that used to collect
        # nothing at all: the list is never called, only its members are.
        policy = torch.nn.Module()
        policy.diffusion = torch.nn.Module()
        policy.diffusion.rgb_encoder = torch.nn.ModuleList([Encoder(), Encoder()])
        trunks, mode = grads.cam_trunks(TrunkInference(policy, ["a", "b"]))
        self.assertEqual(mode, "per_module")
        self.assertEqual(len(trunks), 2)

    def test_a_shared_encoder_is_one_interleaved_call(self):
        policy = torch.nn.Module()
        policy.diffusion = torch.nn.Module()
        policy.diffusion.rgb_encoder = Encoder()
        trunks, mode = grads.cam_trunks(TrunkInference(policy, ["a", "b"]))
        self.assertEqual(mode, "interleaved")
        self.assertEqual(len(trunks), 1)

    def test_the_hooked_object_is_the_trunk_and_not_the_encoder(self):
        # Hooking the encoder gives a pooled (B, D) vector: no map to draw.
        encoder = Encoder()
        policy = torch.nn.Module()
        policy.diffusion = torch.nn.Module()
        policy.diffusion.rgb_encoder = torch.nn.ModuleList([encoder])
        trunks, _ = grads.cam_trunks(TrunkInference(policy, ["a"]))
        self.assertIs(trunks[0], encoder.backbone)

    def test_a_token_model_says_why_rather_than_how(self):
        policy = torch.nn.Module()
        with self.assertRaises(grads.NoFeatureMap) as caught:
            grads.cam_trunks(TrunkInference(policy, ["a"]))
        message = str(caught.exception)
        self.assertIn("integrated gradients", message)

    def test_running_out_of_memory_is_not_a_missing_feature_map(self):
        """The one confusion this exception type exists to prevent.

        ``torch.cuda.OutOfMemoryError`` subclasses ``RuntimeError``, so a caller
        that catches the broad type records a full GPU as "Grad-CAM
        unavailable" -- and a reader takes that to mean the method does not
        apply to the policy. MEASURED: a diffusion pass sharing a card with a
        pi0.5 LoRA run wrote numbers, no figures, and no complaint.
        """
        self.assertTrue(issubclass(torch.cuda.OutOfMemoryError, RuntimeError))
        self.assertFalse(issubclass(torch.cuda.OutOfMemoryError, grads.NoFeatureMap))
        self.assertTrue(issubclass(grads.NoFeatureMap, RuntimeError))


class InterleavedSplitTest(unittest.TestCase):
    """The shared encoder sees `b s n -> (b s n)`; the split must undo exactly that."""

    def test_each_camera_gets_its_own_frame_back(self):
        b, s, n = 1, 2, 3
        # Mark every (step, camera) so a wrong split cannot look right.
        marks = torch.arange(b * s * n, dtype=torch.float32)
        whole = marks.view(-1, 1, 1, 1).repeat(1, 2, 4, 4)
        inference = TrunkInference(torch.nn.Module(), ["a", "b", "c"], n_obs_steps=s)
        pairs = grads._cam_activations(inference, [whole], n, "interleaved")
        self.assertEqual(len(pairs), n)
        # b=0, s=0 means flat indices 0, 1, 2 -- one per camera, in order.
        self.assertEqual([float(a[0, 0, 0]) for a, _ in pairs], [0.0, 1.0, 2.0])

    def test_a_batch_that_does_not_divide_is_refused_not_guessed(self):
        whole = torch.zeros(5, 2, 4, 4)
        inference = TrunkInference(torch.nn.Module(), ["a", "b", "c"], n_obs_steps=2)
        self.assertEqual(
            grads._cam_activations(inference, [whole], 3, "interleaved"), []
        )

    def test_per_module_drops_the_window_axis(self):
        act = torch.zeros(2, 2, 4, 4)
        inference = TrunkInference(torch.nn.Module(), ["a"], n_obs_steps=2)
        pairs = grads._cam_activations(inference, [act], 1, "per_module")
        self.assertEqual(tuple(pairs[0][0].shape), (2, 4, 4))


# -- pi0.5's second decorator --------------------------------------------------


class SamplerUnwrapTest(unittest.TestCase):
    """The class must be left exactly as it was found."""

    def make(self):
        from actoris_harena.analysis.inference import Inference

        class Model(torch.nn.Module):
            @torch.no_grad()
            def sample_actions(self, x):
                return x * 2

        inference = Inference.__new__(Inference)
        inference.policy = torch.nn.Module()
        inference.policy.model = Model()
        return inference

    def test_gradients_flow_only_inside_the_context(self):
        inference = self.make()
        x = torch.ones(2, requires_grad=True)
        self.assertIsNone(inference.policy.model.sample_actions(x).grad_fn)
        with inference._grad_through_sampler():
            self.assertIsNotNone(inference.policy.model.sample_actions(x).grad_fn)
        self.assertIsNone(inference.policy.model.sample_actions(x).grad_fn)

    def test_nothing_is_left_bound_on_the_instance(self):
        inference = self.make()
        with inference._grad_through_sampler():
            pass
        self.assertNotIn("sample_actions", vars(inference.policy.model))

    def test_it_restores_even_when_the_body_raises(self):
        inference = self.make()
        with self.assertRaises(ValueError):
            with inference._grad_through_sampler():
                raise ValueError("boom")
        self.assertNotIn("sample_actions", vars(inference.policy.model))

    def test_a_policy_with_no_such_sampler_is_left_alone(self):
        from actoris_harena.analysis.inference import Inference

        inference = Inference.__new__(Inference)
        inference.policy = torch.nn.Module()
        with inference._grad_through_sampler():
            pass  # must not raise: ACT and diffusion take this path


# -- where an analysis is written ---------------------------------------------


class AnalysisPathTest(unittest.TestCase):
    def test_the_day_and_the_content_are_both_in_the_path(self):
        out = paths.analysis_dir("diffusion-all", date="2026-09-07")
        self.assertEqual(out.parent.name, "2026-09-07")
        self.assertEqual(out.name, "diffusion-all")
        self.assertEqual(out.parent.parent.name, "analysis")

    def test_it_honours_the_output_directory_variable(self):
        # The analysis tools were the one part of the repo that ignored it, so
        # running one from anywhere but the repo root scattered results.
        original = os.environ.get("SO101_OUTPUT_DIR")
        os.environ["SO101_OUTPUT_DIR"] = "/tmp/somewhere"
        try:
            out = paths.analysis_dir("act-all", date="2026-01-02")
            self.assertEqual(out, Path("/tmp/somewhere/analysis/2026-01-02/act-all"))
        finally:
            if original is None:
                del os.environ["SO101_OUTPUT_DIR"]
            else:
                os.environ["SO101_OUTPUT_DIR"] = original

    def test_a_camera_set_joins_the_way_a_run_directory_joins_it(self):
        self.assertEqual(
            paths.content_name("pi05", "central+left_arm_left_gripper"),
            "pi05-central+left_arm_left_gripper",
        )

    def test_a_date_that_is_not_a_day_is_refused(self):
        for bad in ("today", "2026-9-7", "", "2026-09-07/../.."):
            with self.assertRaises(ValueError, msg=bad):
                paths.analysis_dir("act-all", date=bad)

    def test_a_name_that_would_escape_its_directory_is_refused(self):
        for bad in ("../elsewhere", "with space", "", "/absolute"):
            with self.assertRaises(ValueError, msg=bad):
                paths.analysis_dir(bad)

    def test_today_is_a_day(self):
        paths.check_date(paths.today())


# -- a deck that does not overstate what it measured ---------------------------


def deck(policy: str, cameras, per_stream, attention=None) -> dict:
    """A minimal attribution payload: one episode, one frame, occlusion shares."""
    frame = {"occlusion": {"streams": {k: {"share": v} for k, v in per_stream.items()}}}
    if attention is not None:
        frame["attention"] = {"mean": attention, "deviation": attention}
    return {
        "policy": policy,
        "cameras": list(cameras),
        "streams": ["state", *cameras],
        "baseline": "mean",
        "direction": "leave_one_out",
        "source": "stand-in",
        "episodes": {"0": {"frames": [frame]}},
    }


class SpreadTest(unittest.TestCase):
    """Rank agreement alone let a near-uniform method look like a good one."""

    def test_it_reproduces_the_measured_figures(self):
        from actoris_harena.analysis import slides

        # MEASURED on the finished ACT checkpoint: occlusion 42x, attention 1.14x.
        occlusion = {"central": 0.571, "tip": 0.0136}
        attention = {"central": 0.219, "tip": 0.192}
        self.assertAlmostEqual(slides.spread(occlusion), 42.0, delta=1.0)
        self.assertAlmostEqual(slides.spread(attention), 1.14, delta=0.02)

    def test_a_signed_quantity_has_no_ratio(self):
        # The `vs uniform` row goes negative by construction. Dropping the
        # negatives gave it a flattering 1.0x from the one camera left standing.
        from actoris_harena.analysis import slides

        self.assertEqual(slides.spread({"a": 0.02, "b": -0.008}), 0.0)
        self.assertEqual(slides.spread({}), 0.0)

    def test_the_table_carries_the_column(self):
        from actoris_harena.analysis import slides

        headers, _ = slides.method_table(
            deck("act", ["central", "tip"], {"central": 0.9, "tip": 0.1})
        )
        self.assertIn("spread", headers)


class ComparingDecksTest(unittest.TestCase):
    def test_a_stream_one_policy_lacks_is_not_zero(self):
        """`n/a` and `0 %` are different claims and must not be confused.

        pi0.5 has three image slots, so it was trained on two fingertips where
        ACT had four. Reporting the missing two as 0 % would say the policy was
        shown them and ignored them.
        """
        from actoris_harena.analysis import slides

        five = deck(
            "act",
            ["central", "a", "b"],
            {"state": 0.4, "central": 0.5, "a": 0.05, "b": 0.05},
        )
        three = deck("pi05", ["central", "a"], {"state": 0.5, "central": 0.4, "a": 0.1})
        headers, rows = slides.compare_table([five, three])
        self.assertEqual(headers, ["stream", "act", "pi05"])
        by_name = {r[0]: r for r in rows}
        self.assertEqual(by_name["b"][2], "n/a")
        self.assertNotEqual(by_name["b"][1], "n/a")

    def test_differing_camera_sets_are_called_out(self):
        from actoris_harena.analysis import slides

        notes = " ".join(
            slides.compare_caveats(
                [
                    deck("act", ["central", "a", "b"], {"central": 1.0}),
                    deck("pi05", ["central", "a"], {"central": 1.0}),
                ]
            )
        )
        self.assertIn("NOT a like-for-like", notes)
        self.assertIn("three pretrained image slots", notes)

    def test_the_normalisation_warning_is_always_present(self):
        # Even when the decks match, a share is normalised within its own deck.
        from actoris_harena.analysis import slides

        one = deck("act", ["central"], {"central": 1.0})
        notes = " ".join(slides.compare_caveats([one, one]))
        self.assertIn("normalised within a deck", notes)

    def test_a_different_baseline_is_called_out(self):
        from actoris_harena.analysis import slides

        one = deck("act", ["central"], {"central": 1.0})
        other = dict(deck("diffusion", ["central"], {"central": 1.0}), baseline="zeros")
        notes = " ".join(slides.compare_caveats([one, other]))
        self.assertIn("baseline is part of the result", notes)
