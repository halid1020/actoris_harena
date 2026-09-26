"""Pinning a stochastic sampler, and the layout of a third architecture.

The measured stakes: on a real diffusion checkpoint, two plans from ONE
unchanged observation differ by 32.15 in commanded units, while the occlusion
effect being measured is 0.39. Unpinned, an attribution study measures the
sampler 83 times over and draws plausible bars while doing it.

These tests use a stand-in rather than a checkpoint -- the behaviour under test
is the pinning and the index arithmetic, neither of which needs weights, and the
real checkpoint is exercised in the integration tier.
"""

from __future__ import annotations

import unittest

import torch

from actoris_harena.analysis import diffusion as sampler
from actoris_harena.analysis.streams import check_layout, token_layout


class FakeInference:
    """Just enough of Inference: a torch handle and a sampler that draws."""

    def __init__(self):
        self.torch = torch
        self.device = "cpu"
        self.action_dim = 4

        class Config:
            horizon = 6
            chunk_size = 6

        self.cfg = Config()

        class Policy:
            def predict_action_chunk(self, batch, noise=None):  # noqa: D102, ARG002
                return torch.randn(1, 6, 4)

        self.policy = Policy()

    def chunk_from(self, batch):  # noqa: ARG002
        # Draws from the GLOBAL rng, exactly as a DDPM scheduler's step() does.
        return torch.randn(6, 4).numpy()


class PinningTest(unittest.TestCase):
    def setUp(self):
        self.inference = FakeInference()

    def test_unpinned_plans_differ(self):
        """The problem, stated as a test so the fix cannot be vacuous."""
        first = self.inference.chunk_from({})
        second = self.inference.chunk_from({})
        self.assertFalse((first == second).all())

    def test_pinned_plans_are_identical(self):
        first = sampler.plan(self.inference, {})
        second = sampler.plan(self.inference, {})
        self.assertTrue((first == second).all())

    def test_a_different_seed_gives_a_different_plan(self):
        """Guards the guard: pinning to a constant would also pass the test above."""
        first = sampler.plan(self.inference, {}, seed=0)
        second = sampler.plan(self.inference, {}, seed=1)
        self.assertFalse((first == second).all())

    def test_the_global_rng_state_is_restored(self):
        """Seeding globally is only acceptable because nothing else observes it."""
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()
        sampler.plan(self.inference, {})
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_the_state_is_restored_even_when_the_body_raises(self):
        torch.manual_seed(99)
        before = torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with sampler.pinned(self.inference):
                raise RuntimeError("boom")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_pinning_does_not_freeze_the_stream_inside_the_block(self):
        """Two draws inside one pinned block must still differ -- pinning fixes
        where the stream STARTS, it does not make every draw the same."""
        with sampler.pinned(self.inference):
            first, second = torch.randn(4), torch.randn(4)
        self.assertFalse(torch.equal(first, second))


class OwnGeneratorTest(unittest.TestCase):
    """A policy that seeds its own generator from its config, as FastWAM does."""

    class OwnSeeded:
        def __init__(self):
            self.torch = torch

            class Config:
                inference_seed = 42

            class Policy:
                config = Config()

            self.policy = Policy()

        def chunk_from(self, batch):  # noqa: ARG002
            seed = self.policy.config.inference_seed
            generator = torch.Generator().manual_seed(seed)
            return torch.randn(6, 4, generator=generator).numpy()

    def test_different_seeds_reach_a_policy_that_ignores_the_global_rng(self):
        # The failure: three seeds, one answer to every digit.
        inference = self.OwnSeeded()
        a = sampler.plan(inference, {}, seed=0)
        b = sampler.plan(inference, {}, seed=1)
        self.assertFalse((a == b).all())

    def test_one_seed_still_gives_one_answer(self):
        inference = self.OwnSeeded()
        self.assertTrue(
            (
                sampler.plan(inference, {}, seed=3)
                == sampler.plan(inference, {}, seed=3)
            ).all()
        )

    def test_the_configs_own_seed_is_restored(self):
        inference = self.OwnSeeded()
        sampler.plan(inference, {}, seed=7)
        self.assertEqual(inference.policy.config.inference_seed, 42)


class StochasticProbeTest(unittest.TestCase):
    def test_a_drawing_policy_is_detected_from_behaviour(self):
        self.assertTrue(sampler.is_stochastic(FakeInference(), batch={}))

    def test_a_deterministic_policy_is_not(self):
        inference = FakeInference()
        inference.chunk_from = lambda batch: torch.zeros(6, 4).numpy()  # noqa: ARG005
        self.assertFalse(sampler.is_stochastic(inference, batch={}))

    def test_without_a_batch_it_asks_the_signature(self):
        """A policy that accepts a starting noise is one that starts from noise."""
        self.assertTrue(sampler.is_stochastic(FakeInference()))


class FixedNoiseTest(unittest.TestCase):
    def test_it_is_shaped_from_horizon_not_from_the_executed_prefix(self):
        """Diffusion plans `horizon` and executes `n_action_steps` of it; the
        shorter number is a shape error deep inside the scheduler."""
        noise = sampler.fixed_noise(FakeInference())
        self.assertEqual(tuple(noise.shape), (1, 6, 4))

    def test_it_is_the_same_every_call(self):
        inference = FakeInference()
        self.assertTrue(
            torch.equal(sampler.fixed_noise(inference), sampler.fixed_noise(inference))
        )


class SamplerSpreadTest(unittest.TestCase):
    def test_it_measures_what_pinning_removes(self):
        """Deliberately unpinned: it is the floor an effect must clear."""
        spread = sampler.sampler_spread(FakeInference(), {}, trials=8)
        self.assertGreater(spread, 0.0)


class TokenLayoutTest(unittest.TestCase):
    """pi0.5's prefix: image patches per camera, then one state token."""

    def setUp(self):
        from lerobot.configs.types import FeatureType, PolicyFeature

        class Config:
            input_features = {
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(12,)),
                "observation.images.a": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 8, 8)
                ),
                "observation.images.b": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 8, 8)
                ),
            }
            output_features: dict = {}
            robot_state_feature = input_features["observation.state"]
            env_state_feature = None
            image_features = {
                k: v
                for k, v in input_features.items()
                if k.startswith("observation.images.")
            }

        self.config = Config()

    def test_cameras_own_contiguous_equal_runs_then_the_state(self):
        spans = token_layout(self.config, tokens_per_camera=16)
        by_name = {span.stream.name: span for span in spans}
        self.assertEqual(by_name["a"].ranges, [(0, 16)])
        self.assertEqual(by_name["b"].ranges, [(16, 32)])
        # One token for the whole state, however wide the vector.
        self.assertEqual(by_name["state"].ranges, [(32, 33)])

    def test_the_spans_tile_the_prefix(self):
        spans = token_layout(self.config, tokens_per_camera=16)
        total = sum(span.width for span in spans)
        self.assertEqual(check_layout(spans, total), [])

    def test_camera_order_follows_the_config(self):
        """Sorting would silently re-map which camera is which."""
        spans = token_layout(self.config, tokens_per_camera=4)
        cameras = [s.stream.name for s in spans if s.stream.is_camera]
        self.assertEqual(cameras, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
