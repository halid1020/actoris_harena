"""The flow-matching objective, and the policy built on it.

The one invariant worth guarding here is the TIME DIRECTION. There are two
conventions in the literature running opposite ways, and a model trained in one
and sampled in the other trains perfectly and emits noise -- the loss curve looks
fine the whole way. The cheap guard is that integrating the true velocity
recovers the clean sample exactly, because the flow-matching path is straight.
The expensive guard, on a trained model, is in the integration tier.
"""

from __future__ import annotations

import unittest

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from actoris_harena.policies.common import flow


def tiny_config():
    """A model small enough to build in a unit test, shaped like the real one."""
    from actoris_harena.policies.flowmatch.configuration_flowmatch import (
        HarenaFlowmatchConfig,
    )

    config = HarenaFlowmatchConfig(
        chunk_size=4,
        n_action_steps=4,
        dim_model=32,
        n_layers=1,
        n_heads=2,
        dim_feedforward=64,
        crop_shape=(32, 32),
        dropout=0.0,
        num_inference_steps=3,
        device="cpu",
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
        "observation.images.central": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 48, 48)
        ),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))
    }
    return config


def tiny_batch(batch_size: int = 2):
    return {
        OBS_STATE: torch.randn(batch_size, 12),
        "observation.images.central": torch.rand(batch_size, 3, 48, 48),
        ACTION: torch.randn(batch_size, 4, 12),
    }


class FlowObjectiveTest(unittest.TestCase):
    def test_interpolation_endpoints(self):
        clean, noise = torch.randn(3, 5, 12), torch.randn(3, 5, 12)
        at_zero = flow.interpolate(clean, noise, torch.zeros(3))
        at_one = flow.interpolate(clean, noise, torch.ones(3))
        # pi0.5's convention: t=0 is the CLEAN sample, t=1 is pure noise.
        self.assertTrue(torch.allclose(at_zero, clean))
        self.assertTrue(torch.allclose(at_one, noise))

    def test_interpolation_broadcasts_over_any_rank(self):
        # An action chunk is (B, H, A); a video latent is (B, T, C, H, W). One
        # implementation has to serve both or DreamZero needs a second copy.
        for shape in [(2, 5, 12), (2, 4, 3, 8, 8)]:
            with self.subTest(shape=shape):
                clean, noise = torch.randn(*shape), torch.randn(*shape)
                out = flow.interpolate(clean, noise, torch.rand(2))
                self.assertEqual(out.shape, clean.shape)

    def test_velocity_target_is_constant_in_time(self):
        """Why the objective is a regression and not a schedule to match."""
        clean, noise = torch.randn(2, 5, 12), torch.randn(2, 5, 12)
        self.assertTrue(
            torch.allclose(flow.velocity_target(clean, noise), noise - clean)
        )

    def test_integrating_the_true_velocity_recovers_the_clean_sample(self):
        """The direction guard. The path is straight, so ONE step is exact."""
        clean, noise = torch.randn(4, 6, 12), torch.randn(4, 6, 12)
        truth = flow.velocity_target(clean, noise)
        for steps in (1, 2, 10):
            with self.subTest(steps=steps):
                got = flow.integrate(lambda x, t: truth, noise, steps=steps)
                self.assertLess((got - clean).abs().max().item(), 1e-5)

    def test_integrating_the_wrong_way_does_not(self):
        """Guards the guard: if the assertion above passed either way it is vacuous."""
        clean, noise = torch.randn(4, 6, 12), torch.randn(4, 6, 12)
        truth = flow.velocity_target(clean, noise)
        x = noise.clone()
        for step in range(10):  # upwards, i.e. DreamZero's direction
            x = x + (1.0 / 10) * truth
        self.assertGreater((x - clean).abs().mean().item(), 1.0)

    def test_sampled_time_stays_inside_the_open_interval(self):
        time = flow.sample_time(4096, "cpu")
        self.assertGreater(time.min().item(), 0.0)
        self.assertLessEqual(time.max().item(), 1.0)
        # Beta(1.5, 1) has mean 0.6: training leans towards the noisy end.
        self.assertAlmostEqual(time.mean().item(), 0.6, delta=0.05)

    def test_sampled_time_is_reproducible_with_a_generator(self):
        generator = torch.Generator().manual_seed(7)
        first = flow.sample_time(64, "cpu", generator=generator)
        generator = torch.Generator().manual_seed(7)
        self.assertTrue(
            torch.equal(first, flow.sample_time(64, "cpu", generator=generator))
        )

    def test_integrate_refuses_zero_steps(self):
        with self.assertRaises(ValueError):
            flow.integrate(lambda x, t: x, torch.randn(1, 2, 3), steps=0)


class FlowmatchPolicyTest(unittest.TestCase):
    def setUp(self):
        from actoris_harena.policies.flowmatch.modeling_flowmatch import (
            HarenaFlowmatchPolicy,
        )

        torch.manual_seed(0)
        self.config = tiny_config()
        self.policy = HarenaFlowmatchPolicy(self.config)

    def test_forward_returns_a_scalar_loss(self):
        loss, parts = self.policy.forward(tiny_batch())
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss", parts)

    def test_predicted_chunk_has_the_configured_shape(self):
        chunk = self.policy.predict_action_chunk(tiny_batch(3))
        self.assertEqual(tuple(chunk.shape), (3, self.config.chunk_size, 12))

    def test_select_action_drains_the_chunk_before_replanning(self):
        batch = tiny_batch(1)
        self.policy.reset()
        first = self.policy.select_action(dict(batch))
        self.assertEqual(tuple(first.shape), (1, 12))
        # n_action_steps actions come from ONE plan; the queue empties, not refills.
        self.assertEqual(len(self.policy._queue), self.config.n_action_steps - 1)
        for _ in range(self.config.n_action_steps - 1):
            self.policy.select_action(dict(batch))
        self.assertEqual(len(self.policy._queue), 0)

    def test_reset_clears_the_queue(self):
        batch = tiny_batch(1)
        self.policy.select_action(dict(batch))
        self.assertGreater(len(self.policy._queue), 0)
        self.policy.reset()
        self.assertEqual(len(self.policy._queue), 0)

    def test_padded_actions_are_excluded_from_the_loss(self):
        """A chunk running off the end of an episode is padded; training on the
        padding teaches the policy to reproduce it."""
        torch.manual_seed(0)
        batch = tiny_batch(2)
        batch["action_is_pad"] = torch.zeros(2, 4, dtype=torch.bool)
        unpadded, _ = self.policy.forward(dict(batch))

        # Same batch, but the last two steps are declared padding and given
        # absurd values. A loss that ignores them cannot move much.
        batch2 = {k: v.clone() for k, v in batch.items()}
        batch2["action_is_pad"][:, 2:] = True
        batch2[ACTION][:, 2:] = 1e3
        torch.manual_seed(0)
        padded, _ = self.policy.forward(dict(batch2))
        self.assertLess(abs(unpadded.item() - padded.item()) / unpadded.item(), 0.6)

    def test_camera_order_follows_the_config_not_sorting(self):
        """Every layout in common/analysis assumes config order; sorting would
        silently re-map which camera is which."""
        self.assertEqual(self.policy.camera_keys, list(self.config.image_features))

    def test_optim_params_are_groups_of_tensors(self):
        """A dict here makes torch iterate the KEYS and reject a str as a param."""
        groups = self.policy.get_optim_params()
        self.assertIsInstance(groups, list)
        for group in groups:
            self.assertIn("params", group)
            for param in group["params"]:
                self.assertIsInstance(param, torch.Tensor)
        # The vision trunk trains slower than the expert on top of it.
        self.assertTrue(any("lr" in group for group in groups))


if __name__ == "__main__":
    unittest.main()
