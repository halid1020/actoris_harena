"""Building and running the world action model.

Needs the frozen image VAE, which is fetched from the Hub on first use and is
83.7M parameters, so this is the integration tier rather than the unit one. The
pure configuration arithmetic -- including the video/action time alignment -- is
in ``test/unit/test_dreamzero_model.py`` and needs none of it.

Every test here SKIPS rather than fails when the VAE is not on the machine.
"""

from __future__ import annotations

import unittest

import torch
from actoris_harena.policies.dreamzero.configuration_dreamzero import (
    HarenaDreamzeroConfig,
)
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE


def tiny_config(**overrides) -> HarenaDreamzeroConfig:
    settings = dict(
        chunk_size=8,
        n_action_steps=8,
        latent_frames_per_chunk=2,
        n_chunks=3,
        n_context_chunks=1,
        image_size=64,
        patch_size=2,
        dim_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        num_inference_steps=2,
        device="cpu",
    )
    settings.update(overrides)
    config = HarenaDreamzeroConfig(**settings)  # type: ignore[arg-type]
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
        "observation.images.scene": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 48, 64)
        ),
        "observation.images.wrist": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 48, 64)
        ),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))
    }
    return config


def tiny_batch(config: HarenaDreamzeroConfig, batch_size: int = 2) -> dict:
    steps = config.n_chunks * config.latent_frames_per_chunk
    batch = {
        OBS_STATE: torch.randn(batch_size, steps, 12),
        ACTION: torch.randn(batch_size, config.n_chunks * config.chunk_size, 12),
    }
    for key in config.image_features:
        batch[key] = torch.rand(batch_size, steps, 3, 48, 64)
    return batch


def make_policy(config):
    from actoris_harena.policies.dreamzero.modeling_dreamzero import (
        HarenaDreamzeroPolicy,
    )

    return HarenaDreamzeroPolicy(config)


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.config = tiny_config()
        try:
            cls.policy = make_policy(cls.config)
            cls.policy.vae  # noqa: B018 - force the download now, so a failure skips
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"frozen VAE unavailable: {exc}") from exc
        cls.batch = tiny_batch(cls.config)

    def test_patchify_round_trips_exactly(self):
        latents = torch.randn(2, 4, 4, self.config.latent_size, self.config.latent_size)
        tokens = self.policy.patchify(latents)
        self.assertTrue(torch.equal(self.policy.unpatchify(tokens, 4), latents))

    def test_cameras_are_tiled_into_one_square_frame(self):
        """The paper's multi-view recipe: concatenate the views, do not touch the
        backbone."""
        tiled = self.policy.tile_cameras(self.batch)
        steps = self.config.n_chunks * self.config.latent_frames_per_chunk
        self.assertEqual(
            tuple(tiled.shape),
            (2, steps, 3, self.config.image_size, self.config.image_size),
        )

    def test_a_camera_without_a_time_axis_is_named_in_the_error(self):
        batch = dict(self.batch)
        key = list(self.config.image_features)[1]
        batch[key] = batch[key][:, 0]
        with self.assertRaises(ValueError) as caught:
            self.policy.tile_cameras(batch)
        self.assertIn(key, str(caught.exception))

    def test_forward_returns_a_finite_loss_with_both_halves(self):
        loss, parts = self.policy.forward(dict(self.batch))
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("action_loss", parts)
        self.assertIn("video_loss", parts)

    def test_prediction_shapes(self):
        video, actions = self.policy.predict_future(dict(self.batch))
        frames = self.config.predicted_chunks * self.config.latent_frames_per_chunk
        self.assertEqual(
            tuple(video.shape),
            (2, frames, 4, self.config.latent_size, self.config.latent_size),
        )
        self.assertEqual(
            tuple(actions.shape),
            (2, self.config.predicted_chunks, self.config.chunk_size, 12),
        )

    def test_predicted_frames_are_images(self):
        frames = self.policy.predict_future_frames(dict(self.batch))
        self.assertEqual(
            frames.shape[-2:], (self.config.image_size, self.config.image_size)
        )
        self.assertGreaterEqual(frames.min().item(), 0.0)
        self.assertLessEqual(frames.max().item(), 1.0)

    def test_select_action_drains_one_plan(self):
        self.policy.reset()
        first = self.policy.select_action(dict(self.batch))
        self.assertEqual(tuple(first.shape), (2, 12))
        self.assertEqual(len(self.policy._queue), self.config.n_action_steps - 1)

    def test_the_frozen_vae_is_not_written_into_the_checkpoint(self):
        """83.7M identical parameters in every save, otherwise."""
        self.assertTrue(self.policy._vae is not None)
        self.assertFalse(
            any(key.startswith("_vae.") for key in self.policy.state_dict())
        )

    def test_the_vae_is_frozen(self):
        self.assertFalse(any(p.requires_grad for p in self.policy.vae.parameters()))

    def test_optim_params_are_groups_of_tensors(self):
        groups = self.policy.get_optim_params()
        self.assertIsInstance(groups, list)
        self.assertTrue(all(isinstance(p, torch.Tensor) for p in groups[0]["params"]))


class GradientTest(unittest.TestCase):
    """adaLN-Zero starts with closed gates; the point is that they open."""

    def test_every_parameter_receives_gradient_after_one_step(self):
        torch.manual_seed(0)
        config = tiny_config()
        try:
            policy = make_policy(config)
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"frozen VAE unavailable: {exc}") from exc
        batch = tiny_batch(config)
        optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-3)

        # Step 0: the zero-initialised gates block gradient to everything behind
        # them. That is adaLN-Zero working, not a bug -- a freshly built block is
        # the identity so the residual stream reaches the head unmangled.
        loss, _ = policy.forward(dict(batch))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss, _ = policy.forward(dict(batch))
        optimiser.zero_grad()
        loss.backward()
        trainable = [p for p in policy.parameters() if p.requires_grad]
        without = [
            index
            for index, p in enumerate(trainable)
            if p.grad is None or p.grad.abs().sum().item() == 0
        ]
        self.assertEqual(
            without, [], f"{len(without)} parameters still receive no gradient"
        )


if __name__ == "__main__":
    unittest.main()
