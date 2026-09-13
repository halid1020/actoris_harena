"""The world action model's configuration: derived sizes and the refusals.

Pure arithmetic over a config, so it is fast and needs neither weights nor a
network. The load-bearing one is that a chunk's video and its actions describe
the SAME interval of time -- the alignment no shape check catches, and whose
absence would look like a merely mediocre model rather than a bug.

Building the model needs the frozen VAE, which is fetched from the Hub, so those
tests live in ``test/integration/test_dreamzero_model.py``.
"""

from __future__ import annotations

import unittest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from actoris_harena.policies.dreamzero.configuration_dreamzero import (
    HarenaDreamzeroConfig,
)


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


class ConfigTest(unittest.TestCase):
    """Derived sizes, and the refusals that stop a misaligned run."""

    def test_video_and_actions_of_a_chunk_span_the_same_time(self):
        """THE alignment. The paper gets it from a temporal VAE folding four raw
        frames into one latent; a per-frame VAE has to subsample instead. Get it
        wrong and the objective aligns a video of one moment with actions of
        another -- which trains, and is simply wrong."""
        config = tiny_config()
        frames = config.observation_delta_indices
        actions = config.action_delta_indices
        per_chunk = config.latent_frames_per_chunk
        for chunk in range(config.n_chunks):
            with self.subTest(chunk=chunk):
                chunk_frames = frames[chunk * per_chunk : (chunk + 1) * per_chunk]
                chunk_actions = actions[
                    chunk * config.chunk_size : (chunk + 1) * config.chunk_size
                ]
                # The chunk's frames start where its actions start, and the last
                # frame falls inside the action window rather than past it.
                self.assertEqual(chunk_frames[0], chunk_actions[0])
                self.assertLess(chunk_frames[-1], chunk_actions[-1] + 1)
                self.assertGreaterEqual(chunk_frames[-1], chunk_actions[0])

    def test_context_chunks_sit_in_the_past(self):
        config = tiny_config()
        first_predicted = config.n_context_chunks * config.chunk_size
        self.assertLess(config.action_delta_indices[0], 0)
        self.assertEqual(config.action_delta_indices[first_predicted], 0)

    def test_frame_stride_divides_the_chunk(self):
        config = tiny_config()
        self.assertEqual(
            config.frame_stride * config.latent_frames_per_chunk, config.chunk_size
        )

    def test_a_chunk_that_cannot_be_evenly_subsampled_is_refused(self):
        with self.assertRaises(ValueError):
            tiny_config(chunk_size=7, n_action_steps=7, latent_frames_per_chunk=2)

    def test_an_image_size_the_vae_cannot_take_is_refused(self):
        with self.assertRaises(ValueError):
            tiny_config(image_size=60)  # not divisible by 8 * patch_size

    def test_a_config_that_predicts_nothing_is_refused(self):
        with self.assertRaises(ValueError):
            tiny_config(n_chunks=1, n_context_chunks=1)

    def test_derived_token_counts(self):
        config = tiny_config()
        self.assertEqual(config.latent_size, 8)
        self.assertEqual(config.patches_per_frame, 16)
        self.assertEqual(config.video_tokens_per_chunk, 32)
        self.assertEqual(config.predicted_chunks, 2)


if __name__ == "__main__":
    unittest.main()
