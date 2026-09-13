"""The flow-matching policy can actually learn, and its sampler agrees with it.

The unit tier checks the objective's algebra on the TRUE velocity. This checks
the thing that algebra cannot: that the velocity a trained network predicts,
integrated the way the sampler integrates it, lands on the actions it was
trained on. A time-direction error passes every unit test and fails here by an
order of magnitude, which is exactly what the control below measures.

MEASURED once at 1500 steps on one batch: 9.1% of target scale at 5 inference
steps against 359.9% integrating the other way. The thresholds here are loose
enough to survive a different seed and still be nowhere near a direction error.
"""

from __future__ import annotations

import unittest
from typing import Any

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

TRAIN_STEPS = 600


def _policy_and_batch():
    from actoris_harena.policies.flowmatch.configuration_flowmatch import (
        HarenaFlowmatchConfig,
    )
    from actoris_harena.policies.flowmatch.modeling_flowmatch import (
        HarenaFlowmatchPolicy,
    )

    torch.manual_seed(0)
    config = HarenaFlowmatchConfig(
        chunk_size=6,
        n_action_steps=6,
        dim_model=96,
        n_layers=2,
        n_heads=4,
        dim_feedforward=192,
        crop_shape=(48, 48),
        dropout=0.0,
        num_inference_steps=10,
        device="cpu",
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
        "observation.images.central": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 64, 64)
        ),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))
    }
    policy = HarenaFlowmatchPolicy(config)
    batch = {
        OBS_STATE: torch.randn(2, 12),
        "observation.images.central": torch.rand(2, 3, 64, 64),
        ACTION: torch.randn(2, 6, 12),
    }
    return policy, batch


class FlowmatchLearnsTest(unittest.TestCase):
    # Declared here because mypy does not infer attributes assigned in
    # setUpClass, and the training run is far too slow to repeat per test.
    policy: Any
    batch: dict
    first_loss: float
    final_loss: float
    scale: float

    @classmethod
    def setUpClass(cls):
        cls.policy, cls.batch = _policy_and_batch()
        optimiser = torch.optim.AdamW(cls.policy.parameters(), lr=1e-3)
        cls.policy.train()
        cls.first_loss = None
        for _ in range(TRAIN_STEPS):
            loss, _ = cls.policy.forward(dict(cls.batch))
            if cls.first_loss is None:
                cls.first_loss = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        cls.final_loss = loss.item()
        cls.policy.eval()
        cls.scale = cls.batch[ACTION].abs().mean().item()

    def _error(self, steps: int) -> float:
        self.policy.config.num_inference_steps = steps
        torch.manual_seed(1)
        chunk = self.policy.predict_action_chunk(dict(self.batch))
        return (chunk - self.batch[ACTION]).abs().mean().item() / self.scale

    def test_the_loss_falls(self):
        self.assertLess(self.final_loss, self.first_loss * 0.5)

    def test_the_sampler_recovers_what_was_trained_on(self):
        self.assertLess(self._error(steps=10), 0.5)

    def test_few_step_sampling_is_not_catastrophic(self):
        """The DreamZero-Flash comparison rests on this degrading gracefully."""
        self.assertLess(self._error(steps=1), 0.9)

    def test_integrating_the_other_way_is_far_worse(self):
        """The control. Without it the tests above could pass a broken direction
        that merely happens to land near zero."""
        from actoris_harena.policies.common import (  # noqa: F401  (documents the pairing)
            flow,
        )

        torch.manual_seed(1)
        noise = torch.randn(2, 6, 12)
        tokens = self.policy._observation_tokens(self.policy._prepare(dict(self.batch)))
        x_t, dt = noise.clone(), 1.0 / 20
        with torch.no_grad():
            for step in range(20):
                x_t = x_t + dt * self.policy.expert(
                    tokens, x_t, torch.full((2,), step * dt)
                )
        wrong = (x_t - self.batch[ACTION]).abs().mean().item() / self.scale
        self.assertGreater(wrong, 3 * self._error(steps=20))


if __name__ == "__main__":
    unittest.main()
