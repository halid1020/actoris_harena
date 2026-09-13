"""A ported policy reproduces a real trained checkpoint, and trains like it.

This is the claim that decides whether moving the code cost anything, and it has
two halves. The finished 80 000-step ACT run on
``fold-short-from-flattend-tactile`` must plan exactly what it planned before --
and it must also compute the same LOSS and the same GRADIENTS, which the action
comparison cannot see: ``predict_action_chunk`` runs under ``no_grad`` in
``eval()`` mode, so dropout, ACT's VAE sampling and diffusion's noise draw are
all on a branch it never enters.

What this still does not cover is everything ``lerobot-train`` assembles around
the model -- processors, optimiser preset, dataloader, plugin discovery. That is
``tool/compare_port_training.py``, which runs the real trainer twice; the source
half of the argument is ``test/unit/test_policy_ports.py``.

Needs the weights on this machine and a couple of minutes of CPU, which is why
it is here and not in the unit tier. Both tests skip rather than fail when the
checkpoint is not on this machine.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from actoris_harena.outputs import output_root

# The checkpoints belong to whichever RIG trained them, not to this package, so
# they are looked up under the shared output root rather than beside this file.
# Point RIG_OUTPUT_DIR at a rig's outputs to run this against its real
# checkpoints:
#
#     RIG_OUTPUT_DIR=~/Projects/so101_garment/outputs make test-integration
#
# Without that these tests SKIP, and a skip here is not a pass: this is the test
# that makes "a checkpoint trained by either implementation loads into the other"
# a checkable claim rather than an intention.
REPO = output_root()

ACT_CHECKPOINT = REPO / "policies/fold-short-from-flattend-tactile__all-act"
DIFFUSION_CHECKPOINT = (
    REPO
    / "vla_real_long/tmp_remote_diffusion/train/diffusion/checkpoints/000005/pretrained_model"
)


def _readable(path: Path) -> bool:
    """Is this file actually there? An autofs mount whose drive is detached
    raises OSError from is_file() rather than returning False."""
    try:
        return path.is_file()
    except OSError:
        return False


class CheckpointEquivalenceTest(unittest.TestCase):
    """A port that cannot reproduce a trained checkpoint is a rewrite, not a port."""

    def _compare(self, checkpoint: Path, upstream_type: str, ported_type: str) -> None:
        import torch
        from actoris_harena.policies.loading import config_as, ensure_registered
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        ensure_registered()
        path = str(checkpoint)
        upstream_config = PreTrainedConfig.from_pretrained(path)
        self.assertEqual(upstream_config.type, upstream_type)
        upstream_config.pretrained_path, upstream_config.device = path, "cpu"
        upstream = get_policy_class(upstream_type).from_pretrained(
            path, config=upstream_config
        )

        ported_config = config_as(upstream_config, ported_type)
        ported_config.pretrained_path, ported_config.device = path, "cpu"
        ported = get_policy_class(ported_type).from_pretrained(
            path, config=ported_config
        )

        left, right = upstream.state_dict(), ported.state_dict()
        self.assertEqual(set(left), set(right), "state_dict keys differ")
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), f"{key} differs")

        upstream.eval()
        ported.eval()
        torch.manual_seed(0)
        # A policy that reads a WINDOW of observations (diffusion's n_obs_steps)
        # wants them stacked on a step axis; one that reads a single frame (ACT)
        # must not be given that axis at all.
        steps = int(getattr(upstream_config, "n_obs_steps", 1) or 1)
        shape = (1, steps) if steps > 1 else (1,)
        batch = {
            key: torch.rand(*shape, *feature.shape)
            for key, feature in upstream_config.input_features.items()
        }

        # Seeded IMMEDIATELY BEFORE each call, not once before both: a stochastic
        # sampler (diffusion denoises over 100 timesteps) draws from the global
        # RNG, so the first call leaves the second somewhere else entirely and the
        # two would differ for a reason that has nothing to do with the port.
        def plan(policy):
            torch.manual_seed(0)
            with torch.no_grad():
                return policy.predict_action_chunk(dict(batch))

        self.assertTrue(
            torch.equal(plan(upstream), plan(ported)),
            "the two implementations planned different actions from one observation",
        )
        self._compare_loss(upstream, ported, upstream_config, batch)

    def _compare_loss(self, upstream, ported, config, batch) -> None:
        """The TRAINING forward, which the inference comparison never touches.

        ``predict_action_chunk`` runs under ``no_grad`` in ``eval()`` mode, so
        it exercises none of the code that decides how a run trains: dropout,
        ACT's VAE sampling, diffusion's noise and timestep draws, and the loss
        itself are all on the training branch only. Without this, "our act
        trains like LeRobot's act" rests entirely on the two files being the
        same bytes -- a strong argument, but one that says nothing about the
        processors and the optimiser preset the factory builds around them.
        """
        import torch

        action = config.output_features["action"]
        horizon = int(
            getattr(config, "horizon", None) or getattr(config, "chunk_size", 1) or 1
        )
        train_batch = dict(batch)
        train_batch["action"] = torch.rand(1, horizon, *action.shape)
        train_batch["action_is_pad"] = torch.zeros(1, horizon, dtype=torch.bool)

        # Seeded immediately before each, for the same reason the inference
        # comparison is: the training branch draws from the global RNG, and the
        # first call would otherwise leave the second in a different place.
        def loss(policy):
            policy.train()
            torch.manual_seed(0)
            value, _ = policy.forward(dict(train_batch))
            return value

        left, right = loss(upstream), loss(ported)
        self.assertTrue(
            torch.equal(left, right),
            f"the two implementations computed different losses "
            f"({left.item()} vs {right.item()}) from one batch — they would "
            "train to different places",
        )

        # And the gradients, which is what the optimiser actually consumes. Two
        # implementations can agree on a scalar and disagree on where it came
        # from; only this compares the step that would be taken.
        def grads(policy):
            policy.zero_grad(set_to_none=True)
            policy.train()
            torch.manual_seed(0)
            value, _ = policy.forward(dict(train_batch))
            value.backward()
            return {
                name: p.grad.clone()
                for name, p in policy.named_parameters()
                if p.grad is not None
            }

        left_grads, right_grads = grads(upstream), grads(ported)
        self.assertEqual(
            set(left_grads), set(right_grads), "different parameters moved"
        )
        self.assertTrue(left_grads, "no gradients were produced at all")
        for name in left_grads:
            self.assertTrue(
                torch.equal(left_grads[name], right_grads[name]),
                f"gradient for {name} differs",
            )

    def test_act_reproduces_the_trained_checkpoint(self) -> None:
        if not _readable(ACT_CHECKPOINT / "model.safetensors"):
            self.skipTest(f"no ACT checkpoint at {ACT_CHECKPOINT}")
        self._compare(ACT_CHECKPOINT, "act", "harena_act")

    def test_diffusion_reproduces_a_trained_checkpoint(self) -> None:
        if not _readable(DIFFUSION_CHECKPOINT / "model.safetensors"):
            self.skipTest(f"no diffusion checkpoint at {DIFFUSION_CHECKPOINT}")
        self._compare(DIFFUSION_CHECKPOINT, "diffusion", "harena_diffusion")


if __name__ == "__main__":
    unittest.main()
