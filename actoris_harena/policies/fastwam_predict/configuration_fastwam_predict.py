#!/usr/bin/env python
"""FastWAM, described in the vocabulary the world-model scorer speaks.

FastWAM already decodes future video -- ``wan.modular.infer_joint`` returns
``{"video": ..., "action": ...}`` -- and nothing in the policy API ever calls
it: ``predict_action_chunk`` only reaches ``infer_action``. So the port owns a
video prior it cannot be scored on, while ``harena_dreamzero`` can be scored and
owns no prior. They are not comparable, and that is a naming problem rather than
a modelling one.

This config adds nothing the model uses. It supplies the two fields
``tool/eval_world_model.py`` reads to know how much of the window is context --
DreamZero counts context in latent CHUNKS, FastWAM conditions on a single first
frame -- so one scorer can address both. The port itself is untouched, which is
what keeps ``tool/port_policies.py --check`` green.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..fastwam.configuration_fastwam import HarenaFastwamConfig


@PreTrainedConfig.register_subclass("harena_fastwam_predict")
@dataclass
class HarenaFastwamPredictConfig(HarenaFastwamConfig):
    """HarenaFastwamConfig, plus the context arithmetic the scorer asks for."""

    #: Denoising steps for a scored prediction. Kept apart from whatever an
    #: action rollout uses: a video comparison wants the same budget every
    #: frame, and a PSNR that moved because the step count moved is not a
    #: measurement of the model.
    predict_inference_steps: int = 20

    #: Seed for a scored prediction. ``infer_joint`` samples, so two passes over
    #: one observation disagree; a scorer comparing an arm against another arm
    #: has to hold that fixed or it measures the sampler. Same reasoning as
    #: ``analysis.diffusion.plan``.
    predict_seed: int = 0

    @property
    def n_context_chunks(self) -> int:
        """One. FastWAM conditions on a single first frame, not on a chunk.

        DreamZero's context is ``n_context_chunks * latent_frames_per_chunk``
        latent frames; FastWAM's is ``input_image``, the first frame of the
        window, and the rest of the window is what it predicts. Reporting 1
        here is not a convention chosen for convenience -- it is the number of
        observed frames the prediction is conditioned on, which is what the
        scorer slices off before comparing.
        """
        return 1

    @property
    def latent_frames_per_chunk(self) -> int:
        """One, for the same reason: FastWAM's context is one frame, not a chunk."""
        return 1


# Registered under the legacy prefix too, so a checkpoint or a run matrix that
# names the old spelling still resolves. The new name is registered FIRST,
# which is what makes ``config.type`` report it -- draccus's ``get_choice_name``
# returns the first registered name for a class. See LEGACY_PREFIX in _port.py.
PreTrainedConfig.register_subclass("so101_fastwam_predict", HarenaFastwamPredictConfig)
