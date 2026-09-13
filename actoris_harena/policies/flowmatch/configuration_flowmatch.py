#!/usr/bin/env python
"""Configuration for the flow-matching policy.

pi0.5's objective and action expert, on the diffusion policy's vision trunk.
See ``modeling_flowmatch.py`` for why that pairing is the interesting one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("harena_flowmatch")
@dataclass
class HarenaFlowmatchConfig(PreTrainedConfig):
    """A small flow-matching policy: ResNet vision, transformer action expert.

    The vision fields are named exactly as the diffusion policy names them,
    because ``DiffusionRgbEncoder`` is reused verbatim and reads them off this
    object. Renaming one here would break it in a way that only shows up at
    construction time.
    """

    # ── what it reads and writes ────────────────────────────────────────────
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ── vision: the diffusion policy's encoder, field for field ─────────────
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    resize_shape: tuple[int, int] | None = None
    #: One trunk per camera, or one shared by all of them. Separate trunks cost
    #: ~11M parameters each; shared is the default because this rig has five
    #: cameras and the point of the model is to be small.
    use_separate_rgb_encoder_per_camera: bool = False

    # ── the action expert ───────────────────────────────────────────────────
    dim_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # ── flow matching ───────────────────────────────────────────────────────
    #: Denoising steps at inference. The whole family degrades gracefully as this
    #: falls, which is what the DreamZero-Flash comparison measures, so it is a
    #: config field rather than a constant.
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001

    # ── training ────────────────────────────────────────────────────────────
    optimizer_lr: float = 1e-4
    #: The ResNet trunk trains slower than the expert on top of it; see
    #: HarenaFlowmatchPolicy.get_optim_params.
    optimizer_lr_backbone: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed "
                f"chunk_size ({self.chunk_size}): the policy does not plan that far."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"n_obs_steps must be 1 (got {self.n_obs_steps}); this policy "
                "conditions on a single observation, as pi0.5 does."
            )
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {self.num_inference_steps}"
            )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError(
                "a flow-matching policy needs at least one camera or an env state"
            )
        if "action" not in self.output_features:
            raise ValueError("'action' is required as an output feature")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


# Registered under its former name as well, so a checkpoint written before this
# package was shared, a run matrix that names it, and a training run resuming
# from its own train_config.json all still resolve. The NEW name is registered
# first, which is what makes ``config.type`` report it -- draccus's
# ``get_choice_name`` returns the first registered name for a class. See
# LEGACY_PREFIX in _port.py.
PreTrainedConfig.register_subclass("so101_flowmatch", HarenaFlowmatchConfig)
