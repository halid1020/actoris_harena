#!/usr/bin/env python
"""Configuration for the world action model.

A small, faithful DreamZero: the paper's objective, architecture and schedules,
at a size a 24 GB card can train. Every place we differ is named in
``modeling_dreamzero``'s docstring and in ``documents/world_action_model.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from actoris_harena.policies.common.vae import DEFAULT_VAE


@PreTrainedConfig.register_subclass("harena_dreamzero")
@dataclass
class HarenaDreamzeroConfig(PreTrainedConfig):
    """DreamZero at rig scale.

    The chunking defaults are the paper's (Appendix C): ``K = 2`` latent frames
    per chunk, ``M = 4`` chunks, one of which is context. The sizes are not --
    a 14B DiT on a Wan backbone is out of reach here, and the point of this
    implementation is to test the paper's CLAIMS rather than reproduce its
    numbers.
    """

    # ── what it reads and writes ────────────────────────────────────────────
    n_obs_steps: int = 1
    #: Actions executed per planned chunk. The paper uses H = 48 at 30 Hz
    #: (1.6 s); this rig records at 30 Hz too, so the default matches.
    chunk_size: int = 48
    n_action_steps: int = 24

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,  # the VAE wants raw [0, 1]
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ── the chunked sequence (Appendix C) ───────────────────────────────────
    latent_frames_per_chunk: int = 2  # K; the paper found 2 beats 1
    n_chunks: int = 4  # M
    n_context_chunks: int = 1  # what the rollout starts from
    #: Frames are square and all views are tiled into ONE of them, which is how
    #: the paper handles multiple cameras without touching the backbone.
    image_size: int = 224
    #: Patch 4 over a 28x28 latent gives 49 tokens a frame, so the whole chunked
    #: sequence is ~1000 tokens rather than ~3000. The paper does not state its
    #: patch size; at 14B it can afford one that we cannot -- attention is
    #: quadratic and this has to fit beside its activations on 24 GB.
    patch_size: int = 4

    # ── the DiT ─────────────────────────────────────────────────────────────
    dim_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    dim_feedforward: int = 2048
    dropout: float = 0.0

    # ── flow matching ───────────────────────────────────────────────────────
    #: DreamZero-Flash: bias the VIDEO timestep towards noise while leaving the
    #: action timestep uniform, so the model learns to read actions off a still
    #: noisy visual context -- which is what few-step inference actually asks of
    #: it. False is the paper's coupled schedule (Eq. 4).
    flash: bool = False
    flash_alpha: float = 7.0
    flash_beta: float = 1.0
    num_inference_steps: int = 4
    #: Weight on the video half of the joint objective. 1.0 is the paper's
    #: unweighted sum; lowering it trades video fidelity for action accuracy and
    #: is the knob to reach for if the actions lag the video.
    video_loss_weight: float = 1.0
    #: Savitzky-Golay smoothing of the returned chunk (Appendix D.3). Off for
    #: prediction-accuracy measurement, on when driving an arm.
    smooth_actions: bool = True

    # ── the frozen VAE ──────────────────────────────────────────────────────
    vae_model_id: str = DEFAULT_VAE

    # ── training ────────────────────────────────────────────────────────────
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 50000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_context_chunks < 1:
            raise ValueError("n_context_chunks must be at least 1")
        if self.n_chunks <= self.n_context_chunks:
            raise ValueError(
                f"n_chunks ({self.n_chunks}) must exceed n_context_chunks "
                f"({self.n_context_chunks}), or nothing is predicted"
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed "
                f"chunk_size ({self.chunk_size})"
            )
        if self.image_size % (8 * self.patch_size):
            raise ValueError(
                f"image_size ({self.image_size}) must divide by the VAE's 8x "
                f"compression times patch_size ({self.patch_size})"
            )
        if self.dim_model % self.n_heads:
            raise ValueError(
                f"dim_model {self.dim_model} is not divisible by {self.n_heads} heads"
            )
        if self.chunk_size % self.latent_frames_per_chunk:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must divide by latent_frames_per_chunk "
                f"({self.latent_frames_per_chunk}): a chunk's frames and its actions have to "
                "span the SAME interval of time, so the frames are subsampled to fit"
            )

    # ── derived sizes, so nothing recomputes them by hand ───────────────────

    @property
    def latent_size(self) -> int:
        return self.image_size // 8

    @property
    def patches_per_frame(self) -> int:
        side = self.latent_size // self.patch_size
        return side * side

    @property
    def video_tokens_per_chunk(self) -> int:
        return self.latent_frames_per_chunk * self.patches_per_frame

    @property
    def predicted_chunks(self) -> int:
        return self.n_chunks - self.n_context_chunks

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("a world model needs at least one camera to predict")
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
    def frame_stride(self) -> int:
        """Frames between the latent frames of one chunk.

        A chunk's ``latent_frames_per_chunk`` frames and its ``chunk_size``
        actions must describe the SAME interval, or the joint objective is
        asked to align a video of one moment with actions of another. The paper
        gets this from Wan's temporal VAE, which folds four raw frames into one
        latent frame; a per-frame VAE has to subsample instead. At the defaults
        that is one frame every 24, spanning 1.6 s per chunk at 30 Hz -- the same
        1.6 s the paper uses.
        """
        return self.chunk_size // self.latent_frames_per_chunk

    @property
    def observation_delta_indices(self) -> list:
        """Frame offsets for every chunk, context and predicted alike.

        Time zero is the start of the FIRST PREDICTED chunk: context chunks sit
        at negative offsets, which is what makes them the past.
        """
        return [
            (chunk - self.n_context_chunks) * self.chunk_size
            + frame * self.frame_stride
            for chunk in range(self.n_chunks)
            for frame in range(self.latent_frames_per_chunk)
        ]

    @property
    def action_delta_indices(self) -> list:
        """Action offsets for every chunk, negative for the context ones.

        The context chunks' actions are part of the conditioning -- Algorithm 1
        writes the clean context as ``{(z_1^j, a_1^j)}``, video AND action -- so
        the past actions are fetched rather than zeroed.
        """
        return [
            (chunk - self.n_context_chunks) * self.chunk_size + step
            for chunk in range(self.n_chunks)
            for step in range(self.chunk_size)
        ]

    @property
    def reward_delta_indices(self) -> None:
        return None


# Registered under its former name as well, so a checkpoint written before this
# package was shared, a run matrix that names it, and a training run resuming
# from its own train_config.json all still resolve. The NEW name is registered
# first, which is what makes ``config.type`` report it -- draccus's
# ``get_choice_name`` returns the first registered name for a class. See
# LEGACY_PREFIX in _port.py.
PreTrainedConfig.register_subclass("so101_dreamzero", HarenaDreamzeroConfig)
