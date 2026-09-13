#!/usr/bin/env python
"""A small flow-matching policy: pi0.5's objective, the diffusion policy's eyes.

**Why this model exists.** It is the control for the world-action model. It
shares DreamZero's training objective -- conditional flow matching over an
action chunk -- and shares nothing else: no video prediction, no world model, no
pretrained video prior. So the gap between the two measures the world-modelling
objective with everything else held fixed, which is a cleaner comparison than
either paper runs against its own baselines.

**What is borrowed, and from where.**

* *Vision* is ``DiffusionRgbEncoder`` from ``actoris_harena.policies.diffusion`` --
  literally that class, not a reimplementation: a ResNet trunk with its
  BatchNorms swapped for GroupNorm, ending in a spatial-softmax keypoint pool.
  Reusing it is what makes "same backbone" a fact rather than a claim.
* *The objective and the action expert* are pi0.5's: a transformer that reads the
  noisy action chunk conditioned on a timestep, trained to regress the
  flow-matching velocity, sampled by Euler integration from noise to action.
  ``actoris_harena.policies.common.flow`` holds that objective, in pi0.5's time
  convention, and says so.

**What is deliberately not borrowed.** pi0.5 is a 4.1B vision-language-action
model; its action expert attends to a PaliGemma prefix carrying image and
language tokens. There is no language here and no VLM. The prefix is the
observation: one token per camera plus one for proprioception. That is the whole
point -- it isolates the objective from the pretrained semantics, so a difference
in results cannot be attributed to a language model this rig never had.
"""

from __future__ import annotations

import math
from collections import deque

import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from actoris_harena.policies.common import flow
from actoris_harena.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder

from .configuration_flowmatch import HarenaFlowmatchConfig


def timestep_embedding(time: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Sinusoidal embedding of a continuous timestep in [0, 1].

    The usual transformer formula, with ``time`` scaled up because it lives in
    the unit interval rather than in the thousands of diffusion step indices the
    constant was chosen for -- without the scaling every frequency collapses to
    nearly the same value and the model cannot tell one timestep from another.
    """
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=time.device)
        / half
    )
    angles = time[:, None].float() * 1000.0 * frequencies[None, :]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FlowmatchActionExpert(nn.Module):
    """Transformer over the noisy chunk, conditioned on observation and timestep.

    One encoder stack over ``[observation tokens ; action tokens]`` with full
    attention. pi0.5 splits the two into a frozen prefix with a KV cache and a
    trained suffix because its prefix is a 3B VLM that must not be re-run per
    denoising step; at this size the prefix is a handful of tokens and caching it
    would buy nothing but a chance to get the masking wrong.
    """

    def __init__(
        self, config: HarenaFlowmatchConfig, n_observation_tokens: int, action_dim: int
    ):
        super().__init__()
        self.config = config
        self.action_in = nn.Linear(action_dim, config.dim_model)
        self.action_out = nn.Linear(config.dim_model, action_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.SiLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )
        # Learned, because neither axis is a sequence in time: the observation
        # tokens are an unordered set of streams and the action tokens index a
        # chunk whose length never changes.
        self.position = nn.Parameter(
            torch.zeros(1, n_observation_tokens + config.chunk_size, config.dim_model)
        )
        nn.init.normal_(self.position, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and torch warns
        # about it on every construction; say so once here instead.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.dim_model)
        self.n_observation_tokens = n_observation_tokens

    def forward(
        self, observation_tokens: Tensor, noisy_actions: Tensor, time: Tensor
    ) -> Tensor:
        """Predict the flow-matching velocity for every action in the chunk."""
        actions = self.action_in(noisy_actions)
        # Added to every token, observation included: the observation's meaning
        # to the model genuinely does depend on how far the chunk has denoised.
        conditioning = self.time_mlp(timestep_embedding(time, self.config.dim_model))
        tokens = (
            torch.cat([observation_tokens, actions], dim=1) + conditioning[:, None, :]
        )
        tokens = tokens + self.position[:, : tokens.shape[1]]
        out = self.norm(self.transformer(tokens))
        return self.action_out(out[:, self.n_observation_tokens :])


class HarenaFlowmatchPolicy(PreTrainedPolicy):
    """Flow matching over an action chunk, from images and proprioception."""

    config_class = HarenaFlowmatchConfig
    name = "harena_flowmatch"

    def __init__(self, config: HarenaFlowmatchConfig, dataset_stats=None, **kwargs):
        # make_policy passes dataset_stats and dataset_meta when it has them;
        # normalisation is the processors' job here, so they are accepted and
        # ignored rather than allowed to raise.
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.camera_keys = list(config.image_features)
        if config.use_separate_rgb_encoder_per_camera:
            self.rgb_encoder = nn.ModuleList(
                DiffusionRgbEncoder(config) for _ in self.camera_keys
            )
            feature_dim = self.rgb_encoder[0].feature_dim
        else:
            self.rgb_encoder = DiffusionRgbEncoder(config)
            feature_dim = self.rgb_encoder.feature_dim

        self.image_proj = nn.Linear(feature_dim, config.dim_model)

        state_feature = config.robot_state_feature
        self.state_dim = int(state_feature.shape[0]) if state_feature is not None else 0
        if self.state_dim:
            self.state_proj = nn.Linear(self.state_dim, config.dim_model)

        self.action_dim = int(config.output_features[ACTION].shape[0])
        n_tokens = len(self.camera_keys) + (1 if self.state_dim else 0)
        self.expert = FlowmatchActionExpert(config, n_tokens, self.action_dim)

        self.reset()

    def get_optim_params(self) -> list:
        """Parameter groups for the optimiser.

        A LIST of groups, not a dict: ``cfg.optimizer.build`` passes this
        straight to torch, which iterates it -- handed a dict it would iterate
        the KEYS and complain that a str is not a Tensor.

        The vision trunk gets a lower rate than the expert, as ACT does for its
        backbone: a ResNet reaching a useful representation is a slower business
        than a randomly initialised transformer finding a velocity field, and one
        rate for both overshoots on the trunk.
        """
        vision = [
            p
            for name, p in self.named_parameters()
            if name.startswith("rgb_encoder") and p.requires_grad
        ]
        rest = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("rgb_encoder") and p.requires_grad
        ]
        return [
            {"params": rest},
            {"params": vision, "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self) -> None:
        """Clear the queue of actions already planned but not yet executed."""
        self._queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    # ── conditioning ────────────────────────────────────────────────────────

    def _observation_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        images = batch[OBS_IMAGES]
        if isinstance(images, Tensor):
            # Stacked on a camera axis by some callers, a list by others. Both
            # reach here, so accept both rather than making the caller care.
            images = list(images.unbind(dim=-4))

        features = []
        for index, image in enumerate(images):
            encoder = (
                self.rgb_encoder[index]
                if self.config.use_separate_rgb_encoder_per_camera
                else self.rgb_encoder
            )
            features.append(self.image_proj(encoder(image)))
        tokens = torch.stack(features, dim=1)

        if self.state_dim:
            state = self.state_proj(batch[OBS_STATE])
            tokens = torch.cat([state[:, None, :], tokens], dim=1)
        return tokens

    def _prepare(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Give the batch the OBS_IMAGES list the encoders expect.

        The camera ORDER is ``config.image_features``, which is also the order
        every layout in ``common/analysis`` assumes. It is not arbitrary and must
        not be re-derived by sorting.
        """
        batch = dict(batch)
        if OBS_IMAGES not in batch:
            batch[OBS_IMAGES] = [batch[key] for key in self.camera_keys]
        return batch

    # ── the objective ───────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """One training step: regress the flow-matching velocity. Eq. as pi0.5."""
        batch = self._prepare(batch)
        actions = batch[ACTION]
        tokens = self._observation_tokens(batch)

        noise = torch.randn_like(actions)
        time = flow.sample_time(
            actions.shape[0],
            actions.device,
            alpha=self.config.time_sampling_beta_alpha,
            beta=self.config.time_sampling_beta_beta,
            scale=self.config.time_sampling_scale,
            offset=self.config.time_sampling_offset,
        )
        noisy = flow.interpolate(actions, noise, time)
        predicted = self.expert(tokens, noisy, time)

        per_element = flow.loss(predicted, actions, noise)
        pad = batch.get("action_is_pad")
        if pad is not None:
            # A chunk that runs off the end of an episode is padded. Training on
            # the padding teaches the policy to reproduce it.
            per_element = per_element * (~pad).unsqueeze(-1)
            loss = per_element.sum() / ((~pad).sum() * self.action_dim).clamp(min=1)
        else:
            loss = per_element.mean()
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Integrate from noise to a chunk of actions."""
        batch = self._prepare(batch)
        tokens = self._observation_tokens(batch)
        reference = tokens
        noise = torch.randn(
            reference.shape[0],
            self.config.chunk_size,
            self.action_dim,
            device=reference.device,
            dtype=reference.dtype,
        )
        return flow.integrate(
            lambda x_t, time: self.expert(tokens, x_t, time),
            noise,
            steps=self.config.num_inference_steps,
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """One action per call, re-planning only when the queue runs dry."""
        self.eval()
        if not self._queue:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Queued along time so popleft yields the next action, which is what
            # the transpose is for.
            self._queue.extend(chunk.transpose(0, 1))
        return self._queue.popleft()
