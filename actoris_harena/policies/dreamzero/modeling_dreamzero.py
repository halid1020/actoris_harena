#!/usr/bin/env python
"""DreamZero: a world action model that predicts video and actions together.

Implements *World Action Models are Zero-shot Policies* (arXiv 2602.15922) --
its objective, architecture and schedules -- at a size a 24 GB card can train.

**The idea.** A policy trained only to map observations to actions has to infer
the dynamics of the world implicitly, from action labels alone. A world action
model is trained to predict the future *frames* as well, so it learns dynamics
from every consecutive frame pair rather than only from what an action label
happens to reveal. Actions and video are denoised jointly under ONE flow
matching objective, which is what keeps them aligned: the paper's own failure
analysis is that when DreamZero fails, it is usually the video prediction that
was wrong and the actions faithfully executed it.

**Faithful to the paper:**

* the joint flow matching objective over ``[video latents ; actions]``, Eq. 2-3;
* chunk-wise teacher forcing -- a chunk is denoised conditioned on the CLEAN
  earlier chunks, never its own answer (``masking.py``, Algorithm 1 line 9);
* the between-chunk causal attention mask of Figure 14, with the training and
  inference masks proven to agree;
* autoregressive rollout with a KV cache, and -- the step that makes it a policy
  rather than a video generator -- the real observation replacing the predicted
  latent after each executed chunk (Algorithm 2 line 28-32);
* DreamZero-Flash's decoupled noise schedules, Eq. 5-6;
* Savitzky-Golay chunk smoothing, Appendix D.3;
* multiple cameras tiled into ONE frame rather than changed in the backbone.

**Deviations, every one of them deliberate:**

* **No video pretraining.** The paper's whole argument is that a WAM inherits
  physical priors from web-scale video. We train from scratch, so what this
  tests is the OBJECTIVE and the ARCHITECTURE, not the prior. It is the single
  biggest reason our absolute numbers are not theirs.
* **A per-frame image VAE**, not Wan's temporal video VAE. See ``common/vae.py``:
  a latent frame is a frame, so the same token budget buys a shorter visual
  context.
* **~1/100th the parameters.** The paper's own 5B ablation scored 21% against
  14B's 50%, so scale is known to matter here and our numbers sit below both.
* **Language conditioning is a learned task embedding**, not a frozen text
  encoder. Every dataset on this rig carries one task string, so a text encoder
  would be an expensive way to look up a constant.
* **adaLN modulation is per token, not per sample.** It has to be: Flash gives
  video and action tokens different timesteps within one forward pass, and a
  single conditioning vector per sample cannot express that.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from torch import Tensor, nn

from actoris_harena.policies.common import flow
from actoris_harena.policies.common.smoothing import smooth_chunk
from actoris_harena.policies.common.vae import FrozenImageVae

from .configuration_dreamzero import HarenaDreamzeroConfig
from .masking import ChunkLayout, to_additive, training_mask


def timestep_embedding(time: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Sinusoidal embedding of continuous timesteps of any shape.

    ``time`` may be ``(B,)`` or ``(B, N)``; the embedding is appended as a last
    axis. Scaled by 1000 because these timesteps live in the unit interval, not
    in the thousands of step indices the constant was chosen for.
    """
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=time.device)
        / half
    )
    angles = time.float().unsqueeze(-1) * 1000.0 * frequencies
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class DiTBlock(nn.Module):
    """One DiT block: adaLN-Zero modulation, masked self-attention, MLP.

    The modulation is computed PER TOKEN because DreamZero-Flash gives the video
    and action tokens of a single forward pass different timesteps. Standard DiT
    takes one conditioning vector per sample and could not represent that.

    adaLN-*Zero*: the output gates start at zero, so a freshly initialised block
    is the identity and the residual stream reaches the head unmangled. Without
    it a deep stack of these trains slowly and unstably.
    """

    def __init__(
        self, dim: int, n_heads: int, dim_feedforward: int, dropout: float = 0.0
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
        )
        self.modulation = nn.Linear(dim, 6 * dim)
        # Zero here is what makes it adaLN-Zero rather than adaLN.
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, conditioning: Tensor, attn_mask: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(
            conditioning
        ).chunk(6, dim=-1)
        batch, tokens, dim = x.shape

        h = self.norm1(x) * (1 + scale1) + shift1
        qkv = self.qkv(h).reshape(batch, tokens, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        # SDPA rather than a materialised attention matrix: at ~1000 tokens the
        # explicit one is hundreds of megabytes per layer.
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        x = x + gate1 * self.attn_out(attended)

        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


class HarenaDreamzeroPolicy(PreTrainedPolicy):
    """Joint video-and-action flow matching over a chunked, causally masked sequence."""

    config_class = HarenaDreamzeroConfig
    name = "harena_dreamzero"

    def __init__(self, config: HarenaDreamzeroConfig, dataset_stats=None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.camera_keys = list(config.image_features)
        self.action_dim = int(config.output_features[ACTION].shape[0])
        state_feature = config.robot_state_feature
        self.state_dim = int(state_feature.shape[0]) if state_feature is not None else 0

        # The VAE is frozen and identical in every checkpoint, so it is built
        # lazily and kept out of the state dict -- 83.7M parameters that would
        # otherwise be written into every save for nothing.
        self._vae: "FrozenImageVae | None" = None

        dim = config.dim_model
        patch_dim = 4 * config.patch_size * config.patch_size
        self.video_in = nn.Linear(patch_dim, dim)
        self.video_out = nn.Linear(dim, patch_dim)
        self.action_in = nn.Linear(self.action_dim, dim)
        self.action_out = nn.Linear(dim, self.action_dim)

        self.layout = ChunkLayout(
            n_chunks=config.n_chunks,
            video_tokens_per_chunk=config.video_tokens_per_chunk,
            action_tokens_per_chunk=config.chunk_size,
            n_context=config.n_context_chunks,
        )
        # The mask is a constant of the architecture. Registered as a buffer so
        # it follows the model to the GPU, and non-persistent so it is not
        # written into the checkpoint.
        self.register_buffer(
            "attn_mask",
            to_additive(training_mask(self.layout)),
            persistent=False,
        )

        self.position = nn.Parameter(torch.zeros(1, self.layout.total, dim))
        nn.init.normal_(self.position, std=0.02)
        # Video and action tokens are different KINDS of thing occupying one
        # sequence; without this the model has to infer which is which from
        # position alone.
        self.modality = nn.Parameter(torch.zeros(2, dim))
        nn.init.normal_(self.modality, std=0.02)

        if self.state_dim:
            self.state_encoder = nn.Sequential(
                nn.Linear(self.state_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
            )
        self.time_encoder = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        # One task string per dataset on this rig, so a learned embedding is the
        # honest form of "language conditioning" here.
        self.task_embedding = nn.Parameter(torch.zeros(1, dim))

        self.blocks = nn.ModuleList(
            DiTBlock(dim, config.n_heads, config.dim_feedforward, config.dropout)
            for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)

        self.reset()

    # ── plumbing ────────────────────────────────────────────────────────────

    def get_optim_params(self) -> list:
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def reset(self) -> None:
        self._queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    @property
    def vae(self) -> FrozenImageVae:
        if self._vae is None:
            self._vae = FrozenImageVae(
                self.config.vae_model_id, device=self.parameters_device
            )
        return self._vae

    @property
    def parameters_device(self) -> torch.device:
        return next(self.parameters()).device

    # ── patching ────────────────────────────────────────────────────────────

    def patchify(self, latents: Tensor) -> Tensor:
        """``(B, T, C, h, w)`` to ``(B, T * patches, C * p * p)``."""
        patch = self.config.patch_size
        batch, steps, channels, height, width = latents.shape
        x = latents.reshape(
            batch, steps, channels, height // patch, patch, width // patch, patch
        )
        # -> (B, T, h/p, w/p, C, p, p) so each patch's channels stay together
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        return x.reshape(batch, steps * (height // patch) * (width // patch), -1)

    def unpatchify(self, tokens: Tensor, steps: int) -> Tensor:
        """The exact inverse of :meth:`patchify`."""
        patch = self.config.patch_size
        side = self.config.latent_size // patch
        batch = tokens.shape[0]
        x = tokens.reshape(batch, steps, side, side, 4, patch, patch)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)
        return x.reshape(batch, steps, 4, side * patch, side * patch)

    # ── assembling the sequence ─────────────────────────────────────────────

    def tile_cameras(self, batch: dict[str, Tensor]) -> Tensor:
        """Every camera into ONE square frame, the paper's multi-view recipe.

        "For robot training data that contains multiple views, we concatenate all
        views into a single frame instead of making architectural changes to the
        backbone." Laid out on the smallest square grid that fits them, in
        ``config.image_features`` order, with any spare cell left black.

        Input frames are ``(B, T, 3, H, W)`` per camera; the output is
        ``(B, T, 3, image_size, image_size)``.
        """
        cells = self.camera_cells()
        frames = []
        for key in self.camera_keys:
            frame = batch[key]
            if frame.ndim != 5:
                # A world model reads a SEQUENCE of frames, so every camera needs
                # the time axis that observation_delta_indices asks for. A batch
                # built by hand that delta-timestamps only some cameras arrives
                # here mixed, and torch's own complaint names an interpolation
                # size rather than the missing axis.
                raise ValueError(
                    f"{key} is {tuple(frame.shape)}; expected (batch, time, 3, h, w). "
                    "Every camera needs the frames named by "
                    "config.observation_delta_indices, not just one of them."
                )
            frames.append(frame)
        batch_size, steps = frames[0].shape[:2]
        canvas = frames[0].new_zeros(
            batch_size, steps, 3, self.config.image_size, self.config.image_size
        )
        for frame, (top, left, size) in zip(frames, cells.values()):
            resized = F.interpolate(
                frame.flatten(0, 1),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (batch_size, steps))
            canvas[..., top : top + size, left : left + size] = resized
        return canvas

    def camera_cells(self) -> "dict[str, tuple[int, int, int]]":
        """Where each camera sits in the tiled frame: ``{key: (top, left, size)}``.

        The single source of the grid, so tiling and un-tiling cannot disagree --
        and un-tiling is what makes prediction accuracy reportable PER CAMERA,
        which is the whole point on a rig where four of five views are tactile
        and behave nothing like the overhead one.
        """
        grid = math.ceil(math.sqrt(len(self.camera_keys)))
        size = self.config.image_size // grid
        cells = {}
        for index, key in enumerate(self.camera_keys):
            row, column = divmod(index, grid)
            cells[key] = (row * size, column * size, size)
        return cells

    def untile_cameras(self, tiled: Tensor) -> "dict[str, Tensor]":
        """Split a tiled frame back into one tensor per camera.

        The inverse of :meth:`tile_cameras` up to the resize, which is lossy --
        so a prediction and its ground truth are BOTH compared at cell
        resolution, never one upscaled to meet the other.
        """
        return {
            key: tiled[..., top : top + size, left : left + size]
            for key, (top, left, size) in self.camera_cells().items()
        }

    def _chunk_tokens(self, video: Tensor, actions: Tensor, block: int) -> Tensor:
        """Embed one chunk's video patches and actions into ``dim_model``.

        ``block`` indexes the chunk WITHIN the tensor handed in, which is not the
        absolute chunk index: the noisy tensors hold only the predicted chunks,
        so their first block is chunk ``n_context``. Conflating the two reads off
        the end of the noisy tensor and produces an empty slice.
        """
        frames = self.config.latent_frames_per_chunk
        start = block * frames
        video_tokens = self.video_in(self.patchify(video[:, start : start + frames]))
        video_tokens = video_tokens + self.modality[0]
        action_tokens = self.action_in(actions) + self.modality[1]
        return torch.cat([video_tokens, action_tokens], dim=1)

    def _chunk_state(self, state: "Tensor | None", chunk: int) -> "Tensor | None":
        """The proprioceptive state of chunk ``k`` -- the paper's ``q_k``.

        LeRobot applies ``observation_delta_indices`` to EVERY observation key,
        so the state arrives as one reading per requested frame, ``(B, T, D)``.
        Each chunk takes the reading at its own first frame, which is what makes
        it that chunk's state rather than a single global one. A caller handing
        in a lone ``(B, D)`` reading gets it used for every chunk.
        """
        if state is None or not self.state_dim:
            return None
        if state.ndim == 2:
            return state
        index = min(chunk * self.config.latent_frames_per_chunk, state.shape[1] - 1)
        return state[:, index]

    def _conditioning(
        self, video_time: Tensor, action_time: Tensor, state: "Tensor | None"
    ) -> Tensor:
        """Per-token conditioning: its timestep, its chunk's state, the task.

        Clean context tokens are given timestep 0 -- they carry no noise, and
        saying so explicitly is what lets one stack of blocks process context and
        noisy tokens in a single pass.
        """
        layout = self.layout
        batch = video_time.shape[0]
        device = video_time.device
        times = torch.zeros(batch, layout.total, device=device)

        for chunk in layout.predicted_chunks:
            video_start, video_end = layout.noisy_video_span(chunk)
            action_start, action_end = layout.noisy_action_span(chunk)
            times[:, video_start:video_end] = video_time[:, None]
            times[:, action_start:action_end] = action_time[:, None]

        conditioning = self.time_encoder(
            timestep_embedding(times, self.config.dim_model)
        )
        conditioning = conditioning + self.task_embedding

        if state is not None and self.state_dim:
            for chunk in range(layout.n_chunks):
                embedded = self.state_encoder(self._chunk_state(state, chunk))[
                    :, None, :
                ]
                start, end = layout.clean_span(chunk)
                conditioning[:, start:end] = conditioning[:, start:end] + embedded
                if chunk in layout.predicted_chunks:
                    start, end = layout.noisy_span(chunk)
                    conditioning[:, start:end] = conditioning[:, start:end] + embedded
        return conditioning

    def _denoise(
        self,
        clean_video: Tensor,
        clean_actions: Tensor,
        noisy_video: Tensor,
        noisy_actions: Tensor,
        video_time: Tensor,
        action_time: Tensor,
        state: "Tensor | None",
    ) -> "tuple[Tensor, Tensor]":
        """One forward pass. Returns predicted velocities ``(video, actions)``.

        ``clean_*`` are the teacher-forced context for EVERY chunk; the mask is
        what stops a predicted chunk seeing its own.
        """
        layout = self.layout
        pieces = [
            self._chunk_tokens(clean_video, clean_actions[:, chunk], chunk)
            for chunk in range(layout.n_chunks)
        ]
        pieces += [
            self._chunk_tokens(
                noisy_video,
                noisy_actions[:, chunk - layout.n_context],
                chunk - layout.n_context,
            )
            for chunk in layout.predicted_chunks
        ]
        x = torch.cat(pieces, dim=1) + self.position

        conditioning = self._conditioning(video_time, action_time, state)
        mask = self.attn_mask.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, conditioning, mask)

        shift, scale = self.final_modulation(conditioning).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift

        video_out, action_out = [], []
        for chunk in layout.predicted_chunks:
            video_start, video_end = layout.noisy_video_span(chunk)
            action_start, action_end = layout.noisy_action_span(chunk)
            video_out.append(
                self.unpatchify(
                    self.video_out(x[:, video_start:video_end]),
                    self.config.latent_frames_per_chunk,
                )
            )
            action_out.append(self.action_out(x[:, action_start:action_end]))
        return torch.cat(video_out, dim=1), torch.stack(action_out, dim=1)

    # ── the objective ───────────────────────────────────────────────────────

    def _sample_times(self, batch: int, device) -> "tuple[Tensor, Tensor]":
        if self.config.flash:
            return flow.sample_flash_times(
                batch,
                device,
                alpha=self.config.flash_alpha,
                beta=self.config.flash_beta,
            )
        # Coupled: ONE timestep shared by both modalities (Eq. 4).
        shared = flow.sample_uniform_time(batch, device)
        return shared, shared

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Joint flow matching on video latents and actions (Eq. 3)."""
        config = self.config
        frames = self.tile_cameras(batch)
        latents = self.vae.encode(frames).to(self.parameters_device)

        context = self.layout.n_context
        # The batch carries every chunk's actions -- the context chunks' at
        # negative deltas -- so the clean context is the real past, as Algorithm
        # 1's C_k = {(z_1^j, a_1^j)} requires, rather than a zeroed placeholder.
        clean_actions = batch[ACTION][
            :, : config.n_chunks * config.chunk_size
        ].unflatten(1, (config.n_chunks, config.chunk_size))
        actions = clean_actions[:, context:]

        # The video targets are the latent frames of the PREDICTED chunks.
        target_video = latents[:, context * config.latent_frames_per_chunk :]
        video_noise = torch.randn_like(target_video)
        action_noise = torch.randn_like(actions)

        video_time, action_time = self._sample_times(actions.shape[0], actions.device)
        noisy_video = flow.interpolate(target_video, video_noise, video_time)
        noisy_actions = flow.interpolate(actions, action_noise, action_time)

        video_velocity, action_velocity = self._denoise(
            latents,
            clean_actions,
            noisy_video,
            noisy_actions,
            video_time,
            action_time,
            batch.get(OBS_STATE),
        )

        video_loss = flow.loss(video_velocity, target_video, video_noise).mean()
        per_action = flow.loss(action_velocity, actions, action_noise)
        pad = batch.get("action_is_pad")
        if pad is not None:
            pad = pad[:, : config.n_chunks * config.chunk_size].unflatten(
                1, (config.n_chunks, config.chunk_size)
            )[:, context:]
            per_action = per_action * (~pad).unsqueeze(-1)
            action_loss = per_action.sum() / ((~pad).sum() * self.action_dim).clamp(
                min=1
            )
        else:
            action_loss = per_action.mean()

        loss = action_loss + config.video_loss_weight * video_loss
        return loss, {
            "loss": loss.item(),
            "action_loss": action_loss.item(),
            "video_loss": video_loss.item(),
        }

    # ── rollout ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Denoise one chunk of video and actions, and return the actions.

        The video half is computed and discarded here -- it is the visual plan the
        actions were produced alongside, and :meth:`predict_future` returns it for
        anyone measuring prediction accuracy.
        """
        _, actions = self.predict_future(batch, **kwargs)
        chunk = actions[:, 0]  # the first predicted chunk is the one to execute
        if self.config.smooth_actions:
            chunk = smooth_chunk(chunk)
        return chunk

    @torch.no_grad()
    def predict_future(
        self, batch: dict[str, Tensor], steps: "int | None" = None
    ) -> "tuple[Tensor, Tensor]":
        """``(video_latents, actions)`` for every predicted chunk.

        This is the world model's actual output, and the reason the class exists:
        what it thinks will happen, and what it plans to do about it. Euler
        integration from noise, in the direction ``common/flow`` documents.
        """
        config = self.config
        frames = self.tile_cameras(batch)
        latents = self.vae.encode(frames).to(self.parameters_device)
        context = self.layout.n_context
        batch_size = latents.shape[0]
        device = latents.device

        # At rollout the context chunks' actions are known (they have been
        # executed) and the rest are what we are solving for. The mask means a
        # predicted chunk never reads the clean slots of its own or later
        # chunks, so what sits there does not reach it; zeros are honest for the
        # unknown part rather than merely convenient.
        clean_actions = torch.zeros(
            batch_size,
            config.n_chunks,
            config.chunk_size,
            self.action_dim,
            device=device,
        )
        known = batch.get(ACTION)
        if known is not None:
            past = known[:, : context * config.chunk_size]
            clean_actions[:, :context] = past.unflatten(1, (context, config.chunk_size))
        video_shape = (
            batch_size,
            config.predicted_chunks * config.latent_frames_per_chunk,
            self.vae.latent_channels,
            config.latent_size,
            config.latent_size,
        )
        video = torch.randn(video_shape, device=device)
        actions = torch.randn(
            batch_size,
            config.predicted_chunks,
            config.chunk_size,
            self.action_dim,
            device=device,
        )

        total = steps or config.num_inference_steps
        dt = -1.0 / total
        state = batch.get(OBS_STATE)
        for step in range(total):
            t = 1.0 + step * dt
            time = torch.full((batch_size,), t, device=device)
            video_velocity, action_velocity = self._denoise(
                latents, clean_actions, video, actions, time, time, state
            )
            video = video + dt * video_velocity
            actions = actions + dt * action_velocity
        return video, actions

    @torch.no_grad()
    def predict_future_frames(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """The predicted future as IMAGES, for measuring against what happened."""
        video, _ = self.predict_future(batch, **kwargs)
        return self.vae.decode(video)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if not self._queue:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._queue.extend(chunk.transpose(0, 1))
        return self._queue.popleft()

    # ── keep the frozen VAE out of the checkpoint ───────────────────────────

    def state_dict(self, *args: Any, **kwargs: Any):  # noqa: D102
        state = super().state_dict(*args, **kwargs)
        return {
            key: value for key, value in state.items() if not key.startswith("_vae.")
        }
