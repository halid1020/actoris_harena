#!/usr/bin/env python
"""DinoViTacFormer: ACT's CVAE with ViTacFormer's visuo-tactile fusion.

See the config's docstring for the method. The model keeps ACT's VAE encoder,
latent and state projections, transformer encoder and decoder and action head --
it subclasses ``ACT`` and inherits them -- and replaces what the encoder reads:

    overhead -> frozen DINOv2 patch tokens ----------\
                                                      +-> fusion_forecast -> encoder -> 8 queries -> forecasts
    fingertips -> ResNet-18 (scratch, shared) -------/                                                 |
                                                       true future (curriculum) or forecast <---------/
                                                                 |
    overhead + fingertips + future-signal tokens -> fusion_action -> ACT encoder -> ACT decoder -> actions

Each fusion is a stack of bidirectional cross-attention layers (vision queries
touch, touch queries vision, simultaneously), then the two token sets are
concatenated -- ViTacFormer's Fig. 3. With no layers it is plain concatenation.

The ACT port is not edited (``tool/port_policies.py --check`` holds it to
upstream); ``forward`` is written out here, its VAE half copied from ACT's.
"""

from __future__ import annotations

import copy
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter

from ..act.modeling_act import (
    ACT,
    ACTDecoder,
    ACTEncoder,
    ACTTemporalEnsembler,
    HarenaActPolicy,
    get_activation_fn,
)
from .configuration_dinovitac import HarenaDinovitacConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_dino(model_id: str) -> nn.Module:
    """The pretrained DINOv2, from the Hugging Face cache (jobs run offline)."""
    from transformers import Dinov2Model

    return Dinov2Model.from_pretrained(model_id)


class CrossAttentionLayer(nn.Module):
    """One direction of the fusion: ``query`` attends to ``context``.

    Pre-norm, with ACT's convention of adding the position embedding to the
    query and the key but not the value. Sequence-first, as ACT is.
    """

    def __init__(self, config: HarenaDinovitacConfig):
        super().__init__()
        d = config.dim_model
        self.attn = nn.MultiheadAttention(d, config.n_heads, dropout=config.dropout)
        self.norm_query = nn.LayerNorm(d)
        self.norm_context = nn.LayerNorm(d)
        self.norm_ffn = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, d)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)

    def forward(
        self, query: Tensor, context: Tensor, query_pos: Tensor, context_pos: Tensor
    ) -> Tensor:
        q = self.norm_query(query)
        kv = self.norm_context(context)
        x = self.attn(q + query_pos, kv + context_pos, value=kv)[0]
        query = query + self.dropout1(x)
        x = self.linear2(
            self.dropout(self.activation(self.linear1(self.norm_ffn(query))))
        )
        return query + self.dropout2(x)


class CrossModalFusion(nn.Module):
    """Bidirectional cross-attention, then concatenation.

    Both directions of a layer read the PREVIOUS layer's tokens, so neither
    modality sees the other's update first. Returns ``(tokens, pos)`` with the
    vision tokens first.
    """

    def __init__(self, config: HarenaDinovitacConfig, n_layers: int):
        super().__init__()
        self.vision_from_touch = nn.ModuleList(
            CrossAttentionLayer(config) for _ in range(n_layers)
        )
        self.touch_from_vision = nn.ModuleList(
            CrossAttentionLayer(config) for _ in range(n_layers)
        )

    def forward(
        self, vision: Tensor, touch: Tensor, vision_pos: Tensor, touch_pos: Tensor
    ) -> "tuple[Tensor, Tensor]":
        for v_layer, t_layer in zip(self.vision_from_touch, self.touch_from_vision):
            vision, touch = (
                v_layer(vision, touch, vision_pos, touch_pos),
                t_layer(touch, vision, touch_pos, vision_pos),
            )
        return torch.cat([vision, touch]), torch.cat([vision_pos, touch_pos])


def _present(x: Tensor, ndim: int) -> Tensor:
    """Frame 0 of a windowed observation, or the observation itself."""
    return x[:, 0] if x.ndim == ndim + 1 else x


class DinoViTacACT(ACT):
    """ACT with DINOv2 + scratch-ResNet tokens, fused, and a tactile forecast."""

    def __init__(self, config: HarenaDinovitacConfig, dino: "nn.Module | None" = None):
        super().__init__(config)
        # ACT's single ResNet and its projection are replaced below.
        del self.backbone
        del self.encoder_img_feat_input_proj
        d = config.dim_model

        self.image_keys = list(config.image_features)
        cameras = set(config.tactile_cameras)
        self.is_tactile = [k.rsplit(".", 1)[-1] in cameras for k in self.image_keys]
        self.n_tactile = sum(self.is_tactile)
        self.offsets = tuple(config.tactile_forecast_offsets)

        self.dino = dino if dino is not None else build_dino(config.dino_model_id)
        self.dino.requires_grad_(False)
        self.dino.eval()
        self.vision_proj = nn.Linear(self.dino.config.hidden_size, d)

        # "Untrained": random initialisation, and GroupNorm because a scratch
        # network has no running statistics worth freezing.
        resnet = torchvision.models.resnet18(
            weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c)
        )
        self.tactile_backbone = IntermediateLayerGetter(
            resnet, return_layers={"layer4": "feature_map"}
        )
        self.tactile_proj = nn.Conv2d(resnet.fc.in_features, d, kernel_size=1)
        # ACT gives every camera the same 2D position embedding; four fingertips
        # through one encoder would then be indistinguishable.
        self.camera_embed = nn.Embedding(len(self.image_keys), d)

        layers = config.cross_attention_layers
        self.fusion_forecast = CrossModalFusion(config, layers)
        self.fusion_action = CrossModalFusion(config, layers)

        forecast_config = copy.copy(config)
        forecast_config.n_encoder_layers = config.forecast_encoder_layers
        forecast_config.n_decoder_layers = 1
        self.forecast_encoder = ACTEncoder(forecast_config)
        self.forecast_decoder = ACTDecoder(forecast_config)
        n_future = self.n_tactile * len(self.offsets)
        h, w = config.tactile_forecast_size
        signal = 3 * h * w
        # One query per (fingertip, offset), fingertip-major.
        self.forecast_queries = nn.Embedding(n_future, d)
        self.forecast_head = nn.Linear(d, signal)
        self.future_proj = nn.Linear(signal, d)
        self.future_embed = nn.Embedding(n_future, d)

        # Counts training forward passes, so the curriculum knows its phase; a
        # buffer, so a resumed run is restored into the phase it left.
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )
        for p in chain(
            self.forecast_encoder.parameters(),
            self.forecast_decoder.parameters(),
            self.fusion_forecast.parameters(),
            self.fusion_action.parameters(),
        ):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()  # frozen: never in training mode
        return self

    # -- observation tokens --------------------------------------------------

    def _cam_tokens(self, feat: Tensor, camera: int) -> "tuple[Tensor, Tensor]":
        pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)
        pos = pos + self.camera_embed.weight[camera].view(1, -1, 1, 1)
        return (
            einops.rearrange(feat, "b c h w -> (h w) b c"),
            einops.rearrange(pos, "b c h w -> (h w) b c"),
        )

    def _vision_features(self, img: Tensor) -> Tensor:
        size = tuple(self.config.dino_image_size)
        x = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = self.dino(pixel_values=x).last_hidden_state
        skip = 1 + int(getattr(self.dino.config, "num_register_tokens", 0) or 0)
        patch = int(self.dino.config.patch_size)
        h, w = size[0] // patch, size[1] // patch
        feat = self.vision_proj(out[:, skip:])  # (B, h*w, D)
        return einops.rearrange(feat, "b (h w) c -> b c h w", h=h, w=w)

    def observation_tokens(
        self, images: "list[Tensor]"
    ) -> "tuple[Tensor, Tensor, Tensor, Tensor]":
        """``(vision, vision_pos, touch, touch_pos)``, each sequence-first."""
        images = [(img - self.pixel_mean) / self.pixel_std for img in images]
        vision, vision_pos, touch, touch_pos = [], [], [], []
        for camera, (img, tactile) in enumerate(zip(images, self.is_tactile)):
            if not tactile:
                tok, pos = self._cam_tokens(self._vision_features(img), camera)
                vision.append(tok)
                vision_pos.append(pos)
        # The four fingertips go through their shared encoder as one batch.
        tactile_idx = [i for i, t in enumerate(self.is_tactile) if t]
        stacked = torch.cat([images[i] for i in tactile_idx])
        fmaps = self.tactile_proj(self.tactile_backbone(stacked)["feature_map"])
        for fmap, camera in zip(fmaps.chunk(len(tactile_idx)), tactile_idx):
            tok, pos = self._cam_tokens(fmap, camera)
            touch.append(tok)
            touch_pos.append(pos)
        return (
            torch.cat(vision),
            torch.cat(vision_pos),
            torch.cat(touch),
            torch.cat(touch_pos),
        )

    # -- the forecast's targets ----------------------------------------------

    def future_signals(self, batch: dict) -> "tuple[Tensor, Tensor] | None":
        """The true future signals ``(B, n_future, 3*h*w)`` and their validity.

        None when the batch carries no future frames (inference, or a caller
        that handed in single frames). Fingertip-major, as the queries are.
        """
        size = tuple(self.config.tactile_forecast_size)
        targets, valid = [], []
        for key, img, tactile in zip(
            self.image_keys, batch[OBS_IMAGES], self.is_tactile
        ):
            if not tactile:
                continue
            if img.ndim != 5 or img.shape[1] < 1 + len(self.offsets):
                return None
            future = img[:, 1 : 1 + len(self.offsets)]  # (B, F, C, H, W)
            small = F.adaptive_avg_pool2d(future.flatten(0, 1), size)
            targets.append(small.reshape(future.shape[0], future.shape[1], -1))
            pad = batch.get(f"{key}_is_pad")
            if pad is None:
                valid.append(
                    torch.ones(future.shape[:2], dtype=torch.bool, device=img.device)
                )
            else:
                valid.append(~pad[:, 1 : 1 + len(self.offsets)].bool())
        return torch.cat(targets, dim=1), torch.cat(valid, dim=1)

    # -- the forward pass ------------------------------------------------------

    def forward(self, batch: "dict[str, Tensor]"):  # type: ignore[override]
        """Actions, the latent's parameters, and the forecast with its targets.

        Returns ``(actions, (mu, log_sigma_x2), extras)`` where ``extras`` has
        ``forecast`` and, when the batch carries future frames, ``target`` and
        ``valid``. Index 0 is the action chunk, as ACT's is.
        """
        config = self.config
        images = batch[OBS_IMAGES]
        present = [_present(img, 4) for img in images]
        state = _present(batch[OBS_STATE], 2) if OBS_STATE in batch else None
        batch_size = present[0].shape[0]
        device = present[0].device

        # The latent, exactly as ACT draws it.
        if config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
            if config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(state)
                parts = [cls_embed, robot_state_embed.unsqueeze(1), action_embed]
            else:
                parts = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(parts, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if config.robot_state_feature else 1),
                False,
                device=device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, config.latent_dim], dtype=torch.float32, device=device
            )

        # The 1-D tokens: latent, then state.
        one_d = [self.encoder_latent_input_proj(latent_sample)]
        if config.robot_state_feature:
            one_d.append(self.encoder_robot_state_input_proj(state))
        one_d_tokens = torch.stack(one_d)  # (n, B, D)
        one_d_pos = self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)  # (n, 1, D)

        vision, vision_pos, touch, touch_pos = self.observation_tokens(present)

        # Pass 1: forecast the future tactile signals.
        fused, fused_pos = self.fusion_forecast(vision, touch, vision_pos, touch_pos)
        tokens = torch.cat([one_d_tokens, fused])
        pos = torch.cat([one_d_pos, fused_pos])
        memory = self.forecast_encoder(tokens, pos_embed=pos)
        queries = self.forecast_queries.weight.unsqueeze(1)
        decoded = self.forecast_decoder(
            torch.zeros(
                (queries.shape[0], batch_size, config.dim_model),
                dtype=pos.dtype,
                device=device,
            ),
            memory,
            encoder_pos_embed=pos,
            decoder_pos_embed=queries,
        )
        forecast = self.forecast_head(decoded).transpose(0, 1)  # (B, n_future, signal)

        # The curriculum: the truth while it lasts, in training only.
        extras: dict = {"forecast": forecast}
        truth = self.future_signals(batch) if self.training else None
        if truth is not None:
            extras["target"], extras["valid"] = truth
        signal = forecast
        if truth is not None and int(self.train_steps) < config.curriculum_steps:
            signal = truth[0]
        if self.training:
            self.train_steps += 1

        # Pass 2: the action, reading the future signals as touch tokens.
        future = self.future_proj(signal).transpose(0, 1)  # (n_future, B, D)
        future_pos = self.future_embed.weight.unsqueeze(1)
        fused, fused_pos = self.fusion_action(
            vision,
            torch.cat([touch, future]),
            vision_pos,
            torch.cat([touch_pos, future_pos]),
        )
        encoder_in_tokens = torch.cat([one_d_tokens, fused])
        encoder_in_pos_embed = torch.cat([one_d_pos, fused_pos])
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (config.chunk_size, batch_size, config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        actions = self.action_head(decoder_out.transpose(0, 1))
        return actions, (mu, log_sigma_x2), extras


class HarenaDinovitacPolicy(HarenaActPolicy):
    """ACT's policy wrapper around DinoViTacACT, with the forecast's loss."""

    config_class = HarenaDinovitacConfig
    name = "harena_dinovitac"

    def __init__(
        self,
        config: HarenaDinovitacConfig,
        dino: "nn.Module | None" = None,
        **kwargs,
    ):
        # Not HarenaActPolicy.__init__: it would build ACT's model (and its
        # ResNet) only for it to be thrown away.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = DinoViTacACT(config, dino=dino)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )
        self.reset()

    def forward(self, batch: "dict[str, Tensor]") -> "tuple[Tensor, dict]":
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions_hat, (mu_hat, log_sigma_x2_hat), extras = self.model(batch)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
        loss = l1_loss
        loss_dict = {"l1_loss": l1_loss.item()}

        if self.config.use_vae and log_sigma_x2_hat is not None:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = loss + mean_kld * self.config.kl_weight

        if "target" in extras:
            err = (
                (extras["forecast"] - extras["target"]).abs().mean(-1)
            )  # (B, n_future)
            valid = extras["valid"].to(err.dtype)
            tactile_l1 = (err * valid).sum() / valid.sum().clamp_min(1)
            loss_dict["tactile_l1"] = tactile_l1.item()
            loss = loss + self.config.tactile_forecast_weight * tactile_l1
        return loss, loss_dict
