#!/usr/bin/env python
"""FastWAM with its predicted future exposed, so it can be scored.

Three methods, and all three exist because ``tool/eval_world_model.py`` was
written against DreamZero and asks for DreamZero's shapes. Nothing about the
model changes: ``predict_future_frames`` calls the ``infer_joint`` the port
already has, and the tiling pair is a rename of what FastWAM already does to
put two cameras in one frame.

The plan for this stage said ``predict_future_frames`` was all the scorer
required. It was not -- checked rather than assumed: the scorer also calls
``tile_cameras`` and ``untile_cameras`` and reads two config fields. Shipping
only the one method would have failed on the first line of ``evaluate_frame``.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..fastwam.modeling_fastwam import (
    HarenaFastwamPolicy,
    _batch_to_infer_kwargs,
    _stack_video_from_images,
    batch_device,
)
from .configuration_fastwam_predict import HarenaFastwamPredictConfig


def image_keys_of(batch: dict[str, Tensor]) -> "list[str]":
    """The camera keys, in the order FastWAM concatenates them.

    Sorted, and WITHOUT the ``*_is_pad`` companions that delta-timestamp
    loading adds beside each camera -- they carry the same prefix and are not
    frames. This repeats ``_stack_video_from_images``'s own rule on purpose:
    untiling has to split on exactly the order tiling joined on, and deriving
    it twice from one written-down rule is what keeps them from drifting.
    """
    return sorted(
        key
        for key in batch
        if key.startswith("observation.images.") and not key.endswith("_is_pad")
    )


def decoded_as_tensor(video: Any) -> Tensor:
    """What ``infer_joint`` decoded, as ``[C, T, H, W]`` floats in ``[0, 1]``.

    The Wan core's ``_decode_latents`` returns a LIST OF PIL IMAGES, one per
    frame, ``H x W x 3`` in ``0..255`` -- a preview format, not a tensor.
    ``torch.as_tensor`` cannot read a PIL image at all, and a naive stack would
    produce ``0..255`` against a dataset frame in ``0..1``, so every PSNR would
    be scored against the wrong peak and still look like a number. MEASURED on
    Viking, 2026-09-19: the first run that got this far died here, after the
    model had denoised.

    A tensor is passed through, so a future port that returns one is not
    rescaled twice.
    """
    if isinstance(video, Tensor):
        return video
    frames = [
        torch.from_numpy(np.asarray(frame, dtype=np.uint8).copy()) for frame in video
    ]
    if not frames:
        raise ValueError("infer_joint decoded no frames")
    # [T, H, W, C] uint8 -> [C, T, H, W] float in [0, 1].
    return torch.stack(frames).permute(3, 0, 1, 2).float() / 255.0


class HarenaFastwamPredictPolicy(HarenaFastwamPolicy):
    """HarenaFastwamPolicy that will also hand back the video it predicted."""

    config_class = HarenaFastwamPredictConfig
    name = "harena_fastwam_predict"

    # ── the scorer's vocabulary ─────────────────────────────────────────────

    def tile_cameras(self, batch: dict[str, Tensor]) -> Tensor:
        """Every camera in one frame, ``[B, T, C, H, W]``.

        FastWAM tiles by concatenating cameras along WIDTH, which is already a
        1xN tile; this is that, transposed. ``_stack_video_from_images``
        returns ``[B, C, T, H, W]`` because the Wan core wants channels first,
        and the scorer slices time on axis 1, so the axes are swapped back
        here rather than the scorer being taught a second layout.
        """
        video = _stack_video_from_images(batch, self.config)
        return video.permute(0, 2, 1, 3, 4)

    def untile_cameras(self, tiled: Tensor) -> "dict[str, Tensor]":
        """Split a tiled ``[B, T, C, H, W]`` back into one entry per camera.

        The inverse of :meth:`tile_cameras` and only that: it divides the width
        into equal columns in the order the keys were joined. It needs the key
        list, which a bare tensor does not carry, so the names come from the
        config's ``image_features`` -- the same set, sorted the same way.
        """
        names = sorted(self.config.image_features)
        if not names:
            raise ValueError("no image features on this config to untile into")
        width = tiled.shape[-1]
        if width % len(names):
            raise ValueError(
                f"a tile {width} wide does not divide into {len(names)} camera(s); "
                "the tensor was not produced by tile_cameras on this config"
            )
        each = width // len(names)
        return {
            name: tiled[..., index * each : (index + 1) * each]
            for index, name in enumerate(names)
        }

    @torch.no_grad()
    def reconstruct_frames(self, frames: Tensor) -> Tensor:
        """``[B, T, C, H, W]`` frames in [0, 1] through the Wan VAE and back.

        The ceiling on :meth:`predict_future_frames`, which decodes through the
        same VAE. Wan's VAE is TEMPORAL -- it compresses four frames into one
        latent after the first -- so a clip's length must be 1 + 4k; a single
        observed frame, which is FastWAM's whole context, always is. One clip at
        a time, because ``_decode_latents`` returns the first clip's frames only.
        """
        steps = frames.shape[1]
        if (steps - 1) % 4:
            raise ValueError(f"the Wan VAE takes 1 + 4k frames, got {steps}")
        clips = []
        for clip in frames:
            video = clip.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
            latents = self.model._encode_video_latents(video.to(self.model.device))
            decoded = decoded_as_tensor(self.model._decode_latents(latents))
            clips.append(decoded.permute(1, 0, 2, 3))  # [T, C, H, W]
        return torch.stack(clips).to(device=frames.device, dtype=torch.float32)

    @torch.no_grad()
    def predict_future_frames(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """The video FastWAM thinks comes next, ``[B, T, C, H, W]``.

        ``infer_joint`` decodes both the video and the action; the action is
        dropped here. It runs one observation at a time because the upstream
        signature takes a single prompt and a single first frame, which is also
        how ``predict_action_chunk`` handles a batch.

        The FIRST predicted frame is the conditioning frame reproduced --
        ``infer_joint`` pins ``latents_video[:, :, 0:1]`` to the first frame's
        latents at every denoising step. The scorer compares against the window
        AFTER the context, so that frame is dropped here; leaving it in would
        score the model on reproducing an image it was handed, and it would
        flatter every horizon curve at step one.
        """
        self.eval()
        infer_kwargs = _batch_to_infer_kwargs(batch=batch, config=self.config)
        # The scorer loads a WINDOW of states, because `observation_delta_indices`
        # names the future frames the model is supervised on -- [0, 4, ... 32].
        # `infer_joint` conditions on a single state, so the window is reduced
        # to its FIRST entry: index 0 is the current observation, and the last
        # entry is thirty-two steps into the future. Taking the last would
        # condition the prediction on what it is supposed to predict.
        proprio = infer_kwargs.get("proprio")
        if proprio is not None and proprio.ndim == 3:
            infer_kwargs["proprio"] = proprio[:, 0]
        infer_kwargs.update(
            num_video_frames=self.config.model_video_frames,
            action_horizon=self.config.action_horizon,
            num_inference_steps=self.config.predict_inference_steps,
            seed=self.config.predict_seed,
            # infer_joint cross-checks its action against infer_action's and
            # warns if they differ. That is a second full action rollout for a
            # video measurement that never looks at the action.
            test_action_with_infer_action=False,
        )
        # The shared builder assembles arguments for `infer_action`, and
        # `infer_joint` takes a SUBSET of them -- `compile_action_infer` belongs
        # to the action path alone and reaches here as an unexpected keyword.
        # Filtering against the signature keeps this working as the port tracks
        # upstream, where the two argument lists have drifted before.
        accepted = set(inspect.signature(self.model.infer_joint).parameters)
        unknown = sorted(set(kwargs) - accepted)
        if unknown:
            # A caller's own keyword is never dropped silently: that would turn
            # a misspelling into a setting that appears to apply and does not.
            raise TypeError(
                f"infer_joint does not take {', '.join(unknown)}; it takes "
                f"{', '.join(sorted(accepted))}"
            )
        infer_kwargs.update(kwargs)
        infer_kwargs = {k: v for k, v in infer_kwargs.items() if k in accepted}
        out = self.model.infer_joint(**infer_kwargs)
        video = decoded_as_tensor(out["video"] if isinstance(out, dict) else out[0])
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(
                f"infer_joint returned video with shape {tuple(video.shape)}; "
                "expected [B, C, T, H, W] or [C, T, H, W]"
            )
        video = video.permute(0, 2, 1, 3, 4)
        context = self.config.n_context_chunks * self.config.latent_frames_per_chunk
        return video[:, context:].to(device=batch_device(batch), dtype=torch.float32)
