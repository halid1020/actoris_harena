"""One vector per observation, from a policy's own vision encoder.

What :mod:`actoris_harena.analysis.recordings` compares. Each policy is asked for
the representation its action head is conditioned on, taken at the last point
where it is still a per-image summary:

* **ACT** -- the shared ResNet backbone's feature map, averaged over space.
* **Diffusion / flow matching** -- each camera's ``DiffusionRgbEncoder`` output
  (spatial-softmax keypoints through a linear layer), which is what conditions
  the denoiser.
* **pi0.5** -- the SigLIP image tokens after PaliGemma's projector, averaged
  over the tokens.
* **DreamZero** -- its frozen image VAE's latent of the tiled frame, pooled to a
  6x6 grid (two cells per camera tile across).
* **FastWAM** -- the Wan VAE latent of its side-by-side frame, pooled to 4x8.

Every camera's vector is scaled to unit length before the cameras are joined,
so a camera whose encoder happens to output larger numbers does not dominate
the distance. The two world models' VAEs are frozen pretrained weights: their
features say what the model was SHOWN, not what our training taught it.

The input is the batch AFTER the checkpoint's own preprocessor, so a crop or a
rename that ran in training runs here too (``Inference.batch`` does that).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

WAM_GRID = {"dreamzero": (6, 6), "fastwam": (4, 8)}


def encoder_kind(policy) -> str:
    """Which of the six encoders this policy has, by structure, not by name."""
    model = getattr(policy, "model", None)
    if getattr(model, "infer_joint", None) is not None:
        return "fastwam"
    if hasattr(policy, "tile_cameras") and hasattr(policy, "vae"):
        return "dreamzero"
    if getattr(model, "paligemma_with_expert", None) is not None:
        return "pi05"
    if getattr(model, "backbone", None) is not None:
        return "act"
    if getattr(policy, "diffusion", None) is not None:
        return "diffusion"
    if hasattr(policy, "rgb_encoder") and hasattr(policy, "image_proj"):
        return "flowmatch"
    raise ValueError(f"no feature adapter for {type(policy).__name__}")


def _unit(x: Tensor) -> Tensor:
    return F.normalize(x.flatten(1).float(), dim=1)


def _last_step(image: Tensor) -> Tensor:
    """``[B, C, H, W]``; a stacked observation window keeps its newest frame."""
    return image[:, -1] if image.ndim == 5 else image


def _camera_images(policy, batch: dict) -> "list[Tensor]":
    return [_last_step(batch[key]) for key in policy.config.image_features]


def _rgb_encoders(policy, encoders, count: int) -> list:
    separate = getattr(policy.config, "use_separate_rgb_encoder_per_camera", False)
    return list(encoders) if separate else [encoders] * count


@torch.no_grad()
def frame_features(policy, batch: dict) -> Tensor:
    """``[B, D]`` features for a preprocessed batch; see the module docstring."""
    kind = encoder_kind(policy)
    if kind == "act":
        maps = [
            policy.model.backbone(image)["feature_map"]
            for image in _camera_images(policy, batch)
        ]
        parts = [_unit(m.mean(dim=(-2, -1))) for m in maps]
    elif kind in ("diffusion", "flowmatch"):
        images = _camera_images(policy, batch)
        root = policy.diffusion if kind == "diffusion" else policy
        encoders = _rgb_encoders(policy, root.rgb_encoder, len(images))
        parts = [_unit(enc(image)) for enc, image in zip(encoders, images)]
    elif kind == "pi05":
        images, _ = policy._preprocess_images(batch)
        embed = policy.model.paligemma_with_expert.embed_image
        parts = [_unit(embed(image).mean(dim=1)) for image in images]
    elif kind == "dreamzero":
        frames = {
            key: batch[key] if batch[key].ndim == 5 else batch[key].unsqueeze(1)
            for key in policy.config.image_features
        }
        tiled = policy.tile_cameras(frames)[:, -1:]
        latent = policy.vae.encode(tiled.to(policy.parameters_device))[:, 0]
        parts = [_unit(F.adaptive_avg_pool2d(latent.float(), WAM_GRID[kind]))]
    else:  # fastwam
        from actoris_harena.policies.fastwam.modeling_fastwam import (
            _input_image_from_batch,
        )

        images = _input_image_from_batch(batch, policy.config)
        latents = []
        for image in images:  # the Wan encoder takes one image at a time
            z = policy.model._encode_input_image_latents_tensor(image.unsqueeze(0))
            latents.append(z.reshape(z.shape[0], z.shape[1], *z.shape[-2:]))
        latent = torch.cat(latents)
        parts = [_unit(F.adaptive_avg_pool2d(latent.float(), WAM_GRID[kind]))]
    return torch.cat([p.cpu() for p in parts], dim=1)


def feature_groups(
    kind: str, image_keys: "list[str]", feature_dim: int
) -> "dict[str, list[int]]":
    """Which entries of a frame's feature vector belong to which camera.

    The policies join one equal block per camera, in ``config.image_features``
    order (what :func:`frame_features` iterates). The two world models pool one
    latent of the whole tiled frame onto a grid, flattened channel-major
    (``[C, rows, cols]``), so a camera is a set of grid CELLS:

    * DreamZero tiles its cameras row-major on the smallest square grid that
      fits them; on the 6x6 pooling grid each of its 3x3 tiles is 2x2 cells.
    * FastWAM sets its inputs side by side in sorted key order; on the 4x8 grid
      each of its two inputs is a 4x4 half.
    """
    names = [key.split(".")[-1] for key in image_keys]
    if not names:
        raise ValueError("no cameras to split the features by")
    if kind in WAM_GRID:
        rows, cols = WAM_GRID[kind]
        cells = rows * cols
        if feature_dim % cells:
            raise ValueError(f"{feature_dim} features do not fill a {rows}x{cols} grid")
        channels = feature_dim // cells
        if kind == "dreamzero":
            side = int(np.ceil(np.sqrt(len(names))))
            if rows % side or cols % side:
                raise ValueError(
                    f"a {rows}x{cols} grid does not split into {side}x{side} tiles"
                )
            tile_r, tile_c = rows // side, cols // side
            boxes = {
                name: (i // side * tile_r, i % side * tile_c, tile_r, tile_c)
                for i, name in enumerate(names)
            }
        else:  # fastwam: side by side, sorted
            ordered = sorted(names)
            if cols % len(ordered):
                raise ValueError(
                    f"{cols} columns do not split into {len(ordered)} inputs"
                )
            width = cols // len(ordered)
            boxes = {
                name: (0, i * width, rows, width) for i, name in enumerate(ordered)
            }
        out = {}
        for name, (top, left, height, width) in boxes.items():
            out[name] = [
                c * cells + r * cols + k
                for c in range(channels)
                for r in range(top, top + height)
                for k in range(left, left + width)
            ]
        return out
    if feature_dim % len(names):
        raise ValueError(
            f"{feature_dim} features do not split into {len(names)} cameras"
        )
    size = feature_dim // len(names)
    return {name: list(range(i * size, (i + 1) * size)) for i, name in enumerate(names)}


#: Camera names that mean the overhead view, under any model's naming.
OVERHEAD_NAMES = ("central", "base_0_rgb")
