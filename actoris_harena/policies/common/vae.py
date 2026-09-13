"""The frozen image VAE DreamZero's latents live in.

DreamZero builds on Wan2.1-I2V-14B, whose VAE compresses video in TIME as well
as space, and freezes it ("we update all DiT blocks, the state encoder, action
encoder, and action decoder, while freezing the text encoder, image encoder, and
VAE"). We keep the freezing and drop the temporal compression: this is a
per-frame image VAE, Stable Diffusion's, at 8x spatial compression.

**That is a deviation and it is stated as one.** A temporal VAE would let one
latent frame stand for several raw frames, which is how the paper fits 33 raw
frames into 8 latent frames. Per-frame, a latent frame is a frame. The
consequence is a shorter visual context for the same token budget, not a
different objective -- but it is a reason our absolute numbers are not theirs,
and it belongs in the limitations rather than in a footnote.

The VAE is never trained, so it is excluded from the checkpoint: it is 83.7M
parameters that would be written into every save and are identical in all of
them. It is fetched from the Hub on first use, which means a compute node needs
it staged -- see ``hpc/README.md``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

#: Stable Diffusion's fine-tuned VAE. 8x spatial compression, 4 latent channels:
#: a 224x224 frame becomes 4x28x28, which is 196 tokens at patch size 2.
DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"
DOWNSAMPLE = 8
LATENT_CHANNELS = 4


class FrozenImageVae(nn.Module):
    """Encode frames to latents and back. Frozen, eval-mode, never checkpointed."""

    def __init__(
        self, model_id: str = DEFAULT_VAE, device: "str | torch.device" = "cpu"
    ):
        super().__init__()
        from diffusers import AutoencoderKL

        self.model_id = model_id
        vae = AutoencoderKL.from_pretrained(model_id)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(device)
        # The latent distribution of this VAE has a standard deviation far from
        # 1; the scaling factor is what puts it near 1 so a diffusion model sees
        # a well-conditioned target. Skipping it trains, badly.
        self.scaling = float(vae.config.scaling_factor)

    def train(self, mode: bool = True):  # noqa: D102 - stays in eval, always
        return super().train(False)

    @property
    def latent_channels(self) -> int:
        return LATENT_CHANNELS

    def latent_hw(self, height: int, width: int) -> "tuple[int, int]":
        if height % DOWNSAMPLE or width % DOWNSAMPLE:
            raise ValueError(
                f"{height}x{width} is not divisible by the VAE's {DOWNSAMPLE}x "
                "compression; pick a frame size that is"
            )
        return height // DOWNSAMPLE, width // DOWNSAMPLE

    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        """``(B, T, 3, H, W)`` in [0, 1] to ``(B, T, 4, H/8, W/8)``, scaled.

        The MEAN of the latent distribution, not a sample: the stochasticity that
        makes a VAE a VAE would inject noise into the very target the flow
        matching objective is trying to regress.
        """
        batch, steps = frames.shape[:2]
        flat = frames.flatten(0, 1)
        # The VAE was trained on [-1, 1]; our datasets are [0, 1].
        latents = self.vae.encode(flat * 2.0 - 1.0).latent_dist.mean * self.scaling
        return latents.unflatten(0, (batch, steps))

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """``(B, T, 4, h, w)`` back to frames in [0, 1], for looking at."""
        batch, steps = latents.shape[:2]
        flat = latents.flatten(0, 1) / self.scaling
        frames = (self.vae.decode(flat).sample + 1.0) / 2.0
        return frames.clamp(0.0, 1.0).unflatten(0, (batch, steps))
