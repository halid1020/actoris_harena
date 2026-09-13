"""How good is a predicted future?

The question the world action model exists to answer, and the one the paper
never measures directly -- it reports task progress, from which prediction
quality can only be inferred. In the twin the ground-truth future exists, so it
can be compared frame for frame.

**Report against a baseline or do not report.** A predictor that simply repeats
the current frame scores extremely well on a scene that barely moves -- and a
tactile gel image barely moves until the moment of contact, which is precisely
the moment worth predicting. :func:`held_frame_baseline` is that bar, and
:func:`compare` puts it beside the model on every figure. Without it the numbers
flatter the model exactly where it matters least.

Pure tensor arithmetic: no policy, no GPU requirement, no plotting.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Frames are in [0, 1], so the peak signal is 1.
PEAK = 1.0
#: Below this, PSNR is reported as infinite rather than a huge number that is
#: really just floating-point noise.
_EXACT = 1e-12


def mse(predicted: Tensor, actual: Tensor, per_frame: bool = True) -> Tensor:
    """Mean squared error. ``(B, T, C, H, W)`` in, ``(B, T)`` or scalar out."""
    if predicted.shape != actual.shape:
        raise ValueError(
            f"shapes differ: {tuple(predicted.shape)} vs {tuple(actual.shape)}"
        )
    squared = (predicted - actual) ** 2
    return squared.mean(dim=(-3, -2, -1)) if per_frame else squared.mean()


def psnr(predicted: Tensor, actual: Tensor, per_frame: bool = True) -> Tensor:
    """Peak signal-to-noise ratio in dB. Higher is better; 20 dB is recognisable."""
    error = mse(predicted, actual, per_frame=per_frame)
    return 10.0 * torch.log10(PEAK**2 / error.clamp(min=_EXACT))


def ssim(
    predicted: Tensor, actual: Tensor, window: int = 11, sigma: float = 1.5
) -> Tensor:
    """Structural similarity, per frame, in [-1, 1]. Higher is better.

    Gaussian-windowed, as Wang et al. define it -- a uniform window is the common
    shortcut and it produces visible blocking in the map. Implemented here rather
    than pulled from scikit-image so it runs batched on whatever device the
    frames are already on, which matters when this is called per rollout step.
    """
    if predicted.shape != actual.shape:
        raise ValueError(
            f"shapes differ: {tuple(predicted.shape)} vs {tuple(actual.shape)}"
        )
    batch, steps, channels, height, width = predicted.shape
    if min(height, width) < window:
        raise ValueError(
            f"frames are {height}x{width}, smaller than the {window}px window"
        )

    coords = (
        torch.arange(window, dtype=predicted.dtype, device=predicted.device)
        - window // 2
    )
    gaussian = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] @ gaussian[None, :]).expand(channels, 1, window, window)

    import torch.nn.functional as F  # noqa: N812  (local: keeps the module import-light)

    def blur(x: Tensor) -> Tensor:
        return F.conv2d(x, kernel, groups=channels)

    left = predicted.flatten(0, 1)
    right = actual.flatten(0, 1)
    mu_left, mu_right = blur(left), blur(right)
    mu_left_sq, mu_right_sq = mu_left**2, mu_right**2
    mu_cross = mu_left * mu_right

    sigma_left = blur(left * left) - mu_left_sq
    sigma_right = blur(right * right) - mu_right_sq
    sigma_cross = blur(left * right) - mu_cross

    c1, c2 = (0.01 * PEAK) ** 2, (0.03 * PEAK) ** 2
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_left_sq + mu_right_sq + c1) * (sigma_left + sigma_right + c2)
    return (numerator / denominator).mean(dim=(1, 2, 3)).unflatten(0, (batch, steps))


def held_frame_baseline(context: Tensor, horizon: int) -> Tensor:
    """The last observed frame, repeated -- "nothing changes".

    The bar every prediction must clear. On a near-static view it is a strong
    predictor, which is the point: a model that only matches it has learned the
    scene's stillness and not its dynamics.
    """
    return context[:, -1:].expand(-1, horizon, -1, -1, -1).contiguous()


def compare(predicted: Tensor, actual: Tensor, context: Tensor) -> "dict[str, Tensor]":
    """Model and held-frame baseline on the same frames, per horizon step.

    Every value is ``(T,)`` -- averaged over the batch, kept per horizon step,
    because prediction quality falling with horizon is the shape of the result
    and a single mean hides it.
    """
    baseline = held_frame_baseline(context, actual.shape[1])
    return {
        "psnr": psnr(predicted, actual).mean(dim=0),
        "psnr_baseline": psnr(baseline, actual).mean(dim=0),
        "ssim": ssim(predicted, actual).mean(dim=0),
        "ssim_baseline": ssim(baseline, actual).mean(dim=0),
        "mse": mse(predicted, actual).mean(dim=0),
        "mse_baseline": mse(baseline, actual).mean(dim=0),
    }


def beats_baseline(result: "dict[str, Tensor]") -> Tensor:
    """Per horizon step: did the model actually beat "nothing changes"?

    The headline number. A world model that does not clear this on a given
    horizon has not learned that part of the dynamics, whatever its PSNR says.
    """
    return result["psnr"] > result["psnr_baseline"]
