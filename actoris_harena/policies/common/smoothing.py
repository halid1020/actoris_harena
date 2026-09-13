"""Savitzky-Golay smoothing of a generated action chunk.

DreamZero, Appendix D.3: "first upsampling the action chunk to 2x resolution via
cubic interpolation, then applying a Savitzky-Golay filter (window size 21,
polynomial order 3) to suppress noise while preserving trajectory shape, and
finally downsampling to original resolution."

Why it is needed at all: a chunk that comes out of an iterative denoiser carries
high-frequency noise that a position-controlled arm will happily track, and the
result is audible. Upsampling first is what lets a 21-sample window cover a
short chunk without flattening it.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

WINDOW = 21
POLYNOMIAL_ORDER = 3
UPSAMPLE = 2


def smooth_chunk(
    chunk: torch.Tensor,
    window: int = WINDOW,
    order: int = POLYNOMIAL_ORDER,
    upsample: int = UPSAMPLE,
) -> torch.Tensor:
    """Smooth ``(B, H, A)`` along the horizon. Returns the same shape and device.

    A chunk too short for the window is returned untouched rather than smoothed
    with a silently shrunken one: a filter that adapts its own window is no
    longer the filter the paper specifies, and quietly changing it per chunk
    length would make two runs incomparable.
    """
    if chunk.ndim != 3:
        raise ValueError(f"expected (batch, horizon, action), got {tuple(chunk.shape)}")
    batch, horizon, _ = chunk.shape
    if horizon < 2:
        return chunk

    length = horizon * upsample
    if length < window or window <= order:
        return chunk

    device, dtype = chunk.device, chunk.dtype
    values = chunk.detach().to("cpu", torch.float64).numpy()

    source = np.arange(horizon, dtype=np.float64)
    dense = np.linspace(0.0, horizon - 1, length, dtype=np.float64)
    # cubic needs four points; fall back to linear on a chunk shorter than that
    # rather than raising in the middle of a rollout.
    kind = "cubic" if horizon >= 4 else "linear"

    out = np.empty_like(values)
    for index in range(batch):
        upsampled = interp1d(source, values[index], axis=0, kind=kind)(dense)
        filtered = savgol_filter(
            upsampled, window_length=window, polyorder=order, axis=0
        )
        out[index] = interp1d(dense, filtered, axis=0, kind=kind)(source)
    return torch.from_numpy(out).to(device=device, dtype=dtype)
