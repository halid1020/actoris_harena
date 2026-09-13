"""Conditional flow matching, in the one convention this repo uses.

**Read this before writing a second flow-matching model.** There are two
conventions in the literature and they run time in OPPOSITE directions, so a
model written in one and sampled in the other trains perfectly and predicts
nothing:

* **pi0.5 / openpi** (``lerobot/policies/pi05/modeling_pi05.py:744``) puts NOISE
  at ``t = 1`` and the clean action at ``t = 0``: ``x_t = t*noise + (1-t)*x``,
  target velocity ``u = noise - x``, and sampling integrates *downwards* from 1
  to 0 with a negative step.
* **DreamZero** (arXiv 2602.15922, Eq. 2) puts the CLEAN sample at ``t = 1``:
  ``z_t = t*z_1 + (1-t)*z_0`` with ``z_0`` noise, velocity ``[z_1,a_1]-[z_0,a_0]``,
  integrating *upwards*.

This module implements **pi0.5's**, because the whole point of ``flowmatch`` is to
be pi0.5's objective on a different vision trunk, and because a policy that
matches the reference implementation can be checked against it. DreamZero's
chunk-wise variant reuses :func:`interpolate` and :func:`velocity_target` with its
own time direction, which it states at its own call site rather than silently.

Nothing here knows what is being denoised. Actions, video latents and the two
concatenated all work, which is what lets one implementation serve both models.
"""

from __future__ import annotations

import torch

#: pi0.5's defaults (``configuration_pi05.py:45-48``). Beta(1.5, 1.0) leans
#: towards t = 1, i.e. towards the noisy end, spending more of training where
#: the model has the least to go on. The scale and offset keep t off the exact
#: endpoints, where the velocity target is degenerate.
BETA_ALPHA = 1.5
BETA_BETA = 1.0
TIME_SCALE = 0.999
TIME_OFFSET = 0.001


def sample_time(
    batch: int,
    device: "torch.device | str",
    alpha: float = BETA_ALPHA,
    beta: float = BETA_BETA,
    scale: float = TIME_SCALE,
    offset: float = TIME_OFFSET,
    generator: "torch.Generator | None" = None,
) -> torch.Tensor:
    """One timestep per batch element, in ``[offset, offset + scale]``.

    Beta is sampled on the CPU because ``_sample_dirichlet`` has no MPS kernel --
    the same reason openpi does it there.
    """
    distribution = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32),
        torch.tensor(beta, dtype=torch.float32),
    )
    if generator is None:
        drawn = distribution.sample((batch,))
    else:
        # Beta has no generator argument, so go through the inverse-CDF-free
        # route that does: two Gammas make a Beta. Only reachable when a caller
        # asks for reproducibility, which the samplers and the tests do.
        first = torch._standard_gamma(
            torch.full((batch,), alpha), generator=generator  # type: ignore[call-arg]
        )
        second = torch._standard_gamma(
            torch.full((batch,), beta), generator=generator  # type: ignore[call-arg]
        )
        drawn = first / (first + second)
    return (drawn * scale + offset).to(device=device, dtype=torch.float32)


def interpolate(
    clean: torch.Tensor, noise: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    """``x_t = t*noise + (1-t)*clean`` -- pi0.5's direction, so t=1 is pure noise.

    ``time`` is one value per batch element; it is broadcast over every trailing
    dimension, so this works for an action chunk ``(B, H, A)`` and for a video
    latent ``(B, T, C, H, W)`` alike.
    """
    shape = (time.shape[0],) + (1,) * (clean.ndim - 1)
    expanded = time.reshape(shape)
    return expanded * noise + (1.0 - expanded) * clean


def velocity_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """``u = noise - clean``: what the model is asked to predict, at every t.

    It does not depend on t, which is what makes the objective a regression
    rather than a schedule to be matched.
    """
    return noise - clean


def loss(
    predicted: torch.Tensor, clean: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """Per-element squared error against the velocity target.

    Returned UNREDUCED so a caller can mask padded action dimensions before it
    means anything -- this rig's 12 dimensions sit inside pi0.5's padded 32, and
    averaging over the padding would quietly scale the loss by 12/32.
    """
    return (predicted - velocity_target(clean, noise)) ** 2


@torch.no_grad()
def integrate(
    velocity_fn,
    noise: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Euler-integrate from noise (t=1) to a clean sample (t=0).

    ``velocity_fn(x_t, t)`` returns the predicted velocity; ``t`` is a scalar
    tensor broadcast to the batch, matching how the model saw it in training.
    The step is NEGATIVE because this convention runs time downwards -- the one
    detail that silently produces noise if it is copied from a paper that runs it
    the other way.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    x_t = noise
    dt = -1.0 / steps
    for step in range(steps):
        t = 1.0 + step * dt
        time = torch.full(
            (noise.shape[0],), t, dtype=torch.float32, device=noise.device
        )
        x_t = x_t + dt * velocity_fn(x_t, time)
    return x_t


# ── DreamZero's schedules ────────────────────────────────────────────────────
#
# CONVERSION WARNING. The paper states its schedules in ITS convention, where
# t = 1 is the CLEAN sample. This module runs pi0.5's, where t = 1 is NOISE. So
# every timestep quoted from the paper is mirrored here as ``t_ours = 1 - t_paper``
# and the constants below are given in OUR convention with the paper's number in
# the comment. Copying the paper's formula in unchanged is the single most
# likely way to break this model, and it would show up only as poor results.

#: DreamZero-Flash biases the VIDEO timestep towards high noise (Eq. 5).
#: The paper writes ``t_video = 1 - eta`` with ``eta ~ Beta(7, 1)``, giving
#: ``E[t_video] = 0.125`` in ITS convention -- predominantly noisy. Mirrored into
#: ours that is simply ``t_video ~ Beta(7, 1)``, mean 0.875, likewise noisy.
FLASH_ALPHA = 7.0
FLASH_BETA = 1.0


def sample_uniform_time(
    batch: int, device: "torch.device | str", generator: "torch.Generator | None" = None
) -> torch.Tensor:
    """``t ~ U(0, 1)``: DreamZero's coupled schedule (Eq. 4), and Flash's actions."""
    return torch.rand(batch, device=device, generator=generator, dtype=torch.float32)


def sample_flash_times(
    batch: int,
    device: "torch.device | str",
    alpha: float = FLASH_ALPHA,
    beta: float = FLASH_BETA,
    generator: "torch.Generator | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """``(video_time, action_time)`` for DreamZero-Flash (Eq. 5), in OUR convention.

    The point of decoupling: at inference with very few denoising steps, actions
    must be read off a video context that is still noisy. Training them at the
    SAME timestep never shows the model that situation, so the paper biases video
    towards noise while leaving actions uniform. It recovers most of the 4-step
    quality at 1 step -- 74% against 52% on their table bussing task.

    Returns video time near 1 (noisy) and action time spread over the interval.
    """
    video = torch._standard_gamma(  # type: ignore[call-arg]
        torch.full((batch,), alpha), generator=generator
    )
    other = torch._standard_gamma(  # type: ignore[call-arg]
        torch.full((batch,), beta), generator=generator
    )
    video_time = (video / (video + other)).to(device=device, dtype=torch.float32)
    return video_time, sample_uniform_time(batch, device, generator=generator)
