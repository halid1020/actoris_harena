"""What a stochastic multi-step sampler needs that ACT does not.

ACT plans deterministically at eval: its VAE latent is sampled only
``if self.training``, so two identical observations give identical chunks and an
occlusion delta is entirely the perturbation's doing. **A diffusion policy is
not like that**, and by a wide margin. MEASURED on a real checkpoint: two plans
from ONE unchanged observation differ by 32.4 in commanded units, while
occluding the overhead camera entirely moves the plan by 32.5. Attribute without
pinning the sampler and **99 % of what you measure is the sampler**.

The same applies to any flow-matching policy (pi0.5, ``so101_flowmatch``,
``so101_dreamzero``): they integrate from noise too.

**How it is pinned, and why not the obvious way.** The obvious fix is to hand
the policy a fixed starting ``noise`` -- both ``predict_action_chunk`` and
``generate_actions`` accept one. That is not enough, and believing it is leaves
the numbers just as wrong while looking fixed. The scheduler is a
``DDPMScheduler``: it draws FRESH noise at every one of its denoising steps
(``modeling_diffusion.py:265``, the variance term), and only
``conditional_sample`` takes a ``generator`` -- which neither wrapper forwards.
VERIFIED: identical starting noise and identical conditioning still give
different plans.

So the pin has to cover every draw wherever it happens, which means the global
RNG. :func:`pinned` sets it and then **restores the state it found**, so nothing
else in the process observes a jump -- the usual objection to seeding globally,
answered rather than ignored.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import numpy as np

#: The seed every attribution run uses unless told otherwise. Its value does not
#: matter; that it is the same across the runs being compared does.
DEFAULT_SEED = 0


#: Config fields a policy seeds its OWN generator from, ignoring the global one.
#: FastWAM draws its noise from ``torch.Generator().manual_seed(inference_seed)``,
#: so pinning only the global seed changes nothing: MEASURED 2026-09-26, three
#: "seeds" gave one answer to every digit, which read as a sampler with no
#: spread and was in fact a seed that never arrived.
SEED_FIELDS = ("inference_seed", "predict_seed")


@contextlib.contextmanager
def pinned(inference: Any, seed: int = DEFAULT_SEED) -> Iterator[None]:
    """Make everything inside draw the same randoms, and leave no trace.

    The state is saved on the way in and restored on the way out -- including on
    an exception -- so a caller's own RNG stream is exactly where it was. Without
    that, attributing a policy would silently reseed whatever else shares the
    process, which is the reason seeding globally is usually a bad idea.
    """
    torch = inference.torch
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    config = getattr(getattr(inference, "policy", None), "config", None)
    saved = {
        name: getattr(config, name) for name in SEED_FIELDS if hasattr(config, name)
    }
    try:
        torch.manual_seed(seed)
        for name in saved:
            setattr(config, name, seed)
        yield
    finally:
        for name, value in saved.items():
            setattr(config, name, value)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def is_stochastic(inference: Any, batch: "dict | None" = None) -> bool:
    """Does this policy plan differently twice from one observation? MEASURED.

    A probe rather than a list of policy names: the list would be wrong the day
    a policy is added, and wrong quietly. Given no batch it falls back to asking
    the signature whether a starting ``noise`` can be supplied at all, which is
    the same question asked of the interface instead of the behaviour.
    """
    if batch is None:
        import inspect

        try:
            signature = inspect.signature(inference.policy.predict_action_chunk)
        except (TypeError, ValueError):
            return False
        return "noise" in signature.parameters

    first = inference.chunk_from(dict(batch))
    second = inference.chunk_from(dict(batch))
    return not np.allclose(first, second, atol=1e-6)


def fixed_noise(inference: Any, batch_size: int = 1, seed: int = DEFAULT_SEED):
    """The starting noise, identical every call.

    Shaped from the policy's own config, so it is right before any forward pass
    has happened. ``horizon`` is diffusion's name for the length it denoises,
    which is NOT ``n_action_steps`` -- it plans a longer chunk and executes a
    prefix -- and using the shorter number is a shape error deep inside the
    scheduler.

    On its own this pins only the starting point; see the module docstring.
    """
    torch = inference.torch
    config = inference.cfg
    horizon = int(
        getattr(config, "horizon", None) or getattr(config, "chunk_size", None) or 1
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (batch_size, horizon, inference.action_dim),
        generator=generator,
        dtype=torch.float32,
    )
    return noise.to(inference.device)


def plan(inference: Any, batch: dict, seed: int = DEFAULT_SEED) -> np.ndarray:
    """One chunk, repeatably, whatever the policy's sampler does.

    Use this anywhere two plans are compared -- which is every method in this
    package except the gradients, and those go through
    :meth:`Inference.chunk_tensor`, which pins the same way.
    """
    with pinned(inference, seed):
        return inference.chunk_from(batch)


def sampler_spread(inference: Any, batch: dict, trials: int = 8) -> float:
    """How much this policy's answer moves when NOTHING changes.

    The floor under every attribution on this checkpoint: an occlusion effect
    smaller than this is indistinguishable from the sampler. Deliberately
    UNPINNED -- it is measuring exactly what :func:`plan` removes -- and reported
    beside the results rather than assumed negligible, because on a diffusion
    policy it is not.
    """
    plans = [inference.chunk_from(dict(batch)) for _ in range(trials)]
    stacked = np.stack(plans)
    return float(np.abs(stacked - stacked.mean(axis=0)).mean())
