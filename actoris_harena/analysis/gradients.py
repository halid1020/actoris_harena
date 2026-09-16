"""Attribution by differentiating the plan with respect to what produced it.

Occlusion (:mod:`actoris_harena.analysis.perturb`) answers "does this policy use that
stream" by taking it away. Gradients answer a finer question -- *which pixels*,
and how much each contributed -- and they do it at a resolution occlusion
cannot reach without one forward pass per region.

Three methods, and each is here for a reason the others do not cover.

**Integrated gradients** (Sundararajan, Taly & Yan, 2017) is the primary
quantitative one, because of its completeness axiom: the attributions sum to
``f(x) - f(baseline)``. That is what makes per-stream totals COMPARABLE -- five
cameras' shares are five parts of one whole rather than five unrelated numbers,
which is exactly what a plain gradient cannot give. It is also what makes the
implementation testable: :func:`completeness_error` checks the axiom holds, and
if it does not, the integration path is wrong.

**SmoothGrad** (Smilkov et al., 2017) averages gradients over noised copies of
the input. Plain gradients on a ResNet trunk are too noisy to read as a picture;
this is what makes an overlay legible, and it is offered for looking at, not for
ranking.

**Grad-CAM** (Selvaraju et al., 2016) weights the last convolutional feature map
by the gradient flowing into it. It answers *where in this tactile image*, at
the resolution the network actually reasons at, for the price of one backward
pass. It needs a convolutional trunk, so it runs on ACT and on diffusion but
NOT on a token model: pi0.5 and the flow-matching policies have no feature map
to weight, and integrated gradients is what they get instead. Where the trunk
is, and how many times it runs, differs between the two -- :func:`cam_trunks`.

Hand-rolled on torch autograd rather than pulled from captum: each is a few
dozen lines, and this repo's venv is heavy enough already.

NOTHING HERE IS TRUSTED ON ITS OWN. Saliency methods can be insensitive to the
model entirely -- see :func:`randomise_weights` and the sanity check it exists
for -- and are reported beside the occlusion effect they should agree with.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def target_norm(chunk):
    """The whole plan's magnitude. The default scalar to attribute.

    A chunk is 100 actions of 12 channels and a gradient needs one number.
    This asks "what made the policy plan a movement of this size", which is the
    broadest honest summary; :func:`target_gripper` asks the narrower question
    that a grasp actually turns on.
    """
    return chunk.norm()


def target_gripper(chunk, columns=(5, 11)):
    """The gripper channels alone: what opened or closed the hands.

    Separate because across these datasets the gripper channels span a few
    tenths of open fraction and never reach either end, so their contribution to
    a norm over twelve channels is negligible -- and a grasp is won or lost
    there. An attribution of the norm can be dominated by a large arm sweep
    while saying nothing about the fingers.
    """
    index = [c for c in columns if c < chunk.shape[-1]]
    return chunk[..., index].abs().sum()


TARGETS: "dict[str, Callable]" = {"norm": target_norm, "gripper": target_gripper}


class NoFeatureMap(RuntimeError):
    """Grad-CAM cannot run on this policy, and no amount of memory would help.

    Distinct from a plain ``RuntimeError`` on purpose. ``torch.cuda.OutOfMemory
    Error`` IS a ``RuntimeError``, so a caller catching that broad type recorded
    an OOM as "Grad-CAM unavailable" -- which reads as *this method does not
    apply to this model*, the one conclusion a reader must not draw from a full
    GPU. MEASURED: a diffusion pass sharing a card with a pi0.5 LoRA run wrote a
    deck of numbers and no figures, and said nothing was wrong.
    """


def _grad_of(inference, batch, keys, target):
    """One forward + backward: the gradient of ``target`` at each named key."""
    torch = inference.torch
    live = dict(batch)
    for key in keys:
        live[key] = live[key].detach().clone().requires_grad_(True)
    inference.policy.zero_grad(set_to_none=True)
    scalar = target(inference.chunk_tensor(live))
    grads = torch.autograd.grad(scalar, [live[k] for k in keys], allow_unused=True)
    return (
        {
            k: (g.detach() if g is not None else torch.zeros_like(live[k]))
            for k, g in zip(keys, grads)
        },
        float(scalar.detach()),
    )


def integrated_gradients(
    inference,
    batch: dict,
    baseline_batch: dict,
    keys: "list[str] | None" = None,
    steps: int = 64,
    target: "Callable | str" = "norm",
) -> "dict[str, Any]":
    """Attribute the plan to each input, with attributions that sum to the change.

    The path runs in the NORMALISED tensor space the model actually sees, from
    ``baseline_batch`` to ``batch``. That is deliberate: interpolating raw pixels
    and normalising each step would follow a different path than the one the
    axiom is stated over, and completeness would then fail for a reason that
    looks like a bug in the integration.

    ``steps`` is the Riemann sum's resolution, and the default was MEASURED
    rather than picked. On the five-camera ACT checkpoint, one real frame, the
    completeness error falls 0.29 (16 steps) -> 0.22 (32) -> 0.15 (64) ->
    0.017 (128) -> 0.008 (256): the axiom holds and this model is simply
    non-linear enough to need a fine path. Over 206 frames of six episodes the
    error at 64 steps averages 0.20 and reaches 0.87 on the worst frame, so it
    varies a great deal with the observation and is worth reading per result
    rather than assuming from that one convergence sweep.

    The per-stream SHARES -- what is actually reported -- converge far sooner
    than the sum does: the largest share moves by 0.0016 between 64 steps and
    128, and by 0.0001 between 128 and 256. So 64 is the default (about 16 s a
    frame on a laptop GPU) and the completeness error is returned with every
    result, so the reader can see how coarse the path was rather than take the
    default on trust. Raise it for a figure that has to carry the axiom.
    """
    torch = inference.torch
    if isinstance(target, str):
        target = TARGETS[target]
    keys = list(keys if keys is not None else inference.image_keys())

    totals = {k: torch.zeros_like(batch[k]) for k in keys}
    for step in range(steps):
        # Midpoints: the trapezoid's error is O(1/steps^2) against the left
        # rule's O(1/steps), for the same number of forward passes.
        alpha = (step + 0.5) / steps
        point = dict(batch)
        for key in keys:
            point[key] = baseline_batch[key] + alpha * (
                batch[key] - baseline_batch[key]
            )
        grads, _ = _grad_of(inference, point, keys, target)
        for key in keys:
            totals[key] += grads[key]

    attributions = {
        key: ((batch[key] - baseline_batch[key]) * totals[key] / steps).detach()
        for key in keys
    }
    with torch.no_grad():
        f_x = float(target(inference.chunk_tensor(batch)))
        f_base = float(target(inference.chunk_tensor(baseline_batch)))
    summed = float(sum(a.sum() for a in attributions.values()))
    return {
        "attributions": attributions,
        "f_x": f_x,
        "f_baseline": f_base,
        "sum": summed,
        "completeness_error": completeness_error(f_x, f_base, summed),
        "steps": steps,
    }


def completeness_error(f_x: float, f_baseline: float, summed: float) -> float:
    """How far the attributions are from summing to ``f(x) - f(baseline)``.

    Relative, so it can be compared across targets of different magnitudes. The
    axiom is exact for the true integral; what is measured here is the Riemann
    sum's error, so a few per cent at 32 steps is expected and a large value
    means the path is wrong rather than merely coarse.
    """
    expected = f_x - f_baseline
    scale = max(abs(expected), 1e-9)
    return abs(summed - expected) / scale


def smoothgrad(
    inference,
    batch: dict,
    keys: "list[str] | None" = None,
    samples: int = 16,
    noise: float = 0.15,
    target: "Callable | str" = "norm",
) -> "dict[str, Any]":
    """Gradients averaged over noised copies, so a map can be looked at.

    ``noise`` is a fraction of each input's own range, following the paper: a
    fixed absolute sigma means something different for a normalised image than
    for a state vector in degrees.
    """
    torch = inference.torch
    if isinstance(target, str):
        target = TARGETS[target]
    keys = list(keys if keys is not None else inference.image_keys())
    spread = {
        key: float(noise * (batch[key].max() - batch[key].min()).abs()) for key in keys
    }
    totals = {k: torch.zeros_like(batch[k]) for k in keys}
    for _ in range(samples):
        noisy = dict(batch)
        for key in keys:
            noisy[key] = batch[key] + torch.randn_like(batch[key]) * spread[key]
        grads, _ = _grad_of(inference, noisy, keys, target)
        for key in keys:
            totals[key] += grads[key]
    return {
        "attributions": {k: (v / samples).detach() for k, v in totals.items()},
        "samples": samples,
        "noise": noise,
    }


def cam_trunks(inference) -> "tuple[list[Any], str]":
    """The module(s) whose output is a spatial feature map, and how they run.

    Three shapes, because the two architectures reach their trunk differently
    and diffusion reaches it two ways depending on one config flag:

    ``"per_call"`` -- ACT. ``policy.model.backbone`` is ONE module the encoder
    calls once per camera in a loop, so the hook fires once per camera in the
    policy's own order.

    ``"per_module"`` -- diffusion with ``use_separate_rgb_encoder_per_camera``
    (what this rig trains). ``rgb_encoder`` is an ``nn.ModuleList``, one encoder
    per camera, and the list itself is never called -- hooking it collects
    nothing, which is why this used to raise rather than draw. Each encoder's
    ``.backbone`` is hooked instead.

    ``"interleaved"`` -- diffusion with one shared encoder. It runs ONCE on a
    batch flattened ``b s n -> (b s n)``, so a single activation carries every
    camera and has to be de-interleaved.

    Note it is the encoder's ``.backbone`` that is hooked, never the encoder:
    ``DiffusionRgbEncoder.forward`` returns a pooled ``(B, D)`` vector with no
    spatial dimensions left to draw.
    """
    policy = inference.policy
    backbone = getattr(getattr(policy, "model", None), "backbone", None)
    if backbone is not None:
        return [backbone], "per_call"
    encoder = getattr(getattr(policy, "diffusion", None), "rgb_encoder", None)
    if encoder is None:
        raise NoFeatureMap(
            f"no ResNet trunk found on a '{inference.type}' policy. A token model "
            "(pi0.5, the flow-matching policies) has no convolutional feature map "
            "for Grad-CAM to weight; use integrated gradients on it instead."
        )
    if isinstance(encoder, inference.torch.nn.ModuleList):
        return [e.backbone for e in encoder], "per_module"
    return [encoder.backbone], "interleaved"


def grad_cam(
    inference,
    batch: dict,
    target: "Callable | str" = "norm",
) -> "dict[str, np.ndarray]":
    """Where in each camera's frame the plan came from, at feature-map resolution.

    A forward hook on the ResNet trunk collects one activation per camera, IN
    THE ORDER the policy stacks them, and the backward pass gives the matching
    gradients. The map is the gradient-weighted channel sum, rectified: the
    negative part answers a different question (what would have increased the
    target had it been absent) and mixing the two makes an unreadable picture.

    See :func:`cam_trunks` for how the trunk is found -- it is not the same
    object, nor the same number of calls, on ACT and on diffusion.
    """
    if isinstance(target, str):
        target = TARGETS[target]
    trunks, mode = cam_trunks(inference)

    activations: "list[Any]" = []

    def keep(_module, _inputs, output):
        tensor = output["feature_map"] if isinstance(output, dict) else output
        tensor.retain_grad()
        activations.append(tensor)
        return output

    handles = [trunk.register_forward_hook(keep) for trunk in trunks]
    try:
        live = dict(batch)
        inference.policy.zero_grad(set_to_none=True)
        scalar = target(inference.chunk_tensor(live))
        scalar.backward()
    finally:
        for handle in handles:
            handle.remove()

    names = [k.split(".")[-1] for k in inference.image_keys()]
    per_camera = _cam_activations(inference, activations, len(names), mode)
    if len(per_camera) != len(names):
        raise NoFeatureMap(
            f"the trunk ran {len(activations)} time(s) and yielded "
            f"{len(per_camera)} map(s) for {len(names)} camera(s) in '{mode}' "
            "mode: the map cannot be matched to a camera"
        )
    maps: "dict[str, np.ndarray]" = {}
    for name, (activation, grad) in zip(names, per_camera):
        if grad is None:
            continue
        weights = grad.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * activation).sum(dim=0).clamp(min=0)
        cam = cam.detach().to("cpu").numpy()
        peak = cam.max()
        maps[name] = (cam / peak) if peak > 0 else cam
    return maps


def _cam_activations(inference, activations, n_cameras: int, mode: str):
    """One ``(activation, gradient)`` pair per camera, each ``(C, H, W)``.

    The leading batch axis is dropped here rather than in the caller because
    where it comes from differs per mode: ACT and the per-camera encoders are
    given ``(b*s, C, H, W)`` and the observation window holds copies of one
    frame, so index 0 is that frame; the shared encoder is given every camera
    of every step at once and has to be reshaped before any of it means
    anything.
    """
    if mode == "interleaved":
        if len(activations) != 1:
            return []
        whole = activations[0]
        grad = whole.grad
        steps = int(inference.n_obs_steps) or 1
        rest = whole.shape[1:]
        batch = whole.shape[0] // max(steps * n_cameras, 1)
        if batch * steps * n_cameras != whole.shape[0]:
            return []
        shaped = whole.view(batch, steps, n_cameras, *rest)[0, 0]
        shaped_grad = (
            grad.view(batch, steps, n_cameras, *rest)[0, 0]
            if grad is not None
            else None
        )
        return [
            (shaped[i], shaped_grad[i] if shaped_grad is not None else None)
            for i in range(n_cameras)
        ]
    return [(a[0], a.grad[0] if a.grad is not None else None) for a in activations]


def per_stream(attributions: "dict[str, Any]", inference) -> "dict[str, float]":
    """Fold per-pixel attributions up to one number per stream.

    The SUM of absolute values, not the mean: a stream is being asked how much
    it contributed in total, and a camera with more pixels genuinely does carry
    more of the input. Dividing by pixel count would answer "how attributed is
    the average pixel", which is a different question and would rank a small
    frame above a large one for the same total influence.
    """
    out: "dict[str, float]" = {}
    for key, value in attributions.items():
        out[key.split(".")[-1]] = float(value.abs().sum())
    return out


def shares(totals: "dict[str, float]") -> "dict[str, float]":
    """Per-stream attributions as fractions of the whole. Sums to one."""
    whole = sum(totals.values())
    return {k: (v / whole if whole else 0.0) for k, v in totals.items()}


def randomise_weights(inference, seed: int = 0) -> None:
    """Re-initialise the policy's weights, IN PLACE, for the sanity check.

    From Adebayo et al., *Sanity Checks for Saliency Maps* (2018): a method
    whose output barely changes when the model is randomised is measuring the
    input, not what the model learned from it -- and several widely used methods
    fail this. The check is only meaningful on a throwaway copy, so this is
    deliberately destructive and the caller is expected to reload afterwards.
    """
    torch = inference.torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in inference.policy.parameters():
            if parameter.dim() > 1:
                flat = torch.empty(parameter.shape, device="cpu")
                torch.nn.init.xavier_uniform_(flat, generator=generator)
                parameter.copy_(flat.to(parameter.device))
            else:
                parameter.zero_()


def rank_agreement(a: "dict[str, float]", b: "dict[str, float]") -> float:
    """Spearman correlation between two rankings of the same streams.

    How an attribution method is scored: against the occlusion effect, which is
    behaviour rather than inference about behaviour. Agreement is evidence the
    method is reading the policy; disagreement is a result to report, not a
    number to bury.
    """
    names = sorted(set(a) & set(b))
    if len(names) < 2:
        return float("nan")
    ranks_a = _ranks([a[n] for n in names])
    ranks_b = _ranks([b[n] for n in names])
    first = np.asarray(ranks_a) - np.mean(ranks_a)
    second = np.asarray(ranks_b) - np.mean(ranks_b)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator else float("nan")


def _ranks(values: "list[float]") -> "list[float]":
    """Ranks with ties averaged, which is what Spearman requires."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2.0
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks
