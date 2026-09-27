"""Take a stream away and see what the policy plans instead.

The behavioural ground truth of this package, and the thing every gradient
method here is scored against. It asks the question directly -- give the policy
the same observation with one camera replaced, and measure how far the plan
moves -- so it needs no assumption about architecture, differentiability or
where a feature lives. What it costs is one forward pass per stream.

TWO DIRECTIONS, and they answer different questions.

*Leave one out* replaces a single stream and keeps the rest. A large effect
means the policy depends on that stream; a SMALL one does not mean it is
unused, only that the others carry the same information. This measures
redundancy, and on a rig with four fingertip cameras there is a lot of it.

*Only one in* replaces every stream but one. A small effect means that stream
alone nearly determines the plan. This measures sufficiency, and it is the half
that catches a camera whose leave-one-out score is near zero purely because its
neighbour says the same thing.

THE BASELINE IS PART OF THE RESULT. There is no neutral image. Zeroing is the
usual choice and is off-manifold -- a policy has never seen a black frame, so
its reaction says as much about surprise as about dependence. Four are offered
and every figure names the one it used; where two baselines disagree about a
stream, that disagreement is the honest answer.

EVERY PLAN HERE IS PINNED. The method compares plans, and a diffusion or
flow-matching policy plans differently twice from one unchanged observation --
MEASURED at 32.4 commanded units against an occlusion effect of 32.5, so
unpinned this measures the sampler and not the input. Each forward pass
therefore goes through :func:`actoris_harena.analysis.diffusion.plan`, which seeds the
global RNG and restores it. On ACT it changes nothing: its latent is sampled
only ``if self.training``, so the plan was already deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from actoris_harena.action_layout import gripper_columns
from actoris_harena.analysis.diffusion import DEFAULT_SEED, plan

#: The gripper channels of a 12-D action, from the recorder's own layout:
#: ``[left5 deg, left_grip, right5 deg, right_grip]`` (``eval_sim_policy
#: .decode_action``). Reported separately everywhere because across these
#: datasets they span a few tenths of open fraction and never reach either end,
#: so a grasp is won or lost in a number a mean over twelve joints hides.

BASELINES = ("zeros", "mean", "blur", "shuffle")


def baseline_frame(
    frame: np.ndarray, kind: str, other: "np.ndarray | None" = None
) -> np.ndarray:
    """One camera's frame, replaced.

    ``zeros``   black. Off-manifold, and the conventional choice.
    ``mean``    this frame's own mean colour: the same brightness, no structure.
                Nearer the manifold than black, and isolates *texture* from
                *illumination* -- which matters on a gel camera, where the whole
                signal is texture.
    ``blur``    heavily blurred: keeps the coarse layout, destroys fine detail.
                The right baseline for asking whether a tactile camera is being
                read for contact texture or merely for "something is there".
    ``shuffle`` a real frame from elsewhere in the episode. Fully on-manifold --
                it is a picture the policy could have been given -- so it
                answers "does it matter WHICH frame this is", which is the
                question a saliency map is usually taken to be answering.
    """
    if kind == "zeros":
        return np.zeros_like(frame)
    if kind == "mean":
        colour = frame.reshape(-1, frame.shape[-1]).mean(axis=0)
        return np.broadcast_to(colour.astype(frame.dtype), frame.shape).copy()
    if kind == "blur":
        import cv2

        # A kernel that is a large fraction of the frame: the point is to leave
        # only the coarse layout, not to denoise.
        size = max(3, (min(frame.shape[:2]) // 8) | 1)
        return cv2.GaussianBlur(frame, (size, size), 0)
    if kind == "shuffle":
        if other is None:
            raise ValueError("the 'shuffle' baseline needs another frame to use")
        if other.shape != frame.shape:
            raise ValueError(
                f"shuffle frame is {other.shape}, not {frame.shape}: it must "
                "come from the same camera"
            )
        return np.asarray(other).copy()
    raise ValueError(f"unknown baseline {kind!r} (want one of {', '.join(BASELINES)})")


def baseline_state(
    state: np.ndarray, kind: str, other: "np.ndarray | None" = None
) -> np.ndarray:
    """The proprioceptive vector, replaced. ``blur`` has no meaning here.

    A state of zeros is not a neutral posture -- it is the arms folded into
    themselves, which is both off-manifold and a specific pose. ``mean`` is the
    safer default and this says so rather than leaving it to be discovered.
    """
    if kind in ("zeros", "blur"):
        return np.zeros_like(state)
    if kind == "mean":
        return np.full_like(state, float(np.mean(state)))
    if kind == "shuffle":
        if other is None:
            raise ValueError("the 'shuffle' baseline needs another state to use")
        return np.asarray(other, dtype=state.dtype).copy()
    raise ValueError(f"unknown baseline {kind!r}")


def chunk_delta(reference: np.ndarray, perturbed: np.ndarray) -> "dict[str, Any]":
    """How far one plan moved from another, in the units the arms are commanded in.

    ``l2`` is the mean per-step Euclidean distance over all twelve channels --
    the headline number. ``per_joint`` and ``gripper`` are there because the
    headline hides exactly what a grasp turns on, and ``cosine`` separates "the
    plan changed shape" from "the plan changed scale", which a distance alone
    cannot.
    """
    reference = np.asarray(reference, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    if reference.shape != perturbed.shape:
        raise ValueError(
            f"chunks differ in shape: {reference.shape} vs {perturbed.shape}"
        )
    difference = perturbed - reference
    per_joint = np.sqrt((difference**2).mean(axis=0))
    grip = [c for c in gripper_columns() if c < reference.shape[1]]
    flat_a, flat_b = reference.ravel(), perturbed.ravel()
    denominator = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    return {
        "l2": float(np.linalg.norm(difference, axis=1).mean()),
        "l2_max": float(np.linalg.norm(difference, axis=1).max()),
        "per_joint": [float(v) for v in per_joint],
        "gripper": float(per_joint[grip].mean()) if grip else 0.0,
        "cosine": float(flat_a @ flat_b / denominator) if denominator else 1.0,
        # Where in the chunk the plans diverge most. Early means the policy
        # reacted at once; late means it changed its mind about the approach.
        "argmax_step": int(np.linalg.norm(difference, axis=1).argmax()),
    }


def _replace(
    state: np.ndarray,
    images: "dict[str, np.ndarray]",
    names: "list[str]",
    kind: str,
    alternative: "dict | None" = None,
) -> "tuple[np.ndarray, dict]":
    """A copy of the observation with the named streams replaced."""
    alternative = alternative or {}
    out_images = dict(images)
    out_state = state
    for name in names:
        if name == "state":
            out_state = baseline_state(state, kind, alternative.get("state"))
        elif name in out_images:
            out_images[name] = baseline_frame(
                out_images[name], kind, alternative.get(name)
            )
    return out_state, out_images


def plan_of(
    inference,
    state: np.ndarray,
    images: "dict[str, np.ndarray]",
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """One plan from one observation, with a stochastic sampler pinned.

    The whole of this module compares plans, so nothing here may call
    ``Inference.chunk`` directly -- that is the unpinned door, and its own
    docstring says so.
    """
    return plan(inference, inference.batch(state, images), seed)


def occlusion(
    inference,
    state: np.ndarray,
    images: "dict[str, np.ndarray]",
    baseline: str = "mean",
    alternative: "dict | None" = None,
    direction: str = "leave_one_out",
    seed: int = DEFAULT_SEED,
) -> "dict[str, Any]":
    """Every stream's effect on this one observation's plan.

    ``direction`` is ``leave_one_out`` (replace this stream) or ``only_one_in``
    (replace every OTHER stream). ``alternative`` supplies the frames the
    ``shuffle`` baseline uses, keyed by stream name.

    ``share`` normalises each effect by the total, so streams can be compared
    across observations whose plans move by wildly different amounts -- but it
    is a share of the measured effects and not of anything conserved, so two
    redundant cameras can each show a small share while together mattering a
    great deal. Read it beside the raw ``l2``.
    """
    names = [s.name for s in inference.streams]
    reference = plan_of(inference, state, images, seed)
    effects: "dict[str, dict]" = {}
    for name in names:
        targets = (
            [name] if direction == "leave_one_out" else [n for n in names if n != name]
        )
        moved_state, moved_images = _replace(
            state, images, targets, baseline, alternative
        )
        effects[name] = chunk_delta(
            reference, plan_of(inference, moved_state, moved_images, seed)
        )

    total = sum(e["l2"] for e in effects.values())
    for effect in effects.values():
        effect["share"] = (effect["l2"] / total) if total else 0.0
    return {
        "direction": direction,
        "baseline": baseline,
        "seed": seed,
        "reference": reference,
        "streams": effects,
        # Everything gone at once: the ceiling the individual effects sit under,
        # and a sanity check -- if removing one stream moves the plan further
        # than removing all of them, the baseline is doing something strange.
        "all": chunk_delta(
            reference,
            plan_of(
                inference,
                *_replace(state, images, names, baseline, alternative),
                seed,
            ),
        ),
    }


def ranking(result: "dict[str, Any]") -> "list[tuple[str, float]]":
    """Streams by effect, largest first. What a bar chart is drawn from."""
    return sorted(
        ((name, effect["l2"]) for name, effect in result["streams"].items()),
        key=lambda pair: -pair[1],
    )


def patch_occlusion(
    inference,
    state: np.ndarray,
    images: "dict[str, np.ndarray]",
    cameras: "list[str]",
    grid: "tuple[int, int]" = (10, 10),
    seed: int = DEFAULT_SEED,
) -> "dict[str, np.ndarray]":
    """WHERE in each frame the plan comes from, for any model at all.

    Occlusion at the scale of a patch (Zeiler and Fergus, 2014): each cell of a
    ``grid`` over the frame is painted with the frame's own mean colour -- the
    ``mean`` baseline, so brightness stays and structure goes -- and the plan is
    re-drawn. The cell's value is how far the plan moved (``chunk_delta``'s
    ``l2``). Grad-CAM needs a convolutional feature map, which pi0.5's vision
    transformer and the world models' video backbones do not offer; this needs
    only a forward pass, so every model gets the same map, measured the same way.

    The cells are laid over the RAW frame, before the model's own pre-processor
    crops anything, so a cropped model's map is in the same coordinates as its
    uncropped twin's and a cell it never sees scores exactly zero. For a tiled
    composite (FastWAM's fingertips) pass a grid that is a multiple of the tile
    grid, so no cell straddles two sensors.
    """
    reference = plan_of(inference, state, images, seed)
    rows, cols = grid
    maps: "dict[str, np.ndarray]" = {}
    for camera in cameras:
        frame = np.asarray(images[camera])
        height, width = frame.shape[:2]
        colour = frame.reshape(-1, frame.shape[-1]).mean(axis=0).astype(frame.dtype)
        values = np.zeros(grid)
        for row in range(rows):
            for col in range(cols):
                top, bottom = row * height // rows, (row + 1) * height // rows
                left, right = col * width // cols, (col + 1) * width // cols
                moved = frame.copy()
                moved[top:bottom, left:right] = colour
                values[row, col] = chunk_delta(
                    reference,
                    plan_of(inference, state, {**images, camera: moved}, seed),
                )["l2"]
        maps[camera] = values
    return maps
