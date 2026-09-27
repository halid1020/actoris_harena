"""The figures. Everything here reads a result dict and draws; nothing infers.

Kept apart from the methods so a figure can be redrawn from ``attribution.json``
without loading a checkpoint or touching a GPU -- which is what makes the numbers
worth writing down separately from the pictures.

matplotlib only, no seaborn: it is already a dependency and this needs nothing
it does not have. Colours are assigned per stream and reused across every figure
in a run, so `central` is the same colour in the time plot and the bar chart.
"""

from __future__ import annotations

import numpy as np

#: One colour per stream, assigned in the policy's own camera order so the
#: overhead camera keeps the same colour whatever else is enabled. Chosen to
#: stay distinguishable in greyscale print, which is where these end up.
PALETTE = (
    "#1f77b4",  # central
    "#d62728",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#7f7f7f",
)
PHASE_INK = {
    "reaching": "#00000000",
    "closing": "#f59e0b33",
    "holding": "#22c55e33",
    "opening": "#3b82f633",
    "released": "#00000000",
}


def colours(names: "list[str]") -> "dict[str, str]":
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}


def _figure(width=9.0, height=4.5):
    import matplotlib

    matplotlib.use("Agg")  # no display on a compute node, and none wanted
    import matplotlib.pyplot as plt

    return plt, plt.subplots(figsize=(width, height))


def shade_phases(axis, labels, spans_of, x=None) -> None:
    """Wash the gripper phases behind a time plot, and name them once each.

    `spans_of` counts in POSITIONS along `labels`, while the curves in front are
    drawn against whatever x the caller plotted -- a frame number, a time. Pass
    that same x here and the bands are converted onto it. Without it the bands
    are drawn at 0, 1, 2, ... and land in the corner of a plot whose x runs to
    tens of thousands, which is a wrong figure rather than an ugly one: it
    invites a reader to date a change to the wrong moment.
    """
    seen = set()
    for phase, start, stop in spans_of(labels):
        ink = PHASE_INK.get(phase, "#00000000")
        if ink.endswith("00"):
            continue
        left, right = start, stop - 1
        if x is not None and len(x):
            left = x[min(start, len(x) - 1)]
            right = x[min(stop - 1, len(x) - 1)]
        axis.axvspan(
            left,
            right,
            color=ink,
            lw=0,
            label=phase if phase not in seen else None,
        )
        seen.add(phase)


def contribution_over_time(
    path, frames, series, labels=None, spans_of=None, title="", ylabel="share of effect"
) -> str:
    """Each stream's contribution across an episode, with the phases behind it.

    The headline figure: it is where "tactile matters at contact and nowhere
    else" either shows up or fails to.
    """
    plt, (fig, axis) = _figure(height=5.2)
    ink = colours(list(series))
    if labels and spans_of:
        shade_phases(axis, labels, spans_of, x=frames)
    for name, values in series.items():
        axis.plot(frames, values, label=name, color=ink[name], lw=1.8)
    axis.set_xlabel("frame")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25, lw=0.5)
    # Outside the axes on purpose: these curves cross the whole plot area, and a
    # legend inside it covered the very frames where the streams trade places.
    axis.legend(
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def stream_bars(path, ranked, title="", xlabel="effect on the plan") -> str:
    """Streams by effect, largest at the top. What occlusion is read from."""
    plt, (fig, axis) = _figure(height=0.5 * len(ranked) + 1.8)
    names = [name for name, _ in ranked][::-1]
    values = [value for _, value in ranked][::-1]
    ink = colours(list(reversed(names)))
    axis.barh(names, values, color=[ink[n] for n in names])
    for index, value in enumerate(values):
        axis.text(value, index, f" {value:.3g}", va="center", fontsize=8)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def stream_joint_heatmap(path, per_joint, joint_names, title="") -> str:
    """Stream x joint: which input drives which channel. Grippers marked.

    Rows are normalised individually, because one stream moving every joint a
    lot and another moving one joint a little are both worth seeing, and a
    shared scale hides the second entirely.
    """
    plt, (fig, axis) = _figure(width=9.0, height=0.45 * len(per_joint) + 2.2)
    names = list(per_joint)
    grid = np.array([per_joint[n] for n in names], dtype=float)
    peaks = grid.max(axis=1, keepdims=True)
    grid = np.divide(grid, peaks, out=np.zeros_like(grid), where=peaks > 0)
    image = axis.imshow(grid, aspect="auto", cmap="magma")
    axis.set_xticks(range(len(joint_names)))
    axis.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=8)
    for column, label in enumerate(joint_names):
        if "grip" in label:
            axis.axvline(column, color="#22d3ee", lw=1.2, alpha=0.8)
    axis.set_title(title + "  (rows normalised; gripper columns marked)")
    fig.colorbar(image, ax=axis, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def within_chunk(path, series, title="") -> str:
    """Contribution against position in the plan: does reliance shift along it?"""
    plt, (fig, axis) = _figure()
    ink = colours(list(series))
    for name, values in series.items():
        axis.plot(values, label=name, color=ink[name], lw=1.6)
    axis.set_xlabel("action index within the chunk")
    axis.set_ylabel("attention (deviation from uniform)")
    axis.axhline(0.0, color="#888", lw=0.8, ls="--")
    axis.set_title(title)
    axis.grid(alpha=0.25, lw=0.5)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def overlays(path, frames, maps, title="") -> str:
    """Grad-CAM over each camera's frame, side by side at one moment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in frames if n in maps]
    if not names:
        return ""
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.4))
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names):
        frame = frames[name]
        axis.imshow(frame)
        heat = maps[name]
        axis.imshow(
            heat,
            cmap="jet",
            alpha=0.45,
            interpolation="bilinear",
            extent=(0, frame.shape[1], frame.shape[0], 0),
        )
        axis.set_title(name, fontsize=8)
        axis.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def agreement(path, methods, title="") -> str:
    """Every method's per-stream share, side by side. Disagreement is the point."""
    plt, (fig, axis) = _figure()
    names = sorted({s for shares in methods.values() for s in shares})
    width = 0.8 / max(len(methods), 1)
    for index, (method, shares) in enumerate(methods.items()):
        offset = (index - (len(methods) - 1) / 2) * width
        axis.bar(
            [i + offset for i in range(len(names))],
            [shares.get(n, 0.0) for n in names],
            width=width,
            label=method,
        )
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axis.set_ylabel("share of the total")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25, lw=0.5)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)
