#!/usr/bin/env python3
"""How much of a tactile frame is sensor, and how much is the light leaking in.

The crop fraction the cropped policies train at has to come from the data. This
measures the border rather than guessing at it, and prints the numbers the
choice rests on so the choice can be argued with.

    venv/bin/python tool/measure_tactile_border.py \\
        --dataset ~/.cache/huggingface/lerobot/local/fold-short-from-flattend-tactile \\
        --episodes 0-5

WHAT IT MEASURES, and why these two statistics. A vision-based tactile sensor's
useful signal is *deformation*: the gel moves when it touches something, so the
pixels that carry information are the ones whose value CHANGES over an episode.
The light leaking in at the boundary does the opposite -- it is bright and it is
temporally flat, because it tracks the room rather than the object. So:

* **temporal standard deviation** per pixel, over the frames sampled. High in
  the gel, low at the housing edge.
* **mean luminance** per pixel. High at the leak, mid-range in the gel.

A row or column is "border" when its temporal variation falls below a fraction
of that profile's own PEAK -- relative, not absolute, because exposure differs
per camera and the four fingertips are not identically lit. The reported
fraction is the largest CENTRED box that excludes every such row and column,
and it is reported per camera so a single bad sensor is visible rather than
averaged away.

MEASURED on `fold-short-from-flattend-tactile`, and it changes how this output
should be read: **the border is a smooth vignette, not a band.** Luminance falls
monotonically from both edges to the middle (0.675 to 0.559 on
`right_arm_right_gripper`, a 21 % rise at the edge) and temporal variation rises
the same way (0.045 to 0.068). There is no sharp boundary to find, so no
threshold at which the answer stops moving -- which is why the report prints a
SENSITIVITY row. Read that row before quoting the headline number: it is where
the judgement lives, and a crop fraction chosen without looking at it is an
opinion wearing a measurement's clothes.

Reports, never writes: the number goes into `So101ActCropConfig.tactile_crop`
by hand, because it is a decision about an experiment and deserves to be made
once and written down.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from actoris_harena.policies.common.tactile import TACTILE_CAMERAS  # noqa: E402


def profiles(frames: np.ndarray) -> "dict[str, np.ndarray]":
    """Per-row and per-column temporal variation and brightness.

    ``frames`` is ``(T, H, W, 3)`` in [0, 1]. Reduced to one number per row and
    per column because that is the shape of the decision -- a centred crop takes
    whole rows and columns off, so a per-pixel map would have to be reduced to
    this anyway, and reducing first makes the numbers printable.
    """
    grey = frames.mean(axis=-1)
    variation = grey.std(axis=0)
    brightness = grey.mean(axis=0)
    return {
        "row_var": variation.mean(axis=1),
        "col_var": variation.mean(axis=0),
        "row_lum": brightness.mean(axis=1),
        "col_lum": brightness.mean(axis=0),
    }


def keep_fraction(border: np.ndarray) -> float:
    """Largest centred fraction with no border line at either end.

    ``border`` is a boolean per line. Only a RUN of border lines reaching the
    edge is trimmed: a border line in the middle of the frame is not a border,
    it is the criterion misfiring, and trimming to exclude it would eat the
    frame from the outside for something that is not at the outside.

    The larger of the two margins wins, because a centred crop cannot be
    lopsided -- trimming by the smaller one would leave the leak on one side,
    which is the whole thing being removed.
    """
    n = len(border)
    low = 0
    while low < n and border[low]:
        low += 1
    if low == n:  # every line flagged: the criterion has told us nothing
        return 1.0
    high = n - 1
    while high > low and border[high]:
        high -= 1
    margin = max(low, n - 1 - high)
    return max(0.0, (n - 2 * margin) / n)


#: Fractions of the PEAK temporal variation to try as the "this is still sensor"
#: line. A range rather than one value because, MEASURED on this rig, the border
#: is a smooth vignette and not a band: there is no threshold at which the answer
#: stops moving, so quoting a single fraction would hide the judgement inside it.
SENSITIVITY = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15)


def measure(frames: np.ndarray, excess: float) -> "dict[str, Any]":
    """One camera's answer, the evidence beside it, and its sensitivity.

    ``excess`` is how much brighter than the frame's dimmest line a line may be
    before it counts as border. Luminance and not temporal variation: MEASURED
    on this rig, variation peaks near both SIDES of the frame and is lowest in
    the middle, because the gel responds in two lobes -- so a centred crop keyed
    on it would remove the responsive part and keep the quiet part, which is
    backwards. The leak is a brightness phenomenon and is measured as one.
    """
    prof = profiles(frames)
    edge = min(len(prof["row_var"]), len(prof["col_var"])) // 20 or 1

    def at(excess: float) -> "tuple[float, float]":
        """Trim while a line is more than ``excess`` brighter than the plateau."""
        return tuple(  # type: ignore[return-value]
            keep_fraction(prof[key] > float(prof[key].min()) * (1.0 + excess))
            for key in ("row_lum", "col_lum")
        )

    height, width = at(excess)
    return {
        "keep_height": height,
        "keep_width": width,
        "keep": min(height, width),
        # The evidence for the claim that the border is bright and flat. If these
        # do not separate, the crop is not justified by this dataset and the
        # fraction should stay 1.0 rather than being invented.
        "border_variation": float(np.mean(prof["row_var"][:edge])),
        "centre_variation": float(np.max(prof["row_var"])),
        "border_luminance": float(np.mean(prof["row_lum"][:edge])),
        "centre_luminance": float(np.min(prof["row_lum"])),
        # Reported, NOT used as the criterion. MEASURED on this rig: the column
        # profile peaks near BOTH sides and is quietest in the middle -- the gel
        # responds in two lobes, as the Grad-CAM figures show -- so a centred
        # crop keyed on variation would cut away the responsive part. Variation
        # is evidence about where the sensor works; luminance is evidence about
        # the leak, and the leak is what is being removed.
        "variation_is_centred": bool(
            prof["col_var"].argmax() > len(prof["col_var"]) * 0.25
            and prof["col_var"].argmax() < len(prof["col_var"]) * 0.75
        ),
        # How much the answer depends on where the line is drawn. A flat row
        # would mean a real edge; a moving one means a gradient and a judgement.
        "sensitivity": {f"{f:.2f}": min(at(f)) for f in SENSITIVITY},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="A LeRobotDataset root")
    parser.add_argument("--episodes", default="0-5", help="e.g. 0-5 or 0,3,7")
    parser.add_argument("--every", type=int, default=10, help="Sample 1 frame in N")
    parser.add_argument(
        "--cameras",
        default=",".join(TACTILE_CAMERAS),
        help="Comma list of camera short names to measure",
    )
    parser.add_argument(
        "--quiet",
        dest="excess",
        type=float,
        default=0.06,
        help="A line is border when it is this much brighter than the frame's "
        "dimmest line (default 0.06 = 6%%). The border here is a gradient, not "
        "a band, so this is a judgement -- read the sensitivity table under the "
        "result before quoting the headline number",
    )
    parser.add_argument("--json", default=None, help="Also write the numbers here")
    args = parser.parse_args()

    from actoris_harena.analysis.sources import DatasetSource

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    source = DatasetSource(args.dataset, args.episodes, args.every, cameras)
    print(f"📁 {source.describe()}")

    # `rows()` returns row INDICES; `observations()` is the one that yields
    # frames, as (index, state, images, action) with images already HWC uint8.
    stacks: "dict[str, list[np.ndarray]]" = {c: [] for c in cameras}
    for episode in source.episodes:
        for _index, _state, images, _action in source.observations(episode):
            for camera, frame in images.items():
                if camera in stacks:
                    stacks[camera].append(np.asarray(frame, dtype=np.float32))

    results = {}
    for camera in cameras:
        if not stacks[camera]:
            print(f"⚠️  {camera}: no frames")
            continue
        frames = np.stack(stacks[camera])
        if frames.shape[1] in (1, 3):  # (T, C, H, W) -> (T, H, W, C)
            frames = np.transpose(frames, (0, 2, 3, 1))
        if frames.max() > 1.5:
            frames = frames / 255.0
        results[camera] = measure(frames, args.excess)
        r = results[camera]
        print(
            f"  {camera:26s} keep {r['keep']:.2f} "
            f"(h {r['keep_height']:.2f} w {r['keep_width']:.2f})  "
            f"variation edge {r['border_variation']:.4f} vs peak "
            f"{r['centre_variation']:.4f}  luminance edge "
            f"{r['border_luminance']:.3f} vs dimmest {r['centre_luminance']:.3f} "
            f"(+{100 * (r['border_luminance'] / r['centre_luminance'] - 1):.0f}%)"
        )

    if results:
        print()
        print("  sensitivity — keep fraction against where the line is drawn:")
        print("    threshold " + " ".join(f"{f:>6.2f}" for f in SENSITIVITY))
        for camera, r in results.items():
            row = " ".join(f"{r['sensitivity'][f'{f:.2f}']:>6.2f}" for f in SENSITIVITY)
            print(f"    {camera:24s} {row}")
        # The tightest camera decides: one fraction is trained, and a crop that
        # keeps a leak on one sensor has not solved the problem it exists for.
        chosen = min(r["keep"] for r in results.values())
        print()
        print(f"👉 tactile_crop = {chosen:.2f}   (the tightest of the cameras)")
        separated = all(
            r["border_variation"] < r["centre_variation"] for r in results.values()
        )
        if not separated:
            print(
                "⚠️  the border is NOT quieter than the centre on every camera. "
                "This dataset does not support a crop; leave tactile_crop at 1.0 "
                "rather than cropping on the strength of a figure."
            )
    if args.json and results:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"📝 {args.json}")


if __name__ == "__main__":
    main()
