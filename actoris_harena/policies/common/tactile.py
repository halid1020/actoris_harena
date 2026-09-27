"""Crop a vision-based tactile sensor down to the part of it that touches things.

WHY. Grad-CAM on the finished five-camera ACT checkpoint shows the policy
attending to the **edges** of the fingertip images -- `right_arm_right_gripper`
saturates along its left edge and top-right corner while the gel centre stays
cold -- and it does so in frames where nothing is in contact. That is the
signature of light leaking in at the gel boundary, a known failure of
vision-based tactile sensors: the border is bright, it moves with the ambient
light rather than with the object, and a network will happily learn it.

AND WHY IT CROPS ROWS ONLY. Measuring the frames rather than assuming the
obvious shape changed the design. The rim is real -- the outermost lines are
7.4 % to 22.2 % brighter than the frame's dimmest -- but it is a smooth vignette
with no boundary to find, and, more awkwardly, **horizontally the bright rim and
the responsive region are the same pixels**: on two of the four sensors the
columns whose temporal variation is in the top quartile run right to the frame
edge. A centred width crop cannot take the leak without taking signal with it.
Vertically every camera has room to spare. So the default crops height and
leaves width alone, and the numbers behind that are in `DEFAULT_CROP`.

WHAT IT DOES, and the one decision worth arguing about: it crops the central
fraction of each tactile frame and then **resizes back to the source size**.
Resizing back costs a little interpolation blur and buys the thing that makes
this a one-file change instead of a survey:

* ACT's per-camera token count stays 300, so `analysis/streams.py` keeps
  tiling. It assumes every camera contributes the same token width, and
  cropping only the fingertips would break that assumption for the very
  analysis this change is measured by.
* The diffusion encoder sizes its feature dimension from a dummy input at
  construction time; an unchanged input shape cannot disagree with it.
* pi0.5 letterboxes with `resize_with_pad`, so a changed aspect ratio would
  change how much padding each slot gets -- a second, uncontrolled difference
  between the cropped run and its baseline.

AND THE ARM THAT SEPARATES THEM. Because the crop removes the rim and stretches
what is left in one operation, every result about cropping is a result about
both. `resize=False` does the first alone, at the cost of the three conveniences
above. MEASURED on a CPU before any of it reached a queue: the action-chunking
family builds and runs with the fingertip cameras shorter than the overhead one;
the diffusion family REFUSES, because LeRobot validates that every camera has
the same shape and raises before a model is built. So the arm exists for one
family and not the other, and that is a property of the architectures rather
than of this rig. pi0.5 would letterbox the shorter image differently, which is
a second uncontrolled difference, so it is left on the default too.

WHERE IT RUNS. As a processor step at index 0 of the preprocessor, ahead of
`RenameObservationsProcessorStep`. That ordering is load-bearing on pi0.5,
whose rename map turns `observation.images.left_arm_left_gripper` into
`observation.images.left_wrist_0_rgb` -- run the crop afterwards and it would
look for camera names that no longer exist and silently do nothing. The
preprocessor runs during training as well as inference
(`lerobot_train.py` calls it on every batch), so this changes what the policy is
trained on, not merely what it is shown at deployment.

WHAT IS NOT HERE. Contact gating -- feeding tactile only once contact is
established -- was considered and deliberately left out. "Stable contact" is a
temporal predicate and training shuffles frames, so a processor step sees a
batch and never an episode; the repo's own contact segmentation
(`analysis/phases.py`) needs a whole episode because it thresholds on quantiles
of that episode's own gripper channels. A training-time gate would have to be
re-derived from the tactile image against a no-contact reference, with its own
threshold to justify. That is a separate piece of work, not a flag on this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn.functional as F
from lerobot.processor import ObservationProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_IMAGES

#: The four fingertip cameras on this rig. There is no `tactile` flag on a
#: camera anywhere -- `dataset_view.COMPOSITES` hard-codes the same four names
#: for the same reason -- so the set is named here and overridable per run.
TACTILE_CAMERAS: "tuple[str, ...]" = (
    "left_arm_left_gripper",
    "left_arm_right_gripper",
    "right_arm_left_gripper",
    "right_arm_right_gripper",
)

#: Fraction of HEIGHT and of WIDTH kept, centred. MEASURED on
#: `fold-short-from-flattend-tactile` with `tool/measure_tactile_border.py`, and
#: the asymmetry is the measurement's doing, not a convenience:
#:
#:   camera                    safe row crop   safe col crop   edge brightness
#:   left_arm_left_gripper              0.62            1.00           +12.2 %
#:   left_arm_right_gripper             0.45            0.82           +10.9 %
#:   right_arm_left_gripper             0.53            0.81            +7.4 %
#:   right_arm_right_gripper            0.55            1.00           +22.2 %
#:
#: "Safe" is the tightest centred crop that keeps every line whose temporal
#: variation is in the frame's top quartile -- the lines where the gel actually
#: responds. HORIZONTALLY there is no headroom: on two of the four sensors the
#: most active columns run to the frame edge, so a centred width crop removes
#: signal and rim together. Vertically every camera tolerates 0.62 or tighter.
#:
#: So: crop rows, keep columns. 0.80 sits comfortably inside the 0.62 bound on
#: the tightest camera and removes the brightest rows on all four.
DEFAULT_CROP = (0.80, 1.00)
#: Where the kept box sits, as a (row, column) fraction of the image.
CENTRED = (0.5, 0.5)


def as_fractions(fraction) -> "tuple[float, float]":
    """Accept one number or a (height, width) pair; return the pair.

    A scalar still works because "crop both sides by this much" is the obvious
    thing to reach for and refusing it would be pedantic -- but the default is a
    pair, because on this rig the two axes have genuinely different headroom.
    """
    if isinstance(fraction, (int, float)):
        return (float(fraction), float(fraction))
    height, width = fraction
    return (float(height), float(width))


def crop_box(
    height: int, width: int, fraction, centre=CENTRED
) -> "tuple[int, int, int, int]":
    """The box keeping the given fractions: ``(top, left, h, w)``.

    Centred unless ``centre`` -- the box's centre as a ``(row, column)``
    fraction of the image -- says otherwise; a box that would run off the image
    is slid back inside it. Rounded to at least one pixel, so a silly fraction
    produces a small image rather than an empty tensor and a shape error three
    modules away.
    """
    fh, fw = as_fractions(fraction)
    for value in (fh, fw):
        if not 0 < value <= 1:
            raise ValueError(
                f"tactile crop fraction must be in (0, 1], got {fraction!r}. "
                "1.0 keeps that axis whole; (1.0, 1.0) is the uncropped baseline."
            )
    keep_h = max(1, int(round(height * fh)))
    keep_w = max(1, int(round(width * fw)))
    cy, cx = as_fractions(centre)
    if (cy, cx) == CENTRED:
        # The integer arithmetic every centred result so far was measured with.
        return ((height - keep_h) // 2, (width - keep_w) // 2, keep_h, keep_w)
    for value in (cy, cx):
        if not 0 <= value <= 1:
            raise ValueError(f"tactile crop centre must be in [0, 1], got {centre!r}")
    top = min(max(0, int(round(cy * height - keep_h / 2))), height - keep_h)
    left = min(max(0, int(round(cx * width - keep_w / 2))), width - keep_w)
    return (top, left, keep_h, keep_w)


def crop_and_restore(image: torch.Tensor, fraction, centre=CENTRED) -> torch.Tensor:
    """Crop (centred by default) and resize straight back, so the shape never changes.

    Accepts any leading batch dimensions; the last three are ``(C, H, W)``.
    """
    if min(as_fractions(fraction)) >= 1.0:
        return image
    height, width = image.shape[-2], image.shape[-1]
    top, left, keep_h, keep_w = crop_box(height, width, fraction, centre)
    cropped = image[..., top : top + keep_h, left : left + keep_w]
    lead = cropped.shape[:-3]
    flat = cropped.reshape(-1, *cropped.shape[-3:])
    resized = F.interpolate(
        flat.float(), size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.reshape(*lead, *resized.shape[-3:]).to(image.dtype)


def crop_only(image: torch.Tensor, fraction, centre=CENTRED) -> torch.Tensor:
    """Crop (centred by default) and leave it cropped, so the shape DOES change.

    The arm that separates the two things the default crop does at once. Every
    result about cropping in this work is a result about rim removal AND the
    anisotropic stretch that resizing back applies; this removes the rim and
    nothing else, at the cost of the three conveniences the module docstring
    lists. Accepts any leading batch dimensions; the last three are ``(C, H, W)``.
    """
    if min(as_fractions(fraction)) >= 1.0:
        return image
    height, width = image.shape[-2], image.shape[-1]
    top, left, keep_h, keep_w = crop_box(height, width, fraction, centre)
    return image[..., top : top + keep_h, left : left + keep_w]


def crop_tiles_and_restore(
    image: torch.Tensor, fraction, rows: int, cols: int, centre=CENTRED
) -> torch.Tensor:
    """Crop every tile of a composite on its own, then re-tile.

    FastWAM reads the four fingertips as ONE 2x2 composite (``tactile_quad``,
    built at staging by ``recording.dataset_view``). A centred crop of the whole
    composite would trim the outer edge of each tile and leave the four inner
    edges -- where the tiles meet, and where each sensor's rim is just as bright
    -- untouched. So each tile is cropped about its own centre and resized back
    into its own quadrant. The tile grid is ``height // rows`` by
    ``width // cols``, the same arithmetic the composite was written with.
    """
    if min(as_fractions(fraction)) >= 1.0:
        return image
    height, width = image.shape[-2], image.shape[-1]
    tile_h, tile_w = height // rows, width // cols
    out = image.clone()
    for row in range(rows):
        for col in range(cols):
            rs = slice(row * tile_h, (row + 1) * tile_h)
            cs = slice(col * tile_w, (col + 1) * tile_w)
            out[..., rs, cs] = crop_and_restore(image[..., rs, cs], fraction, centre)
    return out


@ProcessorStepRegistry.register(name="so101_tactile_crop")
@dataclass
class HarenaTactileCropProcessorStep(ObservationProcessorStep):
    """Centre-crop the tactile cameras, leave every other camera alone.

    Registered so it round-trips through `policy_preprocessor.json`: a
    checkpoint that was trained cropped must be *served* cropped, and the
    pipeline is rebuilt from that file by name. `actoris_harena.policies` has to be
    imported for the name to resolve, which
    `--policy.discover_packages_path=actoris_harena.policies` already guarantees
    everywhere these policies are trained or served.
    """

    fraction: "float | tuple[float, float]" = DEFAULT_CROP
    cameras: "tuple[str, ...]" = field(default_factory=lambda: TACTILE_CAMERAS)
    #: Resize back to the source size after cropping. True is the default and
    #: what every result so far was measured with. False leaves the image
    #: cropped, which separates rim removal from the stretch -- and gives up
    #: the three conveniences the module docstring lists, so it is an
    #: experiment and not a better setting.
    resize: bool = True
    #: Composite cameras and their tile grid, ``{"tactile_quad": (2, 2)}``: each
    #: tile is cropped about its own centre (see `crop_tiles_and_restore`).
    #: Empty for every policy that reads the fingertips as separate cameras.
    tiled: "dict[str, tuple[int, int]]" = field(default_factory=dict)
    #: Centre of the kept box, ``(row, column)`` as fractions of the image (of
    #: each tile, for a composite). Off-centre lets a crop sit to one side of a
    #: feature, such as the vertical ridge a third of the way across every gel.
    centre: "tuple[float, float]" = CENTRED

    def __post_init__(self):
        crop_box(64, 64, self.fraction, self.centre)  # refuse a bad box now
        self.fraction = as_fractions(self.fraction)
        self.centre = as_fractions(self.centre)
        self.cameras = tuple(self.cameras)
        self.tiled = {name: (int(r), int(c)) for name, (r, c) in self.tiled.items()}
        if self.tiled and not self.resize:
            # A tile cannot shrink inside a composite without moving its
            # neighbours; refusing here beats a shape error in the model.
            raise ValueError("a tiled composite can only be cropped with resize=True")

    def _is_tactile(self, key: str) -> bool:
        # Matched on the SHORT name so the same step works whether the key is
        # `observation.images.<name>` or a bare `<name>`.
        return key.rsplit(".", 1)[-1] in self.cameras

    def observation(self, observation: "dict[str, Any]") -> "dict[str, Any]":
        if min(as_fractions(self.fraction)) >= 1.0:
            return observation
        out = dict(observation)
        for key, value in observation.items():
            if not isinstance(value, torch.Tensor):
                continue
            if not key.startswith(f"{OBS_IMAGES}."):
                continue
            grid = self.tiled.get(key.rsplit(".", 1)[-1])
            if grid is not None:
                out[key] = crop_tiles_and_restore(
                    value, self.fraction, *grid, centre=self.centre
                )
            elif self._is_tactile(key):
                out[key] = (
                    crop_and_restore(value, self.fraction, self.centre)
                    if self.resize
                    else crop_only(value, self.fraction, self.centre)
                )
        return out

    def get_config(self) -> "dict[str, Any]":
        return {
            "fraction": list(as_fractions(self.fraction)),
            "cameras": list(self.cameras),
            "resize": bool(self.resize),
            "tiled": {name: list(grid) for name, grid in self.tiled.items()},
            "centre": list(self.centre),
        }

    def transform_features(self, features):
        # With `resize` on, the shape is deliberately unchanged -- see the
        # module docstring -- and there is nothing to declare. With it off the
        # tactile cameras really are smaller, and a policy built from features
        # that still claimed the source size would construct its encoder around
        # an image it never receives.
        if self.resize or min(as_fractions(self.fraction)) >= 1.0:
            return features
        for key, feature in features.items():
            if not (key.startswith(f"{OBS_IMAGES}.") and self._is_tactile(key)):
                continue
            shape = tuple(feature.shape)
            if len(shape) != 3:
                continue
            channels, height, width = shape
            _, _, keep_h, keep_w = crop_box(height, width, self.fraction, self.centre)
            features[key] = replace(feature, shape=(channels, keep_h, keep_w))
        return features
