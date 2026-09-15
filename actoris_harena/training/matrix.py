"""One training run as a row, and every reason it cannot work.

``hpc/runs.tsv`` has always been the run matrix; this is the same file read in
Python, so the console, ``tool/train_launch.py`` and ``hpc/submit_real.sh`` all
agree about what a row means instead of each parsing it their own way.

The point of the module is ``row_refusals``. A row that cannot work should be
refused HERE -- while the operator is still choosing, before a dataset is
copied and long before a GPU is reserved. Every failure this checks for has
actually happened: a camera that the dataset does not have, a pi0.5 run with
more cameras than the base has slots, a batch size that OOMs a particular card,
and a policy the installed LeRobot has never heard of.

Nothing here opens a dataset, a connection or a GPU: it reads an already-loaded
``info.json`` mapping and a destination dict. That is what lets the same rules
run in a unit test, in the terminal and in a web request.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Iterable, Sequence

from actoris_harena.recording.camera_profile import profile
from actoris_harena.recording.dataset_view import (
    PI05_SLOT_ORDER,
    available_camera_names,
    camera_keys,
    short_name,
)

# The columns of hpc/runs.tsv, in order. `extra` is last because it is the only
# one that may contain spaces -- everything before it is read field by field.
COLUMNS = ("dataset", "policy", "cameras", "steps", "batch", "hours", "slots", "extra")
DEFAULTED = frozenset({"steps", "batch", "slots", "extra"})  # `-` is allowed

# Per-policy defaults. These are the SAME numbers test/system/long_vla_real.sh
# uses; they live there because that script is what runs, and are repeated here
# only so a page can show what a `-` will become. A change belongs in both.
POLICIES: "dict[str, dict[str, Any]]" = {
    "act": {
        "steps": 80000,
        "batch": 8,
        "hours": 36,
        "scratch": True,
        "max_cameras": None,
    },
    "diffusion": {
        "steps": 100000,
        "batch": 32,
        "hours": 36,
        "scratch": True,
        "max_cameras": None,
    },
    "pi05": {
        "steps": 30000,
        "batch": 8,
        "hours": 36,
        "scratch": False,
        # pi0.5 finetunes a base with exactly three pretrained image slots. A
        # fourth camera has nowhere to go; a slot with no camera is padded and
        # masked by pi0.5 itself, which is what an ablation wants.
        "max_cameras": len(PI05_SLOT_ORDER),
    },
    "fastwam": {
        # The docs show 300000 steps, which is sized for a large multi-task
        # corpus; this dataset is tens of thousands of frames, so the default
        # here is finetune-scale and a row may say otherwise.
        "steps": 30000,
        "batch": 8,
        "hours": 36,
        "scratch": True,
        "max_cameras": None,
        # Every image feature must be this high, and their widths must sum to
        # this: two features of 224x224, or one of 224x448. Not a suggestion --
        # the Wan backbone concatenates them into one tensor of this shape.
        "image_size": (224, 448),
        "module": "lerobot.policies.fastwam",
    },
}

# The same three policies again, implemented in THIS repo rather than in LeRobot
# (src/actoris_harena.policies/). They are ports -- the upstream module tree moved, not
# rewritten -- so they train to the same shape and their checkpoints are
# interchangeable with the originals; test/unit/test_policy_ports.py and
# test/integration/test_policy_ports_checkpoints.py hold them to that. They take
# their sizing from the policy they were ported from, because a run that means to
# compare the two must not also change the budget.
#
# `local` marks a policy this repo defines. It changes only what an unavailable
# one is told to do about it: bumping LeRobot cannot fix a module that lives here.
# harena_fastwam is the odd one: its twin is not in the installed LeRobot at all,
# because FastWAM landed upstream after LEROBOT_COMMIT. `fastwam` stays in this
# table and stays refused by the module probe, which is the honest report -- the
# port is what can actually be trained today.
for _ported, _from in (
    ("harena_act", "act"),
    ("harena_diffusion", "diffusion"),
    ("harena_pi05", "pi05"),
    ("harena_fastwam", "fastwam"),
):
    POLICIES[_ported] = {
        **POLICIES[_from],
        "local": True,
        "ported_from": _from,
        "module": f"actoris_harena.policies.{_from}",
    }
del _ported, _from

# pi0.5's flow-matching objective on the diffusion policy's ResNet trunk, and
# nothing else -- no video model, no language. It is the CONTROL for the world
# action model: it shares DreamZero's loss and shares nothing else, so the gap
# between them measures the world-modelling objective with the rest held fixed.
# Small enough to train from scratch in hours, hence a step count between ACT's
# and pi0.5's rather than either.
POLICIES["harena_flowmatch"] = {
    "steps": 60000,
    "batch": 16,
    "hours": 24,
    "scratch": True,
    "max_cameras": None,
    "local": True,
    "module": "actoris_harena.policies.flowmatch",
}

# The world action model. Jointly predicts future video and actions under one
# flow-matching objective, so it learns dynamics from every consecutive frame
# pair rather than only from what an action label reveals. A video target makes
# each step dearer than a plain policy's, hence the smaller batch, and it is
# trained from scratch, hence the step count.
POLICIES["harena_dreamzero"] = {
    "steps": 80000,
    "batch": 8,
    "hours": 36,
    "scratch": True,
    "max_cameras": None,
    "local": True,
    "module": "actoris_harena.policies.dreamzero",
}

# The supervisor's reading of the Grad-CAM figures: the policy attends to the
# EDGES of the fingertip images, including before contact, which is what light
# leaking in at the gel boundary looks like. These crop the tactile cameras to
# the gel centre and are otherwise their twin exactly -- same model, same steps,
# same batch. Holding the budget fixed is not tidiness: `hpc/runs.tsv` requires
# it across an ablation, or capacity confounds input.
for _crop, _twin in (
    ("harena_act_crop", "harena_act"),
    ("harena_diffusion_crop", "harena_diffusion"),
    ("harena_pi05_crop", "harena_pi05"),
):
    POLICIES[_crop] = {
        **POLICIES[_twin],
        "local": True,
        "crop_of": _twin,
        # NOT `ported_from`: these are not ports, and the port tests would then
        # demand a byte-identical upstream file that does not exist.
        "ported_from": None,
        "module": f"actoris_harena.policies.{_crop.removeprefix('harena_')}",
    }
del _crop, _twin

# FastWAM with its predicted future exposed. `wan.modular.infer_joint` decodes
# both video and action and nothing in the policy API calls it, so the port owns
# a video prior it cannot be scored on while `harena_dreamzero` can be scored and
# owns no prior. The subclass renames what FastWAM already does; the model, the
# budget and the two-camera ceiling are its twin's, because a prediction measured
# under a different budget would not be a measurement of the same model.
POLICIES["harena_fastwam_predict"] = {
    **POLICIES["harena_fastwam"],
    "local": True,
    "variant_of": "harena_fastwam",
    # NOT `ported_from`: the subclass is ours, and the port tests would then
    # demand a byte-identical upstream file that does not exist.
    "ported_from": None,
    "module": "actoris_harena.policies.fastwam_predict",
}

POLICY_NAMES = tuple(POLICIES)

#: A repo-local policy and the LeRobot one it was ported from. Their checkpoints
#: load into either implementation, so this is also the map a loader uses to read
#: one through the other.
PORTED_FROM = {
    name: spec["ported_from"]
    for name, spec in POLICIES.items()
    if spec.get("ported_from")
}


#: Every policy that is a variant of another, and which. A measured ceiling
#: carries along this map: a port is the same model as its twin, and a cropped
#: variant is the same model reading an input of the same SHAPE (the crop
#: resizes back), so neither changes what a machine can hold. Chased
#: transitively, because `harena_pi05_crop` reaches `pi05` only through
#: `harena_pi05`.
def _twin_of(spec: dict) -> "str | None":
    """What this policy is a variant OF, whatever kind of variant it is.

    Three kinds so far and the distinction is kept rather than flattened: a
    PORT is upstream's model re-registered, a CROP reads a narrower input of the
    same shape, and a plain VARIANT adds an API its twin lacks. They are the
    same fact for a ceiling -- none of them changes what a machine can hold --
    and different facts everywhere else, which is why `ported_from` is set to
    None on the other two rather than reused for them.
    """
    return spec.get("ported_from") or spec.get("crop_of") or spec.get("variant_of")


TWIN_OF = {name: _twin_of(spec) for name, spec in POLICIES.items() if _twin_of(spec)}


class MatrixError(ValueError):
    """A run matrix that cannot be read. The message names the offending row."""


# ── Whether the installed LeRobot has the policy at all ──────────────────────


def policy_available(policy: str) -> bool:
    """Can this venv's LeRobot train it?

    A capability probe rather than a version comparison: it keeps telling the
    truth after the pinned commit moves, and it cannot go stale the way a
    hard-coded "needs >= X" would.
    """
    module = (POLICIES.get(policy) or {}).get("module")
    if not module:
        return True
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def lerobot_commit() -> "str | None":
    """The short commit of the LeRobot checkout this venv installed, if visible."""
    import subprocess

    try:
        spec = importlib.util.find_spec("lerobot")
        if spec is None or not spec.origin:
            return None
        repo = str(spec.origin).rsplit("/src/lerobot/", 1)[0]
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def unavailable_message(policy: str) -> str:
    """Why this policy cannot be trained here, and what to do about it.

    The advice differs by where the policy lives. Telling someone to bump
    LeRobot when the missing module is one of ours would send them to the wrong
    repository entirely, which is the whole reason `local` exists.
    """
    spec = POLICIES.get(policy) or {}
    module = spec.get("module") or f"lerobot.policies.{policy}"
    if spec.get("local"):
        return (
            f"{policy} is implemented in this repo and its module {module} did "
            "not import. That is a checkout or a PYTHONPATH problem, not a "
            "LeRobot version: check src/actoris_harena.policies/ is present and that "
            "`source setup.sh` has run on the machine that trains."
        )
    have = lerobot_commit()
    at = f"the one this venv installed is at {have}" if have else "this one is not"
    return (
        f"{policy} needs a LeRobot that ships {module}; {at}. "
        "Bump LEROBOT_COMMIT in install.sh and hpc/provision_create.sh, add the "
        f"'{policy}' extra to LEROBOT_EXTRAS, and reinstall on every machine "
        "that trains."
    )


# ── Reading and writing the matrix ───────────────────────────────────────────


def parse_rows(text: str) -> "list[dict[str, str]]":
    """Every data row of a runs.tsv. Raises MatrixError naming the bad line.

    ``extra`` absorbs the rest of the line, so it may contain spaces; nothing
    before it may. A row short of a field is refused rather than shifted along,
    because a silently shifted row trains something nobody asked for.
    """
    rows: "list[dict[str, str]]" = []
    for lineno, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, len(COLUMNS) - 1)
        if len(parts) == len(COLUMNS) - 1:
            # The `slots` column was added after these files were first
            # written. Say which column is missing rather than shifting `extra`
            # into it, which would read a flag string as a camera mapping.
            raise MatrixError(
                f"line {lineno}: {len(parts)} fields, want {len(COLUMNS)} "
                f"({' '.join(COLUMNS)}). The 'slots' column was added — put a "
                f"'-' before the last column:\n    {line}"
            )
        if len(parts) < len(COLUMNS) - 1:
            raise MatrixError(
                f"line {lineno}: {len(parts)} fields, want {len(COLUMNS)} "
                f"({' '.join(COLUMNS)}). A space in the cameras column shifts "
                f"every later one — check that first:\n    {line}"
            )
        row = dict(zip(COLUMNS, parts))
        row["line"] = str(lineno)
        rows.append(_check_shape(row, lineno))
    return rows


def _check_shape(row: "dict[str, str]", lineno: int) -> "dict[str, str]":
    """The refusals that stop a row being READ at all, as opposed to run."""
    for field in ("steps", "batch"):
        value = row[field]
        if value != "-" and not value.isdigit():
            raise MatrixError(
                f"line {lineno}: {field} is {value!r}; want a whole number or '-'. "
                "A space in the cameras column shifts every later one — check "
                "that first."
            )
    hours = row["hours"]
    if not hours.replace(".", "", 1).isdigit():
        raise MatrixError(f"line {lineno}: hours is {hours!r}; want a number")
    if not row["cameras"]:
        raise MatrixError(f"line {lineno}: the cameras column is empty; want 'all'")
    return row


def format_rows(rows: "Sequence[dict[str, str]]", header: bool = True) -> str:
    """Rows back to runs.tsv text, column-aligned. Round-trips ``parse_rows``."""
    lines = []
    if header:
        lines.append("# " + "\t".join(COLUMNS))
    widths = [
        max([len(c)] + [len(str(r.get(c, "-"))) for r in rows]) for c in COLUMNS[:-1]
    ]
    for row in rows:
        cells = [
            str(row.get(col) or "-").ljust(width)
            for col, width in zip(COLUMNS[:-1], widths)
        ]
        lines.append(" ".join(cells).rstrip() + "  " + str(row.get("extra") or "-"))
    return "\n".join(lines) + "\n"


def make_row(
    dataset: str,
    policy: str,
    cameras: str = "all",
    steps: "int | str" = "-",
    batch: "int | str" = "-",
    hours: "int | str | None" = None,
    slots: str = "-",
    extra: str = "-",
) -> "dict[str, str]":
    """One row, with the policy's default wall time when none is given."""
    if hours is None:
        hours = (POLICIES.get(policy) or {}).get("hours", 36)
    return {
        "dataset": str(dataset),
        "policy": str(policy),
        "cameras": str(cameras) or "all",
        "steps": str(steps or "-"),
        "batch": str(batch or "-"),
        "hours": str(hours),
        "slots": str(slots or "-"),
        "extra": str(extra or "-"),
    }


def limits_for(dest: "dict[str, Any] | None", policy: str) -> "dict[str, Any]":
    """What a machine has been measured to carry for this policy.

    A variant falls back to what it is a variant of, following `TWIN_OF` as far
    as it goes. A port is the same model with the same activations, and a
    cropped variant reads an input of the same shape, so a ceiling measured on
    one holds for the other -- and without this a `harena_pi05` row on CREATE
    resolves to batch 8 and reproduces the OutOfMemoryError that `pi05: batch: 4`
    was written down to prevent. `harena_pi05_crop` needs two hops to get there.
    A destination may still name a variant explicitly to override.
    """
    limits = (dest or {}).get("limits") or {}
    seen: "set[str]" = set()
    name: "str | None" = policy
    while name is not None and name not in seen:
        if name in limits:
            return limits[name] or {}
        seen.add(name)
        name = TWIN_OF.get(name)
    return {}


def resolved(
    row: "dict[str, str]", key: str, dest: "dict[str, Any] | None" = None
) -> "int | None":
    """A row's steps/batch, with ``-`` filled in.

    ``-`` does not mean "the policy's default" so much as "whatever fits here",
    so a destination that has MEASURED a lower ceiling supplies it. An explicit
    number is left alone and refused if it is over -- the operator asked for
    that one, and quietly training something else would make the run matrix a
    record of what was requested rather than of what ran.
    """
    value = row.get(key, "-")
    if value and value != "-":
        return int(value)
    default = (POLICIES.get(row.get("policy", "")) or {}).get(key)
    cap = limits_for(dest, row.get("policy", "")).get(key)
    if default is None:
        return cap
    return min(default, cap) if cap is not None else default


# ── Whether a row can work ───────────────────────────────────────────────────


def row_cameras(row: "dict[str, str]") -> "list[str]":
    """The camera names a row asks for. ``all`` stays as itself."""
    text = str(row.get("cameras") or "all").strip()
    if text == "all":
        return ["all"]
    return [c for c in text.split(",") if c]


def parse_slots(row: "dict[str, str]") -> "dict[str, str] | None":
    """``central=base,left_arm_left_gripper=left_wrist`` -> a map, or None."""
    text = str(row.get("slots") or "-").strip()
    if text in ("", "-"):
        return None
    out: "dict[str, str]" = {}
    for pair in text.split(","):
        camera, sep, slot = pair.partition("=")
        if not sep or not camera or not slot:
            raise MatrixError(
                f"slots {text!r}: want camera=slot pairs, e.g. "
                "central=base,left_arm_left_gripper=left_wrist"
            )
        out[camera.strip()] = slot.strip()
    return out


def row_refusals(
    row: "dict[str, str]",
    info: "dict[str, Any] | None" = None,
    dest: "dict[str, Any] | None" = None,
    staged: "Iterable[str] | None" = None,
) -> "list[str]":
    """Every reason this row cannot run, in the order an operator should fix them.

    ``info`` is the dataset's ``meta/info.json`` (None if it has not been read),
    ``dest`` a destination dict, ``staged`` the dataset names already on that
    machine. Each argument only adds refusals: a row can be judged before any of
    them is known, which is what lets the page refuse while you are still typing.
    """
    out: "list[str]" = []
    policy = str(row.get("policy") or "")
    spec = POLICIES.get(policy)

    if spec is None:
        return [
            f"unknown policy {policy!r}; this repo trains " f"{', '.join(POLICY_NAMES)}"
        ]
    if not policy_available(policy):
        out.append(unavailable_message(policy))

    try:
        slots = parse_slots(row)
    except MatrixError as exc:
        out.append(str(exc))
        slots = None

    cameras = row_cameras(row)
    if info is not None:
        out += _camera_refusals(cameras, info, policy, spec)
        # `all` is the dataset's own cameras. A composite is never included by
        # it: tiling several views into one is a deliberate choice about what a
        # policy sees, so it has to be asked for by name.
        if cameras == ["all"]:
            cameras = [short_name(k) for k in camera_keys(info)]

    if slots is not None:
        unknown = [c for c in slots if c not in cameras and cameras != ["all"]]
        if unknown:
            out.append(
                f"slots names {', '.join(sorted(unknown))}, which this row does "
                f"not record; its cameras are {', '.join(cameras)}"
            )

    max_cameras = spec.get("max_cameras")
    if max_cameras and cameras != ["all"] and len(cameras) > max_cameras:
        out.append(
            f"{policy} has {max_cameras} pretrained image slots but this row "
            f"names {len(cameras)} cameras ({', '.join(cameras)}). Drop one, or "
            "map them explicitly in the slots column."
        )

    out += _image_size_refusals(cameras, policy, spec)

    if dest is not None:
        out += _destination_refusals(row, policy, dest, staged)
    return out


def _camera_refusals(
    cameras: "list[str]", info: "dict[str, Any]", policy: str, spec: dict
) -> "list[str]":
    if cameras == ["all"]:
        return []
    have = set(available_camera_names(info))
    missing = [c for c in cameras if c not in have]
    if not missing:
        return []
    return [
        f"camera {', '.join(missing)} is not in this dataset; it records "
        f"{', '.join(sorted(have))}"
    ]


def _image_size_refusals(cameras: "list[str]", policy: str, spec: dict) -> "list[str]":
    """FastWAM's contract: heights equal, widths summing to image_size[1]."""
    size = spec.get("image_size")
    if not size or cameras == ["all"]:
        return []
    height, width = size
    if width % height:
        return []
    slots = width // height
    if len(cameras) == slots:
        return []
    return [
        f"{policy} concatenates its cameras into one {height}x{width} frame, so "
        f"it takes exactly {slots} of them at {height}x{height}; this row names "
        f"{len(cameras)} ({', '.join(cameras)}). "
        f"{', '.join(sorted(profile().composites))} tiles several cameras into one."
    ]


def _destination_refusals(
    row: "dict[str, str]",
    policy: str,
    dest: "dict[str, Any]",
    staged: "Iterable[str] | None",
) -> "list[str]":
    out: "list[str]" = []
    where = str(dest.get("name") or dest.get("ssh") or "that machine")
    limits = limits_for(dest, policy)
    for key in ("batch", "steps"):
        cap = limits.get(key)
        if row.get(key, "-") in ("", "-"):
            continue  # `-` already resolves to the ceiling; see resolved()
        want = resolved(row, key)
        if cap is not None and want is not None and want > cap:
            # Name the twin when the ceiling was measured on it rather than on
            # the port, so the number can be traced to the run that produced it.
            twin = PORTED_FROM.get(policy)
            measured = (
                f"{cap}, measured on {twin}"
                if twin and policy not in ((dest.get("limits") or {}))
                else str(cap)
            )
            out.append(
                f"{policy} {key} {want} is over what {where} has been measured "
                f"to carry ({measured}). Lower it in the {key} column, or "
                "measure a new ceiling and record it in "
                "src/conf/train_destinations.yaml."
            )
    if staged is not None and row.get("dataset") not in set(staged):
        out.append(
            f"{row.get('dataset')} is not staged on {where} yet — stage it "
            "before submitting (the launcher does this for you)."
        )
    out += _device_refusals(dest, where)
    return out


def _device_refusals(dest: "dict[str, Any]", where: str) -> "list[str]":
    """Refuse a run that would quietly become a CPU run.

    ``measured`` is what the machine said about itself when it was asked -- not
    something the destinations file claims, because that file is checked in and
    this console runs on more than one machine. A GPU too small makes
    ``long_vla_real.sh`` fall back to the CPU without failing, which for an
    80 000-step run means a week of work that looks like it is going fine; the
    driver calls that "the worst outcome available" and it is the DEFAULT
    outcome on a 4 GB laptop. An operator who means it says so.
    """
    measured = dest.get("measured") or {}
    if measured.get("device") != "cpu" or dest.get("allow_cpu"):
        return []
    return [
        f"{where} would train on the CPU: {measured.get('why', 'no usable GPU')}. "
        "A run that falls back to the CPU does not fail, it just never "
        "finishes — ask for a CPU run outright if that is what you mean."
    ]
