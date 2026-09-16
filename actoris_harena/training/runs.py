"""Which log, on which machine, and how little of it has to travel.

``actoris_harena.training.progress`` turns a log into a curve. This decides where that
log is, gets it off the machine that holds it, and finds the runs nobody told us
about -- the nine finished directories on CREATE were all started from a
terminal, and a viewer that listed only what the console launched would show an
empty page next to a full drive.

Three things here are deliberate.

**One round trip, not four.** A remote read is a whole SSH, so each question is
one command whose answer carries its own section markers. The log itself is
filtered ON THE FAR SIDE: the live ACT log on thanos is 5.8 MB of tqdm frames
and 723 useful lines, and after the filter it is 190 KB -- roughly constant as
the run grows, because what grows is the part that is dropped.

**The tqdm frame rides along.** After ``\\r`` is translated, a metric line and
the progress frame that precedes it are the SAME line, so grepping for the
metric line brings the exact step with it. Matching progress frames separately
looked equivalent and shipped 1.7 MB instead of 190 KB.

**Standard output only.** CREATE prints an MFA banner and a QR code on stderr at
every login. Anything that falls back to stderr when stdout is empty -- which is
what "no such job" looks like -- renders that banner as a status.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from actoris_harena.training import metrics
from actoris_harena.training.destinations import ssh_argv

# Delimits the sections of the one command runs.py sends over ssh. Built and
# parsed in the same round trip and never written to a file, so unlike the
# metric sentinel this one has nothing to stay compatible with.
SENTINEL = "@@harena:"

# What a filtered log must keep: the metric lines (with their progress frame),
# the exact checkpoint steps, the end, and any way a run can go wrong. ASCII
# only -- this reaches a remote shell, and the driver's tick and cross do not
# survive every locale on the way.
KEEP_PATTERN = (
    "step:|Checkpoint policy after step|End of training|Traceback"
    "|Error|error:|row [0-9]+ FAILED|row [0-9]+ done|reusing checkpoint"
    # A resumed run appends to the log it already had, so without this line the
    # curve would show one continuous run where in fact the box rebooted and the
    # driver picked the checkpoint back up.
    "|resuming"
    # Everything that is a curve but not one of lerobot's tracker fields: our
    # own metric channel (actoris_harena.training.metrics.SENTINEL) and lerobot's
    # held-out validation loss, whose line says `step 12000: eval_loss=...`
    # with a space and so matches none of the above. This grep runs on the FAR
    # side, so a metric missing from here is a metric that never leaves the
    # machine -- there is no second chance to notice it downstream.
    # Every sentinel the channel has ever used, not just the current one: this
    # grep runs on the FAR side over logs that finished runs already wrote, and
    # a metric missing from here is a metric that never leaves the machine.
    + "".join(f"|{t}" for t in (metrics.SENTINEL, *metrics.LEGACY_SENTINELS))
    + "|eval_loss="
)

# 20 000 lines is 2 000 000 steps at the log_freq this repo uses; the longest
# run here is 100 000. A run past that would lose its early points, and with
# them the ordinal the CREATE step count is derived from.
BODY_LINES = 20000
HEAD_BYTES = 20000
# A resolved train_config.json is a few tens of kilobytes; pi0.5's, the largest
# here, is under 40. Capped anyway, because this arrives over the same link the
# log does and a truncated diff is better than a stalled panel.
CONFIG_BYTES = 200000
# One eval_sim_policy result. `episodes` is the long part -- one record per
# rollout -- and the summary this reads is at the top level beside it.
VAL_BYTES = 100000
# One prediction.json. It holds four series per camera over the horizon, so it
# grows with cameras x horizon and not with the length of the run: this rig's
# five cameras at a sixteen-step horizon come to a few kilobytes. Capped at the
# same order as a config for the same reason.
PREDICTION_BYTES = 200000
READ_TIMEOUT_S = 90.0
DISCOVER_TIMEOUT_S = 60.0

# A run directory is `<dataset>__<slug>`, and a slug joins camera names with
# `+`. These names reach a remote shell inside a command, so what they may
# contain is checked here rather than escaped: a directory that cannot be named
# safely is one this rig did not create.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class RunsError(ValueError):
    """A name that cannot be sent, or an answer that cannot be read."""


def check_name(name: str, what: str = "name") -> str:
    if not _NAME_RE.match(str(name or "")):
        raise RunsError(
            f"{what} {name!r} is not a plain name. Run directories and policies "
            "reach the remote shell inside a command, so they may hold only "
            "letters, digits and . _ + -"
        )
    return str(name)


# ── Where the runs are ───────────────────────────────────────────────────────


def out_roots(dest: "dict[str, Any]") -> "list[str]":
    """Every directory on that machine that may hold run directories.

    Normally one: the drivers both put ``SO101_OUTPUT_DIR`` at
    ``<scratch>/so101_outputs``. A destination may name a second in ``outputs``,
    which is what makes the local machine work -- a console started through
    ``setup.sh`` has ``SO101_OUTPUT_DIR`` pointing into the repo, so runs
    started here land beside the older ones rather than in the cache.
    """
    roots = [f"{dest['scratch']}/so101_outputs"]
    extra = dest.get("outputs")
    if extra:
        formatted = str(extra).format(scratch=dest["scratch"], repo=dest["repo"])
        if formatted not in roots:
            roots.append(formatted)
    return roots


def view_dir_name(dataset: str, cameras: str, info: "dict[str, Any] | None") -> str:
    """``<dataset>__<slug>`` -- the directory both drivers name a run after.

    The slug is the camera view's, computed the way ``make_camera_view`` does,
    because that is the name the run directory actually gets. Joining the row's
    camera list would be close and wrong: ``central,wrist_left`` is a row, and
    ``central+wrist_left`` is the directory.
    """
    if info is None:
        return f"{dataset}__{cameras.replace(',', '+')}"
    from actoris_harena.recording.dataset_view import split_selection, view_slug

    keep, composites = split_selection(info, str(cameras or "all").split(","))
    slug = "+".join(filter(None, [view_slug(info, keep) if keep else "", *composites]))
    return f"{dataset}__{slug}"


def run_dir_names(record: "dict[str, Any]") -> "list[str]":
    """Every directory a recorded run may have written into.

    ``--run-tag`` when one was given, else what was recorded at launch, else the
    camera view derived from the row -- which is what the runs launched before
    this branch, and everything started from a terminal, have to fall back to.
    """
    tag = record.get("run_tag")
    if tag:
        return [str(tag)]
    recorded = record.get("run_dir") or record.get("run_dirs")
    if recorded:
        return (
            [str(recorded)] if isinstance(recorded, str) else [str(r) for r in recorded]
        )
    dataset = record["dataset"]
    return [
        view_dir_name(dataset, str(c), None) for c in (record.get("cameras") or ["all"])
    ]


def run_dir_name(record: "dict[str, Any]") -> str:
    """The first directory of ``run_dir_names``. Convenience."""
    return run_dir_names(record)[0]


#: The two trees a driver writes into. They are shaped differently and that
#: difference is load-bearing, so it is named rather than pattern-matched: a
#: REAL run is one policy under `<run>/train/<policy>`, while a SIM run is a
#: cell `<run>/<mode>/<task>/<policy>` and its log is named for all three. Only
#: a sim run has per-checkpoint rollout results, because only a simulator can
#: roll a checkpoint out without a person and a robot in the room.
REAL_TREE = "vla_real_long"
SIM_TREE = "vla_sim_long"


def log_path(out_root: str, run: str, policy: str, tree: str = REAL_TREE) -> str:
    return f"{out_root}/{tree}/{run}/logs/train_{policy}.log"


def cell_path(out_root: str, run: str, cell: str, tree: str = SIM_TREE) -> str:
    """The directory a sim cell's checkpoints and rollout results live in."""
    return f"{out_root}/{tree}/{run}/{cell}"


def train_dir(out_root: str, run: str, policy: str, tree: str = REAL_TREE) -> str:
    """The directory one policy's checkpoints live in, in either tree."""
    if tree == REAL_TREE:
        return f"{out_root}/{tree}/{run}/train/{policy}"
    return f"{out_root}/{tree}/{run}/{policy}"


def prediction_glob(out_root: str, run: str, policy: str, tree: str = REAL_TREE) -> str:
    """Every place ``tool/eval_world_model.py`` may have left a prediction.json.

    Two, and the difference matters. Given ``--out`` the tool writes where it is
    told; given nothing it defaults beside the checkpoint it scored, which is
    ``checkpoints/<step>/prediction/`` -- ``checkpoints/last`` is a symlink, so
    the file lands under the NUMBERED directory and stays attached to the
    checkpoint that produced it even after ``last`` has moved on. That is the
    right place for it and the reason this is a glob rather than a path: a run
    scored twice has two answers, and which checkpoint each belongs to is in
    its directory name, not in the file.
    """
    train = train_dir(out_root, run, policy, tree)
    return f"{train}/prediction/prediction.json {train}/checkpoints/*/prediction/prediction.json"


def config_path(out_root: str, run: str, policy: str, tree: str = REAL_TREE) -> str:
    """Where the run's resolved configuration is, for the runs table's diff.

    lerobot writes it into every checkpoint, so the last one has it. A run with
    no checkpoint yet has no config to compare, which is correct -- nothing has
    been committed to disk to compare against.
    """
    train = (
        f"{out_root}/{tree}/{run}/train/{policy}"
        if tree == REAL_TREE
        else f"{out_root}/{tree}/{run}/{policy}"
    )
    return f"{train}/checkpoints/last/pretrained_model/train_config.json"


# ── The commands ─────────────────────────────────────────────────────────────


def read_command(
    path: str,
    config: "str | None" = None,
    cell: "str | None" = None,
    prediction: "str | None" = None,
) -> str:
    """One command that returns everything needed to draw a run. Pure.

    The sections are the configuration lerobot dumps before training, the
    filtered metric lines, the raw tail (where a traceback that matched nothing
    still shows), and the file's size and modification time against the
    machine's OWN clock -- which is how staleness is judged, because the
    timestamps inside the log carry no timezone.

    ``config``, ``cell`` and ``prediction`` ride along in the SAME command
    rather than costing a second SSH each: the resolved ``train_config.json``
    the runs table diffs, -- for a sim cell only -- the per-checkpoint rollout
    results ``long_vla_sim.sh`` writes, and the world model's ``prediction.json``.
    All are small and all are optional; a run without them simply has no such
    section, which is the common case -- only a world model has a future to be
    scored on.
    """
    parts = [
        f"L={path}; "
        'if [ ! -f "$L" ]; then echo \'' + SENTINEL + "missing'; exit 0; fi; "
        "echo '" + SENTINEL + f'head\'; head -c {HEAD_BYTES} "$L"; echo; '
        "echo '" + SENTINEL + "body'; "
        f"tr '\\r' '\\n' < \"$L\" | grep -aE '{KEEP_PATTERN}' | tail -n {BODY_LINES}; "
        "echo '" + SENTINEL + "tail'; "
        # The `echo` after each section is load-bearing: a progress bar's last
        # frame ends WITHOUT a newline, so the next marker landed on the end of
        # it -- which put `@@so101:stat` in the text shown to the operator and
        # lost the file's age, which is the whole basis for calling a run
        # stalled.
        "tail -c 4000 \"$L\" | tr '\\r' '\\n' | tail -n 4; echo; "
        "echo '" + SENTINEL + "stat'; stat -c '%s %Y' \"$L\"; date +%s"
    ]
    if config:
        parts.append(
            "echo '" + SENTINEL + "config'; "
            f"[ -f {config} ] && head -c {CONFIG_BYTES} {config}; echo"
        )
    if cell:
        # One JSON per checkpoint, each prefixed by the file it came from so
        # the step is read from the NAME rather than guessed from the order --
        # a shell glob sorts step_5000 before step_10000.
        parts.append(
            "echo '" + SENTINEL + "val'; "
            f"for v in {cell}/val/step_*.json {cell}/selected.json; do "
            '[ -f "$v" ] && { echo "VAL $v"; head -c '
            + str(VAL_BYTES)
            + ' "$v"; echo; }; '
            "done"
        )
    if prediction:
        # A glob, and every match is labelled with the file it came from: a run
        # scored at more than one checkpoint has more than one answer, and the
        # step is in the path. Unquoted on purpose -- the shell has to expand
        # it -- which is why the run and policy inside it are name-checked by
        # the caller rather than escaped here.
        parts.append(
            "echo '" + SENTINEL + "prediction'; "
            f"for p in {prediction}; do "
            '[ -f "$p" ] && { echo "PRED $p"; head -c '
            + str(PREDICTION_BYTES)
            + ' "$p"; echo; }; '
            "done"
        )
    return "; ".join(parts)


def discover_command(dest: "dict[str, Any]") -> str:
    """Every run directory and training log on a machine, in one command. Pure.

    Also reports what ``checkpoints/last`` resolves to, which is the only exact
    step written anywhere structured, and the machine's clock, so an age can be
    computed without comparing two machines' timezones.
    """
    parts = []
    for root in out_roots(dest):
        parts.append(
            f"for f in {root}/{REAL_TREE}/*/logs/train_*.log; do "
            '[ -f "$f" ] && stat -c "LOG %n %s %Y" "$f"; done; '
            f"for c in {root}/{REAL_TREE}/*/train/*/checkpoints/last; do "
            '[ -e "$c" ] && echo "CKPT $c $(basename "$(readlink -f "$c")")"; done'
        )
        # The sim tree, whose cells are <mode>/<task>/<policy> rather than
        # train/<policy>. The CELL lines are what pair a log with its rollout
        # results: the log is named train_<mode>_<task>_<policy>.log, and every
        # one of those three may itself contain an underscore (handover_split,
        # so101_act), so the name cannot be split back apart -- the directory
        # is listed instead and the name rebuilt FROM it.
        parts.append(
            f"for f in {root}/{SIM_TREE}/*/logs/train_*.log; do "
            '[ -f "$f" ] && stat -c "SIMLOG %n %s %Y" "$f"; done; '
            f"for d in {root}/{SIM_TREE}/*/*/*/*/; do "
            '[ -d "$d/checkpoints" ] && echo "CELL $d"; done; '
            f"for c in {root}/{SIM_TREE}/*/*/*/*/checkpoints/last; do "
            '[ -e "$c" ] && echo "SIMCKPT $c $(basename "$(readlink -f "$c")")"; done'
        )
    return "; ".join(parts) + "; echo '" + SENTINEL + "now'; date +%s"


def command_argv(dest: "dict[str, Any]", command: str) -> "list[str]":
    """How to run a shell command on that machine. Pure.

    The local machine is not a special case anywhere else in this feature: the
    same driver, the same lock, the same run directories, and here the same
    command -- only without the SSH in front of it.
    """
    if dest.get("kind") == "local":
        return ["bash", "-lc", command]
    return ssh_argv(dest, command)


# ── Running them ─────────────────────────────────────────────────────────────


def run_command(
    dest: "dict[str, Any]", command: str, timeout: float = READ_TIMEOUT_S
) -> "tuple[str, str | None]":
    """``(stdout, problem)``. Never raises, and never returns stderr as output.

    stderr is reported only as a problem, never as an answer: CREATE greets
    every login with an MFA banner and a QR code on stderr, and a caller that
    substitutes it for empty output shows the operator a QR code where a status
    should be.
    """
    argv = command_argv(dest, command)
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            # `head -c` counts BYTES, and a progress bar is drawn with
            # three-byte block characters, so the head of a log is routinely
            # cut through the middle of one. Without this the whole read raises
            # UnicodeDecodeError and the run cannot be watched at all --
            # measured on a real local run, at byte 20012.
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "", f"{dest.get('name', 'the machine')} did not answer in {timeout:.0f}s"
    except OSError as exc:
        return "", f"could not run {argv[0]}: {exc}"
    if proc.returncode != 0 and not proc.stdout.strip():
        return "", (proc.stderr or "").strip() or f"exit {proc.returncode}"
    return proc.stdout, None


# ── Reading the answers ──────────────────────────────────────────────────────


def parse_sections(text: str) -> "dict[str, str]":
    """Split a sentinel-delimited answer. Pure. Unknown sections are kept."""
    out: "dict[str, str]" = {}
    current = ""
    lines: "list[str]" = []
    for line in text.splitlines():
        if line.startswith(SENTINEL):
            if current:
                out[current] = "\n".join(lines)
            current = line[len(SENTINEL) :].strip()
            lines = []
        elif current:
            lines.append(line)
    if current:
        out[current] = "\n".join(lines)
    return out


_VAL_FILE_RE = re.compile(
    r"^VAL \S+/(?:step_(?P<step>\d+)|(?P<selected>selected))\.json$"
)


def parse_val_results(text: str) -> "dict[str, Any]":
    """Per-checkpoint rollout results -> series, plus which one was selected.

    SIM ONLY, and deliberately so. ``long_vla_sim.sh`` rolls every checkpoint
    out on the validation seeds because a simulator can; the real rig's
    equivalent is a person judging trials in ``outputs/policy_runs``, which is
    a different thing measured at a different time and must never be drawn on
    this axis as though the training loop had produced it.
    """
    import json

    series: "dict[str, list[dict[str, Any]]]" = {}
    selected: "dict[str, Any] | None" = None
    current: "re.Match[str] | None" = None
    buffer: "list[str]" = []

    def flush() -> None:
        nonlocal selected
        if current is None or not buffer:
            return
        try:
            payload = json.loads("\n".join(buffer))
        except (ValueError, TypeError):
            return
        if current.group("selected"):
            selected = payload
            return
        step = int(current.group("step"))
        for key, name in (
            ("success_rate", "eval/success_rate"),
            ("place_err_mm_mean", "eval/place_err_mm"),
        ):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                series.setdefault(name, []).append(
                    {"step": step, "time": None, "value": float(value)}
                )

    for line in text.splitlines():
        header = _VAL_FILE_RE.match(line.strip())
        if header:
            flush()
            current, buffer = header, []
        elif current is not None:
            buffer.append(line)
    flush()
    for entries in series.values():
        entries.sort(key=lambda e: e["step"])
    return {"series": series, "selected": selected}


_PRED_FILE_RE = re.compile(r"^PRED (?P<path>\S+)$")
_PRED_STEP_RE = re.compile(r"/checkpoints/(?P<step>\d+)/")


def parse_predictions(text: str) -> "list[dict[str, Any]]":
    """``prediction.json`` -> what the panel draws, one entry per checkpoint.

    The reduction is the whole point and it is done HERE rather than in the
    browser, because the verdict is not a number the file contains. What the
    file has is PSNR and SSIM per camera per horizon step, and beside each the
    same measure for the last observed frame repeated. What a reader needs to
    know is on how many of those steps the model beat holding -- a world model
    that loses to holding has not learned that camera's dynamics, whatever its
    PSNR says, and on this rig four of five cameras are gel images that barely
    move until contact, where a high PSNR means only that nothing happened.

    The step comes from the PATH, because the file does not record it: the tool
    writes beside the checkpoint it scored, and `checkpoints/<step>/prediction`
    is the only place the number appears. A prediction written somewhere else
    with ``--out`` has no step and is reported with none rather than guessed.
    """
    import json

    out: "list[dict[str, Any]]" = []
    current: "str | None" = None
    buffer: "list[str]" = []

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        path, current = current, None
        try:
            payload = json.loads("\n".join(buffer))
        except (ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        cameras = {}
        for camera, values in (payload.get("per_camera") or {}).items():
            if not isinstance(values, dict):
                continue
            model = values.get("psnr") or []
            held = values.get("psnr_baseline") or []
            beats = sum(1 for a, b in zip(model, held) if a > b)
            cameras[camera] = {
                "psnr": model,
                "psnr_baseline": held,
                "ssim": values.get("ssim") or [],
                "ssim_baseline": values.get("ssim_baseline") or [],
                "horizon": len(model),
                "beats": beats,
            }
        if not cameras:
            return
        found = _PRED_STEP_RE.search(path)
        out.append(
            {
                "path": path,
                "step": int(found.group("step")) if found else None,
                "frames": payload.get("frames"),
                "episodes": payload.get("episodes") or [],
                "policy": payload.get("policy"),
                "cameras": cameras,
            }
        )

    for line in text.splitlines():
        header = _PRED_FILE_RE.match(line.strip())
        if header:
            flush()
            current, buffer = header.group("path"), []
        elif current is not None:
            buffer.append(line)
    flush()
    # Unstepped last: a run scored at several checkpoints should read forwards,
    # and a `--out` prediction has no place on that axis.
    out.sort(key=lambda e: (e["step"] is None, e["step"] or 0))
    return out


def read_log(
    dest: "dict[str, Any]",
    path: str,
    config: "str | None" = None,
    cell: "str | None" = None,
    prediction: "str | None" = None,
) -> "dict[str, Any]":
    """Fetch and parse one training log. Blocking; never raises."""
    from actoris_harena.training.progress import (
        parse_log,
        parse_markers,
        parse_tqdm,
        summarise,
    )

    text, problem = run_command(
        dest, read_command(path, config=config, cell=cell, prediction=prediction)
    )
    if problem:
        return {"ok": False, "problem": problem, "path": path}
    sections = parse_sections(text)
    if "missing" in sections or "body" not in sections:
        return {"ok": False, "problem": f"no log at {path}", "path": path}

    age = None
    stat = (sections.get("stat") or "").split()
    if len(stat) >= 3:
        try:
            age = max(0.0, float(stat[2]) - float(stat[1]))
        except ValueError:
            age = None

    # The curve comes from the filtered body ALONE. The tail overlaps it -- it
    # is the last few lines of the same file -- and appending it counted those
    # points twice, which put a finished 30 000-step run at 30 200 and its
    # fraction over one. The tail is still read, for two things the body cannot
    # give: the freshest progress frame (written every step, where a metric line
    # is written every hundredth) and a traceback whose wording matched no
    # filter.
    parsed = parse_log(sections["body"], head_text=sections.get("head"))
    tail_text = sections.get("tail", "").replace("\r", "\n")
    if tail_text.strip():
        frames = parse_tqdm(tail_text)
        if frames:
            parsed["last_frame"] = frames[-1]
        marks = parse_markers(tail_text)
        parsed["finished"] = parsed["finished"] or marks["finished"]
        parsed["failure"] = parsed["failure"] or marks["failure"]

    val = parse_val_results(sections.get("val") or "")
    if val["series"]:
        parsed["series"].update(val["series"])
        parsed["metrics"] = sorted(parsed["series"])

    # Empty for every policy that is not a world model, which is most of them.
    predictions = parse_predictions(sections.get("prediction") or "")

    return {
        "ok": True,
        "path": path,
        "summary": summarise(parsed, age_s=age),
        "points": parsed["points"],
        "series": parsed["series"],
        "metrics": parsed["metrics"],
        "problems": parsed["problems"],
        "failure": parsed["failure"],
        "checkpoints": parsed["checkpoints"],
        "config": parse_config(sections.get("config") or ""),
        "selected": val["selected"],
        "predictions": predictions,
        "tail": sections.get("tail", "").strip(),
    }


def parse_config(text: str) -> "dict[str, Any] | None":
    """The resolved ``train_config.json``, flattened for the runs table's diff.

    Flattened to dotted keys because the interesting differences between two
    runs are nested (``policy.type``, ``dataset.repo_id``, ``batch_size``) and
    a diff over nested dicts either reports a whole subtree as changed or has
    to walk it anyway. Lists are rendered as one value: a camera list that
    differs, differs as a whole.
    """
    import json

    text = text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    flat: "dict[str, Any]" = {}

    def walk(node: "dict[str, Any]", prefix: str) -> None:
        for key, value in node.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict) and value:
                walk(value, f"{name}.")
            elif isinstance(value, (list, tuple)):
                flat[name] = ", ".join(str(v) for v in value)
            else:
                flat[name] = value

    walk(payload, "")
    return flat


_LOG_LINE_RE = re.compile(
    r"^(?P<kind>SIM)?LOG (?P<path>\S+)/(?P<tree>vla_(?:real|sim)_long)/(?P<run>[^/]+)"
    r"/logs/train_(?P<policy>[^/.]+)\.log (?P<size>\d+) (?P<mtime>\d+)$"
)
_CKPT_RE = re.compile(
    r"^CKPT \S+/vla_real_long/(?P<run>[^/]+)/train/(?P<policy>[^/]+)"
    r"/checkpoints/last (?P<step>\S+)$"
)
# `CELL <root>/vla_sim_long/<run>/<mode>/<task>/<policy>/`
_CELL_RE = re.compile(
    r"^CELL \S+/vla_sim_long/(?P<run>[^/]+)/(?P<mode>[^/]+)/(?P<task>[^/]+)"
    r"/(?P<policy>[^/]+)/?$"
)
_SIM_CKPT_RE = re.compile(
    r"^SIMCKPT \S+/vla_sim_long/(?P<run>[^/]+)/(?P<mode>[^/]+)/(?P<task>[^/]+)"
    r"/(?P<policy>[^/]+)/checkpoints/last (?P<step>\S+)$"
)


def parse_discovery(text: str) -> "list[dict[str, Any]]":
    """The listing -> one entry per (run, policy) found on that machine. Pure.

    A sim cell arrives as a directory rather than as a parsed log name, and is
    matched to its log by rebuilding the name the driver gives it. That way the
    three components never have to be split back out of a string in which each
    of them may contain the separator.
    """
    sections = parse_sections(text)
    now = None
    try:
        now = float((sections.get("now") or "").split()[0])
    except (IndexError, ValueError):
        now = None
    body = text.split(SENTINEL, 1)[0]

    found: "dict[tuple, dict[str, Any]]" = {}
    for line in body.splitlines():
        line = line.strip()
        match = _LOG_LINE_RE.match(line)
        if match:
            tree = match["tree"]
            key = (match["run"], match["policy"])
            mtime = float(match["mtime"])
            found.setdefault(
                key,
                {
                    "run": match["run"],
                    "policy": match["policy"],
                    "out_root": match["path"],
                    "tree": tree,
                    "sim": tree == SIM_TREE,
                    "size": int(match["size"]),
                    "mtime": mtime,
                    "age_s": (now - mtime) if now else None,
                    "checkpoint": None,
                },
            )
            continue
        ckpt = _CKPT_RE.match(line)
        if ckpt:
            entry = found.get((ckpt["run"], ckpt["policy"]))
            step = ckpt["step"]
            if entry is not None:
                entry["checkpoint"] = int(step) if step.isdigit() else step
            continue
        cell = _CELL_RE.match(line) or _SIM_CKPT_RE.match(line)
        if cell:
            name = f"{cell['mode']}_{cell['task']}_{cell['policy']}"
            entry = found.get((cell["run"], name))
            if entry is None:
                continue
            entry["cell"] = f"{cell['mode']}/{cell['task']}/{cell['policy']}"
            entry["mode"] = cell["mode"]
            entry["task"] = cell["task"]
            # The log is named for the whole cell; the POLICY is the last part
            # of it, and it is what the runs table and the ports comparison
            # actually want to group by.
            entry["policy_name"] = cell["policy"]
            step = cell.groupdict().get("step")
            if step:
                entry["checkpoint"] = int(step) if step.isdigit() else step
    return sorted(found.values(), key=lambda e: (-(e["mtime"] or 0), e["run"]))


def discover(dest: "dict[str, Any]") -> "dict[str, Any]":
    """What training this machine holds. Blocking; never raises."""
    text, problem = run_command(dest, discover_command(dest), DISCOVER_TIMEOUT_S)
    if problem:
        return {"ok": False, "problem": problem, "runs": []}
    return {"ok": True, "problem": None, "runs": parse_discovery(text)}


# ── What this machine can do (the local destination) ─────────────────────────

# The rule test/system/long_vla_real.sh applies: below this it picks the CPU,
# and an 80 000-step run that quietly becomes a CPU run is worse than one that
# is refused, because it looks like it is working for a week.
MIN_GPU_GB = 8.0


def local_capability(python: "str | Path | None" = None) -> "dict[str, Any]":
    """This machine's GPU, measured now rather than read from a file.

    The destinations file is checked in and this console is meant to run on more
    than one machine, so a VRAM figure written there would be a fact about
    whoever wrote it. Ask torch instead.
    """
    exe = str(
        python or (Path(__file__).resolve().parents[3] / "venv" / "bin" / "python")
    )
    probe = (
        "import json,torch\n"
        "d={'cuda':torch.cuda.is_available()}\n"
        "if d['cuda']:\n"
        "    p=torch.cuda.get_device_properties(0)\n"
        "    d['gpu']=p.name; d['vram_gb']=round(p.total_memory/1e9,2)\n"
        "print(json.dumps(d))"
    )
    try:
        proc = subprocess.run(
            [exe, "-c", probe], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "problem": f"could not ask torch: {exc}"}
    if proc.returncode != 0:
        return {"ok": False, "problem": (proc.stderr or "torch would not load").strip()}
    try:
        import json

        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        return {"ok": False, "problem": f"could not read torch's answer: {exc}"}

    vram = float(data.get("vram_gb") or 0.0)
    device = "cuda" if data.get("cuda") and vram >= MIN_GPU_GB else "cpu"
    return {
        "ok": True,
        "problem": None,
        "cuda": bool(data.get("cuda")),
        "gpu": data.get("gpu"),
        "vram_gb": vram or None,
        "device": device,
        "why": _device_why(bool(data.get("cuda")), vram),
    }


def _device_why(cuda: bool, vram: float) -> str:
    if not cuda:
        return "torch cannot use a GPU here, so training would run on the CPU"
    if vram < MIN_GPU_GB:
        return (
            f"this GPU has {vram:.1f} GB and the driver wants at least "
            f"{MIN_GPU_GB:.0f} GB, so training would fall back to the CPU"
        )
    return f"{vram:.1f} GB of VRAM"


def with_measurements(dest: "dict[str, Any]") -> "dict[str, Any]":
    """The destination, plus what the machine says about itself. Blocking.

    Only the local machine is asked, and only because it is the one whose
    hardware this repo cannot write down: ``train_destinations.yaml`` is checked
    in and the console is meant to run on more than one machine. A remote
    destination's ceilings are MEASURED and recorded in that file, which is the
    right place for a number somebody had to observe.
    """
    if dest.get("kind") != "local":
        return dest
    return {**dest, "measured": local_capability()}
