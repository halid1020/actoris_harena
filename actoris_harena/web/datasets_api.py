"""The Datasets tab, served by the shared console for any rig.

Browsing, playing back and curating a recorded dataset needs no robot -- only
the drive -- so the console does it itself, rather than through a rig's agent.
This was the SO-101 console's alone (``so101_garment/src/common/web/
datasets_api.py``), which is why the shared page's Datasets tab was empty: the
routes it calls did not exist here.

What came across is the rig-neutral part: the list, the episode index, the
recorded video files played as they are, depth frames, and the delete /
restore / compact / repair curation, which already sat on the shared
``dataset_edit``. What stayed behind is what knows the SO-101: its composite
sensor-view render and its end-effector kinematics. The playback says so
(``mp4: false``, ``ee: null``), and the page draws what it has.

A rig is described by its own dataset: the state's feature names are
``<limb>_<joint>.pos``, so the limbs and joints come from ``meta/info.json``
and a seven-channel UR3e and a twelve-channel dual SO-101 play back alike.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from aiohttp import web  # type: ignore[import]

# Read from the local drive only. A dataset whose metadata is momentarily
# incomplete would otherwise send LeRobot to the Hub on its bare name and come
# back as an offline-mode or 401 failure that says nothing about the real one.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from actoris_harena.recording.dataset_check import (  # noqa: E402
    DatasetDamaged,
    dataset_integrity,
    repair_phantom_episodes,
)
from actoris_harena.recording.dataset_edit import (  # noqa: E402
    ReadOnlyDatasetError,
    compact_dataset,
    read_episode_lengths,
    read_soft_deleted,
    saved_episode_total,
    surviving_indices,
    writability_problem,
    write_soft_deleted,
)
from actoris_harena.web.jobs import finish, new_job, refuse_while_busy  # noqa: E402
from actoris_harena.web.lifecycle import (  # noqa: E402
    directory_size,
    is_working_dir,
    read_dataset_meta,
)
from actoris_harena.web.util import in_executor  # noqa: E402

#: How much of an episode's tail the viewer leaves unplayed. Episodes are packed
#: end to end in a shared video file, so decoders kept in step by seeking can
#: stray across the seam into the next recording for a frame; stopping short
#: keeps every displayed frame inside the episode. The VIEW only -- nothing is
#: removed from the record.
_VIEW_END_MARGIN_S = 0.1


# ── Reading a dataset (blocking; run in the executor) ────────────────────────


def saved_episode_count(root: Path) -> int:
    """Episodes written to the dataset at ``root``, from its info.json; 0 if none."""
    info = Path(root) / "meta" / "info.json"
    if not info.is_file():
        return 0
    try:
        return int(json.loads(info.read_text()).get("total_episodes", 0))
    except (ValueError, TypeError, AttributeError):
        return 0


def realsense_meta(root: Path) -> "tuple[str | None, float]":
    """``(depth_name, depth_scale)`` from ``meta/realsense.json``, or (None, 0.001)."""
    p = Path(root) / "meta" / "realsense.json"
    if not p.is_file():
        return None, 0.001
    try:
        meta = json.loads(p.read_text())
    except ValueError:
        return None, 0.001
    return meta.get("depth_name"), float(meta.get("depth_scale_m_per_unit") or 0.001)


def depth_png_path(root: Path, depth_name: str, episode: int, frame: int) -> Path:
    """Where ``recording.depth.DepthWriter`` put one depth frame."""
    return (
        Path(root)
        / "extra"
        / "depth"
        / depth_name
        / f"episode_{episode:06d}"
        / f"{frame:06d}.png"
    )


def state_columns(root: Path) -> "list[dict[str, Any]]":
    """Every state channel as ``{index, limb, joint}``, from the dataset's names.

    ``arm_shoulder_pan.pos`` is limb ``arm``, joint ``shoulder_pan``. A name
    without a limb prefix keeps the whole name under limb ``state``, so a
    dataset that does not follow the convention still plays, ungrouped.
    """
    try:
        info = json.loads((Path(root) / "meta" / "info.json").read_text())
        names = list(info["features"]["observation.state"].get("names") or [])
    except (OSError, ValueError, KeyError, TypeError):
        names = []
    columns = []
    for index, raw in enumerate(names):
        name = str(raw)
        if name.endswith(".pos"):
            name = name[: -len(".pos")]
        limb, sep, joint = name.partition("_")
        if not sep:
            limb, joint = "state", name
        columns.append({"index": index, "limb": limb, "joint": joint})
    return columns


def list_datasets(root: Path) -> "list[dict]":
    """Every immediate sub-directory of ``root``, with what it holds.

    The episode count is what SURVIVES: episodes marked for deletion are left
    out, so the list agrees with the recording pane beside it. A directory with
    no saved episodes is still listed (``stillborn``) -- a session that quit
    before recording leaves one, and deleting it here is the point. The
    console's own working directories are debris and are not listed.
    """
    from actoris_harena.recording.dataset_read import camera_label

    out: list[dict] = []
    if root is None or not Path(root).is_dir():
        return out
    for child in sorted(Path(root).iterdir()):
        if not child.is_dir() or is_working_dir(child.name):
            continue
        n = saved_episode_count(child)
        pending = len([i for i in read_soft_deleted(child) if i < n])
        meta = read_dataset_meta(child)
        out.append(
            {
                "name": child.name,
                "episodes": n - pending,
                "pending": pending,
                "saved": n,
                "fps": meta["fps"],
                "robot_type": meta["robot_type"],
                "streams": [camera_label(k) for k in meta["streams"]],
                "tasks": meta["tasks"],
                "bytes": directory_size(child),
                "stillborn": n == 0,
                "damaged": meta["fps"] is None,
            }
        )
    return out


def dataset_root(root: Path, name: str) -> Path:
    """A dataset directory under ``root``, validated (no traversal)."""
    if "/" in name or name in ("", ".", ".."):
        raise web.HTTPBadRequest(text="bad dataset name")
    path = Path(root) / name
    if not path.is_dir() or saved_episode_count(path) == 0:
        raise web.HTTPNotFound(text=f"no dataset {name!r}")
    return path


def episodes_of(root: Path, name: str) -> dict:
    """The visible recordings, plus how many are marked for deletion.

    Marked episodes are hidden but NOT renumbered: they keep their on-disk index
    until compaction, and the video route needs it. An episode whose length did
    not read is listed with no length rather than taking the pane down.
    """
    path = dataset_root(root, name)
    total = saved_episode_total(path)
    lengths, bad = read_episode_lengths(path)
    if not total:
        total = (max(lengths) + 1) if lengths else 0
    marked = [i for i in read_soft_deleted(path) if 0 <= i < total]
    visible = surviving_indices(total, marked)
    missing = [i for i in range(total) if i not in lengths]
    return {
        "episodes": [{"index": k, "length": lengths.get(k)} for k in visible],
        "pending": len(marked),
        "total": total,
        "damaged": bad,
        "integrity": {
            "ok": not missing,
            "repairable": bool(missing),
            "summary": (
                ""
                if not missing
                else f"episode(s) {', '.join(str(i) for i in missing[:6])}"
                + ("" if len(missing) <= 6 else f" (+{len(missing) - 6} more)")
                + " are counted by the dataset but have no recording behind them"
            ),
            "phantom": missing,
        },
    }


def _dataset_fps(path: Path) -> int:
    try:
        return int(json.loads((path / "meta" / "info.json").read_text()).get("fps", 30))
    except (OSError, ValueError, TypeError, AttributeError):
        return 30


def uncommitted_episode(recording: bool, name: str, episode: int) -> str:
    """Why an episode has no metadata row yet, in the operator's terms. Pure."""
    if recording:
        return (
            f"episode {episode} of '{name}' is still being written: its frames "
            "are on disk but the session has not committed its metadata yet. It "
            "appears here as soon as it does."
        )
    return (
        f"episode {episode} of '{name}' has no metadata row, and no session is "
        "running to write one -- the session that recorded it ended before "
        "committing this episode. If the count does not settle when the dataset "
        "is next opened for recording, Repair drops the empty slot."
    )


def _round_rows(array, digits: int) -> "list[list[float]]":
    """``(N, C)`` → nested lists, rounded and finite (a bare NaN is invalid JSON)."""
    import numpy as np

    clean = np.nan_to_num(
        np.asarray(array, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    return [[round(v, digits) for v in row] for row in clean.tolist()]


def _peaks(array, columns: int) -> "list[float]":
    import numpy as np

    clean = np.nan_to_num(
        np.asarray(array, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    if clean.size == 0:
        return [0.0] * columns
    return [float(v) for v in np.abs(clean).max(axis=0)]


def motion_payload(state, fps: float, columns: "list[dict[str, Any]]") -> dict:
    """Joint velocity and acceleration by finite differences, per channel.

    Rig-neutral on purpose: an end effector needs the robot's kinematic model,
    which this console does not have, so ``ee`` is null and the page says so.
    """
    import numpy as np

    q = np.asarray(state, dtype=float)
    n = q.shape[1] if q.ndim == 2 else len(columns)
    if q.ndim != 2 or q.shape[0] < 2:
        vel = np.zeros((q.shape[0] if q.ndim == 2 else 0, n))
        acc = vel
    else:
        dt = 1.0 / float(fps or 30)
        vel = np.gradient(q, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
    units = ["/s" if c["joint"] == "gripper" else "°/s" for c in columns] or ["/s"] * n
    return {
        "joint_vel": _round_rows(vel, 2),
        "joint_acc": _round_rows(acc, 0),
        "joint_units": units,
        "ee": None,
        "ee_error": "this console has no kinematic model of the robot",
        "peaks": {
            "joint_vel": [round(v, 2) for v in _peaks(vel, n)],
            "joint_acc": [round(v) for v in _peaks(acc, n)],
        },
    }


def episode_playback(
    root: Path, name: str, episode: int, recording: bool = False
) -> dict:
    """Everything the browser needs to play one episode without a render.

    The recorded videos are what a reviewer wants to see and the browser can
    decode them, so each stream names its file and the window inside it this
    episode occupies (several episodes share one file); the window is reported
    INCLUSIVE of its last frame. Joints come along as numbers, with the limbs
    and joint names the page lays them out by, and depth, which is not a video,
    as a per-frame image route.
    """
    from actoris_harena.recording.dataset_read import (
        camera_label,
        playable_window,
        read_episode_joints,
        read_episode_row,
        video_keys,
    )

    path = dataset_root(root, name)
    row = read_episode_row(path, episode)
    if row is None:
        raise web.HTTPConflict(text=uncommitted_episode(recording, name, episode))
    fps = _dataset_fps(path)
    streams = []
    for key in video_keys(path):
        start, end = playable_window(row, key, fps)
        end = max(end - _VIEW_END_MARGIN_S, start)
        streams.append(
            {
                "key": key,
                "label": camera_label(key),
                "url": f"/api/datasets/{name}/episodes/{episode}/video/{key}",
                "from": start,
                "to": end,
            }
        )
    state, action = read_episode_joints(path, row)
    columns = state_columns(path)
    limbs: list[str] = []
    for c in columns:
        if c["limb"] not in limbs:
            limbs.append(c["limb"])
    depth_name, _scale = realsense_meta(path)
    depth = None
    if depth_name:
        folder = depth_png_path(path, depth_name, episode, 0).parent
        frames = len(list(folder.glob("*.png"))) if folder.is_dir() else 0
        if frames:
            depth = {
                "name": depth_name,
                "frames": frames,
                "url": f"/api/datasets/{name}/episodes/{episode}/depth/",
            }
    return {
        "episode": episode,
        "fps": fps,
        "streams": streams,
        "limbs": limbs,
        "columns": columns,
        **motion_payload(state, fps, columns),
        "depth": depth,
        "has_depth": depth is not None,
        # No composite render in this console: that view is the SO-101's own.
        "mp4": False,
        "state": [[round(v, 1) for v in frame] for frame in state.tolist()],
        "action": [[round(v, 1) for v in frame] for frame in action.tolist()],
    }


def episode_video_file(root: Path, name: str, episode: int, key: str) -> Path:
    """The recorded video file holding ``episode``'s frames for one camera."""
    from actoris_harena.recording.dataset_read import (
        episode_video_path,
        read_episode_row,
        video_keys,
    )

    path = dataset_root(root, name)
    if key not in video_keys(path):
        raise web.HTTPNotFound(text=f"'{name}' has no camera stream '{key}'")
    row = read_episode_row(path, episode)
    if row is None:
        raise web.HTTPConflict(
            text=f"episode {episode} of '{name}' is not readable yet"
        )
    video = episode_video_path(path, row, key)
    if not video.is_file():
        raise web.HTTPNotFound(text=f"{video.name} is missing from the dataset")
    return video


def depth_range(depth_mm) -> "tuple[float, float]":
    """The colour scale for one episode's depth, from a frame of it. Pure.

    The 2nd to 98th percentile of the valid readings, so one stray far pixel
    does not wash the whole scene into one colour. Fixed for the episode by the
    caller, so a colour means the same distance from frame to frame.
    """
    import numpy as np

    valid = np.asarray(depth_mm, dtype=float)
    valid = valid[valid > 0]
    if valid.size == 0:
        return 0.0, 1.0
    near, far = np.percentile(valid, [2, 98])
    return float(near), float(max(far, near + 1.0))


def colour_depth_frame(depth, near: float, far: float):
    """One 16-bit depth image as a JPEG, coloured; no reading is black."""
    import cv2  # type: ignore[import]
    import numpy as np

    d = np.asarray(depth, dtype=np.float32)
    grey = np.clip((d - near) * (255.0 / (far - near)), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)
    bgr[d <= 0] = 0
    ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise web.HTTPInternalServerError(text="could not encode the depth frame")
    return jpeg.tobytes()


def depth_frame_jpeg(
    root: Path, name: str, episode: int, frame: int, ranges: "dict[Any, Any]"
) -> bytes:
    import cv2  # type: ignore[import]

    path = dataset_root(root, name)
    depth_name, _scale = realsense_meta(path)
    if not depth_name:
        raise web.HTTPNotFound(text=f"'{name}' records no depth")
    png = depth_png_path(path, depth_name, episode, frame)
    depth = cv2.imread(str(png), cv2.IMREAD_UNCHANGED) if png.is_file() else None
    if depth is None:
        raise web.HTTPNotFound(text=f"no depth frame {frame} in episode {episode}")
    key = (str(path.resolve()), episode)
    if key not in ranges:
        first = cv2.imread(
            str(depth_png_path(path, depth_name, episode, 0)), cv2.IMREAD_UNCHANGED
        )
        ranges[key] = depth_range(depth if first is None else first)
    near, far = ranges[key]
    return colour_depth_frame(depth, near, far)


# ── Routes ───────────────────────────────────────────────────────────────────


def _recording(app: web.Application) -> bool:
    session = app.get("session")
    return bool(session is not None and session.running())


async def handle_datasets(request: web.Request) -> web.Response:
    app = request.app
    return web.json_response(await in_executor(app, list_datasets, app["root"]))


async def handle_episodes(request: web.Request) -> web.Response:
    app = request.app
    data = await in_executor(app, episodes_of, app["root"], request.match_info["name"])
    return web.json_response(data)


async def handle_playback(request: web.Request) -> web.Response:
    app = request.app
    data = await in_executor(
        app,
        episode_playback,
        app["root"],
        request.match_info["name"],
        int(request.match_info["episode"]),
        _recording(app),
    )
    return web.json_response(data)


async def handle_stream(request: web.Request) -> web.StreamResponse:
    """A recorded video file as it is: no decode, no encode, no cache.

    ``FileResponse`` answers range requests, so the browser seeks straight to
    the episode's window instead of pulling the whole file first.
    """
    app = request.app
    video = await in_executor(
        app,
        episode_video_file,
        app["root"],
        request.match_info["name"],
        int(request.match_info["episode"]),
        request.match_info["key"],
    )
    return web.FileResponse(video)


async def handle_depth(request: web.Request) -> web.Response:
    app = request.app
    jpeg = await in_executor(
        app,
        depth_frame_jpeg,
        app["root"],
        request.match_info["name"],
        int(request.match_info["episode"]),
        int(request.match_info["frame"]),
        app["depth_ranges"],
    )
    return web.Response(
        body=jpeg, content_type="image/jpeg", headers={"Cache-Control": "max-age=60"}
    )


def _mark_deleted(root: Path, name: str, indices: "list[int]") -> dict:
    path = dataset_root(root, name)
    marked = sorted(set(read_soft_deleted(path)) | set(indices))
    try:
        write_soft_deleted(path, marked)
    except OSError as exc:
        raise ReadOnlyDatasetError(
            f"cannot mark episodes: {writability_problem(path) or exc}"
        )
    return {"pending": len(marked)}


async def handle_delete(request: web.Request) -> web.Response:
    """Mark episodes deleted. Immediate; the rewrite waits for Compact."""
    app = request.app
    body = await request.json()
    indices = [int(i) for i in body.get("episodes", [])]
    if not indices:
        raise web.HTTPBadRequest(text="no episodes given")
    try:
        result = await in_executor(
            app, _mark_deleted, app["root"], request.match_info["name"], indices
        )
    except ReadOnlyDatasetError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response(result)


async def handle_restore(request: web.Request) -> web.Response:
    """Un-mark every episode marked for deletion (nothing was removed yet)."""
    app = request.app
    path = dataset_root(app["root"], request.match_info["name"])
    try:
        await in_executor(app, write_soft_deleted, path, [])
    except OSError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response({"pending": 0})


async def handle_compact(request: web.Request) -> web.Response:
    """Really remove the marked episodes: a JOB, because it rewrites the dataset."""
    app = request.app
    name = request.match_info["name"]
    path = dataset_root(app["root"], name)
    refuse_while_busy(app)
    marked = await in_executor(app, read_soft_deleted, path)
    if not marked:
        raise web.HTTPBadRequest(text="no episodes are marked for deletion")
    depth_name, _ = await in_executor(app, realsense_meta, path)
    problem = await in_executor(app, writability_problem, Path(app["root"]))
    if problem:
        raise web.HTTPConflict(text=f"cannot rewrite the dataset: {problem}")
    job = new_job(
        app,
        "compact",
        name,
        episodes=len(marked),
        message=f"removing {len(marked)} recording(s) from {name}",
    )

    def run() -> None:
        try:
            total = compact_dataset(path, name, [depth_name] if depth_name else [])
        except BaseException as exc:  # noqa: B036 - reported, not swallowed
            finish(job, "failed", str(exc) or exc.__class__.__name__)
            return
        finish(job, "done", f"{name} now holds {total} recording(s)")

    asyncio.get_running_loop().run_in_executor(app["job_executor"], run)
    return web.json_response(job)


async def handle_repair(request: web.Request) -> web.Response:
    """Drop episodes the dataset counts but never wrote: a JOB, like Compact."""
    app = request.app
    name = request.match_info["name"]
    path = dataset_root(app["root"], name)
    refuse_while_busy(app)
    report = await in_executor(app, dataset_integrity, path)
    if report["ok"]:
        raise web.HTTPBadRequest(text=f"{name} has nothing to repair")
    if not report["repairable"]:
        raise web.HTTPBadRequest(text=report["summary"])
    problem = await in_executor(app, writability_problem, Path(app["root"]))
    if problem:
        raise web.HTTPConflict(text=f"cannot rewrite the dataset: {problem}")
    depth_name, _ = await in_executor(app, realsense_meta, path)
    dropped = len(report["phantom"])
    job = new_job(
        app,
        "repair",
        name,
        episodes=dropped,
        message=f"dropping {dropped} empty episode slot(s) from {name}",
    )

    def run() -> None:
        try:
            result = repair_phantom_episodes(path, [depth_name] if depth_name else [])
        except (DatasetDamaged, OSError, ValueError) as exc:
            finish(job, "failed", str(exc) or exc.__class__.__name__)
            return
        finish(
            job,
            "done",
            f"{name} now holds {result['episodes']} recording(s); "
            f"{result['renumbered']} were renumbered",
        )

    asyncio.get_running_loop().run_in_executor(app["job_executor"], run)
    return web.json_response(job)


def add_dataset_routes(app: web.Application) -> None:
    """Register the browse / playback / curation routes. Needs ``app["root"]``,
    ``jobs`` and ``job_executor``; ``session`` is read if present."""
    app["depth_ranges"] = {}
    app.add_routes(
        [
            web.get("/api/datasets", handle_datasets),
            web.get("/api/datasets/{name}/episodes", handle_episodes),
            web.get(
                "/api/datasets/{name}/episodes/{episode}/playback", handle_playback
            ),
            web.get(
                "/api/datasets/{name}/episodes/{episode}/video/{key}", handle_stream
            ),
            web.get(
                "/api/datasets/{name}/episodes/{episode}/depth/{frame}.png",
                handle_depth,
            ),
            web.post("/api/datasets/{name}/delete", handle_delete),
            web.post("/api/datasets/{name}/restore", handle_restore),
            web.post("/api/datasets/{name}/compact", handle_compact),
            web.post("/api/datasets/{name}/repair", handle_repair),
        ]
    )
