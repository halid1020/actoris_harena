"""Routes for choosing the collection directory, and the guard for having none.

The rules are in ``actoris_harena.web.roots``; this layer reads the request, keeps the
blocking parts (mounting, walking a directory, asking the filesystem how much
room is left) off the event loop, and applies the one rule that belongs here:
the directory may not change under a running session or a running job, both of
which are working inside the current one.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from aiohttp import web  # type: ignore[import]

from actoris_harena.recording.dataset_edit import writability_problem
from actoris_harena.web.roots import (
    format_target,
    load_state,
    looks_like_collection,
    mount_dir,
    mount_remote,
    parse_ssh_target,
    remember,
    root_problem,
    save_state,
    sshfs_available,
    sshfs_mounts,
    unmount,
)
from actoris_harena.web.util import in_executor

# Requests that mean nothing without a collection directory. Everything else --
# the page itself, the job dock, the signal map, a session's status -- answers
# normally while the operator is still choosing one.
ROOT_REQUIRED_PREFIXES = ("/api/datasets",)
ROOT_REQUIRED_PATHS = (
    "/api/preflight",
    "/api/session/plan",
    "/api/session/start",
    # These three read a dataset off the drive. The rest of the Training tab --
    # which runs exist, how one is going, stopping one -- is about machines
    # elsewhere and answers with no directory chosen.
    "/api/training/config",
    "/api/training/plan",
    "/api/training/start",
)
NO_ROOT = "no collection directory chosen — pick one at the top of the page"


def needs_root(path: str) -> bool:
    """Does this request need a collection directory? Pure."""
    return path in ROOT_REQUIRED_PATHS or path.startswith(ROOT_REQUIRED_PREFIXES)


@web.middleware
async def root_required(request: web.Request, handler):
    """Refuse the dataset routes while there is no directory, in one place."""
    if request.app["root"] is None and needs_root(request.path):
        raise web.HTTPConflict(text=NO_ROOT)
    return await handler(request)


def _busy(app: web.Application) -> "str | None":
    """Why the directory cannot change right now."""
    # No supervisor until the first session is planned: the console builds it
    # on demand, once a rig and a drive are chosen.
    session = app["session"]
    if session is not None and session.running():
        return "a collection session is running — stop it first"
    running = [j for j in app["jobs"].values() if j["state"] == "running"]
    if running:
        return f"{running[0]['kind']} {running[0]['name']} is still running"
    return None


def switch_root(
    app: web.Application, path: Path, kind: str, target: "str | None"
) -> None:
    """Point the console at ``path``. The caller has already validated it."""
    app["root"] = Path(path)
    if app["session"] is not None:
        app["session"].root = Path(path)
    app["prerender_tasks"].clear()
    save_state(
        app["roots_file"], remember(load_state(app["roots_file"]), path, kind, target)
    )


def _describe(app: web.Application) -> "dict[str, Any]":
    """The directory pane's whole state. Blocking (it stats the filesystem)."""
    root = app["root"]
    mounts = {m["path"]: m["target"] for m in sshfs_mounts()}
    free = None
    writable = False
    if root is not None:
        try:
            free = shutil.disk_usage(root).free
        except OSError:
            free = None
        writable = not writability_problem(Path(root))
    state = load_state(app["roots_file"])
    return {
        "root": None if root is None else str(root),
        "kind": "ssh" if root is not None and str(root) in mounts else "local",
        "target": None if root is None else mounts.get(str(root)),
        "writable": writable,
        "free_bytes": free,
        "recent": state["recent"],
        "sshfs": sshfs_available(),
        "mounts": [
            {"path": p, "target": t}
            for p, t in sorted(mounts.items())
            if p.startswith(str(app["mount_dir"]))
        ],
        "busy": _busy(app),
    }


async def handle_roots(request: web.Request) -> web.Response:
    return web.json_response(await in_executor(request.app, _describe, request.app))


def _browse(path: Path) -> "dict[str, Any]":
    """The sub-directories of ``path``, flagged if they hold datasets. Blocking."""
    entries = []
    try:
        for child in sorted(Path(path).iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "collection": looks_like_collection(child),
                }
            )
    except OSError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    parent = str(Path(path).parent)
    return {
        "path": str(path),
        "parent": None if parent == str(path) else parent,
        "collection": looks_like_collection(Path(path)),
        "entries": entries[:400],
    }


async def handle_browse(request: web.Request) -> web.Response:
    raw = request.query.get("path") or str(request.app["root"] or Path.home())
    path = Path(raw).expanduser()
    problem = root_problem(path)
    if problem:
        raise web.HTTPBadRequest(text=problem)
    return web.json_response(await in_executor(request.app, _browse, path))


async def handle_use(request: web.Request) -> web.Response:
    """Work on a directory that is already on this machine (mounted or not)."""
    app = request.app
    body = await request.json()
    path = Path(str(body.get("path") or "")).expanduser()
    problem = root_problem(path)
    if problem:
        raise web.HTTPBadRequest(text=problem)
    busy = _busy(app)
    if busy:
        raise web.HTTPConflict(text=busy)
    mounts = {m["path"]: m["target"] for m in sshfs_mounts()}
    switch_root(
        app, path, "ssh" if str(path) in mounts else "local", mounts.get(str(path))
    )
    return web.json_response(await in_executor(app, _describe, app))


async def handle_mount(request: web.Request) -> web.Response:
    """Mount a remote collection directory over SSH, then work on it."""
    app = request.app
    body = await request.json()
    try:
        user, host, remote = parse_ssh_target(str(body.get("target") or ""))
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    busy = _busy(app)
    if busy:
        raise web.HTTPConflict(text=busy)
    port = body.get("port") or None
    identity = str(body.get("identity") or "").strip() or None
    target = format_target(user, host, remote)
    point = mount_dir(app["mount_dir"], host, remote)
    problem = await in_executor(app, mount_remote, target, point, port, identity)
    if problem:
        raise web.HTTPBadRequest(text=problem)
    problem = root_problem(point)
    if problem:
        raise web.HTTPBadRequest(text=problem)
    switch_root(app, point, "ssh", target)
    return web.json_response(await in_executor(app, _describe, app))


async def handle_unmount(request: web.Request) -> web.Response:
    """Release a mount. If it is the one in use, the console lets go of it first."""
    app = request.app
    body = await request.json()
    path = Path(str(body.get("path") or ""))
    busy = _busy(app)
    if busy:
        raise web.HTTPConflict(text=busy)
    if app["root"] is not None and Path(app["root"]) == path:
        app["root"] = None
    problem = await in_executor(app, unmount, path)
    if problem:
        raise web.HTTPConflict(text=problem)
    return web.json_response(await in_executor(app, _describe, app))


def unmount_own(app: web.Application) -> None:
    """Release the mounts this console made. Called when it exits."""
    base = str(app["mount_dir"])
    for mount in sshfs_mounts():
        if mount["path"].startswith(base):
            unmount(Path(mount["path"]))


def initial_root(cli_dir: "str | None", state_file: Path) -> "Path | None":
    """The directory to open on: the flag, else the last one, else none. Pure-ish."""
    if cli_dir:
        return Path(cli_dir).expanduser()
    last = load_state(state_file)["last"]
    if last and os.path.isdir(last):
        return Path(last)
    return None


def add_root_routes(app: web.Application) -> None:
    """Register the directory routes. ``roots_file`` and ``mount_dir`` must exist."""
    app.add_routes(
        [
            web.get("/api/roots", handle_roots),
            web.get("/api/roots/browse", handle_browse),
            web.post("/api/roots/use", handle_use),
            web.post("/api/roots/mount", handle_mount),
            web.post("/api/roots/unmount", handle_unmount),
        ]
    )
