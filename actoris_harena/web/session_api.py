"""The Collect tab's routes: plan a session, run it, drive it, watch it.

TWO SOURCES OF PICTURES, and which one is live depends on whether a session is
running. The rig is owned by one process, so:

* while a session runs, the frames and the controls come from ITS loopback
  monitor, proxied here. Nothing else can open the camera;
* while none is, the console reaches the rig's own agent instead -- a separate
  subprocess in the rig's interpreter, which `agent_api` already proxies. This
  module says which of the two is live and gets out of the way.

NO RIG IS IMPORTED. The interpreter and the entry point come from the selected
`Rig`, the camera list from its `rig.yaml`, and the readiness table from its own
preflight tool, run as a subprocess. That is the whole reason the console can
serve a UR3e whose venv cannot be installed beside the SO-101's.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import aiohttp  # type: ignore[import]
from aiohttp import web  # type: ignore[import]

from actoris_harena.recording.monitor_wire import allowed_keys_for
from actoris_harena.rigs import Rig
from actoris_harena.web.session import SessionSupervisor, resolve_plan
from actoris_harena.web.util import in_executor

# The live view is polled by an image element, so a slow or absent monitor must
# fail fast rather than hold the browser's connection open.
MONITOR_TIMEOUT_S = 3.0

#: How long the rig's preflight may take before the page gives up on it. It
#: opens a camera and knocks on the arm's port, so it is not instant; it is also
#: not allowed to hang the tab.
PREFLIGHT_TIMEOUT_S = 30.0


def schema_json(rig: Rig) -> "dict[str, Any] | None":
    """A rig's channel layout for the page, or ``None`` if it declared none."""
    schema = rig.schema
    if schema is None:
        return None
    return {
        "limbs": list(schema.limbs),
        "body_joints": list(schema.body_joints),
        "gripper": bool(schema.gripper),
    }


def selected_rig(app: web.Application) -> Rig:
    name = app["rig"]
    if name is None:
        raise web.HTTPBadRequest(text="choose a rig first")
    rig = app["rigs"].get(name)
    if rig is None:
        raise web.HTTPBadRequest(text=f"no rig called {name!r}")
    return rig


def supervisor(app: web.Application) -> SessionSupervisor:
    """The one session this console may run, against the selected rig.

    Built on demand and kept until the rig or the drive changes, because it owns
    a subprocess: rebuilding it under a running session would lose the handle
    and leave an orphan holding the cameras.
    """
    rig = selected_rig(app)
    root = app["root"]
    if root is None:
        raise web.HTTPBadRequest(text="choose a collection directory first")
    existing = app.get("session")
    if existing is not None and app.get("session_key") == (rig.name, str(root)):
        return existing
    if existing is not None and existing.running():
        raise web.HTTPConflict(
            text=(
                "a session is running against another rig or drive; stop it "
                "before switching"
            )
        )
    if rig.teleop is None:
        raise web.HTTPBadRequest(
            text=f"rig {rig.name!r} declares no teleop entry point in its rig.yaml"
        )
    app["session"] = SessionSupervisor(
        root=Path(root),
        python=str(rig.python),
        teleop=Path(rig.teleop),
        monitor_port=int(app.get("monitor_port", 8766)),
    )
    app["session_key"] = (rig.name, str(root))
    return app["session"]


def running(app: web.Application) -> bool:
    """Whether a session is up, without building a supervisor to ask."""
    session = app.get("session")
    return session is not None and session.running()


def monitor_url(app: web.Application, path: str) -> str:
    port = app.get("session").monitor_port if app.get("session") else 0
    return f"http://127.0.0.1:{port}{path}"


async def monitor_get(app: web.Application, path: str) -> "dict[str, Any] | None":
    """Read JSON from the running session's monitor, or ``None`` if it is not up."""
    if not running(app):
        return None
    try:
        timeout = aiohttp.ClientTimeout(total=MONITOR_TIMEOUT_S)
        async with app["http"].get(monitor_url(app, path), timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


# ── Configuring the form ─────────────────────────────────────────────────────


async def handle_collect_config(request: web.Request) -> web.Response:
    """What the Collect form offers: this rig's streams and its channels.

    Taken from `rig.yaml`, which every rig declares precisely so the console can
    read it without importing the rig. A camera listed there may still be
    unplugged; the preflight is what says so.
    """
    rig = selected_rig(request.app)
    mode = "quest"
    return web.json_response(
        {
            "rig": rig.name,
            "title": rig.title,
            # `present` is deliberately null: whether the device node is there
            # can only be answered by the process that has the devices, and that
            # is the rig's preflight, not this one. The page leaves the box
            # ticked and the preflight is what turns a row red.
            "cameras": [
                {"name": name, "enabled": True, "present": None} for name in rig.cameras
            ],
            # `schema` is optional in rig.yaml. Null rather than a guess: the
            # page then draws the table from the running session's own payload,
            # which is the authority anyway, instead of inventing columns for a
            # robot that never declared any.
            "schema": schema_json(rig),
            # WHAT THIS PAGE CAN DO, which is not the whole control map. The
            # session's own allow-list is the authority and it admits two keys;
            # everything that moves the robot stays on the headset, where the
            # operator is looking at it. The rig prints its full table into the
            # session log, which this page tails.
            "controls": {
                mode: [
                    {
                        "key": "A",
                        "what": "start an episode; press again to save it",
                        "where": "headset or this page",
                    },
                    {
                        "key": "Q",
                        "what": (
                            "end the session: park, finish any episode, "
                            "close the dataset"
                        ),
                        "where": "headset or this page",
                    },
                    {
                        "key": "",
                        "what": ("enabling, homing and parking stay on the headset"),
                        "where": "headset",
                    },
                ]
            },
            "allowed_keys": sorted(allowed_keys_for(mode)),
        }
    )


def _run_preflight(rig: Rig) -> "dict[str, Any]":
    """The rig's own readiness table, in its own interpreter.

    A subprocess, like everything else that touches a device. The tool prints
    JSON and exits non-zero when something is red; a non-zero exit is therefore
    a RESULT, not a failure, and only a missing tool or a timeout is an error.
    """
    tool = Path(rig.root) / "tool" / "collect_preflight.py"
    if not tool.is_file():
        return {"ok": None, "checks": [], "problem": f"{rig.name} has no {tool.name}"}
    try:
        done = subprocess.run(
            [str(rig.python), str(tool), "--json"],
            capture_output=True,
            text=True,
            timeout=PREFLIGHT_TIMEOUT_S,
            cwd=str(rig.root),
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {"ok": None, "checks": [], "problem": f"preflight failed: {exc}"}
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError:
        tail = (done.stdout + done.stderr).strip().splitlines()[-3:]
        return {"ok": None, "checks": [], "problem": " / ".join(tail) or "no output"}


async def handle_preflight(request: web.Request) -> web.Response:
    app = request.app
    rig = selected_rig(app)
    if running(app):
        # The session holds the devices. Asking now would report them all
        # missing, which is true of this process and false of the rig.
        return web.json_response(
            {"ok": None, "checks": [], "problem": "a session is running"}
        )
    result = await in_executor(app, _run_preflight, rig)
    return web.json_response(result)


# ── Planning and starting ────────────────────────────────────────────────────


async def _plan_for(request: web.Request, body: "dict[str, Any]") -> "dict[str, Any]":
    app = request.app
    rig = selected_rig(app)
    if app["root"] is None:
        raise web.HTTPBadRequest(text="choose a collection directory first")
    # The camera set comes from rig.yaml; `enabled` is the console's choice, so
    # every known stream defaults to on and the operator turns them off.
    config = {"cameras": {name: {"enabled": True} for name in rig.cameras}}
    return await in_executor(
        app,
        resolve_plan,
        Path(app["root"]),
        str(body.get("name") or ""),
        str(body.get("task") or ""),
        body,
        config,
        running(app),
    )


async def handle_session_plan(request: web.Request) -> web.Response:
    body = await request.json() if request.can_read_body else {}
    return web.json_response(await _plan_for(request, body))


async def handle_session(request: web.Request) -> web.Response:
    """The session's state, and the monitor's if one is up."""
    app = request.app
    session = app.get("session")
    state = {"running": False, "tail": []} if session is None else session.state()
    state["rig"] = app["rig"]
    state["monitor"] = await monitor_get(app, "/status")
    return web.json_response(state)


async def handle_session_start(request: web.Request) -> web.Response:
    app = request.app
    body = await request.json() if request.can_read_body else {}
    session = supervisor(app)
    plan = await _plan_for(request, body)
    if plan["refusals"]:
        raise web.HTTPBadRequest(text="\n".join(plan["refusals"]))
    name = str(body.get("name") or "")
    task = str(body.get("task") or "")
    # Let go of the rig's agent first: it holds the cameras for the idle
    # preview, and two processes cannot open one camera. The second open
    # succeeds and then delivers nothing, which is the failure that wastes an
    # operator's afternoon.
    app["agents"].stop(app["rig"])
    try:
        state = await in_executor(app, session.start, name, task, plan, body)
    except RuntimeError as exc:
        raise web.HTTPConflict(text=str(exc))
    return web.json_response(state)


# ── Driving it ───────────────────────────────────────────────────────────────


async def _press(app: web.Application, key: str) -> "dict[str, Any]":
    if not running(app):
        raise web.HTTPConflict(text="no session is running")
    try:
        timeout = aiohttp.ClientTimeout(total=MONITOR_TIMEOUT_S)
        async with app["http"].post(
            monitor_url(app, "/key"), json={"key": key}, timeout=timeout
        ) as resp:
            text = await resp.text()
            if resp.status == 403:
                # The session's allow-list refused it. Pass the sentence
                # through: it explains WHY, which a bare 403 would not.
                raise web.HTTPForbidden(text=text)
            if resp.status != 200:
                raise web.HTTPBadGateway(text=text)
            return json.loads(text)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise web.HTTPBadGateway(text=f"the session's monitor did not answer: {exc}")


async def handle_session_episode(request: web.Request) -> web.Response:
    """Start an episode, or save the one in progress. The A key, remotely."""
    return web.json_response(await _press(request.app, "a"))


async def handle_session_key(request: web.Request) -> web.Response:
    body = await request.json() if request.can_read_body else {}
    key = str(body.get("key") or "").lower()
    return web.json_response(await _press(request.app, key))


async def handle_session_stop(request: web.Request) -> web.Response:
    """End the session: ask, then escalate on the clock.

    Q first, because that is the door that parks the arm, finishes an in-flight
    episode and closes the dataset. SIGKILL is never reached: it would abandon
    an open episode and leave the arm energised.
    """
    app = request.app
    session = app.get("session")
    if session is None or not session.running():
        return web.json_response({"running": False, "action": "done"})
    session.signal_stop()
    try:
        await _press(app, "q")
    except web.HTTPException:
        pass  # the monitor may already be gone; the ladder still applies
    action = await in_executor(app, session.escalate)
    return web.json_response({"running": session.running(), "action": action})


# ── Watching it ──────────────────────────────────────────────────────────────


async def handle_live_streams(request: web.Request) -> web.Response:
    """Which source is live, and what it has. The page picks its tiles from this."""
    app = request.app
    if not running(app):
        return web.json_response({"source": "agent", "streams": []})
    status = await monitor_get(app, "/streams")
    return web.json_response(
        {"source": "session", "streams": (status or {}).get("streams", [])}
    )


async def handle_live_frames(request: web.Request) -> web.Response:
    """Every stream's latest frame in one response, from the running session.

    One batched request rather than one endless stream per camera: a
    multipart/x-mixed-replace response never completes, so a page that gave each
    tile its own would spend one of the browser's ~6 connections per origin on
    each and only the first few would ever load.
    """
    app = request.app
    if not running(app):
        return web.json_response({"source": "agent", "streams": [], "frames": {}})
    batch = await monitor_get(app, "/frames")
    if batch is None:
        return web.json_response({"source": "session", "streams": [], "frames": {}})
    batch["source"] = "session"
    return web.json_response(batch)


async def handle_live_stream(request: web.Request) -> web.StreamResponse:
    """One camera, as MJPEG, proxied from the session's monitor chunk by chunk."""
    app = request.app
    name = request.match_info["name"]
    if not running(app):
        raise web.HTTPConflict(text="no session is running; use the rig agent")
    response: "web.StreamResponse | None" = None
    try:
        async with app["http"].get(
            monitor_url(app, f"/stream/{name}.mjpg")
        ) as upstream:
            if upstream.status != 200:
                raise web.HTTPNotFound(text=f"no stream {name!r}")
            response = web.StreamResponse(
                headers={
                    "Content-Type": upstream.headers.get(
                        "Content-Type", "multipart/x-mixed-replace"
                    ),
                    "Cache-Control": "no-store",
                }
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(64 * 1024):
                await response.write(chunk)
    except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionResetError):
        pass  # the viewer or the session went away
    return response if response is not None else web.Response(status=204)


def add_session_routes(app: web.Application) -> None:
    app.router.add_get("/api/collect/config", handle_collect_config)
    app.router.add_get("/api/preflight", handle_preflight)
    app.router.add_get("/api/session", handle_session)
    app.router.add_post("/api/session/plan", handle_session_plan)
    app.router.add_post("/api/session/start", handle_session_start)
    app.router.add_post("/api/session/episode", handle_session_episode)
    app.router.add_post("/api/session/key", handle_session_key)
    app.router.add_post("/api/session/stop", handle_session_stop)
    app.router.add_get("/api/live/streams", handle_live_streams)
    app.router.add_get("/api/live/frames", handle_live_frames)
    app.router.add_get("/api/live/stream/{name}", handle_live_stream)
