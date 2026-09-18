"""A small loopback HTTP monitor served from inside a collection session.

One process owns the rig: the V4L2 nodes do not open twice and the arm buses are
lock-guarded, so nothing outside the recorder can read a camera while a session
runs. That is the reason this server exists inside it. It publishes what the
session already has in memory -- the latest frame of every stream, the recorder's
state, each stream's age and reconnect count -- over the loopback interface, so
the console can show a live view and the operator can drive an episode from a
browser without a display attached to the rig.

It is deliberately thin:

* frames come from the session's latest-frame store, which hands out a copy
  under its own lock, so serving a viewer cannot slow the record loop or
  interleave with it, and the joint snapshot beside them is read the same way --
  nothing here opens a device or talks to a bus;
* the control surface is an allow-list of keys, defaulting to the episode toggle
  and quit. Enabling, parking and homing drive the arm and stay on the headset
  and the keyboard, where the operator is looking at the rig;
* it runs its own event loop on a daemon thread, so it neither needs nor gets a
  say in the session's shutdown path.

IT KNOWS NOTHING ABOUT ANY PARTICULAR ROBOT, which is what lets one console
drive a dual SO-101 and a single UR3e. It takes a :class:`RobotSchema` for the
shape of the joint table and a :class:`MonitorSource` for the three questions a
frame store cannot answer -- the joints, whether the robot is armed, and how
stale the joint reading is. Everything else is the same on every rig.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from aiohttp import web  # type: ignore[import]

from actoris_harena.recording.features import ACTION_FRESH_S, RobotSchema
from actoris_harena.recording.monitor_wire import (
    BOUNDARY,
    DEFAULT_ALLOWED_KEYS,
    encode_frame_batch,
    encode_jpeg,
    key_refusal,
    mjpeg_part,
)

# The live view is a monitor, not a recording: a low rate and a small frame keep
# it far below the cost of the capture threads it watches, and JPEG at this
# quality is indistinguishable at tile size.
DEFAULT_VIEW_FPS = 10.0
DEFAULT_QUALITY = 70
DEFAULT_MAX_WIDTH = 480


@dataclass
class CollectionStatus:
    """What the recorder is doing, as the console's badge and bar show it.

    The server only ever reads these five fields and puts them into JSON, so a
    rig may pass anything shaped like this -- the SO-101's own view footer has
    used an identical dataclass since before there was a console. It lives here
    because this is the module that dereferences it, and a shared reader with a
    per-rig type is how the fields drift apart.

    ``episodes_goal`` of 0 means no goal, and hides the bar.
    """

    state_label: str
    episodes_done: int
    episodes_goal: int
    current_frames: int
    recording: bool = False


@runtime_checkable
class MonitorSource(Protocol):
    """What a session hands the monitor: its frames, and three robot questions.

    The frame half is the same three methods
    :class:`actoris_harena.recording.frames.FramePublisher` already defines, so
    a rig's existing teleop blackboard satisfies most of this by construction.

    The robot half is where rigs actually differ, and it is deliberately small.
    Asking for a finished ``joints`` table rather than for joint vectors is what
    keeps a dual-arm rig and a single-arm one on the same server: each builds
    its own with :func:`joint_snapshot`, which is pure and takes the schema.
    """

    def get_rgb_camera_names(self) -> "list[str]":
        """Every stream that has published at least one frame."""

    def get_rgb_image(self, name: str) -> Any:
        """The latest frame of one stream, or ``None``. A copy, under a lock."""

    def get_rgb_image_age(self, name: str, now: float) -> "float | None":
        """Seconds since that stream's latest frame was READ from the device."""

    def is_shutdown_requested(self) -> bool:
        """Whether the session is on its way out."""

    def get_teleop_active(self) -> bool:
        """Whether the operator currently has control -- the clutch, usually."""

    def monitor_joints(self, now: float) -> "dict[str, Any]":
        """The joint table, per limb. Built with :func:`joint_snapshot`."""

    def monitor_activity(self) -> str:
        """One word for what the robot is: enabled, disabled, parked, ...

        A string rather than an enum because it is going straight into JSON and
        the console only ever displays it.
        """

    def monitor_joint_drift_s(self, now: float) -> "float | None":
        """How stale the measured joints are, or ``None`` before the first read.

        Takes the same ``now`` as :meth:`monitor_joints` so the whole snapshot
        is read against one clock; the first thing to look at when the table
        stops moving.
        """


def joint_row(
    schema: RobotSchema, values: Any, gripper: "float | None"
) -> "dict[str, float | None]":
    """One limb's ``{joint: value}``: its body joints, then the gripper. Pure."""
    out: "dict[str, float | None]" = {}
    for i, name in enumerate(schema.body_joints):
        out[name] = None if values is None else round(float(values[i]), 3)
    if schema.gripper:
        out["gripper"] = None if gripper is None else round(float(gripper), 3)
    return out


def joint_snapshot(
    schema: RobotSchema,
    measured: Any,
    grippers: "dict[str, float | None]",
    commands: "dict[str, tuple[Any, float | None, float | None]]",
    now: float,
    fresh_s: float = ACTION_FRESH_S,
) -> "dict[str, Any]":
    """Every limb's measured joints beside its last sent command. Pure.

    The pair is the one the recorder STORES -- ``observation.state`` beside
    ``action`` -- so what an operator watches during collection is what a policy
    will later be trained on.

    ``measured`` is the whole rig's joint vector in URDF degrees, limb by limb in
    ``schema.limbs`` order, or ``None`` before the first read. ``commands[limb]``
    is ``(urdf_deg, gripper_open, t_monotonic)``. ``fresh`` says whether that
    command is recent enough to be the frame's action, which is the question an
    operator watching this table is really asking.
    """
    dof = schema.body_dof
    out: "dict[str, Any]" = {}
    for index, limb in enumerate(schema.limbs):
        segment = (
            None if measured is None else measured[index * dof : (index + 1) * dof]
        )
        urdf_deg, gripper_open, t_mono = commands[limb]
        age = None if t_mono is None else max(0.0, now - float(t_mono))
        out[limb] = {
            "state": joint_row(schema, segment, grippers.get(limb)),
            "command": joint_row(schema, urdf_deg, gripper_open),
            "command_age_s": None if age is None else round(age, 3),
            "fresh": age is not None and age < fresh_s,
        }
    return out


class MonitorServer:
    """Serves the live view and the allowed controls of a running session."""

    def __init__(
        self,
        source: MonitorSource,
        schema: RobotSchema,
        captures: "Iterable[Any]" = (),
        status_provider: "Callable[[], Any] | None" = None,
        key_callbacks: "dict[str, Callable[[], None]] | None" = None,
        allowed_keys: "Iterable[str]" = DEFAULT_ALLOWED_KEYS,
        host: str = "127.0.0.1",
        port: int = 8766,
        view_fps: float = DEFAULT_VIEW_FPS,
        quality: int = DEFAULT_QUALITY,
        max_width: int = DEFAULT_MAX_WIDTH,
    ) -> None:
        self.source = source
        self.schema = schema
        self.captures = list(captures)
        self.status_provider = status_provider
        self.key_callbacks = dict(key_callbacks or {})
        self.allowed_keys = frozenset(allowed_keys)
        self.host = host
        self.port = port
        self.view_fps = float(view_fps)
        self.quality = int(quality)
        self.max_width = int(max_width)

        self._thread: "threading.Thread | None" = None
        self._loop: "asyncio.AbstractEventLoop | None" = None
        self._runner: "web.AppRunner | None" = None
        self._started = threading.Event()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def build_app(self) -> web.Application:
        app = web.Application()
        app.add_routes(
            [
                web.get("/health", self.handle_health),
                web.get("/streams", self.handle_streams),
                web.get("/status", self.handle_status),
                web.get("/stream/{name}.mjpg", self.handle_mjpeg),
                web.get("/frames", self.handle_frames),
                web.post("/key", self.handle_key),
            ]
        )
        return app

    def start(self) -> None:
        """Serve on a daemon thread with its own event loop."""
        self._thread = threading.Thread(
            target=self._serve, name="monitor-server", daemon=True
        )
        self._thread.start()
        self._started.wait(timeout=5.0)

    def _serve(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._run())
            self._started.set()
            loop.run_forever()
        except Exception as exc:  # a dead monitor must never end the session
            print(f"⚠️  live monitor stopped: {exc}")
            self._started.set()
        finally:
            loop.close()

    async def _run(self) -> None:
        self._runner = web.AppRunner(self.build_app())
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        print(f"👁  live monitor on http://{self.host}:{self.port}/")

    def stop(self) -> None:
        """Close the site before stopping the loop, then join the thread.

        Stopping the loop outright leaves any in-flight request handler pending
        and asyncio complains about it on the way out -- "Task was destroyed but
        it is pending!", printed into the session's stdout, which the console
        tails and an operator reads. It says nothing anybody can act on, and it
        is the last thing in the log of an otherwise clean session.
        """
        loop = self._loop
        if loop is None:
            return

        # CLEAN UP FIRST, STOP SECOND, and not in one coroutine: a coroutine
        # that stops its own loop never delivers its future, so waiting on it
        # times out and the fallback then touches a loop `_serve` has already
        # closed -- which raises out of here and takes the session's teardown
        # with it. Two steps, each waited on separately.
        async def _close_site() -> None:
            if self._runner is not None:
                await self._runner.cleanup()

        try:
            asyncio.run_coroutine_threadsafe(_close_site(), loop).result(timeout=2.0)
        except Exception:  # noqa: BLE001 - a monitor that will not close is not fatal
            pass
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass  # already closed; nothing to stop
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ── State ────────────────────────────────────────────────────────────────

    def stream_names(self) -> "list[str]":
        """Every stream with a published frame, cameras named in config order."""
        published = set(self.source.get_rgb_camera_names())
        ordered = [c.name for c in self.captures if c.name in published]
        return ordered + sorted(published - set(ordered))

    def status(self) -> "dict[str, Any]":
        """One snapshot of the session, as the console's Collect pane shows it."""
        now = time.monotonic()
        streams = []
        by_name = {c.name: c for c in self.captures}
        for name in self.stream_names():
            capture = by_name.get(name)
            age = self.source.get_rgb_image_age(name, now)
            streams.append(
                {
                    "name": name,
                    "age_s": None if age is None else round(age, 3),
                    "configured_fps": getattr(capture, "fps", None),
                    "disconnects": getattr(capture, "disconnects", 0),
                }
            )
        out: "dict[str, Any]" = {
            "streams": streams,
            "joints": self.source.monitor_joints(now),
            # The console draws the table from the rig's own schema, so it needs
            # to be told which limbs and joints to expect rather than guessing
            # from the first payload -- a rig whose arm has not reported yet
            # would otherwise draw no table at all.
            "schema": {
                "limbs": list(self.schema.limbs),
                "body_joints": list(self.schema.body_joints),
                "gripper": bool(self.schema.gripper),
            },
            "joint_drift_s": self.source.monitor_joint_drift_s(now),
            "arms": self.source.monitor_activity(),
            "teleop_active": bool(self.source.get_teleop_active()),
            "shutdown_requested": bool(self.source.is_shutdown_requested()),
            "keys": sorted(self.key_callbacks),
            "allowed_keys": sorted(self.allowed_keys & set(self.key_callbacks)),
        }
        if self.status_provider is not None:
            collection = self.status_provider()
            out["recorder"] = {
                "state": collection.state_label,
                "episodes_done": collection.episodes_done,
                "episodes_goal": collection.episodes_goal,
                "current_frames": collection.current_frames,
                "recording": bool(collection.recording),
            }
        return out

    # ── Handlers ─────────────────────────────────────────────────────────────

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def handle_streams(self, request: web.Request) -> web.Response:
        return web.json_response({"streams": self.stream_names()})

    async def handle_status(self, request: web.Request) -> web.Response:
        return web.json_response(self.status())

    async def handle_key(self, request: web.Request) -> web.Response:
        body = await request.json()
        key = str(body.get("key") or "").lower()
        refusal = key_refusal(key, self.allowed_keys, self.key_callbacks)
        if refusal:
            raise web.HTTPForbidden(text=refusal)
        # The callbacks are the session's own button handlers, already written
        # to be called from a background thread (the headset reader does the
        # same) and already wrapped so a failing one cannot kill the caller.
        threading.Thread(
            target=self.key_callbacks[key], name=f"monitor-key-{key}", daemon=True
        ).start()
        return web.json_response({"pressed": key})

    async def handle_frames(self, request: web.Request) -> web.Response:
        """Every stream's latest frame in one response. See encode_frame_batch.

        The console polls this for its live tiles instead of opening one endless
        stream per camera, which is what let a seven-camera rig show four tiles.
        """
        names = self.stream_names()
        frames = {n: self.source.get_rgb_image(n) for n in names}
        loop = asyncio.get_running_loop()
        encoded = await loop.run_in_executor(
            None, encode_frame_batch, frames, self.quality, self.max_width
        )
        return web.json_response({"streams": names, "frames": encoded})

    async def handle_mjpeg(self, request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        if name not in self.stream_names():
            raise web.HTTPNotFound(text=f"no stream {name!r}")
        response = web.StreamResponse(
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
                "Cache-Control": "no-store",
            }
        )
        await response.prepare(request)
        period = 1.0 / self.view_fps if self.view_fps > 0 else 0.1
        loop = asyncio.get_running_loop()
        try:
            while True:
                frame = self.source.get_rgb_image(name)
                jpeg = await loop.run_in_executor(
                    None, encode_jpeg, frame, self.quality, self.max_width
                )
                if jpeg is not None:
                    await response.write(mjpeg_part(jpeg))
                await asyncio.sleep(period)
        except (ConnectionResetError, asyncio.CancelledError):
            pass  # the viewer went away; nothing to clean up
        return response
