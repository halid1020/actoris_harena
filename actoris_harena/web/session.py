"""Supervising one collection session, for whichever rig the console selected.

The rig is owned by one process. A collection session is therefore that rig's
unchanged teleoperation recorder, started here as a subprocess with exactly the
flags its command-line front-end would have computed -- the same resolver backs
both, so a session started from the browser and one started from a terminal
cannot drift apart -- plus a monitor port, which is what lets the console show
the cameras and press the two allowed keys while it runs.

WHICH RIG IS NOT DECIDED HERE. The interpreter and the entry point come from the
selected ``Rig``, through ``Rig.teleop_argv``, so this module never imports a
robot and never needs to: a venv with ur-rtde and one with feetech and mujoco
cannot be installed together, which is the whole reason the console runs devices
as subprocesses.

DEVICE REFUSALS ARE THE RIG'S. Whether a camera node is present, whether a
serial port is held, whether the USB bus has the bandwidth -- none of that can
be asked from this process, which is not the one with the hardware. A ``probe``
is passed in; without one the plan carries no device refusals, which is honest
rather than optimistic, and the session's own fail-fast camera open remains the
authority it always was.

Nothing in this module imports aiohttp: what is worth testing is the resolution
of a request into a command, the refusals that stop a session that cannot work,
and the escalation ladder that ends one. The routes are in ``session_api``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from actoris_harena.recording.collection_settings import (
    SelectionError,
    is_resumable_dataset,
    read_existing_streams,
    resolve_new_selection,
    resolve_resume_selection,
    selection_to_teleop_flags,
)
from actoris_harena.recording.dataset_edit import writability_problem
from actoris_harena.web.lifecycle import valid_dataset_name

#: What a rig may add to a plan that this process cannot work out for itself:
#: ``probe(selected_streams, options) -> (warnings, refusals)``.
DeviceProbe = Callable[["set[str]", "dict[str, Any]"], "tuple[list[str], list[str]]"]

# How the console ends a session. Quitting is a request the session honours at
# the top of its loop, which lets it park the arms, finish an in-flight episode
# and close the dataset properly -- so it is given time before anything harsher.
# An interrupt reaches the same teardown by another door. SIGKILL is absent on
# purpose: it would abandon an open episode and leave the arms energised.
QUIT_GRACE_S = 20.0
INTERRUPT_GRACE_S = 40.0


def stop_action(elapsed_s: float) -> str:
    """What ending a session should do next, ``elapsed_s`` after asking. Pure."""
    if elapsed_s < QUIT_GRACE_S:
        return "wait"
    if elapsed_s < INTERRUPT_GRACE_S:
        return "interrupt"
    return "terminate"


def session_refusals(
    root: Path,
    name: str,
    task: str,
    resuming: bool,
    exists: bool,
    running: bool,
) -> "list[str]":
    """Every reason this session cannot start, in the operator's terms. Pure."""
    reasons: list[str] = []
    if running:
        reasons.append("a session is already running — stop it first")
    problem = valid_dataset_name(name)
    if problem:
        reasons.append(problem)
    if not str(task).strip():
        reasons.append(
            "give the dataset an instruction — it is stored with every frame"
        )
    if exists and not resuming:
        reasons.append(
            f"{name!r} exists but holds no saved episodes (a session that quit "
            "before recording). Delete it on the Datasets tab, then start again"
        )
    problem = writability_problem(root)
    if problem:
        reasons.append(problem)
    return reasons


def resolve_plan(
    root: Path,
    name: str,
    task: str,
    options: "dict[str, Any]",
    config: "dict[str, Any]",
    running: bool = False,
    probe: "DeviceProbe | None" = None,
) -> "dict[str, Any]":
    """Turn a start request into the session's stream selection and flags.

    Returns ``{resuming, cameras, depth, ee, fps, warnings, refusals, flags}``.
    A resumed dataset follows its own recorded settings and reports whatever it
    ignored; a new one takes the request, defaulting to the machine's enabled
    cameras. Refusals are collected rather than raised so the form can show
    everything that is wrong at once.
    """
    dataset_root = Path(root) / name
    exists = dataset_root.exists()
    resuming = is_resumable_dataset(dataset_root)
    # Not every rig has a leader arm; the ones that do record a different
    # action space with it, which is why the plan carries the choice.
    leader = options.get("input") == "leader"

    refusals = session_refusals(Path(root), name, task, resuming, exists, running)
    known = set(config.get("cameras") or {})
    default_enabled = {
        n for n, c in (config.get("cameras") or {}).items() if c["enabled"]
    }
    rs_rgb_name = (config.get("realsense") or {}).get("rgb_name")

    warnings: list[str] = []
    selection: dict[str, Any] = {
        "cameras": set(),
        "depth": False,
        "ee": False,
        "fps": None,
    }
    try:
        if resuming:
            selection, warnings = resolve_resume_selection(
                read_existing_streams(dataset_root),
                rs_rgb_name,
                options.get("streams") or [],
                bool(options.get("depth")),
                not options.get("ee", True),
                options.get("fps"),
                leader=leader,
            )
        else:
            selection = resolve_new_selection(
                known,
                default_enabled,
                options.get("streams") or [],
                bool(options.get("depth")),
                bool(options.get("ee", True)) and not leader,
                options.get("fps"),
            )
    except SelectionError as exc:
        refusals.append(str(exc))

    # Whatever the rig can say about its own hardware -- a missing camera node,
    # a held serial port, a USB bus without the bandwidth. It cannot be asked
    # from here: this process is not the one with the devices, and the venv it
    # would need does not coexist with the next rig's. Without a probe the plan
    # simply carries no device refusals, and the session's own fail-fast camera
    # open stays the authority it has always been.
    if probe is not None:
        probe_warnings, probe_refusals = probe(set(selection["cameras"]), options)
        warnings += probe_warnings
        refusals += probe_refusals

    return {
        "resuming": resuming,
        "cameras": sorted(selection["cameras"]),
        "depth": bool(selection["depth"]),
        "ee": bool(selection["ee"]),
        "fps": selection["fps"],
        "warnings": warnings,
        "refusals": refusals,
        "flags": selection_to_teleop_flags(
            known,
            set(selection["cameras"]),
            bool(selection["depth"]),
            bool(selection["ee"]),
            selection["fps"],
        ),
    }


def teleop_argv(
    root: Path,
    name: str,
    task: str,
    plan: "dict[str, Any]",
    options: "dict[str, Any]",
    monitor_port: int,
    teleop: Path,
) -> "list[str]":
    """The command that runs this session. Pure — unit-tested."""
    argv = [
        str(teleop),
        "--input",
        str(options.get("input") or "quest"),
        "--record",
        "--repo-id",
        name,
        "--dataset-root",
        str(Path(root) / name),
        "--task",
        task,
        "--monitor-port",
        str(int(monitor_port)),
        *plan["flags"],
    ]
    if plan["resuming"]:
        argv.append("--resume")
    if options.get("ip_address"):
        argv += ["--ip-address", str(options["ip_address"])]
    if options.get("goal"):
        argv += ["--episode-goal", str(int(options["goal"]))]
    if options.get("sensor_view"):
        argv.append("--sensor-view")
    if options.get("mock"):
        # A REHEARSAL: the rig substitutes its own mocks for every device and
        # records a real dataset from them. It is how the console itself gets
        # tested without a robot, and how an operator checks the drive, the
        # naming and the episode buttons before anything is plugged in.
        argv.append("--mock")
    if options.get("no_streaming_encode"):
        argv.append("--no-streaming-encode")
    return argv


class SessionSupervisor:
    """The one collection session this console may run, and its output tail."""

    def __init__(
        self,
        root: Path,
        python: str,
        teleop: Path,
        monitor_port: int = 8766,
        tail_lines: int = 400,
    ) -> None:
        self.root = Path(root)
        self.monitor_port = int(monitor_port)
        # NOT sys.executable. The console's own interpreter has no robot in it
        # -- that is the point of the split -- so a default here would launch a
        # session that cannot import its own driver, in a venv nobody chose.
        # `Rig.python` and `Rig.teleop` are where these come from, and
        # `rigs.py` resolves them with normpath rather than resolve() because
        # a venv's bin/python is a symlink to the system interpreter.
        self.python = str(python)
        self.teleop = Path(teleop)
        self._proc: subprocess.Popen | None = None
        self._tail: deque = deque(maxlen=tail_lines)
        self._lock = threading.Lock()
        self._info: dict[str, Any] = {}
        self._stopping_since: float | None = None

    # ── State ────────────────────────────────────────────────────────────────

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def state(self) -> "dict[str, Any]":
        proc = self._proc
        with self._lock:
            tail = list(self._tail)
        return {
            "running": self.running(),
            "pid": None if proc is None else proc.pid,
            "returncode": None if proc is None else proc.poll(),
            "monitor_port": self.monitor_port,
            "stopping": self._stopping_since is not None,
            "tail": tail,
            **self._info,
        }

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(
        self, name: str, task: str, plan: "dict[str, Any]", options: "dict[str, Any]"
    ) -> "dict[str, Any]":
        if self.running():
            raise RuntimeError("a session is already running")
        argv = teleop_argv(
            self.root, name, task, plan, options, self.monitor_port, self.teleop
        )
        # The console has no terminal for the session to ask questions on, so
        # anything it would prompt for has to be answered here or not asked.
        argv.append("--yes")
        env = dict(os.environ)
        # Collection is local-only: a stray Hub lookup while creating or
        # resuming the dataset would fail with a misleading credentials error.
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        with self._lock:
            self._tail.clear()
        self._stopping_since = None
        self._proc = subprocess.Popen(
            [self.python, *argv],
            # NO stdin. Inherited, it would be the terminal the console itself
            # was launched in -- and a leader session reads its control keys
            # from stdin, so it would put that terminal into raw mode underneath
            # the operator's shell and then race the shell for every keystroke.
            # Neither reader gets a usable stream. The session is driven from
            # the page instead, which is where the operator is looking.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._info = {
            "name": name,
            "task": task,
            "argv": argv,
            "input": str(options.get("input") or "quest"),
            "resuming": plan["resuming"],
            "cameras": plan["cameras"],
            "depth": plan["depth"],
            "ee": plan["ee"],
            "fps": plan["fps"],
            "started": time.time(),
        }
        threading.Thread(target=self._read_output, daemon=True).start()
        return self.state()

    def _read_output(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                with self._lock:
                    self._tail.append(line.rstrip("\n"))
        finally:
            proc.wait()
            proc.stdout.close()

    def signal_stop(self) -> None:
        """Note that the operator asked to end the session (the key is pressed
        by the caller, through the monitor). The ladder consults the clock."""
        if self._stopping_since is None:
            self._stopping_since = time.monotonic()

    def escalate(self) -> str:
        """Apply the next step of the stop ladder, and say which it was."""
        if not self.running() or self._stopping_since is None:
            return "done"
        action = stop_action(time.monotonic() - self._stopping_since)
        proc = self._proc
        if proc is None:
            return "done"
        if action == "interrupt":
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        elif action == "terminate":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return action
