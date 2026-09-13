"""What a capture thread needs from whatever it publishes into.

The camera threads used to annotate their argument as ``DualDataManager`` -- a
dual-arm blackboard carrying IK targets, controller state, leader mappings and
last-sent commands, none of which a camera has any use for. Three methods is the
whole of what they actually call, so three methods is the contract.

Any object with these satisfies it structurally; nothing has to inherit. The
dual SO-101's ``DualDataManager`` already does, and a single-arm rig writes
whatever suits it -- or uses :class:`FrameStore`.
"""

import threading
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FramePublisher(Protocol):
    """The publishing side of a frame store, as a capture thread uses it.

    Runtime-checkable so a rig can assert its own store fits before a session
    rather than finding out mid-episode. Note what that check does and does not
    cover: ``isinstance`` against a Protocol tests that the METHODS EXIST, not
    that their signatures match, so it catches a forgotten method and not a
    wrong keyword.
    """

    def set_rgb_image(
        self, rgb: np.ndarray, name: str, t_capture: "float | None" = None
    ) -> None:
        """Publish one RGB frame under ``name``, stamped when it was READ.

        ``t_capture`` is a ``time.monotonic()`` reading taken as close to the
        device read as the driver allows. It is what every stream is later
        aligned through, so a frame stamped on arrival instead of on read moves
        the whole episode's alignment by the queueing delay.
        """

    def set_depth_image(
        self, depth16: np.ndarray, name: str, t_capture: "float | None" = None
    ) -> None:
        """Publish one 16-bit depth frame under ``name``, same stamping rule."""

    def is_shutdown_requested(self) -> bool:
        """True once the session is ending, so a capture loop can stop."""


class FrameStore:
    """The last frame per name, with the stamp it was read at.

    A minimal :class:`FramePublisher` for a rig that has no blackboard of its
    own. Deliberately plain: one lock, two dicts, and no history -- anything
    that needs to look a stream up by time uses ``actoris_harena.sync``, which
    is where that reasoning belongs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rgb: "dict[str, tuple[np.ndarray, float | None]]" = {}
        self._depth: "dict[str, tuple[np.ndarray, float | None]]" = {}
        self._shutdown = threading.Event()

    def set_rgb_image(
        self, rgb: np.ndarray, name: str, t_capture: "float | None" = None
    ) -> None:
        with self._lock:
            self._rgb[name] = (rgb, t_capture)

    def set_depth_image(
        self, depth16: np.ndarray, name: str, t_capture: "float | None" = None
    ) -> None:
        with self._lock:
            self._depth[name] = (depth16, t_capture)

    def get_rgb_image(self, name: str) -> "np.ndarray | None":
        with self._lock:
            found = self._rgb.get(name)
        return None if found is None else found[0]

    def get_depth_image(self, name: str) -> "np.ndarray | None":
        with self._lock:
            found = self._depth.get(name)
        return None if found is None else found[0]

    def rgb_camera_names(self) -> "list[str]":
        with self._lock:
            return sorted(self._rgb)

    def is_shutdown_requested(self) -> bool:
        return self._shutdown.is_set()

    def request_shutdown(self) -> None:
        self._shutdown.set()


@runtime_checkable
class TimedFrameSource(Protocol):
    """Frames looked up BY TIME, which is what a recorder needs.

    Distinct from :class:`FramePublisher`, and the distinction matters. A capture
    thread publishes the newest frame; a recorder asks what the newest frame WAS
    at one reference time, so that every stream in a recorded frame is sampled at
    the same instant rather than being a mix of latest values.

    Each call returns ``(frame, drift_seconds)`` -- the drift being how far the
    chosen sample sits from the time that was asked for. That number is recorded
    per frame per stream, because an alignment that is quietly getting worse is
    otherwise invisible until a policy trains badly.
    """

    def get_rgb_image_at(
        self, name: str, t_ref: float
    ) -> "tuple[np.ndarray, float] | None":
        """The RGB frame nearest ``t_ref``, and its drift. None if none yet."""

    def get_depth_image_at(
        self, name: str, t_ref: float
    ) -> "tuple[np.ndarray, float] | None":
        """The depth frame nearest ``t_ref``, and its drift. None if none yet."""

    def get_rgb_image_age(self, name: str, now: float) -> "float | None":
        """Seconds since this stream last delivered, for staleness grading.

        ``now`` is passed in rather than read here so the recorder's whole
        staleness decision uses ONE clock reading: grading three cameras against
        three separate `time.monotonic()` calls would let them disagree about
        which is the stale one.
        """

    def is_shutdown_requested(self) -> bool:
        ...

    def request_shutdown(self) -> None:
        ...


@runtime_checkable
class ObservationBuilder(Protocol):
    """The part of a recorded frame that is about the ROBOT.

    Everything else a recorder does -- pacing, pausing on a stale camera,
    sampling images and depth at one reference time, assembling the frame,
    writing depth beside it, counting, tallying drift -- is the same on any rig.
    This is the part that is not, and it is three methods.

    Each takes the frame's reference time and the ``drifts`` dict to record into,
    so a rig reports its own streams' alignment the same way the cameras do.
    Returning None means "not ready", and the recorder SKIPS THE WHOLE FRAME
    rather than writing a partial one -- a frame missing a stream is worse than a
    frame that does not exist, because every count downstream still agrees with
    it.
    """

    def state_and_action(
        self, t_ref: float, drifts: "dict[str, float]"
    ) -> "tuple[np.ndarray, np.ndarray, bool] | None":
        """``(state, action, action_fell_back)`` at ``t_ref``.

        ``action_fell_back`` is True when the last commanded action was too stale
        to use and the measured state stood in for it. The recorder tallies those
        because they teach a spurious hold.
        """

    def ee(
        self, t_ref: float, drifts: "dict[str, float]"
    ) -> "tuple[np.ndarray, np.ndarray] | None":
        """``(measured, target)`` end-effector vectors, or None if not recorded."""

    def teleop_active(self) -> bool:
        """Whether teleoperation drove this frame, for the maskable phase flag."""

    def armed(self) -> bool:
        """Whether the rig is still enabled to move.

        A rig that was disabled mid-episode -- an e-stop, a released clutch on a
        leader arm, a torque cut -- stops producing meaningful actions, so the
        episode in progress is DISCARDED rather than saved short. Saving it would
        write frames whose action is the measured state, teaching a hold that
        nobody performed.
        """
