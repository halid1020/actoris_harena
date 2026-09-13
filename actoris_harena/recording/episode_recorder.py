"""Episode recorder: the state machine that owns the LeRobot dataset writer.

A single background thread paces at the dataset fps and is the SOLE caller of
``add_frame`` / ``save_episode`` / ``clear_episode_buffer``. Button callbacks
never touch the writer — they only post ``request_start_episode`` /
``request_stop_save`` flags, which the loop consumes.

States: IDLE -> RECORDING -> (SAVING | DISCARDING) -> IDLE, with
RECORDING <-> PAUSED while a camera stream is missing.

Discard triggers while RECORDING:
* shutdown requested (``TimedFrameSource.is_shutdown_requested``) -> park;
* a thread error inside the loop -> park;
* robot activity became DISABLED -> NO park (a torque-off already happened and
  re-torquing unattended arms is riskier);
* a camera stream absent for longer than ``camera_dead_s`` -> NO park (a data
  problem, not a safety problem).
``park_arms`` (injected) is therefore invoked ONLY for the shutdown/thread-error
triggers.

Camera staleness is graded, because a USB camera that drops off the bus is
usually back within a couple of seconds and throwing away a long, otherwise
good episode for a momentary glitch costs the operator far more than the gap
does. A frame staler than one tick but within ``camera_stale_s`` is reused with
a throttled warning. Beyond ``camera_stale_s`` the recorder PAUSES: it stops
adding frames and holds the episode, then resumes it the moment every stream is
fresh again. Only if the stream stays away for ``camera_dead_s`` is the episode
abandoned.

A pause leaves a real hole in the episode, and it is deliberately NOT hidden:
LeRobot derives each frame's timestamp from its index, so the saved episode
would otherwise claim one tick between the frames either side of a multi-second
gap. Every pause is counted and totalled, and the episode summary says so, so an
operator can review or drop an episode whose gap is too large to train on.

Any automatic end (abandon, disabled arms, shutdown, thread error) plays the
stop cue, because an operator watching the arms rather than the terminal
otherwise cannot tell that recording ended and would keep performing the task.

Each saved episode also gets a timestamp id (``new_episode_uid``) recorded
beside it. The episode index cannot serve as a name: deleting one episode
renumbers every later one, so the same index refers to a different recording
afterwards. The id is fixed when recording starts and never moves.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from enum import Enum
from typing import Any, Callable

from actoris_harena.recording import features as feat
from actoris_harena.recording.dataset_edit import (
    commit_episode_metadata,
    new_episode_uid,
    read_episode_lengths,
    write_episode_uid,
)
from actoris_harena.recording.drift import DriftLog
from actoris_harena.recording.fault_report import session_fault_report
from actoris_harena.recording.frames import ObservationBuilder, TimedFrameSource
from actoris_harena.recording.usb_topology import device_location, directory_location

# The AV1 encoder prints a twenty-line configuration banner every time it starts,
# which is once per camera per episode: sixty lines an episode, saying the same
# thing each time and burying the warnings that do not. SVT_LOG=1 keeps its
# errors and drops the rest. Set here, before any encoder is constructed, so it
# applies however the collection tool was launched.
os.environ.setdefault("SVT_LOG", "1")


class RecorderState(Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    PAUSED = "PAUSED"
    SAVING = "SAVING"
    DISCARDING = "DISCARDING"


# States in which an episode is open and would be lost by an abrupt exit.
_IN_FLIGHT = (RecorderState.RECORDING, RecorderState.PAUSED)


class EpisodeRecorder:
    """Threaded, fps-paced episode recorder owning the dataset writer."""

    def __init__(
        self,
        dataset: Any,
        frames: TimedFrameSource,
        observations: ObservationBuilder,
        task: str,
        fps: int,
        camera_names: list[str],
        cameras: list[Any] | None = None,
        sidecar: Any | None = None,
        park_arms: Callable[[], None] | None = None,
        camera_stale_s: float = 0.5,
        camera_dead_s: float = 5.0,
        depth_streams: list[str] | None = None,
        depth_writer: Any | None = None,
        record_ee: bool = False,
        audio_cue: Any | None = None,
    ) -> None:
        self.dataset = dataset
        #: Where the pictures come from, looked up BY TIME so every stream in a
        #: frame is sampled at the same instant.
        self.frames = frames
        #: The part of a frame that is about the robot -- see ObservationBuilder.
        #: Splitting these two is what lets one recorder serve a dual five-joint
        #: rig and a single six-joint one.
        self.observations = observations
        self.task = task
        self.fps = int(fps)
        self.camera_names = list(camera_names)
        self.cameras = list(cameras) if cameras is not None else []
        self.sidecar = sidecar
        self.park_arms = park_arms
        self.camera_stale_s = camera_stale_s
        # How long a stream may stay absent before the episode is abandoned. Must
        # comfortably exceed the camera reopen interval, or a recoverable dropout
        # is given up on before the retry that would have fixed it.
        self.camera_dead_s = camera_dead_s
        # Optional aligned 16-bit depth streams (RealSense), written outside the
        # video encoder by ``depth_writer`` (a DepthWriter), keyed by frame idx.
        self.depth_streams = list(depth_streams) if depth_streams is not None else []
        self.depth_writer = depth_writer
        # Optional audible start/stop cue (AudioCue); None = silent. Best-effort:
        # every call is non-blocking and swallows its own errors.
        self.audio_cue = audio_cue
        # Whether to ask the observation builder for EE-space features. WHAT
        # they are, and what frame they are in, is the builder's business -- this
        # used to compute one rig's world-to-base transforms here, which meant a
        # shared recorder importing a particular robot's sidecar module.
        self.record_ee = record_ee

        self._lock = threading.Lock()
        self._state = RecorderState.IDLE
        self._pending_start = False
        self._pending_stop = False
        self._stop = threading.Event()
        self._external_shutdown = False
        self._thread: threading.Thread | None = None

        # Per-episode diagnostics.
        self._tick_durations: list[float] = []
        self._frame_count = 0
        self._last_reuse_warn = 0.0
        # Frames the recorder had to reuse because no camera had a fresher one,
        # and the worst age among them. Reported per episode rather than per
        # occurrence (see _stale_camera).
        self._reuse_frames = 0
        self._worst_reuse_s = 0.0
        self._last_overrun_warn = 0.0
        # Pause bookkeeping. ``_pause_started`` is the monotonic instant the
        # current pause began (None while recording normally); the count and
        # total describe the whole episode and are reported when it ends.
        self._pause_started: float | None = None
        self._pause_count = 0
        self._paused_total_s = 0.0
        # Stable identity for the episode currently being recorded (see
        # ``new_episode_uid``). Set when recording starts, written out on save.
        self._episode_uid = ""
        # Temporal-alignment telemetry + action-fallback tally.
        self._drift = DriftLog()
        self._fallback_frames = 0
        self.root = getattr(dataset, "root", None)
        # Set when the storage the dataset is being written to stops answering.
        # Once that happens nothing further can be saved, so the session ends
        # rather than reporting the same I/O error once per write attempt.
        self._storage_lost = False

        # Episode index tracking (kept in sync with the dataset, works on
        # resume where num_episodes > 0).
        self._episode_index = int(getattr(dataset, "num_episodes", 0) or 0)

    # ── Public API ───────────────────────────────────────────────────────────

    def get_state(self) -> RecorderState:
        with self._lock:
            return self._state

    def get_episode_count(self) -> int:
        """Episodes saved so far (== the next episode's index)."""
        with self._lock:
            return self._episode_index

    def get_current_frame_count(self) -> int:
        """Frames captured in the episode currently RECORDING (0 when idle)."""
        with self._lock:
            return self._frame_count

    def request_start_episode(self) -> bool:
        """Ask the loop to start recording. Rejected unless currently IDLE."""
        with self._lock:
            if self._state != RecorderState.IDLE:
                print(f"⚠️  cannot start episode: recorder is {self._state.value}")
                return False
            self._pending_start = True
            return True

    def request_stop_save(self) -> bool:
        """Ask the loop to stop and save. Rejected unless recording or paused.

        PAUSED counts: the operator pressed stop on an episode they consider
        finished, and a stream being briefly absent is no reason to refuse to
        keep the frames already captured.
        """
        with self._lock:
            if self._state not in (RecorderState.RECORDING, RecorderState.PAUSED):
                print(f"⚠️  cannot stop episode: recorder is {self._state.value}")
                return False
            self._pending_stop = True
            return True

    def start(self) -> None:
        """Start the sidecar, camera threads and the record loop."""
        if self.sidecar is not None:
            self.sidecar.start()
        if self.depth_writer is not None:
            self.depth_writer.start()
        for cam in self.cameras:
            cam.start(self.frames)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Finish any in-flight save, stop everything, finalize the dataset.

        The in-flight save/discard runs on the loop thread (sole writer owner);
        this method signals it, joins, then tears down the auxiliary threads and
        finalizes so the parquet footers are written.

        A save already in progress must run to completion: encoding a long,
        multi-camera episode takes far more than the discard budget, and if the
        join gave up on it the auxiliary teardown and ``finalize`` below would
        race the unfinished ``save_episode`` — leaving the videos unencoded and
        ``info.json`` never updated, i.e. the just-recorded episode lost. So the
        wait is bounded only while no episode is being written; while the
        recorder is SAVING it keeps waiting until the save finishes.
        """
        self._external_shutdown = True
        self._stop.set()
        if self._thread is not None:
            announced = False
            while self._thread.is_alive():
                if self.get_state() == RecorderState.SAVING:
                    if not announced:
                        print("💾 finishing in-flight episode save before quitting...")
                        announced = True
                    self._thread.join(timeout=1.0)  # save in progress; keep waiting
                    continue
                self._thread.join(timeout=10.0)  # no save pending; bound the wait
                break
        if self.sidecar is not None:
            self.sidecar.stop()
        for cam in self.cameras:
            cam.stop()
        if self.depth_writer is not None:
            self.depth_writer.stop()
        if self._storage_lost:
            # Finalizing writes to the drive that has gone, so it would only
            # raise again. Every episode already saved was committed as it was
            # saved, so say how many survived rather than leaving it in doubt.
            self._report_surviving_episodes()
        else:
            try:
                self.dataset.finalize()
            except Exception as e:
                if not self._note_storage_fault(e):
                    traceback.print_exc()
        self._report_device_faults()

    def _report_surviving_episodes(self) -> None:
        """State how many episodes are safely on disk after a storage failure."""
        if self.root is None:
            return
        try:
            lengths, unreadable = read_episode_lengths(self.root)
        except Exception:
            print("   could not read the dataset back to count what survived")
            return
        print(
            f"   {len(lengths)} episode(s) were committed to disk before the "
            "failure and are intact"
            + (f"; {len(unreadable)} file(s) unreadable" if unreadable else "")
        )

    # ── Record loop ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        from lerobot.utils.robot_utils import precise_sleep

        dt = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                tick_start = time.perf_counter()
                self._step()
                elapsed = time.perf_counter() - tick_start
                if self.get_state() == RecorderState.RECORDING:
                    self._tick_durations.append(elapsed)
                    if elapsed > dt:
                        self._warn_overrun(elapsed, dt)
                precise_sleep(dt - elapsed)
        except Exception:
            print("❌ record loop crashed; discarding in-flight episode")
            traceback.print_exc()
            self.frames.request_shutdown()
            if self.get_state() in _IN_FLIGHT:
                self._close_pause()
                self._discard(reason="thread_error", park=True)
        finally:
            # External shutdown while an episode is in flight -> discard + park.
            if self.get_state() in _IN_FLIGHT:
                self._close_pause()
                self._discard(reason="shutdown", park=self._external_shutdown)

    def _step(self) -> None:
        state = self.get_state()
        if state == RecorderState.IDLE:
            self._step_idle()
        elif state == RecorderState.RECORDING:
            self._step_recording()
        elif state == RecorderState.PAUSED:
            self._step_paused()
        # SAVING / DISCARDING are handled inline where they are entered.

    def _step_idle(self) -> None:
        with self._lock:
            start = self._pending_start
            self._pending_start = False
        if not start:
            return
        if not self._all_cameras_fresh():
            print("⚠️  cannot start episode: a camera stream is not fresh")
            return
        if self.sidecar is not None:
            self.sidecar.begin_episode()
        if self.depth_writer is not None:
            self.depth_writer.begin_episode(self._episode_index)
        self._tick_durations = []
        self._frame_count = 0
        self._drift.reset()
        self._fallback_frames = 0
        self._reuse_frames = 0
        self._worst_reuse_s = 0.0
        self._pause_started = None
        self._pause_count = 0
        self._paused_total_s = 0.0
        self._episode_uid = new_episode_uid()
        with self._lock:
            self._state = RecorderState.RECORDING
        if self.audio_cue is not None:
            self.audio_cue.play("start")
        print(
            f"🔴 recording episode {self._episode_index} "
            f"[{self._episode_uid}] (task: {self.task!r})"
        )

    def _step_recording(self) -> None:
        # A requested stop-save wins over every discard trigger: if the operator
        # pressed A to save and then quit, the episode they asked to keep must be
        # saved, not thrown away by the shutdown/disabled/stale checks below.
        with self._lock:
            stop = self._pending_stop
            self._pending_stop = False
        if stop:
            with self._lock:
                self._state = RecorderState.SAVING
            if self.audio_cue is not None:
                self.audio_cue.play("stop")
            self._save()
            return

        # Discard triggers, highest priority first.
        if self.frames.is_shutdown_requested():
            self._discard(reason="shutdown", park=True)
            self._stop.set()
            return
        if not self.observations.armed():
            self._discard(reason="disabled", park=False)
            return
        stale = self._stale_camera()
        if stale is not None:
            self._enter_pause(stale)
            return

        self._record_frame()

    def _step_paused(self) -> None:
        """Hold the episode while a camera is away; resume it, or give up.

        Mirrors the priority order of ``_step_recording``: an explicit stop-save
        wins, then the safety/shutdown triggers, then recovery.
        """
        with self._lock:
            stop = self._pending_stop
            self._pending_stop = False
        if stop:
            self._close_pause()
            with self._lock:
                self._state = RecorderState.SAVING
            if self.audio_cue is not None:
                self.audio_cue.play("stop")
            self._save()
            return

        if self.frames.is_shutdown_requested():
            self._close_pause()
            self._discard(reason="shutdown", park=True)
            self._stop.set()
            return
        if not self.observations.armed():
            self._close_pause()
            self._discard(reason="disabled", park=False)
            return

        # Resume on the SAME freshness gate that lets an episode start, so a
        # stream must really be delivering again, not merely twitching.
        if self._all_cameras_fresh():
            waited = self._close_pause()
            with self._lock:
                self._state = RecorderState.RECORDING
            print(
                f"▶️  resumed episode {self._episode_index} after {waited * 1e3:.0f} ms "
                f"({self._frame_count} frames so far)"
            )
            return

        started = self._pause_started
        if started is not None and time.monotonic() - started > self.camera_dead_s:
            missing = self._missing_camera() or "unknown"
            self._close_pause()
            self._discard(reason=f"camera_dead:{missing}", park=False)

    # ── Pause / resume ───────────────────────────────────────────────────────

    def _enter_pause(self, camera: str) -> None:
        """Hold the episode because ``camera`` has gone missing."""
        self._pause_started = time.monotonic()
        self._pause_count += 1
        with self._lock:
            self._state = RecorderState.PAUSED
        print(
            f"⏸️  paused episode {self._episode_index}: camera '{camera}' is not "
            f"fresh — holding up to {self.camera_dead_s:.0f} s for it to return"
        )

    def _close_pause(self) -> float:
        """End the current pause, adding it to the episode total. Returns its length."""
        if self._pause_started is None:
            return 0.0
        waited = time.monotonic() - self._pause_started
        self._paused_total_s += waited
        self._pause_started = None
        return waited

    # ── Frame building ───────────────────────────────────────────────────────

    def _record_frame(self) -> None:
        """One frame: every stream sampled at ONE reference time, or none at all.

        The reference time is the whole point. Sampling each stream whenever it
        is asked would make a recorded frame a mix of latest values, and the mix
        would differ from frame to frame -- so a policy would be learning from an
        alignment that drifts. Every stream is instead sampled AT t_ref and
        reports how far its nearest sample sat from it, and those drifts are
        written beside the episode.

        Any stream that is not ready aborts the WHOLE frame. A frame missing one
        stream is worse than a frame that does not exist, because every count
        downstream still agrees with it.
        """
        t_ref = time.monotonic()
        drifts: dict[str, float] = {}

        built = self.observations.state_and_action(t_ref, drifts)
        if built is None:
            return  # the rig has no state yet
        state, action, action_fell_back = built

        images = {}
        for name in self.camera_names:
            res = self.frames.get_rgb_image_at(name, t_ref)
            if res is None:
                return  # guarded by staleness, but never add a None frame
            images[name], drifts[name] = res

        depth_by_stream: dict = {}
        for name in self.depth_streams:
            res = self.frames.get_depth_image_at(name, t_ref)
            if res is None:
                return  # depth stream not ready; keep depth 1:1 with frames
            depth_by_stream[name], drifts[f"depth_{name}"] = res

        ee_pose_vec = ee_target_vec = None
        if self.record_ee:
            ee_res = self.observations.ee(t_ref, drifts)
            if ee_res is None:
                return  # keep the EE streams 1:1 with the frames
            ee_pose_vec, ee_target_vec = ee_res

        # LeRobot assigns frame_index = position in the episode buffer (0-based
        # = frames added so far); use that so depth files and drift rows align
        # 1:1 with the frames.
        frame_idx = self._frame_count
        teleop_active = self.observations.teleop_active()
        frame = feat.assemble_frame(
            state,
            action,
            images,
            self.task,
            teleop_active,
            ee_pose=ee_pose_vec,
            ee_target=ee_target_vec,
        )
        self.dataset.add_frame(frame)
        self._frame_count += 1
        if self.depth_writer is not None and depth_by_stream:
            self.depth_writer.add(frame_idx, depth_by_stream)

        self._drift.add(frame_idx, t_ref, drifts)
        if teleop_active and action_fell_back:
            self._fallback_frames += 1

    def _save(self) -> None:
        # A save that raises has NOT produced an episode, and everything below
        # describes one: side files named after an index the dataset does not
        # have, and a count one higher than what is on disk. A dataset that
        # counts an episode nobody wrote cannot be opened by LeRobot at all
        # (see actoris_harena.recording.dataset_check), so a failure here is treated as
        # a discard that says so, not as a save.
        saved = True
        try:
            self.dataset.save_episode()
        except Exception as e:
            saved = False
            if not self._note_storage_fault(e):
                traceback.print_exc()
        if saved:
            try:
                # Land this episode's metadata on disk now rather than at exit,
                # so an interrupted session keeps every episode it announced as
                # saved and a review tool can open the dataset mid-session. See
                # actoris_harena.recording.dataset_edit.commit_episode_metadata.
                commit_episode_metadata(self.dataset)
            except Exception:
                traceback.print_exc()
        if self.sidecar is not None:
            if saved:
                self.sidecar.end_episode(self._episode_index)
            else:
                self.sidecar.abort_episode()
        if self.depth_writer is not None:
            if saved:
                self.depth_writer.end_episode()
            else:
                self.depth_writer.abort_episode()
        if saved and self.root is not None:
            try:
                self._drift.write_parquet(self.root, self._episode_index)
            except Exception:
                traceback.print_exc()
            try:
                write_episode_uid(
                    self.root, self._episode_index, self._episode_uid, self.task
                )
            except Exception:
                traceback.print_exc()
        self._print_stats(outcome="saved" if saved else "NOT SAVED (write failed)")
        self._episode_index = self._next_episode_index(saved)
        with self._lock:
            self._state = RecorderState.IDLE

    def _next_episode_index(self, saved: bool) -> int:
        """Where the NEXT episode goes: whatever the dataset says, if it says.

        The dataset numbers episodes, not this recorder, and our side files are
        named after that number -- so following its count keeps the two locked
        together even when a save fails, rather than drifting by one and naming
        every later side file after somebody else's episode.
        """
        count = getattr(self.dataset, "num_episodes", None)
        if isinstance(count, int) and count >= 0:
            return count
        return self._episode_index + 1 if saved else self._episode_index

    def _note_storage_fault(self, exc: BaseException) -> bool:
        """Record whether ``exc`` means the storage itself has gone. Returns it.

        A vanished drive is not a per-call error to retry past: every later write
        raises the same thing, so a traceback per attempt buries the one fact that
        matters. It is recognised by the errno the kernel reports for a device
        that is no longer there, and it ends the session.
        """
        errno = getattr(exc, "errno", None)
        if errno not in (5, 19, 116):  # EIO, ENODEV, ESTALE
            return False
        if not self._storage_lost:
            self._storage_lost = True
            print(
                f"❌ the storage holding the dataset stopped responding "
                f"({exc.__class__.__name__}: errno {errno}) — nothing more can "
                "be recorded, ending the session"
            )
            self.frames.request_shutdown()
        return True

    def _report_device_faults(self) -> None:
        """Explain this session's device failures once, with the right culprit."""
        faults = {f"camera {cam.name}": cam.disconnects for cam in self.cameras}
        locations = {
            f"camera {cam.name}": device_location(cam.device) for cam in self.cameras
        }
        if self.root is not None:
            label = f"dataset drive {self.root}"
            faults[label] = 1 if self._storage_lost else 0
            locations[label] = directory_location(self.root)
        for line in session_fault_report(faults, locations):
            print(line)

    def _discard(self, reason: str, park: bool) -> None:
        with self._lock:
            self._state = RecorderState.DISCARDING
        # Every discard is an END the operator did not ask for. Without a cue an
        # operator watching the arms keeps performing a task that is no longer
        # being recorded, and reads the next press of the record button as a stop
        # when it is really a start.
        if self.audio_cue is not None:
            self.audio_cue.play("stop")
        try:
            self.dataset.clear_episode_buffer()
        except Exception as e:
            if not self._note_storage_fault(e):
                traceback.print_exc()
        if self.sidecar is not None:
            self.sidecar.abort_episode()
        if self.depth_writer is not None:
            self.depth_writer.abort_episode()
        self._print_stats(outcome=f"discarded ({reason})")
        if park and self.park_arms is not None:
            try:
                self.park_arms()
            except Exception:
                traceback.print_exc()
        with self._lock:
            self._state = RecorderState.IDLE

    # ── Camera freshness ─────────────────────────────────────────────────────

    def _all_cameras_fresh(self) -> bool:
        now = time.monotonic()
        for name in self.camera_names:
            age = self.frames.get_rgb_image_age(name, now)
            if age is None or age > self.camera_stale_s:
                return False
        return True

    def _missing_camera(self) -> str | None:
        """First camera stale beyond tolerance, else None. Never prints.

        Used while PAUSED, where the reuse warning of ``_stale_camera`` would be
        both wrong (nothing is being reused) and repetitive.
        """
        now = time.monotonic()
        for name in self.camera_names:
            age = self.frames.get_rgb_image_age(name, now)
            if age is None or age > self.camera_stale_s:
                return name
        return None

    def _stale_camera(self) -> str | None:
        """Return the first camera stale beyond tolerance, else None.

        A present-but-slightly-stale frame (within tolerance) is reused with a
        throttled warning; only beyond the tolerance is a camera reported.
        """
        now = time.monotonic()
        reuse_threshold = 1.5 / self.fps
        worst_reuse = 0.0
        for name in self.camera_names:
            age = self.frames.get_rgb_image_age(name, now)
            if age is None or age > self.camera_stale_s:
                return name
            if age > reuse_threshold:
                worst_reuse = max(worst_reuse, age)
        if worst_reuse > 0.0:
            self._reuse_frames += 1
            self._worst_reuse_s = max(self._worst_reuse_s, worst_reuse)
        # A reused frame is routine when a camera runs slower than the dataset
        # rate: it happens on most ticks, says the same thing every time, and
        # printing it buries the messages that are not routine. The episode
        # summary carries the count, and the drift table the distribution. Only a
        # stream approaching the point of pausing the episode is worth
        # interrupting for.
        if (
            worst_reuse > 0.5 * self.camera_stale_s
            and now - self._last_reuse_warn > 1.0
        ):
            self._last_reuse_warn = now
            print(
                f"⚠️  camera frame {worst_reuse * 1e3:.0f} ms old, close to the "
                f"{self.camera_stale_s * 1e3:.0f} ms limit that pauses the episode"
            )
        return None

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def _warn_overrun(self, elapsed: float, dt: float) -> None:
        now = time.monotonic()
        if now - self._last_overrun_warn > 1.0:
            self._last_overrun_warn = now
            print(
                f"⚠️  record tick overran: {elapsed * 1e3:.1f} ms > "
                f"{dt * 1e3:.1f} ms budget"
            )

    def _print_stats(self, outcome: str) -> None:
        n = len(self._tick_durations)
        if n:
            avg = sum(self._tick_durations) / n * 1e3
            worst = max(self._tick_durations) * 1e3
            tick_msg = f"avg tick {avg:.1f} ms, worst {worst:.1f} ms"
        else:
            tick_msg = "no ticks"
        fallback_msg = ""
        if self._frame_count:
            pct = 100.0 * self._fallback_frames / self._frame_count
            fallback_msg = (
                f", action-fallback {self._fallback_frames}/{self._frame_count} "
                f"({pct:.0f}%)"
            )
        depth_msg = ""
        if self.depth_writer is not None and self.depth_writer.dropped:
            depth_msg = f", depth-drops {self.depth_writer.dropped}"
        pause_msg = ""
        if self._pause_count:
            pause_msg = (
                f", paused {self._pause_count}x for {self._paused_total_s:.1f} s"
            )
        reuse_msg = ""
        if self._reuse_frames:
            pct = 100.0 * self._reuse_frames / max(self._frame_count, 1)
            reuse_msg = (
                f", reused {self._reuse_frames} frames ({pct:.0f}%, worst "
                f"{self._worst_reuse_s * 1e3:.0f} ms)"
            )
        uid_msg = f" [{self._episode_uid}]" if self._episode_uid else ""
        print(
            f"⏹️  episode {self._episode_index}{uid_msg} {outcome}: "
            f"{self._frame_count} frames, {tick_msg}{fallback_msg}{depth_msg}"
            f"{reuse_msg}{pause_msg}"
        )
        # A pause is a genuine hole: LeRobot timestamps frames from their index,
        # so the saved episode shows one tick across a gap that really lasted
        # seconds. Say so plainly rather than let a discontinuity reach training.
        if outcome == "saved" and self._pause_count:
            print(
                f"  ⚠️  this episode has {self._pause_count} recording gap(s) "
                f"totalling {self._paused_total_s:.1f} s, which the frame "
                f"timestamps do NOT show — review it before training and delete "
                f"it if the motion jumps."
            )
        if len(self._drift):
            print("  ⏱️  stream drift from tick reference:")
            print(self._drift.format_summary())
