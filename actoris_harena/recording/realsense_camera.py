"""Intel RealSense RGB-D capture thread for data collection.

Mirrors :class:`actoris_harena.recording.cameras.CameraCapture` so the central RGB-D
camera plugs into the same tool/recorder machinery for its COLOUR stream (a
normal ``observation.images.<rgb_name>`` video feature), while additionally
publishing an aligned 16-bit DEPTH stream via
``FramePublisher.set_depth_image``. Colour and depth are stamped with the SAME
capture time so they stay co-timed for the reference-time collector.

Depth is aligned into the colour frame (``rs.align``) so the two share
intrinsics. The colour-stream intrinsics and the depth scale (metres per unit)
are read from the device at :meth:`open` and exposed for the dataset's
``realsense.json`` replicability record.

``pyrealsense2`` is imported at module load, so import this module only when a
RealSense is actually configured (the teleop tool does so behind
``--central-depth`` / ``realsense.enabled``); the base recording package never
imports it.
"""

from __future__ import annotations

import threading
import time
import traceback

import numpy as np
import pyrealsense2 as rs  # type: ignore[import]

from actoris_harena.recording.frames import FramePublisher

# Auto-exposure settling frames before the lock is applied.
_WARMUP_FRAMES = 30

#: How long one wait for a frameset may take before the stream counts as
#: stalled, and how many pipeline restarts in a row are tried before giving up.
#: A D435 handed over from a process that was killed rather than closed -- or
#: one on a USB 2 port -- can start, deliver, and then stop; librealsense then
#: raises out of ``wait_for_frames`` and the thread used to die there, leaving
#: a session recording with no camera for the rest of its life.
_FRAME_TIMEOUT_MS = 3000  # a first frame has been measured at 2.0 s
_MAX_RESTARTS = 5

#: How long the COLOUR frame number may stand still while framesets keep
#: arriving before that counts as a stall. Measured on the UR3e cell's D435 on
#: USB 2.1: a pipeline started right after another was stopped delivered colour
#: frame 2 and then nothing, while depth ran on to frame 104 -- every frameset
#: paired fresh depth with the same colour, nothing timed out, and the recorder
#: wrote 1500 byte-identical pictures. A frame-set timeout cannot see that.
_COLOUR_STALL_S = 1.0

#: Fresh colour frames in a row before a restart counts as having worked. One
#: is not enough: the stall measured above began with a frame, then nothing, so
#: resetting on the first would restart for ever instead of giving up.
_RECOVERED_FRAMES = 30


class RealSenseCapture:
    """Capture thread for one RealSense: colour (RGB) + aligned 16-bit depth."""

    def __init__(
        self,
        rgb_name: str,
        depth_name: str,
        width: int,
        height: int,
        fps: int,
        serial: str = "",
        align_to_color: bool = True,
        lock_auto_exposure: bool = True,
    ) -> None:
        # ``name``/``width``/``height``/``fps`` match CameraCapture so the view
        # and recorder treat the colour stream like any other camera.
        self.name = rgb_name
        self.depth_name = depth_name
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.align_to_color = align_to_color
        self.lock_auto_exposure = lock_auto_exposure

        self._pipeline: "rs.pipeline | None" = None
        self._align: "rs.align | None" = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_frame_mono: float | None = None

        # Replicability metadata, populated at open().
        self.depth_scale: float = 0.0  # metres per depth unit
        self.intrinsics: dict = {}

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _start_pipeline(self):
        """A started pipeline and its profile, for open() and for a restart."""
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )
        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        return pipeline, pipeline.start(config)

    def open(self) -> bool:
        """Start the pipeline and read intrinsics/scale. Fail-fast (bool)."""
        try:
            pipeline, profile = self._start_pipeline()
        except Exception as e:  # pragma: no cover - hardware path
            print(f"❌ RealSense '{self.name}' failed to start: {e}")
            return False

        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color) if self.align_to_color else None
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.intrinsics = {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        }
        print(
            f"  📷 RealSense '{self.name}' + depth '{self.depth_name}' started: "
            f"{self.width}x{self.height}@{self.fps}, depth_scale={self.depth_scale:.6f} m"
        )
        return True

    def start(self, data_manager: FramePublisher) -> None:
        self._thread = threading.Thread(
            target=self._loop, args=(data_manager,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def seconds_since_last_frame(self, now_mono: float) -> float | None:
        with self._lock:
            if self._last_frame_mono is None:
                return None
            return now_mono - self._last_frame_mono

    # ── Capture loop ───────────────────────────────────────────────────────────

    def _maybe_lock_exposure(self, frame_count: int) -> None:
        if not self.lock_auto_exposure or frame_count != _WARMUP_FRAMES:
            return
        try:
            assert self._pipeline is not None
            sensor = (
                self._pipeline.get_active_profile().get_device().first_color_sensor()
            )
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                print(f"  🔒 RealSense '{self.name}' auto-exposure locked")
        except Exception as e:  # pragma: no cover - hardware path
            print(f"⚠️  could not lock RealSense exposure: {e}")

    def _restart(self) -> bool:
        """Stop and start the pipeline after a stall. True if it came back."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001 - already broken
                pass
            self._pipeline = None
        time.sleep(0.5)
        try:
            self._pipeline, _profile = self._start_pipeline()
        except Exception as e:  # pragma: no cover - hardware path
            print(f"⚠️  RealSense '{self.name}' did not restart: {e}")
            return False
        return True

    def _loop(self, data_manager: FramePublisher) -> None:
        frame_count = 0
        stalls = 0
        fresh_run = 0
        last_colour: "int | None" = None
        colour_since = time.monotonic()

        def stall(why: str) -> None:
            """Restart the pipeline, or give up after too many in a row."""
            nonlocal stalls, frame_count, fresh_run, last_colour, colour_since
            stalls += 1
            fresh_run = 0
            if stalls > _MAX_RESTARTS:
                raise RuntimeError(
                    f"{why}, still after {_MAX_RESTARTS} restarts -- unplug and "
                    "replug the camera, ideally into a USB 3 port"
                )
            print(
                f"⚠️  RealSense '{self.name}': {why}; restarting its pipeline "
                f"({stalls}/{_MAX_RESTARTS})"
            )
            # Exposure goes back to auto with a new pipeline; lock it again
            # once the restarted stream has warmed up.
            frame_count = 0
            self._restart()
            last_colour = None
            colour_since = time.monotonic()

        try:
            while not self._stop.is_set() and not data_manager.is_shutdown_requested():
                if self._pipeline is None:
                    stall("the pipeline is not running")
                    continue
                ok, frames = self._pipeline.try_wait_for_frames(_FRAME_TIMEOUT_MS)
                if not ok:
                    stall(f"{_FRAME_TIMEOUT_MS} ms without a frame")
                    continue
                # Stamp the capture instant before alignment/copy so RGB and
                # depth share one time on the collector's reference clock.
                t_capture = time.monotonic()
                if self._align is not None:
                    frames = self._align.process(frames)
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                number = int(color.get_frame_number()) if color else None
                if not depth or number is None or number == last_colour:
                    # A stale pair: publish NEITHER half, so depth stays 1:1
                    # with colour and the recorder's staleness guard pauses
                    # instead of writing a still picture as a fresh one.
                    if time.monotonic() - colour_since > _COLOUR_STALL_S:
                        stuck = (
                            "no colour"
                            if last_colour is None
                            else (f"colour stuck at frame {last_colour}")
                        )
                        stall(f"{stuck} while depth runs")
                    continue
                last_colour = number
                colour_since = t_capture
                fresh_run += 1
                if fresh_run >= _RECOVERED_FRAMES:
                    stalls = 0
                # COPIED, as depth is: a view would keep librealsense's buffer
                # alive for as long as the frame store's history holds it, and
                # the colour sensor's frame pool is small.
                rgb = np.array(color.get_data(), copy=True)  # already RGB (rgb8)
                depth16 = np.asanyarray(depth.get_data()).astype(np.uint16)
                data_manager.set_rgb_image(rgb, self.name, t_capture=t_capture)
                data_manager.set_depth_image(
                    depth16, self.depth_name, t_capture=t_capture
                )
                with self._lock:
                    self._last_frame_mono = t_capture
                frame_count += 1
                self._maybe_lock_exposure(frame_count)
        except Exception as e:  # pragma: no cover - hardware path
            print(f"❌ RealSense '{self.name}' thread error: {e}")
            traceback.print_exc()
        finally:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                self._pipeline = None
            print(f"📷 RealSense '{self.name}' thread stopped")
