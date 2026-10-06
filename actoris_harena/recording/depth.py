"""Per-episode 16-bit depth writer for the data-collection recorder.

LeRobot video features are 3-channel uint8 encoded to AV1, which would corrupt
16-bit depth. Depth is therefore stored OUTSIDE the dataset's video pipeline as
one lossless PNG16 per frame, keyed by the dataset frame index so it aligns 1:1
with the recorded episode:

    <root>/extra/depth/<stream>/episode_<ep:06d>/<frame:06d>.png

Encoding is offloaded to a background worker thread (a bounded queue) so the
30 Hz record loop never blocks on PNG compression. The public API mirrors the
sidecar's lifecycle: :meth:`begin_episode` / :meth:`add` / :meth:`end_episode`
/ :meth:`abort_episode`, all called from the recorder thread.
"""

from __future__ import annotations

import queue
import shutil
import threading
import traceback
from pathlib import Path

import cv2  # type: ignore[import]
import numpy as np


class DepthWriter:
    """Async PNG16 writer for one or more aligned depth streams."""

    def __init__(self, root: str | Path, stream_names: list[str]) -> None:
        self.root = Path(root)
        self.stream_names = list(stream_names)
        self._q: queue.Queue[tuple[Path, np.ndarray] | None] = queue.Queue(maxsize=256)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ep_idx: int | None = None
        self.dropped = 0  # frames dropped because the queue was full

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._q.put(None)  # sentinel → worker drains and exits
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            path, depth = item
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                # PNG16 is lossless for a single-channel uint16 array.
                cv2.imwrite(str(path), depth)
            except Exception:
                traceback.print_exc()
            finally:
                self._q.task_done()

    def flush(self) -> None:
        """Block until every queued write has completed."""
        self._q.join()

    def _episode_dirs(self, ep_idx: int) -> list[Path]:
        return [
            self.root / "extra" / "depth" / name / f"episode_{ep_idx:06d}"
            for name in self.stream_names
        ]

    def _remove_episode(self, ep_idx: int) -> None:
        for d in self._episode_dirs(ep_idx):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

    # ── Episode control (recorder thread) ─────────────────────────────────────

    def begin_episode(self, ep_idx: int) -> None:
        # The dataset reuses an episode index after a discard, so drain any
        # in-flight writes and wipe stale dirs at this index before recording.
        self.flush()
        self._remove_episode(int(ep_idx))
        self._ep_idx = int(ep_idx)

    def add(self, frame_index: int, depth_by_stream: dict[str, np.ndarray]) -> None:
        """Enqueue one frame's depth arrays for writing (non-blocking)."""
        if self._ep_idx is None:
            return
        for name, depth in depth_by_stream.items():
            path = (
                self.root
                / "extra"
                / "depth"
                / name
                / f"episode_{self._ep_idx:06d}"
                / f"{frame_index:06d}.png"
            )
            try:
                self._q.put_nowait((path, np.ascontiguousarray(depth)))
            except queue.Full:
                # Never block the record loop; count the loss so the episode
                # stats surface it (a saturated disk/USB bus).
                self.dropped += 1

    def end_episode(self) -> None:
        """Flush all queued writes for the saved episode."""
        self.flush()
        self._ep_idx = None

    def abort_episode(self) -> None:
        """Discard the in-flight episode: drain queued writes and remove its
        depth directories so the reused episode index starts clean."""
        ep = self._ep_idx
        self._ep_idx = None
        self.flush()
        if ep is not None:
            self._remove_episode(ep)


def write_realsense_meta(root: "str | Path", capture, cfg: dict) -> Path:
    """The RealSense's intrinsics, depth scale and settings, beside its depth.

    Saved once per dataset to ``<root>/meta/realsense.json``, in the field names
    the SO-101 has always written, so depth is interpretable later and the
    shared console's Datasets tab can find and colour it for either rig. The
    camera's extrinsic pose is a physical measurement made separately and is
    left null for the operator to fill in.
    """
    import json

    meta_dir = Path(root) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rgb_name": capture.name,
        "depth_name": capture.depth_name,
        "serial": str(cfg.get("serial") or ""),
        "width": cfg["width"],
        "height": cfg["height"],
        "fps": cfg["fps"],
        "aligned_to_color": bool(cfg.get("align_to_color", True)),
        "depth_scale_m_per_unit": capture.depth_scale,
        "color_intrinsics": capture.intrinsics,
        "extrinsics_camera_to_rig": None,  # measure + fill in for replicability
    }
    path = meta_dir / "realsense.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
