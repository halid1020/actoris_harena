"""Unit tests for the depth writer and the DualDataManager depth API.

No RealSense hardware: fake uint16 frames drive DepthWriter, and depth is
pushed through DualDataManager's pure state API.
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from actoris_harena.recording.depth import DepthWriter, write_realsense_meta


class TestDepthWriter(unittest.TestCase):
    def _frame(self, val: int) -> np.ndarray:
        return np.full((8, 12), val, dtype=np.uint16)

    def test_png16_roundtrip_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = DepthWriter(tmp, ["central_depth"])
            w.start()
            try:
                w.begin_episode(3)
                w.add(0, {"central_depth": self._frame(1000)})
                w.add(1, {"central_depth": self._frame(40000)})  # >8-bit range
                w.end_episode()
            finally:
                w.stop()
            base = Path(tmp) / "extra" / "depth" / "central_depth" / "episode_000003"
            f0 = base / "000000.png"
            f1 = base / "000001.png"
            self.assertTrue(f0.exists() and f1.exists())
            # Read back unchanged (lossless 16-bit).
            got = cv2.imread(str(f1), cv2.IMREAD_UNCHANGED)
            self.assertEqual(got.dtype, np.uint16)
            self.assertEqual(int(got[0, 0]), 40000)

    def test_abort_removes_episode_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = DepthWriter(tmp, ["central_depth"])
            w.start()
            try:
                w.begin_episode(5)
                w.add(0, {"central_depth": self._frame(7)})
                w.abort_episode()
            finally:
                w.stop()
            base = Path(tmp) / "extra" / "depth" / "central_depth" / "episode_000005"
            self.assertFalse(base.exists())

    def test_begin_wipes_reused_index(self):
        # A discarded episode's index is reused by the next take: begin must
        # start from a clean directory.
        with tempfile.TemporaryDirectory() as tmp:
            w = DepthWriter(tmp, ["central_depth"])
            w.start()
            try:
                w.begin_episode(2)
                w.add(0, {"central_depth": self._frame(11)})
                w.add(1, {"central_depth": self._frame(12)})
                w.end_episode()
                # Reuse index 2 with a single frame; the stale 000001.png goes.
                w.begin_episode(2)
                w.add(0, {"central_depth": self._frame(99)})
                w.end_episode()
            finally:
                w.stop()
            base = Path(tmp) / "extra" / "depth" / "central_depth" / "episode_000002"
            self.assertTrue((base / "000000.png").exists())
            self.assertFalse((base / "000001.png").exists())


class TestTheRealSenseMeta(unittest.TestCase):
    def test_it_names_the_depth_stream_the_datasets_tab_looks_for(self):
        import json
        import tempfile
        from types import SimpleNamespace

        capture = SimpleNamespace(
            name="top",
            depth_name="top_depth",
            depth_scale=0.001,
            intrinsics={"fx": 1.0},
        )
        cfg = {"width": 640, "height": 480, "fps": 30, "serial": "825312073060"}
        with tempfile.TemporaryDirectory() as d:
            meta = json.loads(write_realsense_meta(d, capture, cfg).read_text())
        self.assertEqual(meta["depth_name"], "top_depth")
        self.assertEqual(meta["depth_scale_m_per_unit"], 0.001)
        self.assertIsNone(meta["extrinsics_camera_to_rig"])
