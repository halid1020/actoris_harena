"""Unit tests for actoris_harena.recording.collection_settings."""

import json
import tempfile
import unittest
from pathlib import Path

from actoris_harena.recording.collection_settings import (
    is_resumable_dataset,
    read_existing_streams,
    resume_uvc_cameras,
    selection_to_teleop_flags,
    uvc_cameras,
)


def _write_dataset(
    root: Path,
    features: dict,
    fps: int,
    realsense: dict | None = None,
    total_episodes: int = 1,
):
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"fps": fps, "features": features, "total_episodes": total_episodes})
    )
    if realsense is not None:
        (meta / "realsense.json").write_text(json.dumps(realsense))


class TestReadExistingStreams(unittest.TestCase):
    def test_cameras_fps_and_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(
                root,
                features={
                    "observation.state": {},
                    "action": {},
                    "observation.images.scene": {},
                    "observation.images.wrist_camera_left": {},
                },
                fps=30,
            )
            s = read_existing_streams(root)
            self.assertEqual(s["cameras"], {"scene", "wrist_camera_left"})
            self.assertEqual(s["fps"], 30)
            self.assertFalse(s["depth"])
            self.assertIsNone(s["depth_rgb_name"])
            self.assertFalse(s["ee"])

    def test_depth_detected_from_realsense_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(
                root,
                features={"observation.images.central": {}},
                fps=30,
                realsense={"rgb_name": "central", "depth_name": "central_depth"},
            )
            s = read_existing_streams(root)
            self.assertTrue(s["depth"])
            self.assertEqual(s["depth_rgb_name"], "central")

    def test_ee_detected_from_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(
                root,
                features={
                    "observation.images.scene": {},
                    "ee_pose": {},
                    "ee_target": {},
                },
                fps=30,
            )
            self.assertTrue(read_existing_streams(root)["ee"])

    def test_missing_info_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                read_existing_streams(Path(tmp))


class TestUvcCameras(unittest.TestCase):
    def test_excludes_realsense_rgb(self):
        cams = {"scene", "wrist_camera_left", "central"}
        self.assertEqual(uvc_cameras(cams, "central"), {"scene", "wrist_camera_left"})

    def test_none_rgb_keeps_all(self):
        cams = {"scene", "central"}
        self.assertEqual(uvc_cameras(cams, None), {"scene", "central"})


class TestResumeUvcCameras(unittest.TestCase):
    def test_no_depth_keeps_central_uvc_despite_config_name(self):
        # A no-depth dataset with a 'central' UVC camera must NOT drop it just
        # because recording.yaml names the (disabled) RealSense RGB 'central'.
        settings = {
            "cameras": {"central", "wrist_camera_left", "wrist_camera_right"},
            "depth": False,
            "depth_rgb_name": None,
        }
        self.assertEqual(
            resume_uvc_cameras(settings, "central"),
            {"central", "wrist_camera_left", "wrist_camera_right"},
        )

    def test_depth_dataset_excludes_its_rgb(self):
        # A depth dataset splits out its own RealSense RGB name.
        settings = {
            "cameras": {"central", "wrist_camera_left"},
            "depth": True,
            "depth_rgb_name": "central",
        }
        self.assertEqual(resume_uvc_cameras(settings, "central"), {"wrist_camera_left"})

    def test_depth_dataset_falls_back_to_config_rgb_name(self):
        # Older depth dataset whose realsense.json omitted rgb_name: fall back
        # to the config name only because depth is present.
        settings = {
            "cameras": {"central", "wrist_camera_left"},
            "depth": True,
            "depth_rgb_name": None,
        }
        self.assertEqual(resume_uvc_cameras(settings, "central"), {"wrist_camera_left"})


class TestSelectionToTeleopFlags(unittest.TestCase):
    KNOWN = {"scene", "wrist_camera_left", "wrist_camera_right", "tactile_0"}

    def test_enable_disable_split(self):
        flags = selection_to_teleop_flags(
            self.KNOWN, {"scene", "wrist_camera_left"}, depth=False, record_ee=True
        )
        # scene + wrist_camera_left enabled, the other two disabled.
        self.assertEqual(flags.count("--enable-camera"), 2)
        self.assertEqual(flags.count("--disable-camera"), 2)
        i = flags.index("scene")
        self.assertEqual(flags[i - 1], "--enable-camera")
        j = flags.index("wrist_camera_right")
        self.assertEqual(flags[j - 1], "--disable-camera")

    def test_depth_and_no_ee_and_fps(self):
        flags = selection_to_teleop_flags(
            {"scene"}, {"scene"}, depth=True, record_ee=False, fps=30
        )
        self.assertIn("--central-depth", flags)
        self.assertIn("--no-record-ee", flags)
        self.assertEqual(flags[flags.index("--dataset-fps") + 1], "30")

    def test_a_rig_spells_its_own_depth_flag(self):
        flags = selection_to_teleop_flags(
            {"top"}, {"top"}, depth=True, record_ee=True, depth_flag="--record-depth"
        )
        self.assertIn("--record-depth", flags)
        self.assertNotIn("--central-depth", flags)

    def test_ee_on_and_no_depth_omit_flags(self):
        flags = selection_to_teleop_flags(
            {"scene"}, {"scene"}, depth=False, record_ee=True
        )
        self.assertNotIn("--central-depth", flags)
        self.assertNotIn("--no-record-ee", flags)
        self.assertNotIn("--dataset-fps", flags)

    def test_reproduces_existing_dataset_roundtrip(self):
        # A dataset read back should regenerate flags matching its own streams.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(
                root,
                features={
                    "observation.images.scene": {},
                    "observation.images.central": {},
                    "ee_pose": {},
                },
                fps=30,
                realsense={"rgb_name": "central"},
            )
            s = read_existing_streams(root)
            enabled = uvc_cameras(s["cameras"], s["depth_rgb_name"])
            flags = selection_to_teleop_flags(
                {"scene", "wrist_camera_left"}, enabled, s["depth"], s["ee"], s["fps"]
            )
            self.assertIn("--central-depth", flags)  # realsense present
            self.assertNotIn("--no-record-ee", flags)  # ee recorded
            # scene enabled, wrist_camera_left (unused) disabled
            si = flags.index("scene")
            self.assertEqual(flags[si - 1], "--enable-camera")
            wi = flags.index("wrist_camera_left")
            self.assertEqual(flags[wi - 1], "--disable-camera")


class TestIsResumableDataset(unittest.TestCase):
    def test_missing_info_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(is_resumable_dataset(Path(tmp)))

    def test_zero_episodes_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(root, {"observation.images.scene": {}}, 30, total_episodes=0)
            self.assertFalse(is_resumable_dataset(root))

    def test_has_episodes_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(root, {"observation.images.scene": {}}, 30, total_episodes=2)
            self.assertTrue(is_resumable_dataset(root))
