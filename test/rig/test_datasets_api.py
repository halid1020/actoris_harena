"""The Datasets tab, served by the shared console for any rig.

The routes were the SO-101 console's alone, so the shared page's tab was empty.
These pin the rig-neutral parts that moved: a dataset describes its own limbs,
the motion needs no kinematic model, and depth plays as images.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from actoris_harena.web.console import build_app
from actoris_harena.web.datasets_api import (
    depth_png_path,
    depth_range,
    list_datasets,
    motion_payload,
    state_columns,
)

UR3E_NAMES = [
    "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos",
    "arm_elbow.pos",
    "arm_wrist_1.pos",
    "arm_wrist_2.pos",
    "arm_wrist_3.pos",
    "arm_gripper.pos",
]


def _dataset(root: Path, name: str, names, episodes: int = 1) -> Path:
    path = root / name
    (path / "meta").mkdir(parents=True)
    (path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "robot_type": "test",
                "total_episodes": episodes,
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [len(names)],
                        "names": names,
                    },
                    "observation.images.top": {"dtype": "video", "shape": [4, 6, 3]},
                },
            }
        )
    )
    return path


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestADatasetDescribesItsOwnLimbs(unittest.TestCase):
    def test_a_ur3e_dataset_is_one_limb_of_seven_channels(self):
        with tempfile.TemporaryDirectory() as d:
            path = _dataset(Path(d), "ur", UR3E_NAMES)
            columns = state_columns(path)
        self.assertEqual({c["limb"] for c in columns}, {"arm"})
        self.assertEqual(columns[3]["joint"], "wrist_1")
        self.assertEqual(columns[6]["joint"], "gripper")

    def test_a_dual_arm_dataset_is_two_limbs(self):
        names = [f"{s}_{j}.pos" for s in ("left", "right") for j in ("pan", "gripper")]
        with tempfile.TemporaryDirectory() as d:
            columns = state_columns(_dataset(Path(d), "so", names))
        self.assertEqual([c["limb"] for c in columns], ["left"] * 2 + ["right"] * 2)

    def test_names_without_a_limb_still_play_ungrouped(self):
        with tempfile.TemporaryDirectory() as d:
            columns = state_columns(_dataset(Path(d), "x", ["q0", "q1"]))
        self.assertEqual([c["limb"] for c in columns], ["state", "state"])


class TestMotionNeedsNoKinematicModel(unittest.TestCase):
    def test_velocity_by_finite_differences_and_no_end_effector(self):
        columns = [{"index": 0, "limb": "arm", "joint": "pan"}]
        state = np.arange(10, dtype=float).reshape(-1, 1)  # 1 degree per frame
        payload = motion_payload(state, 30, columns)
        self.assertAlmostEqual(payload["joint_vel"][5][0], 30.0)
        self.assertIsNone(payload["ee"])
        self.assertEqual(payload["joint_units"], ["°/s"])
        self.assertAlmostEqual(payload["peaks"]["joint_vel"][0], 30.0)

    def test_a_one_frame_episode_does_not_divide_by_nothing(self):
        columns = [{"index": 0, "limb": "arm", "joint": "pan"}]
        payload = motion_payload(np.zeros((1, 1)), 30, columns)
        self.assertEqual(payload["joint_vel"], [[0.0]])


class TestTheList(unittest.TestCase):
    def test_every_dataset_is_listed_and_an_empty_one_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            _dataset(Path(d), "kept", UR3E_NAMES, episodes=2)
            _dataset(Path(d), "stillborn", UR3E_NAMES, episodes=0)
            rows = {r["name"]: r for r in list_datasets(Path(d))}
        self.assertEqual(rows["kept"]["episodes"], 2)
        self.assertTrue(rows["stillborn"]["stillborn"])
        self.assertEqual(rows["kept"]["streams"], ["top"])

    def test_the_console_serves_it(self):
        # The route the page called and nothing answered.
        with tempfile.TemporaryDirectory() as d:
            _dataset(Path(d), "kept", UR3E_NAMES)

            async def go():
                app = build_app([], collection_dir=d)
                async with TestClient(TestServer(app)) as client:
                    r = await client.get("/api/datasets")
                    self.assertEqual(r.status, 200)
                    return await r.json()

            body = _run(go())
        self.assertEqual([r["name"] for r in body], ["kept"])


class TestDepthPlaysAsImages(unittest.TestCase):
    def test_the_range_ignores_holes_and_strays(self):
        depth = np.zeros((10, 10), np.uint16)
        depth[:, 5:] = 800
        depth[0, 9] = 60000  # one stray far reading
        near, far = depth_range(depth)
        self.assertGreaterEqual(near, 799)
        self.assertLess(far, 60000)

    def test_a_depth_frame_is_served_as_a_jpeg(self):
        with tempfile.TemporaryDirectory() as d:
            path = _dataset(Path(d), "deep", UR3E_NAMES)
            (path / "meta" / "realsense.json").write_text(
                json.dumps({"depth_name": "top_depth", "depth_scale_m_per_unit": 0.001})
            )
            png = depth_png_path(path, "top_depth", 0, 0)
            png.parent.mkdir(parents=True)
            cv2.imwrite(str(png), np.full((4, 6), 700, np.uint16))

            async def go():
                app = build_app([], collection_dir=d)
                async with TestClient(TestServer(app)) as client:
                    r = await client.get("/api/datasets/deep/episodes/0/depth/0.png")
                    missing = await client.get(
                        "/api/datasets/deep/episodes/0/depth/9.png"
                    )
                    return r.status, r.content_type, missing.status

            status, kind, missing = _run(go())
        self.assertEqual((status, kind), (200, "image/jpeg"))
        self.assertEqual(missing, 404)


if __name__ == "__main__":
    unittest.main()
