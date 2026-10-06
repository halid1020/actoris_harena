"""The shared console: it knows the rigs, and it opens none of them.

The property worth guarding is negative. Building the console must not import
any rig's hardware code, must not start a subprocess, and must not open a
device -- because the whole reason the agent exists is that this process cannot
hold two robots' dependency stacks at once.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from aiohttp.test_utils import TestClient, TestServer

from actoris_harena.rigs import RIG_FILE, load_rig
from actoris_harena.web.console import build_app


def _rig_dir(name="demo", **extra) -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "bin").mkdir()
    (root / "bin" / "python").write_text("#!/bin/sh\n")
    (root / "tool").mkdir()
    (root / "tool" / "agent.py").write_text("")
    data = {
        "name": name,
        "python": "bin/python",
        "agent": "tool/agent.py",
        "schema": {"limbs": ["arm"], "body_joints": ["a", "b", "c"]},
        "cameras": ["wrist"],
    }
    data.update(extra)
    (root / RIG_FILE).write_text(yaml.safe_dump(data))
    return root


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestTheConsoleDescribesItself(unittest.TestCase):
    def setUp(self):
        self.rigs = [load_rig(_rig_dir("one")), load_rig(_rig_dir("two"))]

    def test_it_lists_every_rig_with_its_schema(self):
        async def go():
            async with TestClient(TestServer(build_app(self.rigs))) as client:
                body = await (await client.get("/api/console")).json()
                self.assertEqual(
                    sorted(r["name"] for r in body["rigs"]), ["one", "two"]
                )
                self.assertEqual(body["rigs"][0]["schema"]["state_dim"], 4)

        _run(go())

    def test_nothing_is_selected_until_something_selects_it(self):
        # One rig and no choice is NOT a default. A console that silently picked
        # the only rig would look identical to one that had picked the wrong one
        # once a second appeared.
        async def go():
            async with TestClient(TestServer(build_app(self.rigs[:1]))) as client:
                self.assertIsNone(
                    (await (await client.get("/api/console")).json())["rig"]
                )

        _run(go())

    def test_a_preselected_rig_is_reported(self):
        async def go():
            app = build_app(self.rigs, selected="two")
            async with TestClient(TestServer(app)) as client:
                self.assertEqual(
                    (await (await client.get("/api/console")).json())["rig"], "two"
                )

        _run(go())


class TestSelectingARig(unittest.TestCase):
    def setUp(self):
        self.rigs = [load_rig(_rig_dir("one")), load_rig(_rig_dir("two"))]

    def test_a_known_rig_is_selected(self):
        async def go():
            app = build_app(self.rigs)
            async with TestClient(TestServer(app)) as client:
                r = await client.post("/api/console/rig", json={"rig": "two"})
                self.assertEqual(r.status, 200)
                self.assertEqual(app["rig"], "two")

        _run(go())

    def test_an_unknown_rig_is_refused(self):
        async def go():
            async with TestClient(TestServer(build_app(self.rigs))) as client:
                r = await client.post("/api/console/rig", json={"rig": "nope"})
                self.assertEqual(r.status, 400)

        _run(go())

    def test_it_is_refused_while_a_job_is_running(self):
        # Same rule roots_api applies to changing the collection directory: a
        # job holds state belonging to the rig it started under, and switching
        # underneath it would attribute one robot's work to another.
        async def go():
            app = build_app(self.rigs)
            app["jobs"]["j"] = {"state": "running"}
            async with TestClient(TestServer(app)) as client:
                r = await client.post("/api/console/rig", json={"rig": "two"})
                self.assertEqual(r.status, 409)
                self.assertIsNone(app["rig"])

        _run(go())

    def test_selecting_nothing_deselects(self):
        async def go():
            app = build_app(self.rigs, selected="one")
            async with TestClient(TestServer(app)) as client:
                await client.post("/api/console/rig", json={"rig": None})
                self.assertIsNone(app["rig"])

        _run(go())


class TestRegisteringARig(unittest.TestCase):
    """From the page, through the same validation as `rigs add`."""

    def setUp(self):
        self.registry = str(Path(tempfile.mkdtemp()) / "rigs.yaml")

    def app(self, rigs=()):
        return build_app(list(rigs), registry=self.registry)

    def post(self, app, root):
        async def go():
            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/console/rigs", json={"root": root})
                listed = await (await client.get("/api/console")).json()
                return resp.status, await resp.text(), listed

        return _run(go())

    def test_a_good_rig_is_registered_listed_and_not_selected(self):
        root = _rig_dir("new", session={"execute_flag": "--execute"})
        status, _, listed = self.post(self.app(), str(root))
        self.assertEqual(status, 200)
        self.assertEqual([r["name"] for r in listed["rigs"]], ["new"])
        self.assertEqual(listed["rigs"][0]["session"]["execute_flag"], "--execute")
        # Registering opens nothing, so it selects nothing either.
        self.assertIsNone(listed["rig"])
        from actoris_harena.rigs import load_registry

        self.assertEqual(load_registry(self.registry), [root.resolve()])

    def test_registering_twice_is_harmless(self):
        root = str(_rig_dir("new"))

        async def go():
            async with TestClient(TestServer(self.app())) as client:
                for _ in range(2):
                    resp = await client.post("/api/console/rigs", json={"root": root})
                    self.assertEqual(resp.status, 200)
                listed = await (await client.get("/api/console")).json()
                self.assertEqual(len(listed["rigs"]), 1)

        _run(go())

    def test_a_directory_with_no_rig_is_refused_with_the_reason(self):
        status, text, _ = self.post(self.app(), tempfile.mkdtemp())
        self.assertEqual(status, 400)
        self.assertIn("rig.yaml", text)

    def test_a_name_already_loaded_from_elsewhere_is_refused(self):
        loaded = load_rig(_rig_dir("same"))
        status, _, listed = self.post(self.app([loaded]), str(_rig_dir("same")))
        self.assertEqual(status, 409)
        self.assertEqual(listed["rigs"][0]["root"], str(loaded.root))
        from actoris_harena.rigs import load_registry

        self.assertEqual(load_registry(self.registry), [])


class TestChoosingADriveBeforeAnySession(unittest.TestCase):
    def test_a_fresh_console_can_change_its_directory(self):
        # There is no supervisor until a session is planned, and the busy
        # check used to ask a None whether it was running.
        drive = tempfile.mkdtemp()

        async def go():
            app = build_app([load_rig(_rig_dir())])
            app["roots_file"] = Path(tempfile.mkdtemp()) / "roots.json"
            async with TestClient(TestServer(app)) as client:
                resp = await client.post("/api/roots/use", json={"path": drive})
                self.assertEqual(resp.status, 200, await resp.text())
                self.assertEqual(str(app["root"]), drive)

        _run(go())


class TestItOpensNoRobot(unittest.TestCase):
    def test_building_the_console_imports_no_hardware_module(self):
        before = set(sys.modules)
        build_app([load_rig(_rig_dir())])
        new = set(sys.modules) - before
        for hardware in ("feetech", "rtde_control", "pyrealsense2", "mujoco"):
            self.assertFalse(
                any(m == hardware or m.startswith(hardware + ".") for m in new),
                f"building the console imported {hardware}",
            )

    def test_building_the_console_starts_no_subprocess(self):
        import subprocess

        real = subprocess.Popen
        started = []

        def spy(*args, **kwargs):
            started.append(args)
            return real(*args, **kwargs)

        subprocess.Popen = spy
        try:
            build_app([load_rig(_rig_dir())])
        finally:
            subprocess.Popen = real
        self.assertEqual(started, [])


if __name__ == "__main__":
    unittest.main()
