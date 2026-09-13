"""The Collect and Signals tabs, which the console serves without a robot.

Every route here is a proxy to the selected rig's agent. The properties worth
pinning are the refusals and the lifecycle -- a console that quietly held two
agents on one camera, or left one running after a switch, would waste an
operator's time in a way nothing on the page would show.

No agent is actually started in this file: `AgentPool` is given a fake process
factory, because starting one opens devices and these are unit tests.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path

import yaml
from aiohttp.test_utils import TestClient, TestServer

from actoris_harena.rigs import RIG_FILE, load_rig
from actoris_harena.web.agent_api import AgentPool
from actoris_harena.web.console import build_app


def _rig_dir(name="demo") -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "bin").mkdir()
    (root / "bin" / "python").write_text("#!/bin/sh\n")
    (root / "tool").mkdir()
    (root / "tool" / "agent.py").write_text("")
    (root / RIG_FILE).write_text(
        yaml.safe_dump(
            {
                "name": name,
                "python": "bin/python",
                "agent": "tool/agent.py",
                "schema": {"limbs": ["arm"], "body_joints": ["a", "b", "c"]},
            }
        )
    )
    return root


class FakeProcess:
    def __init__(self) -> None:
        self.alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated = True
        self.alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True
        self.alive = False


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestTheRefusals(unittest.TestCase):
    def setUp(self):
        self.rigs = [load_rig(_rig_dir("one")), load_rig(_rig_dir("two"))]

    def test_asking_a_device_question_with_no_rig_selected_is_refused(self):
        # Not a 500 and not an empty answer: the page has to be able to say
        # "choose a rig" rather than "something went wrong".
        async def go():
            async with TestClient(TestServer(build_app(self.rigs))) as client:
                for path in (
                    "/api/agent/status",
                    "/api/agent/cameras",
                    "/api/agent/arm",
                ):
                    r = await client.get(path)
                    self.assertEqual(r.status, 409, path)
                    self.assertIn("no rig is selected", await r.text())

        _run(go())


class TestTheAgentPool(unittest.TestCase):
    """One agent per rig, started on demand and kept."""

    def setUp(self):
        self.pool = AgentPool()
        self.started: "list[list[str]]" = []
        self.processes: "list[FakeProcess]" = []

        def fake_popen(argv, **_kwargs):
            self.started.append(list(argv))
            process = FakeProcess()
            self.processes.append(process)
            return process

        import actoris_harena.web.agent_api as mod

        self._real = mod.subprocess.Popen
        mod.subprocess.Popen = fake_popen
        self.rig = load_rig(_rig_dir("one"))

    def tearDown(self):
        import actoris_harena.web.agent_api as mod

        mod.subprocess.Popen = self._real

    def test_it_starts_the_agent_in_the_rigs_own_interpreter(self):
        # The whole reason the agent exists: this rig's venv and the next one's
        # cannot be installed together.
        self.pool.start(self.rig)
        self.assertEqual(self.started[0][0], str(self.rig.python))
        self.assertEqual(self.started[0][1], str(self.rig.agent))

    def test_it_is_started_once_and_kept(self):
        # Starting one opens devices; doing that per click would make the page
        # unusable and the cameras unhappy.
        first = self.pool.start(self.rig)
        second = self.pool.start(self.rig)
        self.assertEqual(first, second)
        self.assertEqual(len(self.started), 1)

    def test_each_agent_gets_a_port_the_os_gave_out(self):
        other = load_rig(_rig_dir("two"))
        self.assertNotEqual(self.pool.start(self.rig), self.pool.start(other))

    def test_stopping_it_is_what_releases_the_devices(self):
        self.pool.start(self.rig)
        self.pool.stop(self.rig.name)
        self.assertTrue(self.processes[0].terminated)
        self.assertFalse(self.pool.running(self.rig.name))

    def test_a_stopped_agent_starts_again_on_the_next_ask(self):
        self.pool.start(self.rig)
        self.pool.stop(self.rig.name)
        self.pool.start(self.rig)
        self.assertEqual(len(self.started), 2)

    def test_stopping_one_that_is_not_running_is_not_an_error(self):
        self.pool.stop("never started")

    def test_stop_all_releases_every_rig(self):
        self.pool.start(self.rig)
        self.pool.start(load_rig(_rig_dir("two")))
        self.pool.stop_all()
        self.assertTrue(all(p.terminated for p in self.processes))


class TestSwitchingRigs(unittest.TestCase):
    def setUp(self):
        self.rigs = [load_rig(_rig_dir("one")), load_rig(_rig_dir("two"))]

        import actoris_harena.web.agent_api as mod

        self._real = mod.subprocess.Popen
        mod.subprocess.Popen = lambda argv, **k: FakeProcess()

    def tearDown(self):
        import actoris_harena.web.agent_api as mod

        mod.subprocess.Popen = self._real

    def test_switching_stops_the_previous_rigs_agent(self):
        # Two agents holding the same camera is the one way this could waste an
        # operator's time silently: the second open succeeds and then delivers
        # nothing for ever.
        async def go():
            app = build_app(self.rigs, selected="one")
            async with TestClient(TestServer(app)) as client:
                app["agents"].start(app["rigs"]["one"])
                self.assertTrue(app["agents"].running("one"))
                await client.post("/api/console/rig", json={"rig": "two"})
                self.assertFalse(app["agents"].running("one"))

        _run(go())

    def test_reselecting_the_same_rig_does_not_stop_its_agent(self):
        async def go():
            app = build_app(self.rigs, selected="one")
            async with TestClient(TestServer(app)) as client:
                app["agents"].start(app["rigs"]["one"])
                await client.post("/api/console/rig", json={"rig": "one"})
                self.assertTrue(app["agents"].running("one"))

        _run(go())


if __name__ == "__main__":
    unittest.main()
