"""Closing the console ends the collection session it started.

The teleop runs in its own process group so a Ctrl+C meant for the console does
not land mid-episode; the cost was a session left running after the console
had gone, found and killed by hand in htop. These pin the shutdown hook that
ends it the Stop button's way, and the hangup that now reaches that hook.
"""

import asyncio
import signal
import unittest
from unittest import mock

from aiohttp.web_runner import GracefulExit

import actoris_harena.web.session_api as session_api
from actoris_harena.web import console


class FakeSession:
    """Alive until the ladder has been consulted ``lives`` times."""

    def __init__(self, lives: int) -> None:
        self.lives = lives
        self.calls: "list[str]" = []

    def running(self) -> bool:
        return self.lives > 0

    def signal_stop(self) -> None:
        self.calls.append("signal_stop")

    def escalate(self) -> str:
        self.calls.append("escalate")
        self.lives -= 1
        return "wait"

    def state(self):
        return {"pid": 1234}


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestTheSessionEndsWithTheConsole(unittest.TestCase):
    def setUp(self):
        self.pressed: "list[str]" = []

        async def press(_app, key):
            self.pressed.append(key)
            return {"pressed": key}

        patches = [
            mock.patch.object(session_api, "_press", press),
            mock.patch.object(session_api, "EXIT_POLL_S", 0.001),
            mock.patch.object(session_api, "_say", lambda text: None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_q_first_then_it_waits_for_the_process_to_exit(self):
        # Q is the door that saves an in-flight episode and parks the arm; the
        # hook must not return while the teleop is still alive.
        session = FakeSession(lives=3)
        _run(session_api.end_session_on_exit({"session": session}))
        self.assertEqual(self.pressed, ["q"])
        self.assertEqual(session.calls[0], "signal_stop")
        self.assertEqual(session.calls.count("escalate"), 3)
        self.assertFalse(session.running())

    def test_no_session_means_nothing_to_do(self):
        _run(session_api.end_session_on_exit({"session": None}))
        _run(session_api.end_session_on_exit({"session": FakeSession(lives=0)}))
        self.assertEqual(self.pressed, [])

    def test_a_monitor_that_is_gone_does_not_stop_the_ladder(self):
        async def gone(_app, _key):
            raise RuntimeError("monitor gone")

        session = FakeSession(lives=2)
        with mock.patch.object(session_api, "_press", gone):
            _run(session_api.end_session_on_exit({"session": session}))
        self.assertFalse(session.running())

    def test_a_session_that_will_not_die_is_left_not_killed(self):
        session = FakeSession(lives=10**9)
        with mock.patch.object(
            session_api, "INTERRUPT_GRACE_S", 0.0
        ), mock.patch.object(session_api, "EXIT_PATIENCE_S", 0.02):
            _run(session_api.end_session_on_exit({"session": session}))
        self.assertTrue(session.running())

    def test_the_hook_is_registered_on_shutdown(self):
        # on_shutdown runs before on_cleanup, while the HTTP client is open.
        app = console.build_app([])
        self.assertIn(session_api.end_session_on_exit, list(app.on_shutdown))


class TestClosingTheTerminalIsAnOrdinaryExit(unittest.TestCase):
    def test_a_hangup_raises_the_graceful_exit(self):
        with self.assertRaises(GracefulExit):
            console._hang_up(signal.SIGHUP, None)


if __name__ == "__main__":
    unittest.main()
