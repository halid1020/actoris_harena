"""Turning a Collect-tab request into one rig's command, and ending it.

The console imports no hardware code, so a session is a subprocess started with
the SELECTED rig's interpreter and entry point. What is worth pinning here is
everything that happens before that subprocess exists and after it is asked to
stop: the argv, the refusals, the device probe the console cannot run itself,
and the ladder that never reaches SIGKILL.
"""

import unittest
from pathlib import Path

from actoris_harena.recording.features import RobotSchema
from actoris_harena.recording.monitor_server import joint_snapshot
from actoris_harena.web.session import (
    INTERRUPT_GRACE_S,
    QUIT_GRACE_S,
    SessionSupervisor,
    stop_action,
    teleop_argv,
)

PLAN = {"resuming": False, "flags": ["--enable-camera", "wrist"]}


class TestTheCommandOneRigGets(unittest.TestCase):
    def argv(self, **options):
        return teleop_argv(
            Path("/drive"),
            "towel_fold",
            "fold the towel",
            PLAN,
            options,
            8766,
            Path("/rigs/ur3e/tool/quest_teleoperation.py"),
        )

    def test_the_entry_point_is_the_rigs_own(self):
        # Not a constant in this package. The console serves several robots and
        # each brings its own tool; a default here would launch the wrong one.
        self.assertEqual(self.argv()[0], "/rigs/ur3e/tool/quest_teleoperation.py")

    def test_the_dataset_lands_under_the_chosen_drive(self):
        argv = self.argv()
        i = argv.index("--dataset-root")
        self.assertEqual(argv[i + 1], "/drive/towel_fold")

    def test_the_monitor_port_is_passed_so_the_page_can_watch(self):
        argv = self.argv()
        self.assertEqual(argv[argv.index("--monitor-port") + 1], "8766")

    def test_the_plans_stream_flags_are_carried_through(self):
        # Every known camera is named explicitly, so recording.yaml's defaults
        # cannot quietly override what the operator ticked.
        self.assertIn("--enable-camera", self.argv())
        self.assertIn("wrist", self.argv())

    def test_a_resumed_dataset_says_so(self):
        argv = teleop_argv(
            Path("/drive"),
            "towel_fold",
            "t",
            {"resuming": True, "flags": []},
            {},
            8766,
            Path("/t.py"),
        )
        self.assertIn("--resume", argv)

    def test_a_rehearsal_is_requested_not_assumed(self):
        self.assertNotIn("--mock", self.argv())
        self.assertIn("--mock", self.argv(mock=True))


class TestTheSupervisorNeverGuessesAnInterpreter(unittest.TestCase):
    def test_it_takes_the_rigs_python_and_tool(self):
        # sys.executable would be the CONSOLE's interpreter, which has no robot
        # in it -- that is the point of the split. A session started with it
        # could not import its own driver.
        s = SessionSupervisor(
            root=Path("/drive"),
            python="/rigs/ur3e/venv/bin/python",
            teleop=Path("/t.py"),
        )
        self.assertEqual(s.python, "/rigs/ur3e/venv/bin/python")
        self.assertEqual(s.teleop, Path("/t.py"))

    def test_nothing_is_running_before_it_starts(self):
        s = SessionSupervisor(root=Path("/drive"), python="/p", teleop=Path("/t.py"))
        self.assertFalse(s.running())
        self.assertFalse(s.state()["running"])


class TestEndingASession(unittest.TestCase):
    """The ladder, and the step it deliberately does not have."""

    def test_it_waits_first_so_the_session_can_park_and_save(self):
        self.assertEqual(stop_action(0.0), "wait")
        self.assertEqual(stop_action(QUIT_GRACE_S - 0.1), "wait")

    def test_then_it_interrupts(self):
        self.assertEqual(stop_action(QUIT_GRACE_S), "interrupt")
        self.assertEqual(stop_action(INTERRUPT_GRACE_S - 0.1), "interrupt")

    def test_and_terminates_but_never_kills(self):
        # SIGKILL is absent on purpose: it abandons an open episode and leaves
        # the arm energised. Every later time still asks for a teardown.
        for elapsed in (INTERRUPT_GRACE_S, 1e3, 1e6):
            self.assertEqual(stop_action(elapsed), "terminate")


class TestTheJointTableFitsWhicheverRig(unittest.TestCase):
    """One server, a dual arm and a single one. The schema is the difference."""

    def test_a_single_limb_rig_gets_one_column_pair(self):
        schema = RobotSchema(limbs=("arm",), body_joints=("a", "b"))
        snap = joint_snapshot(
            schema, [1.0, 2.0], {"arm": 0.5}, {"arm": (None, None, None)}, 0.0
        )
        self.assertEqual(list(snap), ["arm"])
        self.assertEqual(snap["arm"]["state"], {"a": 1.0, "b": 2.0, "gripper": 0.5})

    def test_a_dual_limb_rig_splits_the_vector_in_order(self):
        schema = RobotSchema(limbs=("left", "right"), body_joints=("a", "b"))
        snap = joint_snapshot(
            schema,
            [1.0, 2.0, 3.0, 4.0],
            {},
            {s: (None, None, None) for s in ("left", "right")},
            0.0,
        )
        self.assertEqual(snap["left"]["state"]["a"], 1.0)
        self.assertEqual(snap["right"]["state"]["a"], 3.0)

    def test_a_command_older_than_a_frame_is_not_fresh(self):
        # A stale command is what makes the recorded action fall back to the
        # measured state, so the table has to show it rather than imply a live
        # one.
        schema = RobotSchema(limbs=("arm",), body_joints=("a",))
        stale = joint_snapshot(schema, [1.0], {}, {"arm": ([2.0], 1.0, 0.0)}, 100.0)
        fresh = joint_snapshot(schema, [1.0], {}, {"arm": ([2.0], 1.0, 99.99)}, 100.0)
        self.assertFalse(stale["arm"]["fresh"])
        self.assertTrue(fresh["arm"]["fresh"])

    def test_a_rig_with_no_gripper_has_no_gripper_row(self):
        schema = RobotSchema(limbs=("arm",), body_joints=("a",), gripper=False)
        snap = joint_snapshot(schema, [1.0], {}, {"arm": (None, None, None)}, 0.0)
        self.assertNotIn("gripper", snap["arm"]["state"])


if __name__ == "__main__":
    unittest.main()
