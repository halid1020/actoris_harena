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
from actoris_harena.rigs import SessionSpec
from actoris_harena.web.session import (
    INTERRUPT_GRACE_S,
    QUIT_GRACE_S,
    SessionSupervisor,
    option_refusals,
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


UR3E = SessionSpec(inputs=("quest",), execute_flag="--execute", sensor_view=False)


class TestDrivingTheRealArmIsAskedForEveryTime(unittest.TestCase):
    """A rig whose teleop rehearses by default moves only on a person's say-so.

    Without the flag a UR3e session sends the arm nothing and records the
    targets anyway -- a dry run that looks like a dataset. With it by default,
    the console would move an arm nobody asked it to.
    """

    def argv(self, execute_flag, **options):
        return teleop_argv(
            Path("/drive"), "d", "t", PLAN, options, 8766, Path("/t.py"), execute_flag
        )

    def test_the_flag_is_absent_unless_asked_for(self):
        self.assertNotIn("--execute", self.argv("--execute"))
        self.assertNotIn("--execute", self.argv("--execute", execute=False))

    def test_only_a_literal_true_asks(self):
        for loose in ("true", 1, "yes", [1]):
            with self.subTest(loose=loose):
                self.assertNotIn("--execute", self.argv("--execute", execute=loose))

    def test_asked_for_it_is_the_rigs_own_flag(self):
        self.assertEqual(self.argv("--drive", execute=True)[-1], "--drive")

    def test_a_rig_with_no_flag_never_gets_one(self):
        argv = self.argv(None, execute=True)
        self.assertNotIn("--execute", argv)
        self.assertNotIn(None, argv)

    def test_the_supervisor_carries_the_rigs_flag(self):
        s = SessionSupervisor(
            root=Path("/drive"),
            python="/p",
            teleop=Path("/t.py"),
            execute_flag="--execute",
        )
        self.assertEqual(s.execute_flag, "--execute")


class TestOptionsARigCannotHonourAreRefused(unittest.TestCase):
    """Refused in the plan, by name. A flag the rig's parser does not know ends
    the session at argparse, after the page said it had started."""

    def test_a_plain_quest_session_is_fine(self):
        self.assertEqual(option_refusals({"input": "quest"}, UR3E), [])
        self.assertEqual(option_refusals({"execute": True}, UR3E), [])

    def test_an_input_the_rig_does_not_offer(self):
        self.assertTrue(option_refusals({"input": "leader"}, UR3E))

    def test_a_desktop_window_the_rig_does_not_have(self):
        self.assertTrue(option_refusals({"sensor_view": True}, UR3E))

    def test_driving_a_rig_that_declares_no_flag(self):
        self.assertTrue(option_refusals({"execute": True}, SessionSpec()))

    def test_driving_the_arm_and_rehearsing_at_once(self):
        self.assertTrue(option_refusals({"execute": True, "mock": True}, UR3E))

    def test_depth_from_a_rig_that_cannot_record_it(self):
        self.assertTrue(option_refusals({"depth": True}, SessionSpec(depth_flag=None)))
        self.assertEqual(option_refusals({"depth": True}, SessionSpec()), [])

    def test_the_defaults_refuse_nothing_the_so101_page_sent(self):
        spec = SessionSpec()
        for options in ({"input": "leader"}, {"sensor_view": True}, {"mock": True}):
            with self.subTest(options=options):
                self.assertEqual(option_refusals(options, spec), [])


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
