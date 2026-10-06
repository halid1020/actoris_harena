"""Which robots the console can drive, and what it refuses to guess.

The console imports no hardware code, so everything it knows about a rig comes
from that rig's rig.yaml. That makes this file the whole contract, and makes
loudness matter: a console that quietly accepted a missing interpreter would
fail later, in a browser, with a message about a subprocess.
"""

import tempfile
import unittest
from pathlib import Path

import yaml

from actoris_harena.rigs import (
    RIG_FILE,
    RigError,
    discover,
    load_registry,
    load_rig,
    register_rig,
    save_registry,
)

_MINIMAL = {"name": "demo", "python": "bin/python", "agent": "tool/agent.py"}


def _rig_dir(**overrides) -> Path:
    """A directory that load_rig will accept, unless an override breaks it."""
    root = Path(tempfile.mkdtemp())
    (root / "bin").mkdir()
    (root / "bin" / "python").write_text("#!/bin/sh\n")
    (root / "tool").mkdir()
    (root / "tool" / "agent.py").write_text("")
    data = dict(_MINIMAL)
    data.update(overrides)
    data = {k: v for k, v in data.items() if v is not None}
    (root / RIG_FILE).write_text(yaml.safe_dump(data))
    return root


class TestLoadingOneRig(unittest.TestCase):
    def test_a_minimal_rig_loads(self):
        rig = load_rig(_rig_dir())
        self.assertEqual(rig.name, "demo")
        self.assertEqual(rig.title, "demo")  # falls back to the name
        self.assertIsNone(rig.schema)

    def test_paths_resolve_against_the_rig_not_the_caller(self):
        # The console may be started from anywhere, so a relative path in a
        # rig.yaml can only sensibly mean "relative to this file".
        root = _rig_dir()
        rig = load_rig(root)
        self.assertEqual(rig.agent.parent.parent, root.absolute())

    def test_a_venv_interpreter_is_not_resolved_through_its_symlink(self):
        # This was wrong first, and is the bug that would have broken the whole
        # design: venv/bin/python is a SYMLINK to the system interpreter, so
        # resolving it hands the subprocess an environment with none of the
        # rig's packages in it.
        root = _rig_dir()
        link = root / "bin" / "linked"
        link.symlink_to(root / "bin" / "python")
        (root / RIG_FILE).write_text(
            yaml.safe_dump(dict(_MINIMAL, python="bin/linked"))
        )
        self.assertEqual(load_rig(root).python, (root / "bin" / "linked").absolute())

    def test_a_missing_rig_file_is_refused(self):
        with self.assertRaises(RigError):
            load_rig(tempfile.mkdtemp())

    def test_a_missing_interpreter_is_refused_and_says_what_to_do(self):
        root = _rig_dir(python="venv/bin/python")
        with self.assertRaises(RigError) as caught:
            load_rig(root)
        self.assertIn("install.sh", str(caught.exception))

    def test_a_missing_agent_is_refused(self):
        root = _rig_dir(agent="tool/nope.py")
        with self.assertRaises(RigError):
            load_rig(root)

    def test_an_unknown_key_is_refused_by_name(self):
        root = _rig_dir(nonsense=1)
        with self.assertRaises(RigError) as caught:
            load_rig(root)
        self.assertIn("nonsense", str(caught.exception))

    def test_a_missing_required_key_is_refused(self):
        root = _rig_dir()
        (root / RIG_FILE).write_text(yaml.safe_dump({"name": "demo"}))
        with self.assertRaises(RigError):
            load_rig(root)

    def test_a_name_that_could_escape_a_path_or_a_url_is_refused(self):
        # A rig name reaches a URL and is compared against what a browser sent.
        for bad in ("../etc", "a/b", "a b", "a?b"):
            with self.assertRaises(RigError, msg=bad):
                load_rig(_rig_dir(name=bad))

    def test_a_schema_becomes_a_robot_schema(self):
        rig = load_rig(
            _rig_dir(schema={"limbs": ["arm"], "body_joints": ["a", "b", "c"]})
        )
        self.assertIsNotNone(rig.schema)
        self.assertEqual(rig.schema.state_dim, 4)
        self.assertEqual(rig.schema.gripper_columns, (3,))

    def test_an_unusable_schema_is_refused_rather_than_half_built(self):
        with self.assertRaises(RigError):
            load_rig(_rig_dir(schema={"limbs": [], "body_joints": ["a"]}))

    def test_an_unknown_schema_key_is_refused(self):
        with self.assertRaises(RigError):
            load_rig(_rig_dir(schema={"limbs": ["a"], "body_joints": ["b"], "dof": 6}))


class TestWhatATeleopAccepts(unittest.TestCase):
    def test_a_rig_that_says_nothing_gets_what_the_console_always_offered(self):
        spec = load_rig(_rig_dir()).session
        self.assertEqual(spec.inputs, ("quest", "leader"))
        self.assertTrue(spec.sensor_view)
        self.assertIsNone(spec.execute_flag)

    def test_a_session_block_is_read(self):
        rig = load_rig(
            _rig_dir(
                session={
                    "inputs": ["quest"],
                    "execute_flag": "--execute",
                    "sensor_view": False,
                }
            )
        )
        self.assertEqual(rig.session.inputs, ("quest",))
        self.assertEqual(rig.session.execute_flag, "--execute")
        self.assertFalse(rig.session.sensor_view)

    def test_an_unknown_session_key_is_refused_by_name(self):
        with self.assertRaisesRegex(RigError, "execute"):
            load_rig(_rig_dir(session={"execute": True}))

    def test_an_execute_flag_must_be_one_flag(self):
        # It is appended to an argv verbatim: a value, or two words, would
        # change what the rig's parser hears.
        for bad in ("execute", "--", "--execute --mock", "-x"):
            with self.subTest(bad=bad):
                with self.assertRaises(RigError):
                    load_rig(_rig_dir(session={"execute_flag": bad}))


class TestWhichFlagRecordsDepth(unittest.TestCase):
    def test_the_default_is_the_so101s_spelling(self):
        self.assertEqual(load_rig(_rig_dir()).session.depth_flag, "--central-depth")

    def test_a_rig_names_its_own(self):
        rig = load_rig(_rig_dir(session={"depth_flag": "--record-depth"}))
        self.assertEqual(rig.session.depth_flag, "--record-depth")

    def test_null_means_depth_is_not_offered(self):
        rig = load_rig(_rig_dir(session={"depth_flag": None}))
        self.assertIsNone(rig.session.depth_flag)

    def test_it_must_be_one_flag(self):
        with self.assertRaises(RigError):
            load_rig(_rig_dir(session={"depth_flag": "--a --b"}))


class TestTheCommandsARigIsDrivenBy(unittest.TestCase):
    def test_the_agent_runs_in_the_rigs_own_interpreter(self):
        rig = load_rig(_rig_dir())
        argv = rig.agent_argv("--port", "8781")
        self.assertEqual(argv[0], str(rig.python))
        self.assertEqual(argv[1], str(rig.agent))
        self.assertEqual(argv[2:], ["--port", "8781"])

    def test_a_rig_with_no_teleop_says_so_rather_than_building_a_broken_command(self):
        rig = load_rig(_rig_dir())
        with self.assertRaises(RigError):
            rig.teleop_argv()


class TestTheRegistry(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "rigs.yaml"

    def test_a_missing_registry_reads_as_empty_not_as_an_error(self):
        self.assertEqual(load_registry(self.path), [])

    def test_what_is_saved_is_what_is_read_back(self):
        save_registry([Path("/a"), Path("/b")], self.path)
        self.assertEqual(load_registry(self.path), [Path("/a"), Path("/b")])

    def test_a_repeat_entry_is_stored_once(self):
        save_registry([Path("/a"), Path("/a")], self.path)
        self.assertEqual(load_registry(self.path), [Path("/a")])

    def test_discover_returns_the_rigs_that_worked_and_names_the_rest(self):
        # A broken rig.yaml must not hide the working rigs: the console lists
        # what it can drive and says plainly what it could not.
        good = _rig_dir()
        bad = Path(tempfile.mkdtemp())
        rigs, problems = discover([good, bad])
        self.assertEqual([r.name for r in rigs], ["demo"])
        self.assertEqual(len(problems), 1)

    def test_registering_remembers_the_rig_once(self):
        root = _rig_dir()
        self.assertEqual(register_rig(root, self.path).name, "demo")
        register_rig(root, self.path)
        self.assertEqual(load_registry(self.path), [root.resolve()])

    def test_registering_an_unusable_directory_is_refused_and_not_saved(self):
        with self.assertRaises(RigError):
            register_rig(Path(tempfile.mkdtemp()), self.path)
        self.assertEqual(load_registry(self.path), [])

    def test_registering_a_second_rig_of_the_same_name_is_refused(self):
        # The console addresses a rig by name; two would make one unreachable.
        register_rig(_rig_dir(), self.path)
        with self.assertRaisesRegex(RigError, "already registered"):
            register_rig(_rig_dir(), self.path)
        self.assertEqual(len(load_registry(self.path)), 1)

    def test_two_rigs_with_one_name_is_reported_not_silently_collapsed(self):
        rigs, problems = discover([_rig_dir(), _rig_dir()])
        self.assertEqual(len(rigs), 1)
        self.assertTrue(any("both called" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
