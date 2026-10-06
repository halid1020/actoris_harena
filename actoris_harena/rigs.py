"""Which robots this console can drive, and what it needs to know about each.

The console is shared and the robots are not. It has to browse a collection
drive, plan a training run and watch a session for a dual SO-101 and for a single
UR3e, whose Python environments cannot even be installed together -- one needs
feetech and mujoco, the other ur-rtde, and each pins its own LeRobot checkout.

So the console imports NO hardware code at all. A rig is a directory with a
``rig.yaml`` in it, and everything that touches a device runs as a SUBPROCESS
started with that rig's own interpreter. This is not a new mechanism: the
collection session already worked exactly this way, supervised as a subprocess
and read back over a loopback monitor.

A rig.yaml says:

    name: so101                       # the id the console addresses it by
    title: Dual SO-101 garment rig    # what a person reads
    python: venv/bin/python           # interpreter, relative to the repo
    agent: tool/rig_agent.py          # the devices, in that interpreter
    teleop: tool/meta_quest_teleopration.py
    conf: src/conf                    # where recording.yaml and friends live
    sensor_map: src/conf/sensor_map.yaml
    cameras: [central, wrist_camera_left, ...]
    schema:
      limbs: [left, right]
      body_joints: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
      gripper: true
    session:                          # optional; what the teleop accepts
      inputs: [quest]                 # operator interfaces (default quest+leader)
      execute_flag: --execute         # see SessionSpec
      sensor_view: false              # whether --sensor-view exists (default true)

Every path is relative to the rig's own directory and is resolved against it,
never against the console's working directory -- the console may be started from
anywhere, and an absolute path in a checked-in file is a path that is wrong on
the next machine.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import]

from actoris_harena.recording.features import RobotSchema

#: The file inside a rig's directory that declares it.
RIG_FILE = "rig.yaml"

#: Where the list of known rigs is remembered. Under the user's config, not in
#: any repo: which robots are on THIS machine is a fact about the machine.
REGISTRY_PATH = Path.home() / ".config" / "actoris_harena" / "rigs.yaml"

_REQUIRED = frozenset({"name", "python", "agent"})
_OPTIONAL = frozenset(
    {
        "title",
        "teleop",
        "conf",
        "sensor_map",
        "cameras",
        "schema",
        "session",
        "output_dir",
    }
)
_SESSION_KEYS = frozenset({"inputs", "execute_flag", "sensor_view"})

# A rig name reaches a URL and a directory name, and is compared for equality
# against what a browser sent. Keeping it to this alphabet means it can never
# need escaping in either place, and never resolve to a parent directory.
_NAME_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


class RigError(ValueError):
    """A rig.yaml, or the registry, says something unusable."""


@dataclass(frozen=True)
class SessionSpec:
    """What a rig's teleop entry point accepts, so the page offers only that.

    The defaults are what the console offered before rigs could say: both
    operator interfaces, the desktop window, and no execute flag.

    ``execute_flag`` is the ONE thing here with safety weight. Some rigs' teleop
    rehearses unless told otherwise -- the UR3e sends the arm nothing without
    ``--execute`` -- and a console that never passed it would record a dry run
    in silence, a dataset of targets no arm was ever sent. A rig that declares
    the flag gets it only when a person ticked the box for THAT session; a rig
    that declares none never receives it.
    """

    inputs: "tuple[str, ...]" = ("quest", "leader")
    execute_flag: "str | None" = None
    sensor_view: bool = True


def _session_spec(path: Path, spec: object) -> SessionSpec:
    if spec is None:
        return SessionSpec()
    if not isinstance(spec, dict):
        raise RigError(f"{path}: session must be a mapping")
    unknown = set(spec) - _SESSION_KEYS
    if unknown:
        raise RigError(f"{path}: session has unknown key(s): {sorted(unknown)}")
    inputs = tuple(str(i) for i in (spec.get("inputs") or SessionSpec.inputs))
    flag = spec.get("execute_flag")
    if flag is not None:
        flag = str(flag)
        # One argv element, and a flag rather than a value: it is appended to
        # a command line verbatim, so anything else would change its meaning.
        if not flag.startswith("--") or len(flag) < 3 or any(c.isspace() for c in flag):
            raise RigError(f"{path}: session.execute_flag {flag!r} must be one --flag")
    return SessionSpec(
        inputs=inputs,
        execute_flag=flag,
        sensor_view=bool(spec.get("sensor_view", True)),
    )


@dataclass(frozen=True)
class Rig:
    """One robot the console can drive."""

    name: str
    root: Path
    title: str
    python: Path
    agent: Path
    teleop: "Path | None"
    conf: "Path | None"
    sensor_map: "Path | None"
    cameras: "tuple[str, ...]"
    schema: "RobotSchema | None"
    session: SessionSpec = SessionSpec()

    def agent_argv(self, *args: str) -> "list[str]":
        """The command that runs this rig's device agent, in ITS interpreter."""
        return [str(self.python), str(self.agent), *args]

    def teleop_argv(self, *args: str) -> "list[str]":
        """The command that runs this rig's teleoperation entry point."""
        if self.teleop is None:
            raise RigError(f"rig {self.name!r} declares no teleop entry point")
        return [str(self.python), str(self.teleop), *args]


def _check_name(name: str) -> str:
    if not name:
        raise RigError("a rig needs a name")
    bad = sorted(set(name) - _NAME_OK)
    if bad:
        raise RigError(f"rig name {name!r} may not contain {bad}")
    return name


def load_rig(root: "Path | str") -> Rig:
    """Read one rig's ``rig.yaml``. Raises :class:`RigError` on anything unusable.

    Strict for the same reason the recording config is: a console that quietly
    accepted a missing interpreter would fail later, in a browser, with a message
    about a subprocess.
    """
    root = Path(root).expanduser()
    path = root / RIG_FILE
    if not path.is_file():
        raise RigError(f"no {RIG_FILE} in {root}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RigError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise RigError(f"{path}: top level must be a mapping")

    keys = set(data)
    missing = _REQUIRED - keys
    unknown = keys - _REQUIRED - _OPTIONAL
    if missing:
        raise RigError(f"{path}: missing required key(s): {sorted(missing)}")
    if unknown:
        raise RigError(f"{path}: unknown key(s): {sorted(unknown)}")

    name = _check_name(str(data["name"]))

    def resolved(key: str) -> "Path | None":
        value = data.get(key)
        if value in (None, ""):
            return None
        # Relative to the RIG, never to the console's working directory: the
        # console may be started from anywhere.
        #
        # Absolute but NOT resolved, which matters and was wrong first: a venv's
        # bin/python is a SYMLINK to the system interpreter, so resolve() turns
        # `venv/bin/python` into `/usr/bin/python3.12` and the subprocess gets
        # none of the venv's packages. normpath collapses `..` without following
        # a link.
        import os.path

        return Path(os.path.normpath(root.absolute() / str(value)))

    python = resolved("python")
    agent = resolved("agent")
    assert python is not None and agent is not None  # required above
    if not python.is_file():
        raise RigError(f"{path}: python {python} does not exist -- has install.sh run?")
    if not agent.is_file():
        raise RigError(f"{path}: agent {agent} does not exist")

    schema = None
    if "schema" in data:
        spec = data["schema"] or {}
        if not isinstance(spec, dict):
            raise RigError(f"{path}: schema must be a mapping")
        unknown_schema = set(spec) - {"limbs", "body_joints", "gripper"}
        if unknown_schema:
            raise RigError(
                f"{path}: schema has unknown key(s): {sorted(unknown_schema)}"
            )
        try:
            schema = RobotSchema(
                limbs=tuple(spec.get("limbs") or ()),
                body_joints=tuple(spec.get("body_joints") or ()),
                gripper=bool(spec.get("gripper", True)),
            )
        except ValueError as exc:
            raise RigError(f"{path}: schema is unusable: {exc}") from exc

    cameras = tuple(str(c) for c in (data.get("cameras") or ()))
    if schema is not None and not cameras:
        # Not fatal: a rig may declare its cameras only in recording.yaml. Said
        # here because the console's ablation view has nothing to offer without
        # them, and an empty list looks like a bug in the page.
        pass

    return Rig(
        name=name,
        root=root.resolve(),
        title=str(data.get("title") or name),
        python=python,
        agent=agent,
        teleop=resolved("teleop"),
        conf=resolved("conf"),
        sensor_map=resolved("sensor_map"),
        cameras=cameras,
        schema=schema,
        session=_session_spec(path, data.get("session")),
    )


def load_registry(path: "Path | str | None" = None) -> "list[Path]":
    """The rig directories this machine knows about. Missing file reads empty."""
    p = Path(path) if path is not None else REGISTRY_PATH
    if not p.is_file():
        return []
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RigError(f"{p}: not valid YAML: {exc}") from exc
    roots = data.get("rigs") if isinstance(data, dict) else data
    if roots is None:
        return []
    if not isinstance(roots, list):
        raise RigError(f"{p}: `rigs` must be a list of directories")
    return [Path(str(r)).expanduser() for r in roots]


def save_registry(roots: "list[Path]", path: "Path | str | None" = None) -> None:
    """Remember these rig directories. Creates the config directory if needed."""
    p = Path(path) if path is not None else REGISTRY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    seen: "list[str]" = []
    for root in roots:
        text = str(Path(root).expanduser())
        if text not in seen:
            seen.append(text)
    p.write_text(yaml.safe_dump({"rigs": seen}, sort_keys=False), encoding="utf-8")


def register_rig(root: "Path | str", path: "Path | str | None" = None) -> Rig:
    """Validate a rig directory and remember it. Returns the rig.

    Shared by ``actoris-harena rigs add`` and the console's register route, so
    a rig accepted in one is accepted in the other. Registering twice is a
    no-op. Raises :class:`RigError` for an unusable rig.yaml, and for a rig
    whose NAME is already registered from another directory -- the console
    addresses a rig by name, so two would make one unreachable.
    """
    rig = load_rig(root)
    roots = load_registry(path)
    known = [r.expanduser().absolute() for r in roots]
    if rig.root in known:
        return rig
    for other in roots:
        try:
            existing = load_rig(other)
        except RigError:
            continue
        if existing.name == rig.name:
            raise RigError(
                f"a rig called {rig.name!r} is already registered from "
                f"{existing.root}; remove it first or rename one"
            )
    roots.append(rig.root)
    save_registry(roots, path)
    return rig


def discover(
    roots: "list[Path] | None" = None, path=None
) -> "tuple[list[Rig], list[str]]":
    """Load every registered rig. Returns the ones that worked, and the refusals.

    A broken rig.yaml must not hide the working rigs: the console lists what it
    can drive and says plainly what it could not, rather than failing to start.
    """
    rigs: "list[Rig]" = []
    problems: "list[str]" = []
    for root in load_registry(path) if roots is None else roots:
        try:
            rigs.append(load_rig(root))
        except RigError as exc:
            problems.append(str(exc))
    by_name: "dict[str, Rig]" = {}
    for rig in rigs:
        if rig.name in by_name:
            problems.append(
                f"two rigs both called {rig.name!r}: {by_name[rig.name].root} and "
                f"{rig.root} -- rename one, because the console addresses a rig "
                f"by name"
            )
            continue
        by_name[rig.name] = rig
    return list(by_name.values()), problems
