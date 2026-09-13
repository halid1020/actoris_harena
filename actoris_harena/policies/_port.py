"""How the ported policies are derived from LeRobot's, in one place.

``act``, ``diffusion``, ``pi05`` and ``fastwam`` are not rewritten here -- they
are the upstream modules moved into this repo so we own and can edit them. Keeping them
mechanically derived from upstream buys two things that a hand-edited copy would
lose within a week:

* **Checkpoints stay interchangeable.** The module tree is untouched, so the
  ``state_dict`` keys are identical and the finished 80 000-step ACT checkpoint
  loads into either implementation. ``test/unit/test_policy_ports.py`` proves it
  on the real file, not in principle.
* **A LeRobot bump is one command.** ``tool/port_policies.py`` re-derives every
  ported file; ``test_policy_ports.py`` fails the moment one drifts by hand.

Everything the port changes is below. Nothing else may change -- if a port needs
a real edit, it stops being a port and the rule for that file comes out of here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

#: Where the LeRobot source checkout lives, relative to this repo.
UPSTREAM = (
    Path(__file__).resolve().parents[2].parent
    / "lerobot"
    / "src"
    / "lerobot"
    / "policies"
)

#: The prefix every policy in this package registers under. One constant, because
#: it appears in the registered type, the class stem, the processor factory's name
#: and the ``name`` field, and a rename that reached three of those four would
#: leave a policy that imports and cannot be built.
PREFIX = "harena"

#: What they registered under BEFORE the package was shared, still accepted for
#: ever. Checkpoints carry the type string in their config.json, run matrices
#: carry it in a column, and a training run in flight resumes by reading its own
#: train_config.json -- so dropping the old name would strand all three. The new
#: name is registered FIRST, which is what makes ``config.type`` report it:
#: draccus's ``get_choice_name`` returns the first registered name for a class.
LEGACY_PREFIX = "so101"

#: ported directory -> (upstream type, upstream config class, upstream policy class, our stem)
PORTS = {
    "act": ("act", "ACTConfig", "ACTPolicy", "HarenaAct"),
    "diffusion": ("diffusion", "DiffusionConfig", "DiffusionPolicy", "HarenaDiffusion"),
    "pi05": ("pi05", "PI05Config", "PI05Policy", "HarenaPi05"),
    "fastwam": ("fastwam", "FastWAMConfig", "FastWAMPolicy", "HarenaFastwam"),
}

#: A port whose source is NOT in the pinned checkout, and the commit it is taken
#: from instead. FastWAM landed upstream after ``LEROBOT_COMMIT`` (3dd19d04), and
#: bumping the pin to reach it would change act, diffusion and pi05 underneath
#: every finished checkpoint -- including the port-parity measurement the three
#: existing ports rest on. Reading one directory out of a named commit costs
#: nothing else, and ``--check`` still re-derives it byte for byte, so the port
#: is exactly as checkable as the ones on disk.
#:
#: A SHA and not a branch: ``origin/main`` moves, and a port that silently
#: re-derives from a different upstream every fetch is not a port.
PORT_REF = {"fastwam": "3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e"}

#: Files a port carries verbatim, beyond the three the naming contract names.
#: FastWAM's model lives in a ``wan/`` subpackage whose imports are already
#: absolute or package-relative, so nothing in it needs rewriting -- but it does
#: need copying, and it needs to be in ``--check`` so a hand-edit is caught.
#: NOT ``__init__.py``: upstream's re-exports ``FastWAMPolicy``, which the port
#: renames, and the modeling rewrite adds no alias for a policy class the way the
#: configuration one does. Carrying it verbatim breaks the import. Ours is
#: written by hand, exactly as the other three ports' are.
PORT_EXTRA = {
    "fastwam": (
        "wan/__init__.py",
        "wan/adapters.py",
        "wan/components.py",
        "wan/model.py",
        "wan/modular.py",
        "wan/video_dit.py",
    )
}

#: A ported module imports from ``lerobot.policies`` what it used to reach by a
#: relative import, because only its home moved.
_RELATIVE = (
    (r"^(\s*)from \.\.pretrained import", r"\1from lerobot.policies.pretrained import"),
    (r"^(\s*)from \.\.utils import", r"\1from lerobot.policies.utils import"),
    (r"^(\s*)from \.\.rtc\.", r"\1from lerobot.policies.rtc."),
    (r"^(\s*)from \.\.pi_gemma import", r"\1from lerobot.policies.pi_gemma import"),
)

_CONFIG_TAIL = """

# The body above is the upstream module, unchanged, so that a checkpoint trained
# by either implementation loads into the other. Only the class name moved --
# LeRobot derives the policy class from it mechanically -- and this alias keeps
# the sibling modules' imports reading as they do upstream.
{old_config} = {stem}Config

# Registered under its former name as well, so a checkpoint written before the
# package was shared, and a run resuming from one, both still resolve. See
# LEGACY_PREFIX in actoris_harena.policies/_port.py for why this is permanent.
PreTrainedConfig.register_subclass("{legacy}_{typ}", {stem}Config)
"""

_PROCESSOR_TAIL = """

# lerobot.policies.factory._make_processors_from_policy_config looks for
# make_<registered type>_pre_post_processors in this module, by name.
make_{prefix}_{typ}_pre_post_processors = make_{typ}_pre_post_processors
# And under the former name, for a config decoded from an older checkpoint that
# somehow reports the legacy type.
make_{legacy}_{typ}_pre_post_processors = make_{typ}_pre_post_processors
"""


#: Names FastWAM imports from ``lerobot.processor`` that the pinned commit does
#: not export. Reproduced in ``common/processor_compat.py`` line for line, so the
#: port keeps the upstream file verbatim and only its import is redirected --
#: which is the same mechanical move ``_RELATIVE`` makes. Delete both when the
#: pin next moves and the port re-derives with no shim at all.
MISSING_FROM_PIN = (
    "make_default_policy_processor_steps",
    "make_policy_processor_pipelines",
)

_MISSING_HELPERS_RE = re.compile(
    r"^from lerobot\.processor import \(\n(?P<body>(?:[^)]*\n)*?)\)$", re.MULTILINE
)


def _missing_helpers(match: "re.Match[str]") -> str:
    """Split one ``from lerobot.processor import (...)`` into pin + shim."""
    names = [n.strip().rstrip(",") for n in match.group("body").splitlines()]
    names = [n for n in names if n]
    theirs = [n for n in names if n not in MISSING_FROM_PIN]
    ours = [n for n in names if n in MISSING_FROM_PIN]
    if not ours:
        return match.group(0)
    block = "from lerobot.processor import (\n"
    block += "".join(f"    {n},\n" for n in theirs)
    block += ")\n"
    block += "from ..common.processor_compat import (\n"
    block += "".join(f"    {n},\n" for n in ours)
    block += ")"
    return block


def port_text(text: str, name: str, kind: str) -> str:
    """Return the ported form of one upstream file. Pure.

    ``kind`` is ``configuration``, ``modeling`` or ``processor``.
    """
    typ, old_config, old_policy, stem = PORTS[name]
    for pattern, repl in _RELATIVE:
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)

    if kind == "configuration":
        text = text.replace(
            f'@PreTrainedConfig.register_subclass("{typ}")',
            f'@PreTrainedConfig.register_subclass("{PREFIX}_{typ}")',
        )
        text = re.sub(
            rf"^class {old_config}\(", f"class {stem}Config(", text, flags=re.MULTILINE
        )
        return text + _CONFIG_TAIL.format(
            old_config=old_config, stem=stem, legacy=LEGACY_PREFIX, typ=typ
        )

    if kind == "modeling":
        text = re.sub(
            rf"^class {old_policy}\(", f"class {stem}Policy(", text, flags=re.MULTILINE
        )
        return re.sub(
            rf'^(    name = )"{typ}"$', rf'\1"{PREFIX}_{typ}"', text, flags=re.MULTILINE
        )

    if kind == "processor":
        # ProcessorStepRegistry is a GLOBAL namespace, so a ported step whose name
        # is unchanged collides with the upstream one the moment both are imported
        # -- which is exactly what the equivalence test does. This is the only
        # rename that is not cosmetic.
        text = text.replace(
            'ProcessorStepRegistry.register(name="pi05_',
            f'ProcessorStepRegistry.register(name="{PREFIX}_pi05_',
        )
        text = _MISSING_HELPERS_RE.sub(_missing_helpers, text)
        return text + _PROCESSOR_TAIL.format(
            typ=typ, prefix=PREFIX, legacy=LEGACY_PREFIX
        )

    if kind == "verbatim":
        # Carried along, not rewritten: the relative imports inside a
        # subpackage still resolve, and its lerobot imports are already absolute.
        return text

    raise ValueError(f"unknown kind {kind!r}")


#: The LeRobot checkout, for the git reads a ref-sourced port needs.
UPSTREAM_REPO = UPSTREAM.parents[2]


def read_upstream(name: str, relative: str) -> str:
    """One upstream file's text, from the working tree or from a pinned commit.

    A port named in :data:`PORT_REF` is read with ``git show`` because the pinned
    checkout does not contain it. The failure when the object is absent is worth
    a sentence of its own: it means the checkout has never fetched the commit,
    which is a ``git fetch`` away and not a code problem.
    """
    ref = PORT_REF.get(name)
    if ref is None:
        return (UPSTREAM / name / relative).read_text(encoding="utf-8")
    path = f"src/lerobot/policies/{name}/{relative}"
    try:
        return subprocess.run(
            ["git", "-C", str(UPSTREAM_REPO), "show", f"{ref}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as problem:
        raise FileNotFoundError(
            f"{path} is not in {UPSTREAM_REPO} at {ref[:9]}. This port is taken "
            f"from a commit newer than LEROBOT_COMMIT, so the checkout needs "
            f"`git -C {UPSTREAM_REPO} fetch origin` before it can be re-derived."
        ) from problem


def ported_files() -> "list[tuple[str, Path, str, str]]":
    """(upstream relative path, our path, port name, kind) for every ported file.

    ``kind`` is ``configuration``, ``modeling``, ``processor`` -- the three the
    naming contract names -- or ``verbatim`` for a file a port carries along
    unchanged, such as FastWAM's ``wan/`` subpackage. A verbatim file is in the
    list so that ``--check`` catches a hand-edit to it too; being unrewritten is
    not the same as being unowned.
    """
    here = Path(__file__).resolve().parent
    out: "list[tuple[str, Path, str, str]]" = []
    for name in PORTS:
        for kind in ("configuration", "modeling", "processor"):
            stem = f"{kind}_{name}.py"
            out.append((stem, here / name / stem, name, kind))
        for extra in PORT_EXTRA.get(name, ()):
            out.append((extra, here / name / extra, name, "verbatim"))
    return out
