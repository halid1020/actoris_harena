"""``actoris-harena`` -- the shared console, and the list of rigs it can drive.

    actoris-harena rigs list
    actoris-harena rigs add ~/Projects/so101_garment
    actoris-harena rigs remove so101
    actoris-harena console --dir /media/hdd/so101 --port 8000

The console serves the parts of the pipeline that are the same on every robot --
the collection drive, the episode browser, the training tab -- in its own
process, and reaches anything that touches a device through the selected rig's
AGENT, a subprocess started with that rig's own interpreter. That indirection is
not architecture for its own sake: this rig's venv needs feetech and mujoco and
the next one needs ur-rtde, and pip cannot satisfy both at once.

Which rig is selected is chosen in the page. See ``actoris_harena.rigs``.
"""

import argparse
import sys
from pathlib import Path

from actoris_harena.rigs import (
    Rig,
    RigError,
    discover,
    load_registry,
    load_rig,
    register_rig,
    save_registry,
)


def _print_rigs(rigs: "list[Rig]", problems: "list[str]") -> None:
    if not rigs and not problems:
        print("no rigs registered. Add one with:")
        print("    actoris-harena rigs add /path/to/a/rig")
        return
    for rig in rigs:
        print(f"  {rig.name:<12} {rig.title}")
        print(f"  {'':<12} {rig.root}")
        if rig.schema is not None:
            print(
                f"  {'':<12} {rig.schema.state_dim} channels, "
                f"{len(rig.cameras)} camera(s)"
            )
    for problem in problems:
        print(f"  ⚠️  {problem}")


def _cmd_rigs(args: argparse.Namespace) -> int:
    if args.action == "list":
        rigs, problems = discover(path=args.registry)
        _print_rigs(rigs, problems)
        return 1 if problems and not rigs else 0

    if args.action == "add":
        root = Path(args.path).expanduser()
        try:
            rig = register_rig(root, args.registry)
        except RigError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 2
        print(f"✅ {rig.name} ({rig.title}) registered from {rig.root}")
        return 0

    # remove
    roots = load_registry(args.registry)
    kept: "list[Path]" = []
    dropped: "list[Path]" = []
    for root in roots:
        try:
            name = load_rig(root).name
        except RigError:
            # A directory that no longer describes a rig is removable BY PATH,
            # which is the only handle left once its rig.yaml is gone.
            name = ""
        (dropped if args.name in (name, str(root)) else kept).append(root)
    if not dropped:
        print(f"❌ no registered rig called {args.name!r}", file=sys.stderr)
        return 2
    save_registry(kept, args.registry)
    print(f"✅ {args.name} removed (its files are untouched)")
    return 0


def _cmd_console(args: argparse.Namespace) -> int:
    from actoris_harena.web.console import serve

    rigs, problems = discover(path=args.registry)
    for problem in problems:
        print(f"⚠️  {problem}", file=sys.stderr)
    if args.rig and args.rig not in {r.name for r in rigs}:
        print(
            f"❌ no registered rig called {args.rig!r}; "
            f"known: {sorted(r.name for r in rigs)}",
            file=sys.stderr,
        )
        return 2
    return serve(
        rigs,
        collection_dir=args.dir,
        port=args.port,
        selected=args.rig,
        registry=args.registry,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="actoris-harena", description=__doc__)
    parser.add_argument(
        "--registry",
        default=None,
        help="rig registry file (default: ~/.config/actoris_harena/rigs.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rigs = sub.add_parser("rigs", help="list, add or remove the robots on this machine")
    rigs_sub = rigs.add_subparsers(dest="action", required=True)
    rigs_sub.add_parser("list", help="show every registered rig")
    add = rigs_sub.add_parser("add", help="register a directory holding a rig.yaml")
    add.add_argument("path")
    remove = rigs_sub.add_parser(
        "remove", help="forget a rig (its files are untouched)"
    )
    remove.add_argument("name")
    rigs.set_defaults(func=_cmd_rigs)

    console = sub.add_parser("console", help="serve the browser console")
    console.add_argument("--dir", default=None, help="collection directory to open")
    console.add_argument("--port", type=int, default=8000)
    console.add_argument("--rig", default=None, help="preselect this rig")
    console.set_defaults(func=_cmd_console)
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
