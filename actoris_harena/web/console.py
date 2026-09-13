"""The shared console: one browser page, many robots, no hardware imports.

What is served here is everything that is the same whichever robot is plugged
in -- the collection drive and which directory is open, the episode browser's
lifecycle operations, the experiment grouping, and the training tab. Each of
those reasons about files and remote machines, never about a device.

Anything that DOES touch a device is reached through the selected rig's agent, a
subprocess started with that rig's own interpreter (see
``actoris_harena.rigs``). The console never imports feetech, ur-rtde, mujoco or
pyrealsense2, and could not: this rig's venv and the next one's cannot both
exist in one environment.

The rig is chosen in the page. ``/api/console`` says which rigs are known and
which is selected; ``/api/console/rig`` selects one.

The Collect and Signals tabs ARE served here, through ``web/agent_api.py``, which
proxies every one of their routes to the selected rig's agent. Nothing in this
process opens a device.

Not served here, and deliberately: the episode browser's PLAYBACK routes. Those
compose a panel grid per arm, colourise depth from that rig's own RealSense
metadata and plot its joint channels -- all genuinely shaped by how many arms a
robot has -- so they stay in each rig repo's ``tool/rig_web.py``. The dataset
MANAGEMENT half (listing, marking, deleting, merging, compacting) is here, in
``lifecycle_api``, because none of it depends on what the pictures show.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import aiohttp  # type: ignore[import]
from aiohttp import web  # type: ignore[import]

from actoris_harena.outputs import output_root
from actoris_harena.rigs import Rig
from actoris_harena.web.agent_api import add_agent_routes
from actoris_harena.web.jobs import add_job_routes
from actoris_harena.web.lifecycle_api import add_lifecycle_routes
from actoris_harena.web.projects_api import add_project_routes
from actoris_harena.web.roots_api import (
    add_root_routes,
    initial_root,
    root_required,
    unmount_own,
)
from actoris_harena.web.training_api import add_training_routes
from actoris_harena.web.util import preinit_tqdm_lock, revalidate_assets


def _rig_json(rig: Rig) -> "dict":
    return {
        "name": rig.name,
        "title": rig.title,
        "root": str(rig.root),
        "cameras": list(rig.cameras),
        "schema": (
            None
            if rig.schema is None
            else {
                "limbs": list(rig.schema.limbs),
                "body_joints": list(rig.schema.body_joints),
                "state_dim": rig.schema.state_dim,
            }
        ),
    }


def build_app(
    rigs: "list[Rig]",
    collection_dir: "str | None" = None,
    selected: "str | None" = None,
) -> web.Application:
    """Compose the console over a set of rigs. Opens no device and no rig."""
    preinit_tqdm_lock()
    app = web.Application(
        client_max_size=1024, middlewares=[revalidate_assets, root_required]
    )
    outputs = output_root()
    app["roots_file"] = outputs / "console_roots.json"
    app["mount_dir"] = outputs / "console_mounts"
    app["root"] = initial_root(collection_dir, app["roots_file"])
    app["executor"] = ThreadPoolExecutor(max_workers=2)
    app["job_executor"] = ThreadPoolExecutor(max_workers=1)
    app["jobs"] = {}
    app["destinations_file"] = None

    app["rigs"] = {rig.name: rig for rig in rigs}
    # A rig may be preselected on the command line, and otherwise the page
    # chooses. One rig and no choice is not the same thing as a default: a
    # console that silently picked the only rig would look identical to one that
    # had picked the wrong one when a second appeared.
    app["rig"] = selected

    async def handle_console(request: web.Request) -> web.Response:
        """What the page needs to describe itself: the drive, and the robots."""
        root = request.app["root"]
        return web.json_response(
            {
                "root": None if root is None else str(root),
                "rig": request.app["rig"],
                "rigs": [_rig_json(r) for r in request.app["rigs"].values()],
            }
        )

    async def handle_select_rig(request: web.Request) -> web.Response:
        """Select a rig, unless something is already running against one.

        The same rule roots_api applies to changing the collection directory,
        for the same reason: a job or a session holds state that belongs to the
        rig it started under, and switching underneath it would attribute one
        robot's work to another.
        """
        body = await request.json() if request.can_read_body else {}
        name = body.get("rig")
        if name is not None and name not in request.app["rigs"]:
            raise web.HTTPBadRequest(text=f"no rig called {name!r}")
        busy = [j for j in request.app["jobs"].values() if j.get("state") == "running"]
        if busy:
            raise web.HTTPConflict(
                text=(
                    f"{len(busy)} job(s) still running; wait for them or stop "
                    f"them before changing rig"
                )
            )
        # Stop the OLD rig's agent before switching. Two agents holding the
        # same camera is the one way this could waste an operator's time
        # silently -- the second open succeeds and then delivers nothing.
        previous = request.app["rig"]
        if previous is not None and previous != name:
            request.app["agents"].stop(previous)
        request.app["rig"] = name
        return web.json_response({"rig": name})

    async def open_client(a: web.Application) -> None:
        a["http"] = aiohttp.ClientSession()

    async def close_client(a: web.Application) -> None:
        await a["http"].close()
        unmount_own(a)

    app.router.add_get("/api/console", handle_console)
    app.router.add_post("/api/console/rig", handle_select_rig)
    add_root_routes(app)
    add_lifecycle_routes(app)
    add_project_routes(app)
    add_job_routes(app)
    add_training_routes(app)
    add_agent_routes(app)
    app.on_startup.append(open_client)
    app.on_cleanup.append(close_client)
    return app


def serve(
    rigs: "list[Rig]",
    collection_dir: "str | None" = None,
    port: int = 8000,
    selected: "str | None" = None,
) -> int:
    """Run the console until interrupted. Returns a process exit code."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    app = build_app(rigs, collection_dir=collection_dir, selected=selected)
    names = ", ".join(sorted(app["rigs"])) or "none registered"
    print(f"🖥️  console on http://127.0.0.1:{port}/   rigs: {names}")
    if app["root"] is not None:
        print(f"   collection: {app['root']}")
    web.run_app(app, host="127.0.0.1", port=port, print=None)
    return 0
