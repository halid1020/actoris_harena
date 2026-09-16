"""The Training tab: pick a dataset, some policies and a machine, and start.

Thin, deliberately. Which runs are possible, what a `-` becomes and every
reason a row cannot work all live in ``actoris_harena.training``; staging and
submission live in ``tool/train_launch``. This module reads a request, runs the
blocking parts off the event loop and turns an exception into a status code --
so a run started here is the same run, checked the same way and recorded in the
same file, as one started from the terminal.

Two shapes are borrowed from the rest of the console. Refusals come back at 200
inside the plan, the way ``/api/session/plan`` does, so the page can show them
while the operator is still choosing; and a launch is a ``jobs.py`` record,
because copying a dataset over SSH takes minutes and must outlive the request
that asked for it (``refuse_while_busy`` then also stops it racing a merge).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from aiohttp import web  # type: ignore[import]

from actoris_harena.recording.camera_profile import profile
from actoris_harena.recording.dataset_view import available_camera_names
from actoris_harena.training.destinations import (
    load_destinations,
    load_runs,
    reachable,
    save_runs,
    ssh_argv,
    stage_dir,
)
from actoris_harena.training.matrix import (
    POLICIES,
    MatrixError,
    make_row,
    policy_available,
    resolved,
    row_refusals,
    unavailable_message,
)
from actoris_harena.training.runs import (
    REAL_TREE,
    SIM_TREE,
    RunsError,
    cell_path,
    check_name,
    config_path,
    discover,
    log_path,
    out_roots,
    prediction_glob,
    read_log,
    run_dir_names,
    with_measurements,
)
from actoris_harena.web.jobs import finish, new_job, refuse_while_busy
from actoris_harena.web.util import in_executor


def _destinations(app: web.Application) -> "dict[str, dict]":
    return load_destinations(app.get("destinations_file"))


def _destination(app: web.Application, name: str) -> "dict[str, Any]":
    """One destination, plus what it said about itself when asked.

    Only the local machine is asked, and only because its hardware is the one
    thing this repo cannot write down -- see actoris_harena.training.runs.
    """
    known = _destinations(app)
    if name not in known:
        raise web.HTTPBadRequest(
            text=f"no destination called {name!r}; this rig knows "
            f"{', '.join(sorted(known))}"
        )
    return with_measurements(known[name])


# ── the cache in front of the machines ───────────────────────────────────────
#
# Every question here costs a whole SSH, and a page with several run cards on
# it asks them repeatedly. So an answer is remembered for a few seconds and one
# in-flight request is shared: how often the browser polls then has no bearing
# on how often a machine is disturbed, and a second tab is free.

PROGRESS_TTL_S = 20.0
DISCOVER_TTL_S = 45.0
MACHINE_TTL_S = 45.0


async def _cached(
    app: web.Application, key: tuple, ttl: float, fetch, fresh: bool = False
) -> "Any":
    cache = app.setdefault("training_cache", {})
    locks = app.setdefault("training_cache_locks", {})
    entry = cache.get(key)
    if entry and not fresh and (time.time() - entry[0]) < ttl:
        return entry[1]
    lock = locks.setdefault(key, asyncio.Lock())
    async with lock:
        entry = cache.get(key)
        if entry and not fresh and (time.time() - entry[0]) < ttl:
            return entry[1]
        value = await in_executor(app, fetch)
        cache[key] = (time.time(), value)
        return value


def _runs_file(app: web.Application) -> Path:
    return app["training_file"]


# ── what the form is built from (blocking; read through in_executor) ─────────


def _datasets(root: Path) -> "list[dict[str, Any]]":
    """Every dataset under the collection directory, with what it records."""
    out = []
    for child in sorted(Path(root).iterdir()):
        info_path = child / "meta" / "info.json"
        if not info_path.is_file():
            continue
        try:
            info = json.loads(info_path.read_text())
        except (OSError, ValueError):
            continue
        out.append(
            {
                "name": child.name,
                "episodes": info.get("total_episodes"),
                "frames": info.get("total_frames"),
                "cameras": available_camera_names(info),
            }
        )
    return out


def _read_info(root: Path, dataset: str) -> "dict[str, Any]":
    path = Path(root) / dataset / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"no dataset called {dataset!r} in this directory")
    return json.loads(path.read_text())


def _config(app: web.Application, root: Path) -> "dict[str, Any]":
    return {
        "datasets": _datasets(root),
        "composites": {
            name: list(parts) for name, parts in profile().composites.items()
        },
        "policies": [
            {
                "name": name,
                "available": policy_available(name),
                # Only ever a sentence about what to change; a policy that is
                # available has nothing to say.
                "problem": None
                if policy_available(name)
                else unavailable_message(name),
                "steps": spec.get("steps"),
                "batch": spec.get("batch"),
                "hours": spec.get("hours"),
                "max_cameras": spec.get("max_cameras"),
                # Where the implementation lives, and -- for a port -- which
                # LeRobot policy it is a port OF. Without these the page shows
                # nine unstructured checkboxes and nothing says that `act` and
                # `so101_act` are the same model run by different code, which
                # is the entire choice this list exists to offer.
                "local": bool(spec.get("local")),
                "ported_from": spec.get("ported_from"),
            }
            for name, spec in POLICIES.items()
        ],
        "destinations": [
            {
                "name": name,
                "kind": dest["kind"],
                "ssh": dest["ssh"],
                "stage": stage_dir(dest),
                "limits": dest.get("limits") or {},
            }
            for name, dest in sorted(_destinations(app).items())
        ],
    }


# ── resolving a request into rows ────────────────────────────────────────────


def _rows_from(body: "dict[str, Any]", dataset: str) -> "list[dict[str, str]]":
    policies = body.get("policies") or []
    if not policies:
        raise web.HTTPBadRequest(text="choose at least one policy")
    return [
        make_row(
            dataset,
            str(policy),
            cameras=str(body.get("cameras") or "all"),
            steps=str(body.get("steps") or "-"),
            batch=str(body.get("batch") or "-"),
            hours=body.get("hours") or None,
            slots=str(body.get("slots") or "-"),
            extra=str(body.get("extra") or "-"),
        )
        for policy in policies
    ]


def _plan(app: web.Application, root: Path, body: "dict[str, Any]") -> "dict[str, Any]":
    """Rows, and every refusal, without touching the destination. Blocking."""
    dataset = str(body.get("dataset") or "").strip()
    if not dataset:
        raise web.HTTPBadRequest(text="choose a dataset")
    name = str(body.get("dest") or "").strip()
    dest = _destination(app, name)
    if body.get("allow_cpu"):
        dest = {**dest, "allow_cpu": True}
    try:
        info = _read_info(root, dataset)
    except (OSError, ValueError) as exc:
        raise web.HTTPNotFound(text=str(exc))

    rows = _rows_from(body, dataset)
    refusals: "list[str]" = []
    for row in rows:
        try:
            refusals += row_refusals(row, info, dest)
        except MatrixError as exc:
            refusals.append(str(exc))
    return {
        "dataset": dataset,
        "episodes": info.get("total_episodes"),
        "rows": [
            {
                **row,
                "resolved_steps": resolved(row, "steps", dest),
                "resolved_batch": resolved(row, "batch", dest),
            }
            for row in rows
        ],
        "refusals": refusals,
        "warnings": [],
    }


# ── routes ───────────────────────────────────────────────────────────────────


async def handle_config(request: web.Request) -> web.Response:
    app = request.app
    try:
        return web.json_response(await in_executor(app, _config, app, app["root"]))
    except (OSError, ValueError) as exc:
        # A destinations file that cannot be read is a configuration error the
        # operator can fix, and the message names the offending key.
        raise web.HTTPBadRequest(text=str(exc))


async def handle_reach(request: web.Request) -> web.Response:
    """Can this machine be reached? Kept off /config because it is an SSH away."""
    body = await request.json()
    known = _destinations(request.app)
    name = str(body.get("dest") or "")
    if name not in known:
        raise web.HTTPBadRequest(text=f"no destination called {name!r}")
    problem = await in_executor(request.app, reachable, known[name])
    return web.json_response({"dest": name, "ok": problem is None, "problem": problem})


async def handle_plan(request: web.Request) -> web.Response:
    body = await request.json()
    plan = await in_executor(request.app, _plan, request.app, request.app["root"], body)
    return web.json_response(plan)


async def handle_start(request: web.Request) -> web.Response:
    app = request.app
    body = await request.json()
    refuse_while_busy(app)

    plan = await in_executor(app, _plan, app, app["root"], body)
    if plan["refusals"]:
        # Re-resolved rather than trusted from the page: the drive, the
        # dataset and the destinations file can all have changed since the
        # plan the operator was shown.
        raise web.HTTPBadRequest(text="; ".join(plan["refusals"]))

    dest = _destination(app, str(body["dest"]))
    if body.get("allow_cpu"):
        dest = {**dest, "allow_cpu": True}
    problem = await in_executor(app, reachable, dest)
    if problem:
        raise web.HTTPBadGateway(text=problem)

    dataset = plan["dataset"]
    # The COLLECTION directory, not the dataset: launch() joins the name. The
    # two were confused once, and the result was every dataset on the drive
    # uploaded under a directory named after the drive.
    collection_dir = Path(app["root"])
    rows = [
        {k: v for k, v in row.items() if not k.startswith("resolved_")}
        for row in plan["rows"]
    ]
    info = await in_executor(app, _read_info, collection_dir, dataset)
    job = new_job(app, "training", f"{dataset} → {dest['name']}", dest=dest["name"])

    def work() -> None:
        # The launcher is the RIG's: staging a dataset and submitting a job means
        # knowing that rig's repo layout and its Slurm scripts. app["launch"] is
        # set by whoever composed these routes -- tool/rig_web.py in a rig repo,
        # or the shared console from the selected rig's entry point.
        launch = app["launch"]
        if launch is None:
            raise RuntimeError(
                "no launcher is configured: set app['launch'] when adding the "
                "training routes"
            )

        try:
            record = launch(
                dest,
                collection_dir,
                dataset,
                rows,
                info,
                restage=bool(body.get("restage")),
                progress=lambda message: job.__setitem__("message", message),
            )
        except BaseException as exc:  # noqa: B036 - reported, not swallowed
            finish(job, "failed", str(exc) or exc.__class__.__name__)
            return
        if record is None:
            # Only a dry run returns nothing, and the console never asks for
            # one -- so this is a bug rather than an outcome, and it says so
            # instead of recording a run that does not exist.
            finish(job, "failed", "the launcher reported no run")
            return
        # The console keeps its own copy so it can list runs without the
        # terminal's output directory, and so a run started here survives a
        # restart of the page.
        _remember(app, record)
        job["run"] = record["id"]
        finish(job, "done", f"launched {record['id']} on {dest['name']}")

    asyncio.get_running_loop().run_in_executor(app["job_executor"], work)
    return web.json_response(job)


def _remember(app: web.Application, record: "dict[str, Any]") -> None:
    from actoris_harena.training.destinations import remember_run

    path = _runs_file(app)
    save_runs(path, remember_run(load_runs(path), record))


async def handle_runs(request: web.Request) -> web.Response:
    """What has been launched from this console. Asks no machine anything."""
    state = await in_executor(request.app, load_runs, _runs_file(request.app))
    return web.json_response(state["runs"])


def _live(dest: dict, record: "dict[str, Any]") -> "dict[str, Any]":
    """Ask the machine how the run is going. Blocking; never raises."""
    problem = reachable(dest)
    if problem:
        return {"ok": False, "text": problem}
    if dest["kind"] == "slurm":
        jobs = record.get("jobs") or []
        if not jobs:
            return {"ok": True, "text": "no Slurm job id was recorded"}
        command = (
            f"squeue --jobs={','.join(jobs)} --noheader --format='%i %T %M' || true"
        )
    else:
        command = f"cd {dest['repo']} && bash hpc/gpu_box_run.sh --status"
    try:
        proc = subprocess.run(
            ssh_argv(dest, command), capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "text": str(exc)}
    # Standard output ONLY. CREATE greets every login with an MFA banner and a
    # QR code on stderr, and squeue prints nothing at all for a job that has
    # finished -- so falling back to stderr rendered that QR code as the status
    # of every completed run on the cluster.
    text = (proc.stdout or "").strip()
    if not text and proc.returncode != 0:
        return {"ok": False, "text": (proc.stderr or "").strip() or "no answer"}
    return {"ok": True, "text": text or "nothing running"}


async def handle_run_status(request: web.Request) -> web.Response:
    app = request.app
    record = await in_executor(app, _record, app, request.match_info["id"])
    dest = _destinations(app).get(record.get("dest", ""))
    if dest is None:
        raise web.HTTPNotFound(text=f"destination {record.get('dest')!r} is gone")
    return web.json_response(await in_executor(app, _live, dest, record))


def _record(app: web.Application, run_id: str) -> "dict[str, Any]":
    for record in load_runs(_runs_file(app))["runs"]:
        if record["id"] == run_id:
            return record
    raise web.HTTPNotFound(text=f"no run called {run_id!r}")


def _stop(dest: dict, record: "dict[str, Any]") -> str:
    if dest["kind"] == "slurm":
        jobs = record.get("jobs") or []
        if not jobs:
            raise web.HTTPBadRequest(text="no Slurm job id was recorded for this run")
        command = f"scancel {' '.join(jobs)}"
    else:
        command = f"cd {dest['repo']} && bash hpc/gpu_box_run.sh --stop"
    proc = subprocess.run(
        ssh_argv(dest, command), capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        raise web.HTTPBadGateway(text=(proc.stderr or proc.stdout or "").strip())
    return (proc.stdout or "stopped").strip()


async def handle_run_stop(request: web.Request) -> web.Response:
    app = request.app
    record = await in_executor(app, _record, app, request.match_info["id"])
    dest = _destinations(app).get(record.get("dest", ""))
    if dest is None:
        raise web.HTTPNotFound(text=f"destination {record.get('dest')!r} is gone")
    return web.json_response({"stopped": await in_executor(app, _stop, dest, record)})


# ── watching a run ───────────────────────────────────────────────────────────


def _machine(app: web.Application, name: str) -> "dict[str, Any]":
    """Whether a machine answers, and what it says about itself. Blocking."""
    dest = _destination(app, name)
    problem = reachable(dest)
    return {
        "name": name,
        "kind": dest["kind"],
        "ok": problem is None,
        "problem": problem,
        "measured": dest.get("measured"),
        "roots": out_roots(dest),
    }


async def handle_machines(request: web.Request) -> web.Response:
    app = request.app
    fresh = request.query.get("refresh") == "1"
    names = sorted(_destinations(app))
    out = await asyncio.gather(
        *[
            _cached(app, ("machine", n), MACHINE_TTL_S, _make(_machine, app, n), fresh)
            for n in names
        ]
    )
    return web.json_response(list(out))


def _make(fn, *args):
    """A no-argument callable for the executor, without a lambda per call."""

    def call():
        return fn(*args)

    return call


def _launched_index(app: web.Application) -> "dict[tuple, dict[str, Any]]":
    """Every recorded launch, keyed by the directory it writes into.

    A run started from a terminal is not in here at all, which is the point of
    discovering runs rather than listing this file: nine of the run directories
    on CREATE predate the launcher entirely.
    """
    index: "dict[tuple, dict[str, Any]]" = {}
    for record in load_runs(_runs_file(app))["runs"]:
        for directory in run_dir_names(record):
            for policy in record.get("policies") or []:
                index.setdefault((record.get("dest"), directory, policy), record)
    return index


def _discover(app: web.Application, name: str) -> "dict[str, Any]":
    """What training one machine holds, annotated with what we launched."""
    found = discover(_destination(app, name))
    index = _launched_index(app)
    for entry in found["runs"]:
        entry["dest"] = name
        record = index.get((name, entry["run"], entry["policy"]))
        if record:
            entry["id"] = record["id"]
            entry["dataset"] = record.get("dataset")
            entry["episodes"] = record.get("episodes")
            entry["started"] = record.get("started")
        elif entry.get("sim"):
            # A sim run's directory is a timestamp, so it names no dataset. What
            # it trained on is the task, which the cell path does name.
            entry["dataset"] = entry.get("task") or entry["run"]
        else:
            # The directory names the dataset it was built from, which is the
            # only thing a run nobody recorded can still say about itself.
            entry["dataset"] = entry["run"].split("__")[0]
    return found


async def handle_discovered(request: web.Request) -> web.Response:
    app = request.app
    fresh = request.query.get("refresh") == "1"
    wanted = request.query.get("dest")
    names = [wanted] if wanted else sorted(_destinations(app))
    results = await asyncio.gather(
        *[
            _cached(
                app, ("discover", n), DISCOVER_TTL_S, _make(_discover, app, n), fresh
            )
            for n in names
        ]
    )
    runs: "list[dict[str, Any]]" = []
    problems = {}
    for name, result in zip(names, results):
        runs += result["runs"]
        if result.get("problem"):
            problems[name] = result["problem"]
    runs.sort(key=lambda r: -(r.get("mtime") or 0))
    return web.json_response({"runs": runs, "problems": problems})


def _progress(
    app: web.Application,
    name: str,
    run: str,
    policy: str,
    tree: str = REAL_TREE,
    cell: "str | None" = None,
) -> "dict":
    dest = _destination(app, name)
    # Both roots are tried because a machine may hold runs in either -- the
    # local one does, since a console started through setup.sh writes into the
    # repo while the launcher stages into the cache.
    last: "dict[str, Any]" = {}
    for root in out_roots(dest):
        result = read_log(
            dest,
            log_path(root, run, policy, tree),
            config=config_path(root, run, cell or policy, tree),
            cell=cell_path(root, run, cell, tree) if cell else None,
            prediction=prediction_glob(root, run, cell or policy, tree),
        )
        if result["ok"]:
            return {**result, "dest": name, "run": run, "policy": policy}
        last = result
    return {**last, "dest": name, "run": run, "policy": policy}


def _cell_param(raw: str) -> "str | None":
    """``mode/task/policy``, each a plain name. Anything else is refused.

    The cell reaches a remote shell inside a command, so it is validated the
    way run and policy names are rather than escaped -- and it is exactly three
    segments because that is what ``long_vla_sim.sh`` writes. A caller that
    sends something else is not describing a cell this rig created.
    """
    if not raw:
        return None
    parts = raw.split("/")
    if len(parts) != 3:
        raise RunsError(
            f"cell {raw!r} is not <mode>/<task>/<policy>; a sim run's results "
            "live three directories deep and nowhere else."
        )
    for part in parts:
        check_name(part, "cell segment")
    return "/".join(parts)


async def handle_progress(request: web.Request) -> web.Response:
    app = request.app
    name = request.query.get("dest") or ""
    try:
        # These land inside a command on the far machine, so what they may
        # contain is checked rather than escaped -- and a name this rig would
        # never have created is a bad request, not a server error.
        run = check_name(request.query.get("run") or "", "run directory")
        policy = check_name(request.query.get("policy") or "", "policy")
        cell = _cell_param(request.query.get("cell") or "")
    except RunsError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    # The tree is named by the caller because discovery already told it which
    # one the run is in, and the two are shaped differently enough that
    # guessing would cost a second SSH per run to find out.
    tree = SIM_TREE if request.query.get("tree") == SIM_TREE else REAL_TREE
    fresh = request.query.get("refresh") == "1"
    out = await _cached(
        app,
        ("progress", name, run, policy, tree, cell),
        PROGRESS_TTL_S,
        _make(_progress, app, name, run, policy, tree, cell),
        fresh,
    )
    return web.json_response(out)


def add_training_routes(app: web.Application) -> None:
    """Register the Training tab. ``training_file`` must exist."""
    app.setdefault("destinations_file", None)
    app.setdefault("launch", None)
    app.add_routes(
        [
            web.get("/api/training/config", handle_config),
            web.post("/api/training/reach", handle_reach),
            web.post("/api/training/plan", handle_plan),
            web.post("/api/training/start", handle_start),
            web.get("/api/training/runs", handle_runs),
            web.get("/api/training/runs/{id}/status", handle_run_status),
            web.post("/api/training/runs/{id}/stop", handle_run_stop),
            web.get("/api/training/machines", handle_machines),
            web.get("/api/training/discovered", handle_discovered),
            web.get("/api/training/progress", handle_progress),
        ]
    )
