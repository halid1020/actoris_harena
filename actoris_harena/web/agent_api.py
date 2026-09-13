"""The selected rig's devices, reached through its agent.

THE COLLECT AND SIGNALS TABS, for a console that cannot import a robot. Every
route here is a PROXY: it finds the selected rig, makes sure that rig's agent is
running, and forwards. The console never opens a camera, never touches a motor
bus, and never imports feetech or ur-rtde -- it could not, since one rig's venv
needs the first and the other's needs the second.

WHY PROXYING AND NOT MIGRATING. The obvious alternative was to move each rig's
session and sensor code into this package behind another protocol. That would
have been the wrong shape: supervising a collection session means knowing one
repo's teleop entry point and its flags, and probing an arm means knowing which
bus it speaks. Those are not two implementations of one idea, they are two
different programs -- so the seam is a PROCESS boundary, and the contract between
them is HTTP on loopback rather than a Python protocol.

It is also not a new boundary. A running collection session has always been a
subprocess the console read over a loopback monitor; the agent is that same
arrangement for the idle case.

An agent is started on demand and kept. Stopping it is what releases the rig's
devices, so switching rigs stops the old one first -- two agents holding the same
camera is the one way this could waste an operator's time silently.
"""

import asyncio
import socket
import subprocess
import time

import aiohttp  # type: ignore[import]
from aiohttp import web  # type: ignore[import]

#: How long to wait for a freshly started agent to answer. Opening a RealSense
#: and a motor bus is slow; a shorter wait reports a working rig as broken.
_START_TIMEOUT_S = 20.0

#: How long a proxied request may take. Long enough for a camera to open, short
#: enough that a hung agent does not hang the page.
_REQUEST_TIMEOUT_S = 25.0


class AgentError(RuntimeError):
    """The selected rig's agent could not be started or could not be reached."""


def _free_port() -> int:
    """A loopback port nothing is using. Asked of the OS, not guessed."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class AgentPool:
    """One agent process per rig, started on demand and kept.

    Deliberately not one per request: starting an agent opens devices, and doing
    that per click would make the page unusable and the cameras unhappy.
    """

    def __init__(self) -> None:
        self._agents: "dict[str, tuple[subprocess.Popen, int]]" = {}

    def port(self, name: str) -> "int | None":
        found = self._agents.get(name)
        return None if found is None else found[1]

    def running(self, name: str) -> bool:
        found = self._agents.get(name)
        return found is not None and found[0].poll() is None

    def start(self, rig) -> int:
        """Start this rig's agent if it is not already up. Returns its port."""
        if self.running(rig.name):
            return self._agents[rig.name][1]
        port = _free_port()
        try:
            process = subprocess.Popen(
                rig.agent_argv("--port", str(port)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            raise AgentError(f"could not start {rig.name}'s agent: {exc}") from exc
        self._agents[rig.name] = (process, port)
        return port

    def stop(self, name: str) -> None:
        """Stop one agent, which is what releases that rig's devices."""
        found = self._agents.pop(name, None)
        if found is None:
            return
        process, _port = found
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # A capture thread sitting in a blocking device read will not answer
            # a terminate. Killing it is better than leaving the camera held.
            process.kill()
            process.wait(timeout=5)

    def stop_all(self) -> None:
        for name in list(self._agents):
            self.stop(name)


async def _wait_until_answering(session, port: int) -> None:
    deadline = time.monotonic() + _START_TIMEOUT_S
    last: "Exception | None" = None
    while time.monotonic() < deadline:
        try:
            async with session.get(
                f"http://127.0.0.1:{port}/status",
                timeout=aiohttp.ClientTimeout(total=2),
            ) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - not up yet is the normal case
            last = exc
        await asyncio.sleep(0.25)
    raise AgentError(
        f"the agent did not answer within {_START_TIMEOUT_S:.0f}s"
        + (f" ({type(last).__name__})" if last else "")
    )


def add_agent_routes(app: web.Application) -> None:
    """Register the Collect and Signals routes, which all proxy to an agent."""
    app["agents"] = AgentPool()

    def _selected(request: web.Request):
        name = request.app["rig"]
        if name is None:
            raise web.HTTPConflict(text="no rig is selected; choose one first")
        rig = request.app["rigs"].get(name)
        if rig is None:
            raise web.HTTPConflict(text=f"rig {name!r} is no longer registered")
        return rig

    async def _agent_port(request: web.Request) -> int:
        rig = _selected(request)
        pool = request.app["agents"]
        already = pool.running(rig.name)
        port = pool.start(rig)
        if not already:
            try:
                await _wait_until_answering(request.app["http"], port)
            except AgentError as exc:
                pool.stop(rig.name)
                raise web.HTTPBadGateway(text=str(exc)) from exc
        return port

    async def _proxy(request: web.Request, path: str, method: str = "GET"):
        port = await _agent_port(request)
        url = f"http://127.0.0.1:{port}{path}"
        body = await request.read() if request.can_read_body else None
        try:
            async with request.app["http"].request(
                method,
                url,
                data=body,
                headers={"Content-Type": "application/json"} if body else None,
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S),
            ) as response:
                return web.Response(
                    body=await response.read(),
                    status=response.status,
                    content_type=response.content_type,
                )
        except asyncio.TimeoutError as exc:
            raise web.HTTPGatewayTimeout(
                text=f"the rig's agent did not answer within "
                f"{_REQUEST_TIMEOUT_S:.0f}s"
            ) from exc

    async def handle_agent_status(request: web.Request) -> web.Response:
        """What the rig says about itself. Starts the agent if it is not up."""
        return await _proxy(request, "/status")

    async def handle_agent_stop(request: web.Request) -> web.Response:
        """Stop the agent, releasing the rig's cameras and buses."""
        rig = _selected(request)
        request.app["agents"].stop(rig.name)
        return web.json_response({"running": False})

    async def handle_cameras(request: web.Request) -> web.Response:
        return await _proxy(request, "/cameras")

    async def handle_cameras_start(request: web.Request) -> web.Response:
        return await _proxy(request, "/cameras/start", "POST")

    async def handle_cameras_stop(request: web.Request) -> web.Response:
        return await _proxy(request, "/cameras/stop", "POST")

    async def handle_camera_stream(request: web.Request) -> web.StreamResponse:
        """MJPEG, streamed through rather than buffered.

        Buffering would hold the whole stream in memory and never end, since an
        MJPEG response has no length.
        """
        port = await _agent_port(request)
        name = request.match_info["name"]
        url = f"http://127.0.0.1:{port}/cameras/{name}"
        async with request.app["http"].get(
            url, timeout=aiohttp.ClientTimeout(total=None)
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                headers={
                    "Content-Type": upstream.headers.get(
                        "Content-Type", "multipart/x-mixed-replace"
                    ),
                    "Cache-Control": "no-store",
                },
            )
            await response.prepare(request)
            try:
                async for chunk in upstream.content.iter_chunked(65536):
                    await response.write(chunk)
            except (ConnectionResetError, asyncio.CancelledError):
                pass  # the tab closed
        return response

    async def handle_arm(request: web.Request) -> web.Response:
        return await _proxy(request, "/arm")

    async def handle_arm_start(request: web.Request) -> web.Response:
        return await _proxy(request, "/arm/start", "POST")

    async def handle_arm_stop(request: web.Request) -> web.Response:
        return await _proxy(request, "/arm/stop", "POST")

    async def close_agents(a: web.Application) -> None:
        """Stop every agent on the way out, releasing every device."""
        a["agents"].stop_all()

    app.router.add_get("/api/agent/status", handle_agent_status)
    app.router.add_post("/api/agent/stop", handle_agent_stop)
    app.router.add_get("/api/agent/cameras", handle_cameras)
    app.router.add_post("/api/agent/cameras/start", handle_cameras_start)
    app.router.add_post("/api/agent/cameras/stop", handle_cameras_stop)
    app.router.add_get("/api/agent/cameras/{name}", handle_camera_stream)
    app.router.add_get("/api/agent/arm", handle_arm)
    app.router.add_post("/api/agent/arm/start", handle_arm_start)
    app.router.add_post("/api/agent/arm/stop", handle_arm_stop)
    app.on_cleanup.append(close_agents)
