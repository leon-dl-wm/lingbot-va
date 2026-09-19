"""Websocket inference server module.

Wraps a policy model (e.g. the wan_va_server inference wrapper, or QwenPiServer from
qwenpi_policy) into a remote inference service over the websocket protocol: after
connecting, a client (simulation eval under evaluation/*, or real-robot deployment; see
websocket_client_policy.py) sends msgpack-serialized observation dicts, and the server
calls ``policy.infer(obs)`` and returns the action dict. Also exposes a ``/healthz``
HTTP health-check endpoint so scripts can probe readiness (e.g. waiting for the service
in script/run_launch_va_server_sync.sh).
"""
import asyncio
import http
import logging
import time
import traceback

import websockets.asyncio.server as _server
import websockets.frames

from .msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.

    Notes (added): websocket-based policy inference server (client implementation in
    websocket_client_policy.py). Protocol: right after the connection opens the server
    sends one metadata message, then enters a "receive observation -> policy.infer ->
    send action" loop; observations/actions are serialized with the msgpack+numpy
    extension. Currently only the `load` and `infer` method semantics are implemented.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Initialize the server.

        Args:
            policy: the wrapped policy object; must implement ``infer(obs: dict) -> dict``
                (optionally ``reset()``; a ``reset=True`` flag inside the observation is
                handled inside policy.infer).
            host (str): listen address, default 0.0.0.0 (all NICs).
            port (int | None): listen port (matches cfg.port in the config, e.g. 29536).
            metadata (dict | None): service metadata sent to the client immediately after
                the connection opens (e.g. model version / config summary); the client
                reads it via get_server_metadata().
        """
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """Blocking synchronous entry point: drives the async run() via asyncio.run."""
        asyncio.run(self.run())

    async def run(self):
        """Start the async websocket service and run forever.

        Key parameters: compression=None (disable compression; images are already uint8
        so gains are low), max_size=None (no message size limit; multi-camera
        observations can be large), ping_interval/timeout=None (disable heartbeats so
        long inference calls are not mistaken for dead connections),
        process_request=_health_check (intercept /healthz health-check requests).
        """
        async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
                ping_timeout=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        """Per-connection handler coroutine: send metadata first, then loop "recv obs -> infer -> send action".

        Args:
            websocket (_server.ServerConnection): the established websocket connection.

        Behavior:
        - the first message after connect is the server metadata (msgpack-serialized);
        - each received observation message triggers ``policy.infer(obs)``, and the
          returned action dict is augmented with ``server_timing`` (server-side infer_ms
          and previous round's prev_total_ms);
        - when the client disconnects (ConnectionClosed) the loop exits normally;
        - on server-side exceptions the traceback is sent back as a string frame (the
          client's infer raises when it receives a str), then the connection is closed
          with INTERNAL_ERROR and the exception is re-raised.
        """
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        # Handshake: send the service metadata to the client right after connect
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                # Receive one msgpack message and deserialize it into the observation dict (image/state/prompt, etc.)
                obs = unpackb(await websocket.recv())

                # Call the wrapped policy's inference interface (blocking; may take seconds)
                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # Attach server-side timing info to the response so clients can analyze latency bottlenecks
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"][
                        "prev_total_ms"] = prev_total_time * 1000

                # Serialize the action dict and send it back; then record this round's total time (including send)
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                # Client disconnected: end this connection's handler loop normally
                logger.info(
                    f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # Server error: send the traceback back as a string frame (client raises RuntimeError on it), then close
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason=
                    "Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection,
                  request: _server.Request) -> _server.Response | None:
    """HTTP health-check hook (process_request): GET /healthz returns 200 "OK" directly.

    Args:
        connection: current connection object, used to build a plain HTTP response.
        request: the incoming HTTP request.

    Returns:
        Response | None: an HTTP 200 response when the path is /healthz (skipping the
        websocket upgrade); None for any other path so the normal websocket handshake
        continues.
    """
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
