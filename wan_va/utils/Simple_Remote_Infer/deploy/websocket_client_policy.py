"""Websocket inference client module.

Calls a remote WebsocketPolicyServer over a synchronous websocket connection: on
construction it blocks until the service is ready and receives the server metadata;
afterwards ``infer(obs)`` sends msgpack-serialized observations and receives actions.
Simulation eval (clients under evaluation/robotwin and evaluation/libero) and real-robot
deployment both use this class; its interface matches a local policy (obs dict in,
action dict out), so eval code does not need to know where the model runs.
"""
import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client
from typing_extensions import override

from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.

    Notes (added): Policy client communicating with the remote inference server over
    websocket (server implementation: WebsocketPolicyServer). Connects on construction
    and caches the metadata; infer() is a synchronous blocking call: send observation
    dict -> wait for server inference -> return the action dict (with server_timing).
    """

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: Optional[int] = None,
                 api_key: Optional[str] = None) -> None:
        """Initialize the client and block until the server is ready.

        Args:
            host (str): server host/IP (matches cfg.host).
            port (Optional[int]): server port (matches cfg.port, e.g. 29536); None uses the default port 80.
            api_key (Optional[str]): optional auth key, sent as an ``Authorization: Api-Key <key>`` header.
        """
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        # Connect to the server (retrying every 5 s on failure) and receive the metadata sent during the handshake
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        """Return the metadata dict sent by the server on connect (e.g. model version / config summary)."""
        return self._server_metadata

    # def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
    #     logging.info(f"Waiting for server at {self._uri}...")
    #     while True:
    #         try:
    #             headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
    #             conn = websockets.sync.client.connect(
    #                 self._uri, compression=None, max_size=None, additional_headers=headers
    #             )
    #             metadata = unpackb(conn.recv())
    #             return conn, metadata
    #         except ConnectionRefusedError:
    #             logging.info("Still waiting for server...")
    #             time.sleep(5)

    def _wait_for_server(
            self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """Block until the inference server is reachable, retrying every 5 seconds.

        Returns:
            Tuple[ClientConnection, Dict]: (established synchronous websocket connection,
            metadata dict sent by the server during the handshake).

        Notes: training/serving startup can be slow, so eval scripts may start the client
        before the server. Connection parameters mirror the server side — compression
        disabled, no message size limit, ping heartbeat disabled (so a single long
        inference call is not mistaken for a timeout).
        """
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {
                    "Authorization": f"Api-Key {self._api_key}"
                } if self._api_key else None
                # 禁用 ping 机制，防止推理时间过长导致超时
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    close_timeout=10)
                # The first message is the server metadata (msgpack-serialized)
                metadata = unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, Exception) as e:
                # Server not ready or network error: log the reason and retry after 5 s
                logging.info(f"Still waiting for server... (Error: {e})")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """Send one observation and synchronously wait for the inference result.

        Args:
            obs (Dict): observation dict, usually containing ``image`` (uint8 arrays per
                camera), ``state`` (robot state), ``prompt`` (list of task instructions);
                also used to pass control signals (e.g. ``dict(reset=True)``).

        Returns:
            Dict: action dict returned by the server (msgpack-deserialized), usually
            containing ``action`` (action chunk of shape [T, action_dim]) and
            ``server_timing`` info.

        Raises:
            RuntimeError: when the server hit an error it sends back a string frame
            (traceback), which is raised here.
        """
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    @override
    def reset(self) -> None:
        """Ask the server to reset inference state (clear KV cache, start a new episode).

        Implemented by sending the special observation ``dict(reset=True)``, handled
        inside the server-side policy.infer.
        """
        self.infer(dict(reset=True))


# Manual debug entry: connect to a local inference server on port 8000, build a random observation and run one infer
if __name__ == "__main__":
    policy_on_device = WebsocketClientPolicy(port=8000)
    import numpy as np
    import torch
    from PIL import Image

    from .image_tools import convert_to_uint8
    device = torch.device("cuda")

    # Build random image observations: uint8 arrays of shape (1, 3, 224, 224) (mock base and left-wrist cameras)
    base_0_rgb = np.random.randint(0,
                                   256,
                                   size=(1, 3, 224, 224),
                                   dtype=np.uint8)
    left_wrist_0_rgb = np.random.randint(0,
                                         256,
                                         size=(1, 3, 224, 224),
                                         dtype=np.uint8)
    # Robot state: float32 vector of shape (1, 8)
    state = np.random.rand(1, 8).astype(np.float32)
    # Task instruction (list form, supports batching)
    prompt = ["do something"]

    # observation = {
    #     "image": {
    #         "base_0_rgb": torch.from_numpy(base_0_rgb).to(device)[None],
    #         "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to(device)[None],
    #     },
    #     "state": torch.from_numpy(state).to(device)[None],
    #     "prompt": prompt,
    # }

    # Assemble the observation dict: image (three uint8 camera views) + state + prompt, then run one remote inference
    observation = {
        "image": {
            "base_0_rgb": convert_to_uint8(base_0_rgb),
            "left_wrist_0_rgb": convert_to_uint8(left_wrist_0_rgb),
            "right_wrist_0_rgb": convert_to_uint8(left_wrist_0_rgb),
        },
        "state": state,
        "prompt": prompt,
    }

    policy_on_device.infer(observation)
    # Drop into an IPython shell to inspect the result
    from IPython import embed
    embed()
