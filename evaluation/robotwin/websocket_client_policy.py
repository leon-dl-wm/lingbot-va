"""Websocket policy client: the communication layer between evaluation scripts and the
remote inference server.

Role in the evaluation loop: evaluation clients (``evaluation/robotwin/eval_polict_client_openpi.py``
/ ``evaluation/libero/client.py``) use :class:`WebsocketClientPolicy` to send observations
(dict of numpy arrays: images / robot state) to the inference server ``wan_va/wan_va_server.py``
(the matching WebsocketPolicyServer); the server runs the "imagine future frames -> infer
actions" AR diffusion inference and returns an action chunk. Message bodies are serialized
with ``msgpack_numpy`` (efficient ndarray transport).

The asynchronous execution protocol is driven entirely by the dict keys of ``infer()``:
    infer(dict(reset=True, prompt=...))       # reset server: clear KV cache, encode the prompt
    infer(dict(obs=initial obs, prompt=...))  # first inference, returns an action chunk (2 latent frames x 16 substeps)
    infer(dict(obs=real obs sequence, compute_kv_cache=True, state=...))
                                              # after robot execution, feed real obs back to replace the imagined-frame cache

Typical usage::

    client = WebsocketClientPolicy(host="127.0.0.1", port=29056)
    ret = client.infer(dict(obs=obs_dict, prompt="stack the bowls"))
    actions = ret["action"]   # [C, F, N]: C=action channels, F=latent frames, N=substeps per frame
"""
import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client
from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.

    Notes: Synchronous websocket client. Construction blocks until the server is ready and
    receives the server metadata; ``infer`` is a synchronous "send one, receive one" call
    (sends msgpack bytes, blocks waiting for the response). The client itself is stateless:
    reset / observation-feedback semantics are all encoded in the request dict and
    interpreted by the server.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        """Initialize the client and establish the connection (blocks until the server is ready).

        Args:
            host: Server hostname/IP, combined into ``ws://{host}[:{port}]``.
            port: Server port; if None the URI carries no port (default 80 is used).
            api_key: Optional authentication key, sent via the ``Authorization: Api-Key ...`` header.
        """
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        """Return the first message pushed by the server upon connection setup — the server
        metadata (policy hyperparameters, etc.)."""
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

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """Wait for the server to come online and establish the connection, retrying every
        5 seconds on failure (waits indefinitely).

        Returns:
            (websocket connection object, server metadata dict).

        Notes:
            - Catches all exceptions, not just ConnectionRefusedError: while the server is
              loading the model it may refuse/drop connections in various ways, and the
              client should be startable before the server (multi-GPU batch script scenario);
            - ``ping_interval=None`` disables the websocket heartbeat: a single AR diffusion
              inference can take far longer than the default ping timeout, so without this
              the connection would be wrongly considered dead.
        """
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                # 禁用 ping 机制，防止推理时间过长导致超时
                conn = websockets.sync.client.connect(
                    self._uri, 
                    compression=None, 
                    max_size=None, 
                    additional_headers=headers,
                    ping_interval=None, 
                    close_timeout=10
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, Exception) as e:
                logging.info(f"Still waiting for server... (Error: {e})")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """Send one inference request and synchronously wait for the response.

        Args:
            obs: Request dict; its keys determine the server behavior (reset / obs /
                compute_kv_cache / prompt / state, etc.), and values may be numpy arrays
                (multi-camera images, joint states, ...), serialized via msgpack_numpy
                before sending.
        Returns:
            Server response dict, usually containing ``action`` (action chunk [C,F,N]);
            with save_visualization it also contains ``video`` (model-imagined future frames).
        Raises:
            RuntimeError: If the server returns a string — by protocol, a normal response is
                bytes, so a string is an error message.
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
        """No-op: session reset happens server-side via ``infer(dict(reset=True, ...))``;
        the client holds no local state to clear."""
        pass

# Connectivity self-test / example code below: builds random observations (3 camera images +
# joint state + prompt) and sends one infer request, to quickly verify the client-server
# link and message format (the inference server must be running first).
if __name__ == "__main__":
    policy_on_device = WebsocketClientPolicy(port=8000)
    import torch
    import numpy as np
    from PIL import Image
    from .image_tools import convert_to_uint8
    device = torch.device("cuda")

    base_0_rgb = np.random.randint(0, 256, size=(1, 3, 224, 224), dtype=np.uint8)
    left_wrist_0_rgb = np.random.randint(0, 256, size=(1, 3, 224, 224), dtype=np.uint8)
    state = np.random.rand(1,8).astype(np.float32)
    prompt = ["do something"]

    # observation = {
    #     "image": {
    #         "base_0_rgb": torch.from_numpy(base_0_rgb).to(device)[None],
    #         "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to(device)[None],
    #     },
    #     "state": torch.from_numpy(state).to(device)[None],
    #     "prompt": prompt,
    # }

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
    from IPython import embed;embed()
