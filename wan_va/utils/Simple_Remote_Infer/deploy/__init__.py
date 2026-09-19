"""Simple_Remote_Infer deployment components package.

Provides everything needed for websocket-based remote inference deployment:
- ``websocket_policy_server``: msgpack-serialized websocket server that wraps a policy
  model and serves inference to remote clients;
- ``websocket_client_policy``: the matching synchronous client (simulation eval and
  real-robot deployment use the same interface);
- ``msgpack_numpy``: msgpack serialization/deserialization support for NumPy arrays;
- ``image_tools``: uint8 image conversion and resize+pad preprocessing utilities;
- ``qwenpi_policy`` / ``replay_policy``: deployment policy wrappers (QwenPI0 VLA
  inference / dataset action replay).
"""
