"""Adds NumPy array support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.

Notes (added): this module adds NumPy array support to msgpack and is the serialization
layer used by this repo's websocket remote inference (server/client) to transmit
observations and actions. Compared with pickle, msgpack is secure (no arbitrary code
execution), cross-language, schema-free, and about 4x faster for large arrays.
Core idea: when packing, an ndarray becomes a dict ``{__ndarray__, data (raw bytes),
dtype, shape}``; when unpacking, an object_hook recognizes the marker and rebuilds the
array from a zero-copy buffer.
"""

import functools

import msgpack
import numpy as np


def pack_array(obj):
    """msgpack serialization hook (default): convert NumPy arrays/scalars into packable dicts.

    Args:
        obj: object to serialize. Supports np.ndarray (arbitrary dimensions) and
            np.generic (NumPy scalars); other types are returned unchanged and handled
            by msgpack's default logic.

    Returns:
        dict | object: ndarray -> ``{b"__ndarray__", b"data" (raw tobytes), b"dtype"
        (e.g. '<f4'), b"shape"}``; scalar -> ``{b"__npgeneric__", b"data", b"dtype"}``;
        non-NumPy objects are returned as-is.

    Raises:
        ValueError: unsupported dtype (structured 'V', object 'O', complex 'c') —
            object arrays may hide arbitrary Python objects, so they are rejected to
            avoid falling back to pickle and its security risks.
    """
    if (isinstance(
            obj,
        (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    """msgpack deserialization hook (object_hook): rebuild NumPy objects from pack_array's dicts.

    Args:
        obj: dict produced by deserialization (msgpack calls this hook for every map).

    Returns:
        np.ndarray | np.generic | dict: if the ``b"__ndarray__"`` marker is present,
        the array is rebuilt zero-copy from data/dtype/shape; if ``b"__npgeneric__"``
        is present, a NumPy scalar is rebuilt; otherwise the dict is returned unchanged.
    """
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"],
                          dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


# Packer/unpacker factories and convenience functions pre-wired with the NumPy hooks:
# Packer/packb handle ndarray via default=pack_array when serializing;
# Unpacker/unpackb restore ndarray via object_hook=unpack_array when deserializing.
# websocket_policy_server / websocket_client_policy import these four symbols directly.
Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
