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

Additional notes: this module adds NumPy array (de)serialization support to msgpack. It is
the data-encoding layer between the evaluation client (``websocket_client_policy.py``) and
the inference server (``wan_va/wan_va_server.py``) for transmitting observation images,
robot states, and action arrays over websocket.

How it works:
- Serialization: an ndarray is packed into a ``{__ndarray__, data (raw memory bytes), dtype,
  shape}`` dict; NumPy scalars (np.generic) are packed into ``{__npgeneric__, data, dtype}``;
- Deserialization: dicts are recognized by the ``__ndarray__``/``__npgeneric__`` markers and
  rebuilt zero-copy from the byte buffer;
- Object/void/complex arrays (dtype kind "V"/"O"/"c") are unsupported and raise ValueError
  directly, avoiding the insecure and slow pickle fallback of the original msgpack-numpy.
"""

import functools

import msgpack
import numpy as np


def pack_array(obj):
    """msgpack serialization hook (used as the ``default`` callback): convert NumPy objects
    into msgpack-representable dicts.

    msgpack calls this function whenever it encounters a type it cannot serialize natively:
    - ``np.ndarray``: packed as ``{__ndarray__, data, dtype, shape}``, where data is the raw
      memory bytes from ``tobytes()`` (the array must be byte-serializable; object arrays are
      rejected earlier by the dtype check above);
    - ``np.generic`` (NumPy scalar): packed as ``{__npgeneric__, data, dtype}``;
    - Any other type: returned unchanged so msgpack handles it natively.

    Args:
        obj: Arbitrary object to serialize.
    Returns:
        The converted msgpack-compatible structure (dict), or the original object.
    Raises:
        ValueError: If the array dtype is void/object/complex ("V"/"O"/"c"), which is unsupported.
    """
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    # ndarray: transmit raw bytes + dtype/shape metadata so the receiver can rebuild it zero-copy
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
    """msgpack deserialization hook (used as ``object_hook``): restore dicts packed by
    :func:`pack_array` back into NumPy objects.

    - Contains the ``__ndarray__`` marker: rebuild the ndarray directly from the byte buffer
      using dtype/shape (zero-copy view);
    - Contains the ``__npgeneric__`` marker: restore the NumPy scalar of the corresponding dtype;
    - Plain dicts: returned unchanged.
    """
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


# Pre-bind the hooks to produce drop-in replacements for msgpack's native interfaces:
# Packer/packb route ndarrays through pack_array automatically; Unpacker/unpackb restore
# dicts through unpack_array automatically.
Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
