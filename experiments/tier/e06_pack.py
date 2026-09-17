"""Lossless compact packing of saved E06 client envelopes.

Native appearance/residual/background bitstreams are stored, not re-encoded.
Envelope, placement and mask arrays are deflated. Wire T is the compact file
length; logical native payload sizes are reported separately and are not
subtracted to invent an overhead remainder.
"""

from __future__ import annotations

import io
import json
from typing import Any
import zipfile

import numpy as np

NATIVE_PREFIXES = (
    "residual_bitstream",
    "background_payload_",
    "encoded_crop_",
    "ref_",
)


def _is_native(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix) for prefix in NATIVE_PREFIXES)


def pack_lossless_compact(payload: bytes) -> bytes:
    """Rewrite an npz so native codecs stay stored and auxiliary arrays deflate."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
    background = metadata.get("background")
    if isinstance(background, dict) and background.get("geometry_header"):
        background = dict(background)
        background["geometry_header"] = ""
        metadata["background"] = background
        arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, array in arrays.items():
            buf = io.BytesIO()
            np.save(buf, np.ascontiguousarray(array), allow_pickle=False)
            compress = zipfile.ZIP_STORED if _is_native(name) else zipfile.ZIP_DEFLATED
            archive.writestr(f"{name}.npy", buf.getvalue(), compress_type=compress)
    return stream.getvalue()


def zip_inventory(payload: bytes) -> dict[str, Any]:
    entries = []
    stored_native = 0
    compressed_aux = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            stored = info.compress_type == zipfile.ZIP_STORED
            item = {
                "name": info.filename,
                "file_size": int(info.file_size),
                "compress_size": int(info.compress_size),
                "stored_native": stored and _is_native(info.filename.replace(".npy", "")),
            }
            entries.append(item)
            if item["stored_native"]:
                stored_native += int(info.compress_size)
            else:
                compressed_aux += int(info.compress_size)
    return {
        "transport_total": len(payload),
        "stored_native_bytes": stored_native,
        "compressed_auxiliary_bytes": compressed_aux,
        "zip_framing_bytes": max(0, len(payload) - stored_native - compressed_aux),
        "entries": entries,
        "note": (
            "T is the compact file length. Native codec payloads are stored as-is. "
            "Do not treat T minus logical B+F+R as envelope remainder."
        ),
    }
