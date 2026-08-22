"""Pack a chunk into msgpack metadata plus named blobs — no I/O.

A network transport reuses this class. Putting JPEG encoding or residual
copying in a medium is how DiskTransport ended up owning serialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import msgpack

from src.components.transport.payload import (
    SCHEDULE_KEY,
    ChunkPayload,
    dump_schedule,
    load_schedule,
)


@dataclass(frozen=True)
class SerializedBundle:
    """Bytes ready for any medium: one metadata blob, plus named sidecars."""

    metadata_bytes: bytes
    blobs: tuple[tuple[str, bytes], ...]

    def blob_map(self) -> dict[str, bytes]:
        return dict(self.blobs)


def _assert_blob_name(name: str) -> None:
    """Sidecar names are filenames, not paths. A slash would escape the chunk dir."""
    if not name or name in {".", ".."}:
        raise ValueError(f"Blob name {name!r} is not a usable filename.")
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(
            f"Blob name {name!r} is not a plain filename. Sidecars live next to "
            f"the metadata; a path component would write outside the chunk directory."
        )


class PayloadSerializer:
    """Msgpack the schedule-bearing metadata; keep sidecar bytes as-is."""

    def dumps(self, payload: ChunkPayload) -> SerializedBundle:
        if not payload.chunk_id:
            raise ValueError("ChunkPayload.chunk_id must be a non-empty string.")
        names = sorted(payload.blobs)
        for name in names:
            _assert_blob_name(name)
        document: dict[str, Any] = {
            "chunk_id": payload.chunk_id,
            SCHEDULE_KEY: dump_schedule(payload.schedule),
            "blob_names": names,
            "extra": dict(payload.extra),
        }
        metadata_bytes: bytes = msgpack.packb(document, use_bin_type=True)
        blobs = tuple((name, bytes(payload.blobs[name])) for name in names)
        return SerializedBundle(metadata_bytes=metadata_bytes, blobs=blobs)

    def loads(self, bundle: SerializedBundle) -> ChunkPayload:
        document = msgpack.unpackb(bundle.metadata_bytes, raw=False)
        if not isinstance(document, dict):
            raise ValueError("Transport metadata is not a mapping.")
        chunk_id = str(document.get("chunk_id", ""))
        if not chunk_id:
            raise ValueError("Transport metadata has no chunk_id.")
        raw_schedule = document.get(SCHEDULE_KEY)
        if not isinstance(raw_schedule, Mapping):
            raise ValueError(
                f"Transport metadata is missing {SCHEDULE_KEY!r}. The temporal "
                f"decision has to travel in the payload; reconstructing it from "
                f"config on the decoder would drift from the encoder."
            )
        named = [str(name) for name in document.get("blob_names", ())]
        available = bundle.blob_map()
        missing = [name for name in named if name not in available]
        if missing:
            raise FileNotFoundError(
                f"Serialized bundle for chunk {chunk_id!r} is missing sidecar(s): "
                + ", ".join(missing)
            )
        extra = document.get("extra") or {}
        if not isinstance(extra, Mapping):
            raise ValueError("Transport metadata extra is not a mapping.")
        return ChunkPayload(
            chunk_id=chunk_id,
            schedule=load_schedule(raw_schedule),
            blobs={name: available[name] for name in named},
            extra=dict(extra),
        )
