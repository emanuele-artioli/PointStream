"""Disk transport: the registered medium, composed with the shared serializer."""

from __future__ import annotations

from pathlib import Path

from src.components.transport.medium import DiskMedium, TransportMedium
from src.components.transport.payload import ChunkPayload
from src.components.transport.serialize import PayloadSerializer, SerializedBundle


class DiskTransport:
    """Serialize a chunk, then write it to a directory.

    ``serializer`` is injected so a network transport can share it. The default
    medium is disk; tests swap in ``MemoryMedium`` without touching packing.
    """

    def __init__(
        self,
        root: str | Path = ".pointstream",
        *,
        serializer: PayloadSerializer | None = None,
        medium: TransportMedium | None = None,
    ) -> None:
        self.serializer = serializer or PayloadSerializer()
        self.medium: TransportMedium = medium if medium is not None else DiskMedium(root)

    def send(self, payload: ChunkPayload) -> None:
        bundle = self.serializer.dumps(payload)
        self.medium.put(payload.chunk_id, bundle)

    def receive(self, chunk_id: str) -> ChunkPayload:
        bundle = self.medium.get(chunk_id)
        return self.serializer.loads(bundle)

    def put_bundle(self, chunk_id: str, bundle: SerializedBundle) -> None:
        """Store an already-serialized bundle. Used when the caller packed once."""
        self.medium.put(chunk_id, bundle)

    def get_bundle(self, chunk_id: str) -> SerializedBundle:
        return self.medium.get(chunk_id)
