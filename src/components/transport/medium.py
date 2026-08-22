"""Where serialized bytes go. Media do not pack, unpack, or interpret them."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.components.transport.serialize import SerializedBundle


class TransportMedium(Protocol):
    """Move an already-serialized bundle. Serialization lives elsewhere."""

    def put(self, chunk_id: str, bundle: SerializedBundle) -> None:
        """Store ``bundle`` under ``chunk_id``."""

    def get(self, chunk_id: str) -> SerializedBundle:
        """Return the bundle previously ``put`` under ``chunk_id``.

        Raises:
            FileNotFoundError: If nothing was stored under that id.
        """


class DiskMedium:
    """Write metadata.msgpack and one file per sidecar under ``root/chunk_<id>/``."""

    def __init__(self, root: str | Path = ".pointstream") -> None:
        self._root = Path(root)

    def chunk_dir(self, chunk_id: str) -> Path:
        return self._root / f"chunk_{chunk_id}"

    def put(self, chunk_id: str, bundle: SerializedBundle) -> None:
        if not chunk_id:
            raise ValueError("chunk_id must be a non-empty string.")
        self._root.mkdir(parents=True, exist_ok=True)
        directory = self.chunk_dir(chunk_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.msgpack").write_bytes(bundle.metadata_bytes)
        blobs_dir = directory / "blobs"
        if bundle.blobs:
            blobs_dir.mkdir(parents=True, exist_ok=True)
        for name, data in bundle.blobs:
            (blobs_dir / name).write_bytes(data)

    def get(self, chunk_id: str) -> SerializedBundle:
        directory = self.chunk_dir(chunk_id)
        metadata_path = directory / "metadata.msgpack"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"No payload found for chunk {chunk_id!r} in {directory}."
            )
        blobs_dir = directory / "blobs"
        blobs: list[tuple[str, bytes]] = []
        if blobs_dir.is_dir():
            for path in sorted(blobs_dir.iterdir()):
                if path.is_file():
                    blobs.append((path.name, path.read_bytes()))
        return SerializedBundle(
            metadata_bytes=metadata_path.read_bytes(),
            blobs=tuple(blobs),
        )


class MemoryMedium:
    """In-process store. Exists so a second medium can reuse the serializer."""

    def __init__(self) -> None:
        self._store: dict[str, SerializedBundle] = {}

    def put(self, chunk_id: str, bundle: SerializedBundle) -> None:
        if not chunk_id:
            raise ValueError("chunk_id must be a non-empty string.")
        self._store[chunk_id] = bundle

    def get(self, chunk_id: str) -> SerializedBundle:
        try:
            return self._store[chunk_id]
        except KeyError:
            raise FileNotFoundError(
                f"No payload found for chunk {chunk_id!r} in memory."
            ) from None
