from __future__ import annotations

from pathlib import Path
import shutil

import msgpack

from src.shared.interfaces import BaseTransport
from src.shared.schemas import EncodedChunkPayload
from src.shared.tags import cpu_bound


class DiskTransport(BaseTransport):
    def __init__(self, root_dir: str | Path = ".pointstream") -> None:
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def _chunk_dir(self, chunk_id: str) -> Path:
        return self._root_dir / f"chunk_{chunk_id}"

    @cpu_bound
    def send(self, payload: EncodedChunkPayload) -> None:
        chunk_dir = self._chunk_dir(payload.chunk.chunk_id)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = chunk_dir / "metadata.msgpack"
        residual_path = chunk_dir / "residual.mp4"

        source_residual = Path(payload.residual.residual_video_uri)
        if not source_residual.exists() or not source_residual.is_file():
            raise FileNotFoundError(
                f"Residual stream not found for chunk '{payload.chunk.chunk_id}': {source_residual}"
            )
        if source_residual.resolve() != residual_path.resolve():
            shutil.copy2(source_residual, residual_path)

        payload_for_disk = payload.model_copy(
            update={
                "residual": payload.residual.model_copy(
                    update={
                        "residual_video_uri": str(residual_path),
                    }
                )
            }
        )

        metadata_bytes = msgpack.packb(payload_for_disk.model_dump(mode="python"), use_bin_type=True)
        metadata_path.write_bytes(metadata_bytes)

    @cpu_bound
    def receive(self, chunk_id: str) -> EncodedChunkPayload:
        chunk_dir = self._chunk_dir(chunk_id)
        metadata_path = chunk_dir / "metadata.msgpack"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No payload found for chunk '{chunk_id}' in '{chunk_dir}'."
            )

        metadata_raw = msgpack.unpackb(metadata_path.read_bytes(), raw=False)
        return EncodedChunkPayload.model_validate(metadata_raw)
