from __future__ import annotations

from pathlib import Path
import shutil

import cv2
import msgpack
import numpy as np

from src.shared.interfaces import BaseTransport
from src.shared.schemas import EncodedChunkPayload, SceneActorReference
from src.shared.tags import cpu_bound
from src.transport.panorama_encoder import BasePanoramaEncoder, build_panorama_encoder


class DiskTransport(BaseTransport):
    def __init__(
        self,
        root_dir: str | Path = ".pointstream",
        panorama_encoder: str | BasePanoramaEncoder | None = None,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._panorama_encoder = build_panorama_encoder(panorama_encoder)

    def _chunk_dir(self, chunk_id: str) -> Path:
        return self._root_dir / f"chunk_{chunk_id}"

    @cpu_bound
    def send(self, payload: EncodedChunkPayload) -> None:
        chunk_dir = self._chunk_dir(payload.chunk.chunk_id)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = chunk_dir / "metadata.msgpack"
        residual_path = chunk_dir / "residual.mp4"
        panorama_path = self._materialize_panorama(payload=payload, chunk_dir=chunk_dir)

        source_residual = Path(payload.residual.residual_video_uri)
        if not source_residual.exists() or not source_residual.is_file():
            raise FileNotFoundError(
                f"Residual stream not found for chunk '{payload.chunk.chunk_id}': {source_residual}"
            )
        if source_residual.resolve() != residual_path.resolve():
            shutil.copy2(source_residual, residual_path)

        materialized_references = self._materialize_actor_references(payload=payload, chunk_dir=chunk_dir)

        payload_for_disk = payload.model_copy(
            update={
                "panorama": payload.panorama.model_copy(
                    update={
                        "panorama_uri": str(panorama_path),
                        "panorama_image": None,
                    }
                ),
                "residual": payload.residual.model_copy(
                    update={
                        "residual_video_uri": str(residual_path),
                    }
                ),
                "actor_references": materialized_references,
            }
        )

        metadata_bytes = msgpack.packb(payload_for_disk.model_dump(mode="python"), use_bin_type=True)
        metadata_path.write_bytes(metadata_bytes)

    def _materialize_panorama(self, payload: EncodedChunkPayload, chunk_dir: Path) -> Path:
        panorama_np = self._resolve_panorama_image(payload=payload)
        return self._panorama_encoder.encode(
            image_bgr=panorama_np,
            output_stem=chunk_dir / "panorama",
        )

    def _materialize_actor_references(self, payload: EncodedChunkPayload, chunk_dir: Path) -> list[SceneActorReference]:
        references_dir = chunk_dir / "actor_references"
        references_dir.mkdir(parents=True, exist_ok=True)

        output_references: list[SceneActorReference] = []
        for reference in payload.actor_references:
            jpeg_bytes = self._resolve_reference_bytes(reference=reference)
            if jpeg_bytes is None:
                continue

            reference_path = references_dir / f"track_{int(reference.track_id):04d}.jpg"
            reference_path.write_bytes(jpeg_bytes)
            output_references.append(
                SceneActorReference(
                    track_id=int(reference.track_id),
                    reference_crop_jpeg=None,
                    reference_crop_uri=str(reference_path),
                )
            )

        return output_references

    def _resolve_reference_bytes(self, reference: SceneActorReference) -> bytes | None:
        if reference.reference_crop_jpeg:
            return bytes(reference.reference_crop_jpeg)

        uri = reference.reference_crop_uri
        if uri is None:
            return None

        reference_path = Path(str(uri))
        if not reference_path.exists() or not reference_path.is_file():
            return None
        return reference_path.read_bytes()

    def _resolve_panorama_image(self, payload: EncodedChunkPayload) -> np.ndarray:
        panorama_pixels = payload.panorama.panorama_image
        if panorama_pixels is not None:
            panorama_np = np.asarray(panorama_pixels, dtype=np.uint8)
            if panorama_np.ndim != 3 or panorama_np.shape[2] != 3:
                raise ValueError(
                    "Invalid panorama image shape in payload: "
                    f"expected [H, W, 3], got {tuple(panorama_np.shape)}"
                )
            return panorama_np

        source_path = Path(str(payload.panorama.panorama_uri))
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(
                "Panorama image is missing: neither panorama_image is populated nor panorama_uri points to a valid file. "
                f"Checked: {source_path}"
            )

        decoded = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise ValueError(f"Failed to decode panorama image from {source_path}")
        return np.asarray(decoded, dtype=np.uint8)

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
