from __future__ import annotations

from pathlib import Path
import shutil
from time import perf_counter
import logging

import cv2
import msgpack
import numpy as np

from src.shared.interfaces import BaseTransport
from src.shared.schemas import EncodedChunkPayload, ResidualMode, SceneActorReference
from src.shared.tags import cpu_bound
from src.transport.panorama_encoder import BasePanoramaEncoder, build_panorama_encoder

_logger = logging.getLogger(__name__)


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
        send_started = perf_counter()
        chunk_dir = self._chunk_dir(payload.chunk.chunk_id)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = chunk_dir / "metadata.msgpack"
        residual_path = chunk_dir / "residual.mp4"
        
        panorama_started = perf_counter()
        panorama_path = self._materialize_panorama(payload=payload, chunk_dir=chunk_dir)
        panorama_elapsed = perf_counter() - panorama_started

        source_residual_uri = str(payload.residual.residual_video_uri or "").strip()
        residual_mode = payload.residual.mode
        if isinstance(residual_mode, str):
            residual_mode = ResidualMode(residual_mode)

        materialized_residual_uri = ""
        residual_copy_started = perf_counter()
        if residual_mode != ResidualMode.NONE and source_residual_uri:
            source_residual = Path(source_residual_uri)
            if not source_residual.exists() or not source_residual.is_file():
                raise FileNotFoundError(
                    f"Residual stream not found for chunk '{payload.chunk.chunk_id}': {source_residual}"
                )
            if source_residual.resolve() != residual_path.resolve():
                shutil.copy2(source_residual, residual_path)
            materialized_residual_uri = str(residual_path)
        residual_copy_elapsed = perf_counter() - residual_copy_started

        references_started = perf_counter()
        materialized_references = self._materialize_actor_references(payload=payload, chunk_dir=chunk_dir)
        references_elapsed = perf_counter() - references_started

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
                        "residual_video_uri": materialized_residual_uri,
                    }
                ),
                "actor_references": materialized_references,
            }
        )

        msgpack_started = perf_counter()
        metadata_bytes = msgpack.packb(payload_for_disk.model_dump(mode="python"), use_bin_type=True)
        metadata_path.write_bytes(metadata_bytes)
        msgpack_elapsed = perf_counter() - msgpack_started

        send_elapsed = perf_counter() - send_started
        _logger.debug(
            f"DiskTransport.send() breakdown: "
            f"panorama={panorama_elapsed:.3f}s, "
            f"residual_copy={residual_copy_elapsed:.3f}s, "
            f"references={references_elapsed:.3f}s, "
            f"msgpack={msgpack_elapsed:.3f}s, "
            f"total={send_elapsed:.3f}s"
        )

    def _materialize_panorama(self, payload: EncodedChunkPayload, chunk_dir: Path) -> Path:
        """Encode and write panorama to disk.
        
        Note: Panorama JPEG encoding is the primary bottleneck in transport latency (~97% of send time).
        If transport latency is critical, consider:
        - Reducing panorama image resolution
        - Decreasing JPEG quality (controlled by --panorama-jpeg-quality, default 50)
        - Using --panorama-codec png and increasing compression for extreme latency reduction
        """
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
            # Some payloads intentionally keep panorama in memory-only URIs.
            # Persist a deterministic blank canvas so transport remains robust.
            frame_height = int(getattr(payload.panorama, "frame_height", payload.chunk.height))
            frame_width = int(getattr(payload.panorama, "frame_width", payload.chunk.width))
            frame_height = max(1, frame_height)
            frame_width = max(1, frame_width)
            return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        decoded = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise ValueError(f"Failed to decode panorama image from {source_path}")
        return np.asarray(decoded, dtype=np.uint8)

    @cpu_bound
    def receive(self, chunk_id: str) -> EncodedChunkPayload:
        receive_started = perf_counter()
        chunk_dir = self._chunk_dir(chunk_id)
        metadata_path = chunk_dir / "metadata.msgpack"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No payload found for chunk '{chunk_id}' in '{chunk_dir}'."
            )

        read_started = perf_counter()
        metadata_raw = metadata_path.read_bytes()
        read_elapsed = perf_counter() - read_started

        unpack_started = perf_counter()
        metadata_dict = msgpack.unpackb(metadata_raw, raw=False)
        payload = EncodedChunkPayload.model_validate(metadata_dict)
        unpack_elapsed = perf_counter() - unpack_started

        receive_elapsed = perf_counter() - receive_started
        _logger.debug(
            f"DiskTransport.receive() breakdown: "
            f"read={read_elapsed:.3f}s, "
            f"unpack={unpack_elapsed:.3f}s, "
            f"total={receive_elapsed:.3f}s"
        )

        return payload
