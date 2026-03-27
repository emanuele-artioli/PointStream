from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.encoder.execution_pool import BaseExecutionPool
from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.mock_extractors import ActorExtractor
from src.encoder.orchestrator import EncoderPipeline
from src.encoder.video_io import probe_video_metadata
from src.shared.schemas import VideoChunk
from src.transport.disk import DiskTransport


def run_mock_pipeline(
    transport_root: str = ".pointstream",
    execution_pool: BaseExecutionPool | None = None,
    source_uri: str | None = None,
    num_frames: int | None = None,
    actor_extractor: ActorExtractor | None = None,
) -> dict[str, object]:
    resolved_source_uri = source_uri or _ensure_mock_source_video()
    source_metadata = probe_video_metadata(resolved_source_uri)
    chunk_frames = source_metadata.num_frames if num_frames is None else min(source_metadata.num_frames, num_frames)

    chunk = VideoChunk(
        chunk_id="0001",
        source_uri=resolved_source_uri,
        start_frame_id=0,
        fps=source_metadata.fps,
        num_frames=chunk_frames,
        width=source_metadata.width,
        height=source_metadata.height,
    )

    encoder = EncoderPipeline(execution_pool=execution_pool, actor_extractor=actor_extractor)
    try:
        payload = encoder.encode_chunk(chunk)
    finally:
        encoder.shutdown()

    transport = DiskTransport(root_dir=transport_root)
    transport.send(payload)
    received_payload = transport.receive(chunk.chunk_id)

    decoder = DecoderRenderer()
    decoded = decoder.process(received_payload)

    summary = {
        "chunk_id": received_payload.chunk.chunk_id,
        "num_actor_packets": len(received_payload.actors),
        "num_rigid_object_packets": len(received_payload.rigid_objects),
        "ball_object_id": received_payload.ball.object_id,
        "residual_uri": received_payload.residual.residual_video_uri,
        "decoded_uri": decoded.output_uri,
    }
    return summary


def _ensure_mock_source_video() -> str:
    project_root = Path(__file__).resolve().parents[1]
    assets_dir = project_root / "assets" / "test_chunks"
    assets_dir.mkdir(parents=True, exist_ok=True)
    source_path = assets_dir / "tennis_chunk_0001.mp4"

    if source_path.exists() and source_path.is_file():
        return str(source_path)

    writer = cv2.VideoWriter(
        str(source_path),
        getattr(cv2, "VideoWriter_fourcc")(*"mp4v"),
        30.0,
        (1280, 720),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create mock source video at: {source_path}")

    for frame_idx in range(60):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        x0 = 100 + (frame_idx * 8) % 900
        y0 = 200 + (frame_idx * 3) % 250
        cv2.rectangle(frame, (x0, y0), (x0 + 90, y0 + 180), (0, 220, 60), thickness=-1)
        cv2.rectangle(frame, (1280 - x0 - 120, 720 - y0 - 220), (1280 - x0 - 40, 720 - y0 - 40), (30, 200, 230), thickness=-1)
        writer.write(frame)

    writer.release()
    return str(source_path)


if __name__ == "__main__":
    print(json.dumps(run_mock_pipeline(), indent=2))
