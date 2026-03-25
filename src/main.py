from __future__ import annotations

import json

from src.encoder.execution_pool import BaseExecutionPool
from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.orchestrator import EncoderPipeline
from src.shared.schemas import VideoChunk
from src.transport.disk import DiskTransport


def run_mock_pipeline(
    transport_root: str = ".pointstream",
    execution_pool: BaseExecutionPool | None = None,
) -> dict[str, object]:
    chunk = VideoChunk(
        chunk_id="0001",
        source_uri="assets/test_chunks/tennis_chunk_0001.mp4",
        start_frame_id=0,
        fps=30.0,
        num_frames=60,
        width=1280,
        height=720,
    )

    encoder = EncoderPipeline(execution_pool=execution_pool)
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


if __name__ == "__main__":
    print(json.dumps(run_mock_pipeline(), indent=2))
