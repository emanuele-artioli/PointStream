from __future__ import annotations

import json

from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.orchestrator import EncoderPipeline
from src.shared.schemas import VideoChunk
from src.transport.disk import DiskTransport


def run_mock_pipeline() -> dict[str, object]:
    chunk = VideoChunk(
        chunk_id="0001",
        source_uri="assets/test_chunks/tennis_chunk_0001.mp4",
        start_frame_id=0,
        fps=30.0,
        num_frames=60,
        width=1280,
        height=720,
    )

    encoder = EncoderPipeline()
    payload = encoder.encode_chunk(chunk)

    transport = DiskTransport(root_dir=".pointstream")
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
