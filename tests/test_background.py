from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.decoder.mock_renderer import DecoderRenderer
from src.transport.disk import DiskTransport


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_real_video_background_stitching_and_transport(real_encoder_pipeline, real_tennis_10f_video: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]

    transport_root = project_root / ".pointstream"
    chunk_id = "background_real_0001"
    chunk_dir = transport_root / f"chunk_{chunk_id}"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)

    debug_panorama_path = project_root / "assets" / "debug_panorama.jpg"
    if debug_panorama_path.exists():
        debug_panorama_path.unlink()

    payload, _decoded_video_tensor = real_encoder_pipeline.encode_video_file(
        video_path=real_tennis_10f_video,
        chunk_id=chunk_id,
        max_frames=10,
    )

    panorama = payload.panorama
    assert len(panorama.homography_matrices) == payload.chunk.num_frames
    assert payload.chunk.num_frames == 10
    assert panorama.selected_frame_indices
    assert panorama.selected_frame_indices[0] == 0
    assert len(panorama.selected_frame_indices) <= payload.chunk.num_frames
    assert f"debug_panorama_{chunk_id}.jpg" in panorama.panorama_uri

    homography_0 = np.asarray(panorama.homography_matrices[0], dtype=np.float64)
    assert homography_0.shape == (3, 3)

    panorama_array = np.asarray(panorama.panorama_image, dtype=np.uint8)
    assert panorama_array.shape[0] >= payload.chunk.height
    assert panorama_array.shape[1] >= payload.chunk.width
    assert panorama_array.shape[2] == 3

    assert panorama.frame_height == panorama_array.shape[0]
    assert panorama.frame_width == panorama_array.shape[1]

    chunk_debug_path = project_root / "assets" / f"debug_panorama_{chunk_id}.jpg"
    assert chunk_debug_path.exists()
    debug_img = cv2.imread(str(chunk_debug_path))
    assert debug_img is not None
    assert int(np.max(debug_img)) > 0

    transport = DiskTransport(root_dir=transport_root)
    transport.send(payload)
    recovered = transport.receive(chunk_id)
    decoded = DecoderRenderer().process(recovered)

    assert recovered.chunk.chunk_id == chunk_id
    assert len(recovered.panorama.homography_matrices) == recovered.chunk.num_frames
    assert (chunk_dir / "metadata.msgpack").exists()
    assert (chunk_dir / "residual.mp4").exists()

    assert decoded.chunk_id == chunk_id
    assert decoded.width == recovered.chunk.width
    assert decoded.height == recovered.chunk.height
