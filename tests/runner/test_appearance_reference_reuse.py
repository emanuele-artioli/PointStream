from __future__ import annotations

import io
import json

import numpy as np

from src.contracts.config import BackendConfig, LatticeConfig, PointstreamConfig
from src.pipeline.reconstruction.reconstruct import ObjectRequest
from src.runner.run import run


def test_track_appearance_is_serialized_once_and_reused_by_frame_placements() -> None:
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    objects = []
    for frame_index, (y0, x0, color) in enumerate(((4, 5, 80), (6, 7, 120))):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[y0 : y0 + 10, x0 : x0 + 10] = 1
        frames[frame_index, y0 : y0 + 10, x0 : x0 + 10] = color
        objects.append(
            ObjectRequest(
                object_id="player-1",
                appearance=np.full((10, 10, 3), color, dtype=np.uint8),
                bbox=(x0, y0, x0 + 10, y0 + 10),
                mask=mask,
                frame_index=frame_index,
                object_class="person",
            )
        )
    config = PointstreamConfig(
        segmenter=BackendConfig(backend="none"),
        lattice=LatticeConfig(
            scene_classification=False,
            detection=True,
            selection=False,
            tracking=False,
            appearance=True,
            motion=False,
            temporal_policy=False,
            pose=False,
            segmentation=True,
            rigid_objects=False,
            background=False,
            generation=False,
            residual=False,
        )
    )
    result = run(
        config,
        (frames,),
        objects=(tuple(objects),),
        heartbeat_interval=None,
        sync_fn=None,
    )
    payload = result.chunks[0].bag["wire_request"]
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
        assert list(metadata["references"]) == ["player-1"]
        assert len(metadata["placements"]) == 2
        assert all(item["crop_key"] is None for item in metadata["placements"])
        assert all(item["encoded_crop_key"] is None for item in metadata["placements"])
        assert metadata["references"]["player-1"]["byte_count"] > 0
