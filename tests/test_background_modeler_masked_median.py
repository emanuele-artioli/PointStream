from __future__ import annotations

import numpy as np
import torch

from src.encoder.background_modeler import BackgroundModeler
from src.shared.schemas import FrameState, SceneActor, VideoChunk


def test_background_modeler_excludes_actor_pixels_with_masked_nan_median() -> None:
    height, width = 64, 96
    background = np.full((height, width, 3), 100, dtype=np.uint8)

    frames: list[np.ndarray] = []
    frame_states: list[FrameState] = []
    actor_boxes = [
        (8, 18, 28, 46),
        (36, 18, 56, 46),
        (64, 18, 84, 46),
    ]
    for frame_idx in range(3):
        frame = background.copy()
        x1, y1, x2, y2 = actor_boxes[frame_idx]
        frame[y1:y2, x1:x2] = np.array([255, 255, 255], dtype=np.uint8)
        frames.append(frame)

        frame_states.append(
            FrameState(
                frame_id=frame_idx,
                actors=[
                    SceneActor(
                        track_id="player_0",
                        class_name="player",
                        bbox=[x1, y1, x2, y2],
                        mask=np.ones((y2 - y1, x2 - x1), dtype=np.uint8).tolist(),
                    )
                ],
            )
        )

    decoded_video_tensor = (
        torch.from_numpy(np.stack(frames, axis=0))
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(torch.uint8)
        .to(torch.float32)
        / 255.0
    )

    modeler = BackgroundModeler()
    chunk = VideoChunk(
        chunk_id="bg_masked_median",
        source_uri="memory://unused",
        start_frame_id=0,
        fps=24.0,
        num_frames=3,
        width=width,
        height=height,
    )

    panorama = modeler.process(
        chunk=chunk,
        decoded_video_tensor=decoded_video_tensor,
        frame_states=frame_states,
        translation_threshold_px=0.0,
    )

    pano_np = np.asarray(panorama.panorama_image, dtype=np.uint8)
    actor_union = pano_np[20:44, 8:84]
    mean_val = float(np.mean(actor_union))
    assert mean_val > 70.0
    assert mean_val < 140.0


def test_background_modeler_does_not_mask_full_bbox_when_actor_mask_missing() -> None:
    modeler = BackgroundModeler()
    frame_states = [
        FrameState(
            frame_id=0,
            actors=[
                SceneActor(
                    track_id="player_0",
                    class_name="player",
                    bbox=[20, 12, 60, 40],
                    mask=None,
                )
            ],
        )
    ]

    exclusion = modeler._build_actor_exclusion_mask(
        frame_idx=0,
        frame_states=frame_states,
        frame_height=64,
        frame_width=96,
    )

    assert int(np.count_nonzero(exclusion)) == 0
