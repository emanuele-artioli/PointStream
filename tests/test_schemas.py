from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.shared.schemas import (
    InterpolateCommandEvent,
    KeyframeEvent,
    ObjectClass,
    StaticCommandEvent,
    VideoChunk,
)


class TestSchemas(unittest.TestCase):
    def test_keyframe_event_requires_object_id(self) -> None:
        with self.assertRaises(ValidationError):
            KeyframeEvent(
                frame_id=0,
                object_class=ObjectClass.PERSON,
                coordinates=[0.1, 0.2],
            )

    def test_video_chunk_valid_contract(self) -> None:
        chunk = VideoChunk(
            chunk_id="chunk_a",
            source_uri="assets/test_chunks/a.mp4",
            start_frame_id=0,
            fps=30.0,
            num_frames=16,
            width=640,
            height=360,
        )
        self.assertEqual(chunk.chunk_id, "chunk_a")
        self.assertEqual(chunk.num_frames, 16)

    def test_video_chunk_rejects_invalid_fps(self) -> None:
        with self.assertRaises(ValidationError):
            VideoChunk(
                chunk_id="chunk_bad",
                source_uri="assets/test_chunks/a.mp4",
                start_frame_id=0,
                fps=0.0,
                num_frames=16,
                width=640,
                height=360,
            )

    def test_interpolate_event_rejects_invalid_method(self) -> None:
        with self.assertRaises(ValidationError):
            InterpolateCommandEvent(
                frame_id=1,
                object_id="person_0",
                object_class=ObjectClass.PERSON,
                target_frame_id=3,
                method="cubic",
            )

    def test_static_event_rejects_negative_hold_frame(self) -> None:
        with self.assertRaises(ValidationError):
            StaticCommandEvent(
                frame_id=1,
                object_id="person_1",
                object_class=ObjectClass.PERSON,
                hold_until_frame_id=-1,
            )


if __name__ == "__main__":
    unittest.main()
