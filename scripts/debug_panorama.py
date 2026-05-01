#!/usr/bin/env python3
"""Diagnostic script to inspect keyframe selection and pose data in real runs."""

from pathlib import Path
import argparse
import json
import msgpack
import numpy as np
import cv2

from src.encoder.background_modeler import BackgroundModeler
from src.encoder.mock_extractors import ActorExtractor
from src.encoder.video_io import probe_video_metadata, decode_video_to_tensor
from src.shared.schemas import VideoChunk


def main():
    parser = argparse.ArgumentParser(description="Debug panorama generation")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--num-frames", type=int, default=24, help="Number of frames to process")
    parser.add_argument("--translation-threshold-px", type=float, default=30.0, help="Translation threshold")
    parser.add_argument("--output-dir", type=str, default="./outputs/debug_panorama", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = probe_video_metadata(input_path)
    chunk = VideoChunk(
        chunk_id="debug_0001",
        source_uri=str(input_path),
        start_frame_id=0,
        fps=metadata.fps,
        num_frames=min(args.num_frames, metadata.num_frames),
        width=metadata.width,
        height=metadata.height,
    )

    print(f"Processing {chunk.num_frames} frames from {input_path}")
    print(f"Input resolution: {chunk.width}x{chunk.height}")

    # Extract actors with pose
    actor_extractor = ActorExtractor(
        detector_backend="yoloe",
        detector_caption="tennis player",
        pose_backend="yolo",
        segmenter_backend="yoloe",
        segmenter_caption="tennis player",
    )
    extraction_result = actor_extractor.process_with_states(chunk)
    frame_states = extraction_result.frame_states

    print(f"\nExtracted {len(frame_states)} frame states")
    for frame_idx, state in enumerate(frame_states):
        num_actors = len(state.actors)
        has_pose = sum(1 for a in state.actors if a.pose_dw is not None)
        has_mask = sum(1 for a in state.actors if a.mask is not None)
        print(f"  Frame {frame_idx}: {num_actors} actors | pose: {has_pose} | mask: {has_mask}")

    # Run background modeler
    decoded_video = decode_video_to_tensor(input_path)
    decoded_tensor = decoded_video.tensor[:chunk.num_frames]

    modeler = BackgroundModeler()
    panorama = modeler.process(
        chunk=chunk,
        decoded_video_tensor=decoded_tensor,
        frame_states=frame_states,
        translation_threshold_px=args.translation_threshold_px,
    )

    print(f"\nPanorama generation:")
    print(f"  Selected keyframes: {panorama.selected_frame_indices}")
    print(f"  Output canvas size: {panorama.frame_width}x{panorama.frame_height}")
    print(f"  Input video size:   {chunk.width}x{chunk.height}")
    if panorama.panorama_image is not None:
        print(f"  Image shape: {np.asarray(panorama.panorama_image, dtype=np.uint8).shape}")

    # Save debug info
    debug_info = {
        "num_frames": chunk.num_frames,
        "input_size": [chunk.width, chunk.height],
        "canvas_size": [panorama.frame_width, panorama.frame_height],
        "selected_keyframes": panorama.selected_frame_indices,
        "num_keyframes": len(panorama.selected_frame_indices),
        "frame_state_summary": [
            {
                "frame_idx": idx,
                "num_actors": len(state.actors),
                "actors_with_pose": sum(1 for a in state.actors if a.pose_dw is not None),
                "actors_with_mask": sum(1 for a in state.actors if a.mask is not None),
                "actors": [
                    {
                        "track_id": a.track_id,
                        "class": a.class_name,
                        "has_pose": a.pose_dw is not None,
                        "has_mask": a.mask is not None,
                        "bbox": a.bbox,
                    }
                    for a in state.actors
                ]
            }
            for idx, state in enumerate(frame_states)
        ]
    }

    debug_file = output_dir / "debug_info.json"
    debug_file.write_text(json.dumps(debug_info, indent=2))
    print(f"\nDebug info saved to {debug_file}")

    # Save panorama image
    if panorama.panorama_image is not None:
        pano_np = np.asarray(panorama.panorama_image, dtype=np.uint8)
        pano_path = output_dir / "panorama_debug.jpg"
        cv2.imwrite(str(pano_path), pano_np)
        print(f"Panorama image saved to {pano_path}")


if __name__ == "__main__":
    main()
