import argparse
import os
from collections import deque
from utils import (
    get_video_properties, 
    extract_frames,
    detect_scene_changes, 
    save_video_segment,
    classify_scene_motion
)

if __name__ == "__main__":
    # --- Argument Parsing ---
    argparser = argparse.ArgumentParser(description="PointStream Server - Scene-based Video Clipper")
    argparser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    args = argparser.parse_args()

    # --- Execution & Logic ---
    os.makedirs("scenes", exist_ok=True)
    print(f"Starting scene detection for '{args.input_video}'...")

    video_path = args.input_video
    threshold = 0.5

    total_frames, fps = get_video_properties(video_path)
    if not total_frames or not fps:
        print("Error: Could not retrieve video properties.")
    else:
        print(f"Video properties: {total_frames} frames, {fps:.2f} FPS.")

        batch_size = 150
        frame_buffer = deque()
        last_cut_frame = 0
        processed_frames_count = 0

        while processed_frames_count < total_frames:
            # 1. Read a large batch of frames for efficient processing
            frame_range = (processed_frames_count, processed_frames_count + batch_size - 1)
            batch_frames = extract_frames(video_path, frame_range)
            if not batch_frames:
                break
            
            frame_buffer.extend(batch_frames)
            print(f"Processing frames {frame_range[0]} to {frame_range[0] + len(batch_frames) - 1}...")

            # 2. Run efficient scene detection on the new batch
            scene_changes = detect_scene_changes(batch_frames, threshold=threshold, analysis_window=25)
            
            # Convert local batch indices to global frame indices
            global_cut_indices = sorted([idx + processed_frames_count for idx in scene_changes.keys()])
            
            for cut_frame in global_cut_indices:
                # 3. A segment is found. Get its frames from the start of the buffer.
                segment_len = cut_frame - last_cut_frame
                if segment_len > 0 and len(frame_buffer) >= segment_len:
                    
                    segment_frames = [frame_buffer.popleft() for _ in range(segment_len)]
                    
                    # 4. Classify the in-memory frames
                    scene_type = classify_scene_motion(segment_frames)
                    
                    # 5. Save the clip
                    output_path = f"scenes/{last_cut_frame}_{cut_frame}_{scene_type}.mp4"
                    save_video_segment(video_path, last_cut_frame, cut_frame, fps, output_path)
                    
                    last_cut_frame = cut_frame
            
            processed_frames_count += len(batch_frames)

        # After the loop, process any remaining frames in the buffer as the final scene
        if frame_buffer:
            final_frame_count = last_cut_frame + len(frame_buffer)
            scene_type = classify_scene_motion(list(frame_buffer))
            output_path = f"scenes/{last_cut_frame}_{final_frame_count}_{scene_type}.mp4"
            save_video_segment(video_path, last_cut_frame, final_frame_count, fps, output_path)
            frame_buffer.clear()

        print(f"Processing complete. Clips saved in 'scenes'.")