import os
import cv2
import numpy as np
import argparse
from natsort import natsorted
from typing import List

# --- Configuration Constants ---
# Factor to downsample frames for faster motion analysis.
MOTION_DOWNSAMPLE_FACTOR = 0.25
# Threshold to classify a chunk as STATIC. Lower is stricter.
MOTION_CLASSIFIER_THRESHOLD = 0.5
# The number of frames to analyze at a time. This is our "chunk" size.
MOTION_ANALYSIS_CHUNK_SIZE = 150
# Sample every N frames within a chunk for motion analysis (speeds up processing)
MOTION_ANALYSIS_SAMPLE_RATE = 5

# --- Core Motion Classifier (Unchanged) ---
def _classify_camera_motion(scene_frames: List[np.ndarray]) -> str:
    """Analyzes a list of frames to classify the global camera motion."""
    if len(scene_frames) < 5:
        return "SIMPLE"

    scale = MOTION_DOWNSAMPLE_FACTOR
    try:
        prev_gray = cv2.cvtColor(cv2.resize(scene_frames[0], (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)        
    except cv2.error:
        return "COMPLEX" # Likely a bad frame

    magnitudes = []
    for i in range(1, len(scene_frames)):
        try:
            curr_gray = cv2.cvtColor(cv2.resize(scene_frames[i], (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(np.mean(magnitude))
            prev_gray = curr_gray
        except cv2.error:
            continue

    avg_motion = np.mean(magnitudes) if magnitudes else 0

    if avg_motion < MOTION_CLASSIFIER_THRESHOLD * 5:
        return "SIMPLE"
    else:
        return "COMPLEX"

# --- SIMPLIFIED Video Processing Function ---
def extract_video_frames(video_path: str, output_path: str, quality: int, max_frames: int):
    """
    Scans a video in chunks, finds static ones, and extracts a sample of frames.
    """
    video_basename = os.path.basename(video_path)
    base_name = os.path.splitext(video_basename)[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file '{video_path}'.")
        return 0

    # Step 1: Scan video in chunks and collect indices of simple motion frames
    print(f"\n--- 📹 Step 1/2: Scanning '{video_basename}' for simple motion chunks ---")
    simple_frame_indices = []
    frame_number = 0
    while True:
        chunk_frames = []
        chunk_start_frame = frame_number
        
        # Read only a subset of frames from the chunk for motion analysis
        frames_read_in_chunk = 0
        for i in range(MOTION_ANALYSIS_CHUNK_SIZE):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only keep every Nth frame for motion analysis to speed up processing
            if i % MOTION_ANALYSIS_SAMPLE_RATE == 0:
                chunk_frames.append(frame)
            
            frames_read_in_chunk += 1
            frame_number += 1
        
        # If the chunk has frames, classify its motion using the sampled frames
        if chunk_frames:
            motion_type = _classify_camera_motion(chunk_frames)
            print(f"  - Chunk {chunk_start_frame}-{chunk_start_frame + frames_read_in_chunk - 1}: Motion is {motion_type} (analyzed {len(chunk_frames)}/{frames_read_in_chunk} frames)")
            
            # If the chunk has simple motion, add ALL frame numbers from this chunk to our candidate pool
            if motion_type == "SIMPLE":
                simple_frame_indices.extend(range(chunk_start_frame, chunk_start_frame + frames_read_in_chunk))
        
        # If we couldn't read any more frames, we're done scanning
        if frames_read_in_chunk == 0 or frames_read_in_chunk < MOTION_ANALYSIS_CHUNK_SIZE:
            break
    
    cap.release()

    if not simple_frame_indices:
        print("\nNo simple motion chunks were found. No frames will be extracted.")
        return 0

    print(f"\nFound {len(simple_frame_indices)} total frames in simple motion parts of the video.")

    # Step 2: Sample from the simple motion frames and save them
    print(f"\n--- 📹 Step 2/2: Sampling and saving final frames ---")
    
    if len(simple_frame_indices) <= max_frames:
        final_indices_to_extract = simple_frame_indices
    else:
        step = len(simple_frame_indices) / max_frames
        sample_indices = [round(i * step) for i in range(max_frames)]
        final_indices_to_extract = [simple_frame_indices[i] for i in sample_indices]

    cap = cv2.VideoCapture(video_path) # Re-open for efficient frame grabbing
    frames_saved_count = 0
    for i, frame_index in enumerate(final_indices_to_extract):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            output_frame_number = i + 1
            new_filename = f"{base_name}_{output_frame_number:05d}.jpg"
            destination_path = os.path.join(output_path, new_filename)
            cv2.imwrite(destination_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frames_saved_count += 1
    
    cap.release()
    print(f"✅ Saved {frames_saved_count} frames from simple motion chunks.")
    return frames_saved_count

# --- Directory Processing Function (Unchanged) ---
def process_directory(input_folder: str, output_path: str, quality: int):
    # This function for processing folders of images remains the same.
    print(f"\n--- 📁 Processing directory: {input_folder} ---")
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    total_frames_processed = 0
    for subfolder_name in sorted(os.listdir(input_folder)):
        subfolder_path = os.path.join(input_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            print(f"\n--- Processing subfolder: {subfolder_name} ---")
            all_files = os.listdir(subfolder_path)
            image_files = natsorted([f for f in all_files if f.lower().endswith(supported_extensions)])
            if not image_files: continue
            for i, filename in enumerate(image_files):
                frame_number = i + 1
                source_frame_path = os.path.join(subfolder_path, filename)
                image = cv2.imread(source_frame_path)
                if image is not None:
                    new_filename = f"{subfolder_name}_{frame_number:05d}.jpg"
                    destination_path = os.path.join(output_path, new_filename)
                    cv2.imwrite(destination_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                else:
                    print(f"⚠️ Warning: Could not read file '{source_frame_path}'.")
            print(f"✅ Saved {len(image_files)} frames from '{subfolder_name}'.")
            total_frames_processed += len(image_files)
    return total_frames_processed

# --- Main Entry Point ---
def main():
    """Main function to parse arguments and decide the processing mode."""
    parser = argparse.ArgumentParser(
        description="Extracts frames from videos (from static chunks) or reprocesses frames from image folders.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_path', type=str, help="Path to the source folder OR a single video file.")
    parser.add_argument('output_path', type=str, help="Path to the flat output folder where frames will be saved.")
    parser.add_argument('-q', '--quality', type=int, default=90, choices=range(0, 101), metavar="[0-100]", help="JPEG compression quality (0-100). Default: 90.")
    parser.add_argument('--max_frames', type=int, default=1000, help="Max frames to extract from a video's static parts. Default: 1000.")
    # The scenedetect threshold is no longer needed
    # parser.add_argument('--scenedetect_threshold', type=float, default=27.0, help="Threshold for PySceneDetect's ContentDetector. Default: 27.0.")

    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"❌ Error: Input path '{args.input_path}' not found.")
        return

    os.makedirs(args.output_path, exist_ok=True)
    
    total_frames = 0
    if os.path.isdir(args.input_path):
        total_frames = process_directory(args.input_path, args.output_path, args.quality)
    elif os.path.isfile(args.input_path):
        # The call is now simpler, without the scenedetect argument
        total_frames = extract_video_frames(args.input_path, args.output_path, args.quality, args.max_frames)
    else:
        print(f"❌ Error: Input path '{args.input_path}' is not a valid file or directory.")
        return

    print(f"\n🎉 All done! Total frames processed: {total_frames}.")

if __name__ == '__main__':
    main()