import os
import cv2
import argparse
from natsort import natsorted

def extract_video_frames(video_path: str, output_path: str, quality: int, max_frames: int):
    """
    Extracts up to max_frames, evenly spaced, from a single video file.
    """
    video_basename = os.path.basename(video_path)
    print(f"\n--- 📹 Extracting frames from video: {video_basename} ---")
    
    base_name = os.path.splitext(video_basename)[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file '{video_path}'.")
        return 0

    # 1. Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"⚠️ Warning: Video '{video_basename}' has no frames or duration. Skipping.")
        cap.release()
        return 0

    print(f"Video has {total_frames} total frames. Will extract up to {max_frames} frames.")

    # 2. Determine which frame indices to extract
    if total_frames <= max_frames:
        # If the video is shorter than the limit, extract every frame
        indices_to_extract = range(total_frames)
    else:
        # If the video is longer, calculate evenly spaced indices
        step = total_frames / max_frames
        indices_to_extract = [round(i * step) for i in range(max_frames)]

    # 3. Extract only the chosen frames by seeking
    frames_saved_count = 0
    for i, frame_index in enumerate(indices_to_extract):
        # Jump the capture to the specific frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ Warning: Could not read frame at index {frame_index}. Skipping.")
            continue

        # The saved filename uses a 1-based counter for the extracted sequence
        output_frame_number = i + 1
        new_filename = f"{base_name}_{output_frame_number:05d}.jpg"
        destination_path = os.path.join(output_path, new_filename)
        
        cv2.imwrite(destination_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frames_saved_count += 1
        
        if frames_saved_count > 0 and frames_saved_count % 100 == 0:
            print(f"Saved {frames_saved_count}/{len(indices_to_extract)} frames...")

    cap.release()
    print(f"✅ Saved {frames_saved_count} frames from '{video_basename}'.")
    return frames_saved_count

def process_directory(input_folder: str, output_path: str, quality: int):
    """
    Processes image frames from subdirectories. (This function is unchanged).
    """
    print(f"\n--- 📁 Processing directory: {input_folder} ---")
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    total_frames_processed = 0

    for subfolder_name in sorted(os.listdir(input_folder)):
        subfolder_path = os.path.join(input_folder, subfolder_name)

        if os.path.isdir(subfolder_path):
            print(f"\n--- Processing subfolder: {subfolder_name} ---")

            all_files = os.listdir(subfolder_path)
            image_files = natsorted([f for f in all_files if f.lower().endswith(supported_extensions)])

            if not image_files:
                print("No supported image frames found. Skipping.")
                continue

            for i, filename in enumerate(image_files):
                frame_number = i + 1
                source_frame_path = os.path.join(subfolder_path, filename)
                image = cv2.imread(source_frame_path)

                if image is not None:
                    new_filename = f"{subfolder_name}_{frame_number:05d}.jpg"
                    destination_path = os.path.join(output_path, new_filename)
                    cv2.imwrite(destination_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                else:
                    print(f"⚠️ Warning: Could not read file '{source_frame_path}'. Skipping.")

            print(f"✅ Saved {len(image_files)} frames from '{subfolder_name}'.")
            total_frames_processed += len(image_files)
            
    return total_frames_processed

def main():
    """
    Main function to parse arguments and decide the processing mode.
    """
    parser = argparse.ArgumentParser(
        description="Extracts frames from a video or reprocesses frames from subfolders into a single flat directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input_path',
        type=str,
        help="Path to the source folder OR a single video file."
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Path to the flat output folder where frames will be saved."
    )
    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=80,
        choices=range(0, 101),
        metavar="[0-100]",
        help="JPEG compression quality (0-100).\nDefault is 80."
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=1000,
        help="Maximum number of frames to extract from a single video.\nFrames will be spaced evenly. Default is 1000."
    )

    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"❌ Error: Input path '{args.input_path}' not found.")
        return

    os.makedirs(args.output_path, exist_ok=True)
    
    total_frames = 0
    if os.path.isdir(args.input_path):
        total_frames = process_directory(args.input_path, args.output_path, args.quality)
    elif os.path.isfile(args.input_path):
        total_frames = extract_video_frames(args.input_path, args.output_path, args.quality, args.max_frames)
    else:
        print(f"❌ Error: Input path '{args.input_path}' is not a valid file or directory.")
        return

    print(f"\n🎉 All done! Total frames processed: {total_frames}.")

if __name__ == '__main__':
    main()