import argparse
from utils import extract_frames, detect_scene_changes
import csv

def main():
    
    # Parse command line arguments
    argparser = argparse.ArgumentParser(description="PointStream Server")
    argparser.add_argument(
        "--input_video",
        type=str,
        help="Path to the input video file."
    )
    args = argparser.parse_args()

    # Parse video frames in batches, find scene changes using SSIM
    video_path = args.input_video
    batch_size = 100
    frame_range = (0, batch_size - 1)

    # Initialize CSV file with header
    with open("scene_changes.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['global_frame_index', 'ssim_score'])

    while True:
        frames = extract_frames(video_path, frame_range)
        print(f"Extracted {len(frames)} frames from {frame_range[0]} to {frame_range[1]}.")
        if not frames:
            break

        scene_changes = detect_scene_changes(frames, threshold=0.2, analysis_window=25)
        if scene_changes:
            print(f"Scene changes detected at frames: {sorted(scene_changes)}")
            with open("scene_changes.csv", 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                for frame_index, ssim_score in scene_changes.items():
                    csv_writer.writerow([frame_index + frame_range[0], ssim_score])

        frame_range = (frame_range[0] + batch_size, frame_range[1] + batch_size)

if __name__ == "__main__":
    main()