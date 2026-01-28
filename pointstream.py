#!/usr/bin/env python3
"""
PointStream Pipeline
Runs A1 (SAM segmentation) and A2 (pose extraction with skeleton generation) 
on all videos in a dataset directory.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import shutil


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True


def process_video(video_path, output_base_dir, model_path, pose_model_path):
    """
    Process a single video through the full pipeline.
    
    Args:
        video_path: Path to the input video file
        output_base_dir: Base directory for output (e.g., pointstream/experiments/dataset/video_name/scene_name)
        model_path: Path to SAM model
        pose_model_path: Path to YOLO pose model
    
    Returns:
        True if successful, False otherwise
    """
    video_name = Path(video_path).stem
    print(f"\n{'#'*80}")
    print(f"# Processing video: {video_name}")
    print(f"{'#'*80}\n")
    
    # Create video-specific output directory
    video_output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Step 1: Run A1 - SAM Segmentation
    # This creates a timestamped experiment directory
    a1_cmd = [
        sys.executable,
        "/home/itec/emanuele/pointstream/A1_segment_with_sam.py",
        "--video_path", video_path,
        "--model_path", model_path
    ]
    
    if not run_command(a1_cmd, f"A1: SAM Segmentation for {video_name}"):
        return False
    
    # Find the most recent experiment directory (created by A1)
    experiments_dir = "/home/itec/emanuele/pointstream/experiments"
    experiment_dirs = [d for d in os.listdir(experiments_dir) 
                       if os.path.isdir(os.path.join(experiments_dir, d)) and d.endswith("_sam_seg")]
    if not experiment_dirs:
        print(f"❌ ERROR: No experiment directory found after A1")
        return False
    
    # Get the most recently created directory
    experiment_dirs.sort()
    latest_experiment = os.path.join(experiments_dir, experiment_dirs[-1])
    print(f"\n📁 Using experiment directory: {latest_experiment}")
    
    # Step 2: Run A2 - Extract Poses
    a2_cmd = [
        sys.executable,
        "/home/itec/emanuele/pointstream/A2_extract_poses_from_crops.py",
        "--experiment_dir", latest_experiment,
        "--pose_model", pose_model_path
    ]
    
    if not run_command(a2_cmd, f"A2: Pose Extraction for {video_name}"):
        return False
    
    # Step 3: Organize outputs into the desired structure
    # Move masked_crops and skeletons to video-specific directory
    crops_src = os.path.join(latest_experiment, "masked_crops")
    poses_src = os.path.join(latest_experiment, "skeletons")
    
    crops_dst = os.path.join(video_output_dir, "crops")
    poses_dst = os.path.join(video_output_dir, "poses")
    
    if os.path.exists(crops_src):
        if os.path.exists(crops_dst):
            shutil.rmtree(crops_dst)
        shutil.move(crops_src, crops_dst)
        print(f"\n📂 Moved crops to: {crops_dst}")
    
    if os.path.exists(poses_src):
        if os.path.exists(poses_dst):
            shutil.rmtree(poses_dst)
        shutil.move(poses_src, poses_dst)
        print(f"\n📂 Moved poses to: {poses_dst}")
    
    # Optionally, clean up the timestamped experiment directory or keep metadata
    # For now, we'll keep it for reference
    print(f"\n✓ Video {video_name} processed successfully!")
    print(f"   Crops: {crops_dst}")
    print(f"   Poses: {poses_dst}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run PointStream pipeline on all videos in a dataset")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="/home/itec/emanuele/Datasets/medvedev_struff",
        help="Directory containing video files to process"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medvedev_struff",
        help="Name of the dataset (used for output directory structure)"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="/home/itec/emanuele/pointstream/experiments/dataset",
        help="Base directory for organized outputs"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/itec/emanuele/models/sam3.pt",
        help="Path to SAM model"
    )
    parser.add_argument(
        "--pose_model",
        type=str,
        default="/home/itec/emanuele/models/yolo11l-pose.pt",
        help="Path to YOLO pose model"
    )
    parser.add_argument(
        "--video_pattern",
        type=str,
        default="*.mp4",
        help="Pattern to match video files (default: *.mp4)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos that already have output directories"
    )
    
    args = parser.parse_args()
    
    # Find all video files in the dataset directory
    dataset_path = Path(args.dataset_dir)
    video_files = sorted(dataset_path.glob(args.video_pattern))
    
    if not video_files:
        print(f"❌ No video files found matching '{args.video_pattern}' in {args.dataset_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"PointStream Pipeline")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Videos found: {len(video_files)}")
    print(f"Output base: {args.output_base}")
    print(f"{'='*80}\n")
    
    # Create output base directory
    output_base_dir = os.path.join(args.output_base, args.dataset_name)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each video
    results = {}
    for i, video_file in enumerate(video_files, 1):
        video_name = video_file.stem
        
        # Check if we should skip this video
        if args.skip_existing:
            video_output_dir = os.path.join(output_base_dir, video_name)
            crops_dir = os.path.join(video_output_dir, "crops")
            poses_dir = os.path.join(video_output_dir, "poses")
            
            if os.path.exists(crops_dir) and os.path.exists(poses_dir):
                print(f"\n⏭️  Skipping {video_name} ({i}/{len(video_files)}) - already processed")
                results[video_name] = "skipped"
                continue
        
        print(f"\n{'='*80}")
        print(f"Processing video {i}/{len(video_files)}: {video_name}")
        print(f"{'='*80}")
        
        success = process_video(
            str(video_file),
            output_base_dir,
            args.model_path,
            args.pose_model
        )
        
        results[video_name] = "success" if success else "failed"
        
        if not success:
            print(f"\n⚠️  Video {video_name} failed. Continuing with next video...")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for v in results.values() if v == "success")
    failed = sum(1 for v in results.values() if v == "failed")
    skipped = sum(1 for v in results.values() if v == "skipped")
    
    print(f"Total videos: {len(video_files)}")
    print(f"✓ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped: {skipped}")
    
    if failed > 0:
        print(f"\nFailed videos:")
        for video_name, status in results.items():
            if status == "failed":
                print(f"  - {video_name}")
    
    print(f"\nOutputs saved to: {output_base_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
