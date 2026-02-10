#!/usr/bin/env python3
"""
PointStream Pipeline - Unified Tennis Video-to-Animation Pipeline

Complete pipeline for processing tennis match videos:
  A1: SAM segmentation - Segment players from video frames
  A2: Pose extraction - Extract YOLO pose skeletons from crops
  A3: Dataset preparation - Convert to fs_vid2vid training format
  A4: LMDB build - Create efficient training database
  A5: Training - Train fs_vid2vid pose transfer model
  A6: Inference - Generate new animations from reference + skeleton

Usage:
  # Full extraction pipeline (A1 + A2)
  python pointstream.py --mode extract --dataset_dir /path/to/videos

  # Prepare for training (A3 + A4)
  python pointstream.py --mode prepare

  # Train model (A5)
  python pointstream.py --mode train

  # Run inference (A6)
  python pointstream.py --mode inference --checkpoint /path/to/checkpoint.pt
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import shutil


POINTSTREAM_DIR = Path("/home/itec/emanuele/pointstream")
IMAGINAIRE_DIR = Path("/home/itec/emanuele/imaginaire")
EXPERIMENTS_DIR = POINTSTREAM_DIR / "experiments"
DATASET_DIR = EXPERIMENTS_DIR / "dataset"


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


def run_extraction(args):
    """Run extraction pipeline (A1 + A2) on videos."""
    # Find all video files in the dataset directory
    dataset_path = Path(args.dataset_dir)
    video_files = sorted(dataset_path.glob(args.video_pattern))
    
    if not video_files:
        print(f"❌ No video files found matching '{args.video_pattern}' in {args.dataset_dir}")
        return False
    
    print(f"\n{'='*80}")
    print(f"PointStream Extraction Pipeline (A1 + A2)")
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
    successful = sum(1 for v in results.values() if v == "success")
    failed = sum(1 for v in results.values() if v == "failed")
    skipped = sum(1 for v in results.values() if v == "skipped")
    
    print(f"\n{'='*80}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos: {len(video_files)}")
    print(f"✓ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"\nOutputs saved to: {output_base_dir}")
    print(f"{'='*80}\n")
    
    return failed == 0


def run_prepare(args):
    """Run dataset preparation pipeline (A3 + A4)."""
    print(f"\n{'='*80}")
    print(f"PointStream Dataset Preparation (A3 + A4)")
    print(f"{'='*80}\n")
    
    # Step A3: Prepare dataset
    a3_cmd = [
        sys.executable,
        str(POINTSTREAM_DIR / "A3_prepare_vid2vid_dataset.py"),
        "--dataset_dir", str(DATASET_DIR),
        "--output_dir", str(IMAGINAIRE_DIR / "datasets" / "tennis_pose"),
        "--image_size", args.image_size,
        "--val_ratio", str(args.val_ratio),
    ]
    
    if not run_command(a3_cmd, "A3: Prepare vid2vid Dataset"):
        return False
    
    # Step A4: Build LMDB
    a4_cmd = [
        sys.executable,
        str(POINTSTREAM_DIR / "A4_build_lmdb.py"),
        "--data_root", str(IMAGINAIRE_DIR / "datasets" / "tennis_pose"),
    ]
    
    if not run_command(a4_cmd, "A4: Build LMDB Database"):
        return False
    
    print(f"\n{'='*80}")
    print(f"Dataset preparation complete!")
    print(f"Ready for training with: python pointstream.py --mode train")
    print(f"{'='*80}\n")
    
    return True


def run_train(args):
    """Run training pipeline (A5)."""
    print(f"\n{'='*80}")
    print(f"PointStream Training (A5)")
    print(f"{'='*80}\n")
    
    a5_cmd = [
        sys.executable,
        str(POINTSTREAM_DIR / "A5_train_vid2vid.py"),
        "--config", args.config,
        "--num_gpus", str(args.num_gpus),
    ]
    
    if args.checkpoint:
        a5_cmd.extend(["--checkpoint", args.checkpoint])
    
    if args.wandb:
        a5_cmd.extend(["--wandb", "--wandb_name", args.wandb_name])
    
    if args.single_gpu:
        a5_cmd.append("--single_gpu")
    
    if not run_command(a5_cmd, "A5: Train fs_vid2vid Model"):
        return False
    
    return True


def run_inference(args):
    """Run inference pipeline (A6)."""
    print(f"\n{'='*80}")
    print(f"PointStream Inference (A6)")
    print(f"{'='*80}\n")
    
    if not args.checkpoint:
        print("❌ ERROR: --checkpoint is required for inference")
        return False
    
    a6_cmd = [
        sys.executable,
        str(POINTSTREAM_DIR / "A6_inference_vid2vid.py"),
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--output_dir", args.output_dir,
    ]
    
    if args.reference_dir:
        a6_cmd.extend(["--reference_dir", args.reference_dir])
    
    if args.driving_dir:
        a6_cmd.extend(["--driving_dir", args.driving_dir])
    
    if not run_command(a6_cmd, "A6: fs_vid2vid Inference"):
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="PointStream - Unified Tennis Video-to-Animation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract poses from tennis videos
  python pointstream.py --mode extract --dataset_dir /path/to/videos

  # Prepare dataset for training
  python pointstream.py --mode prepare

  # Train the model
  python pointstream.py --mode train

  # Run inference
  python pointstream.py --mode inference --checkpoint /path/to/checkpoint.pt
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["extract", "prepare", "train", "inference", "full"],
        default="extract",
        help="Pipeline mode: extract, prepare, train, inference, or full (all steps)"
    )
    
    # Extraction arguments (A1 + A2)
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="/home/itec/emanuele/Datasets/djokovic_federer",
        help="Directory containing video files to process"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="djokovic_federer",
        help="Name of the dataset (used for output directory structure)"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=str(DATASET_DIR),
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
    
    # Preparation arguments (A3 + A4)
    parser.add_argument(
        "--image_size",
        type=str,
        default="512,256",
        help="Output image size as 'width,height'"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    
    # Training arguments (A5)
    parser.add_argument(
        "--config",
        type=str,
        default=str(IMAGINAIRE_DIR / "configs" / "projects" / "fs_vid2vid" / "tennis_pose" / "ampO1.yaml"),
        help="Path to training config"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Use single GPU mode"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint for resume/inference"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="tennis_vid2vid",
        help="W&B project name"
    )
    
    # Inference arguments (A6)
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="",
        help="Directory containing reference image and pose"
    )
    parser.add_argument(
        "--driving_dir",
        type=str,
        default="",
        help="Directory containing driving skeleton sequence"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for generated frames"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print(f"#  PointStream - Tennis Video-to-Animation Pipeline")
    print(f"#  Mode: {args.mode.upper()}")
    print(f"{'#'*80}\n")
    
    success = True
    
    if args.mode == "extract":
        success = run_extraction(args)
    elif args.mode == "prepare":
        success = run_prepare(args)
    elif args.mode == "train":
        success = run_train(args)
    elif args.mode == "inference":
        success = run_inference(args)
    elif args.mode == "full":
        # Run full pipeline
        if not run_extraction(args):
            print("❌ Extraction failed!")
            sys.exit(1)
        if not run_prepare(args):
            print("❌ Preparation failed!")
            sys.exit(1)
        if not run_train(args):
            print("❌ Training failed!")
            sys.exit(1)
        success = True
    
    if success:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
