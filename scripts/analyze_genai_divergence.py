#!/usr/bin/env python3
"""Detailed frame-by-frame comparison of encoder vs decoder GenAI outputs."""

import sys
from pathlib import Path
import json
from collections import defaultdict

import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def load_image(path):
    """Load image safely."""
    if not Path(path).exists():
        return None
    return cv2.imread(str(path))

def compute_frame_pair_diff(debug_dir, stage1="encoder", stage2="decoder"):
    """Compare encoder and decoder frames and generate difference report."""
    stage1_dir = debug_dir / stage1
    stage2_dir = debug_dir / stage2
    
    report_lines = [
        "GenAI Compositor Divergence Analysis - Detailed Frame Comparison",
        "=" * 70,
        f"\nComparing: {stage1} vs {stage2}\n",
    ]
    
    # Collect frame directories
    stage1_frames = defaultdict(lambda: defaultdict(dict))
    stage2_frames = defaultdict(lambda: defaultdict(dict))
    
    if stage1_dir.exists():
        for frame_dir in sorted(stage1_dir.glob("frame_*")):
            frame_name = frame_dir.name
            for actor_dir in frame_dir.glob("actor_*"):
                actor_id = actor_dir.name
                stage1_frames[frame_name][actor_id] = actor_dir
    
    if stage2_dir.exists():
        for frame_dir in sorted(stage2_dir.glob("frame_*")):
            frame_name = frame_dir.name
            for actor_dir in frame_dir.glob("actor_*"):
                actor_id = actor_dir.name
                stage2_frames[frame_name][actor_id] = actor_dir
    
    report_lines.append(f"{stage1} frames with actors: {len(stage1_frames)}")
    report_lines.append(f"{stage2} frames with actors: {len(stage2_frames)}")
    report_lines.append("")
    
    # Compare matching frame/actor pairs
    all_frames = set(stage1_frames.keys()) | set(stage2_frames.keys())
    matched_pairs = 0
    divergence_pairs = 0
    total_diff_sum = 0.0
    total_diff_max = 0.0
    frame_diffs = []
    
    for frame_name in sorted(all_frames):
        stage1_actors = stage1_frames.get(frame_name, {})
        stage2_actors = stage2_frames.get(frame_name, {})
        common_actors = set(stage1_actors.keys()) & set(stage2_actors.keys())
        
        if not common_actors:
            continue
        
        for actor_id in sorted(common_actors):
            stage1_actor_dir = stage1_actors[actor_id]
            stage2_actor_dir = stage2_actors[actor_id]
            
            # Load composited frames
            stage1_comp = load_image(stage1_actor_dir / "05_composited_frame.png")
            stage2_comp = load_image(stage2_actor_dir / "05_composited_frame.png")
            
            if stage1_comp is None or stage2_comp is None:
                continue
            
            matched_pairs += 1
            
            # Compute frame-wise difference
            if stage1_comp.shape == stage2_comp.shape:
                diff = np.abs(stage1_comp.astype(np.float32) - stage2_comp.astype(np.float32))
                mean_diff = np.mean(diff)
                max_diff = np.max(diff)
                pct_nonzero = 100.0 * np.count_nonzero(diff > 1.0) / diff.size
                
                frame_diffs.append({
                    "frame": frame_name,
                    "actor": actor_id,
                    "mean_diff": float(mean_diff),
                    "max_diff": float(max_diff),
                    "pct_nonzero": float(pct_nonzero),
                })
                
                total_diff_sum += mean_diff
                total_diff_max = max(total_diff_max, max_diff)
                
                if mean_diff > 0.5:
                    divergence_pairs += 1
                    
                    # Compare intermediate outputs
                    stage1_ref = load_image(stage1_actor_dir / "00_reference_crop.png")
                    stage1_pose = load_image(stage1_actor_dir / "01_pose_condition.png")
                    stage1_bg = load_image(stage1_actor_dir / "02_warped_background.png")
                    
                    stage2_ref = load_image(stage2_actor_dir / "00_reference_crop.png")
                    stage2_pose = load_image(stage2_actor_dir / "01_pose_condition.png")
                    stage2_bg = load_image(stage2_actor_dir / "02_warped_background.png")
                    
                    details = {
                        "frame": frame_name,
                        "actor": actor_id,
                        "mean_diff": float(mean_diff),
                        "max_diff": float(max_diff),
                        "pct_nonzero": float(pct_nonzero),
                        "inputs_match": {
                            "reference_crop": _compare_images(stage1_ref, stage2_ref),
                            "pose_condition": _compare_images(stage1_pose, stage2_pose),
                            "warped_background": _compare_images(stage1_bg, stage2_bg),
                        }
                    }
                    
                    report_lines.append(f"{frame_name} actor {actor_id}:")
                    report_lines.append(f"  Output difference: mean={mean_diff:.2f}, max={max_diff:.2f}, nonzero={pct_nonzero:.1f}%")
                    report_lines.append(f"  Reference crop match: {details['inputs_match']['reference_crop']}")
                    report_lines.append(f"  Pose condition match: {details['inputs_match']['pose_condition']}")
                    report_lines.append(f"  Warped background match: {details['inputs_match']['warped_background']}")
                    report_lines.append("")
    
    report_lines.append("\nStatistics:")
    report_lines.append(f"  Matched pairs: {matched_pairs}")
    report_lines.append(f"  Divergent pairs (mean_diff > 0.5): {divergence_pairs}")
    if matched_pairs > 0:
        report_lines.append(f"  Average mean_diff: {total_diff_sum / matched_pairs:.2f}")
        report_lines.append(f"  Maximum max_diff: {total_diff_max:.2f}")
    
    return "\n".join(report_lines), frame_diffs

def _compare_images(img1, img2):
    """Check if two images are identical."""
    if img1 is None and img2 is None:
        return "both_missing"
    if img1 is None or img2 is None:
        return "one_missing"
    if img1.shape != img2.shape:
        return f"shape_mismatch ({img1.shape} vs {img2.shape})"
    if np.allclose(img1, img2):
        return "identical"
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    mean_diff = np.mean(diff)
    return f"differs (mean={mean_diff:.2f})"

def main():
    """Main entry point."""
    debug_dir = Path("/home/itec/emanuele/pointstream/outputs/genai_debug_comparison")
    
    if not debug_dir.exists():
        print(f"ERROR: Debug directory does not exist: {debug_dir}")
        return 1
    
    print("Analyzing encoder vs decoder divergence...")
    print(f"Debug dir: {debug_dir}\n")
    
    # Generate detailed comparison report
    report, frame_diffs = compute_frame_pair_diff(debug_dir)
    
    # Write report
    report_file = debug_dir / "DETAILED_DIVERGENCE_REPORT.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(report)
    print(f"\nDetailed report saved to: {report_file}")
    
    # Write JSON summary
    json_file = debug_dir / "frame_differences.json"
    with open(json_file, "w") as f:
        json.dump(frame_diffs, f, indent=2)
    
    print(f"Frame difference data saved to: {json_file}")
    
    # Analyze divergence patterns
    if frame_diffs:
        mean_diffs = [d["mean_diff"] for d in frame_diffs]
        print("\nDifference distribution:")
        print(f"  Min: {min(mean_diffs):.3f}")
        print(f"  Max: {max(mean_diffs):.3f}")
        print(f"  Mean: {np.mean(mean_diffs):.3f}")
        print(f"  Median: {np.median(mean_diffs):.3f}")
        print(f"  Std: {np.std(mean_diffs):.3f}")
        
        # Count frames with significant difference
        sig_diffs = [d for d in frame_diffs if d["mean_diff"] > 1.0]
        print(f"\nFrames with significant differences (mean_diff > 1.0): {len(sig_diffs)}/{len(frame_diffs)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
