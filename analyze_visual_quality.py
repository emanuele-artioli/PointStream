#!/usr/bin/env python3
"""
Visual Quality Analysis for PointStream Pipeline
Analyzes background images and track appearances for quality assessment.
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def analyze_background_quality(bg_path):
    """Analyze background image quality metrics."""
    if not os.path.exists(bg_path):
        return None
    
    img = cv2.imread(bg_path)
    if img is None:
        return None
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate quality metrics
    metrics = {
        'resolution': img.shape[:2],
        'mean_intensity': np.mean(gray),
        'std_intensity': np.std(gray),
        'contrast': np.std(gray),
        'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'brightness': np.mean(img),
        'color_diversity': np.std(img.reshape(-1, 3), axis=0).mean()
    }
    
    # Detect potential artifacts
    artifacts = {
        'low_contrast': metrics['contrast'] < 20,
        'overexposed': metrics['mean_intensity'] > 240,
        'underexposed': metrics['mean_intensity'] < 15,
        'low_sharpness': metrics['sharpness'] < 100
    }
    
    return {
        'path': bg_path,
        'metrics': metrics,
        'artifacts': artifacts,
        'quality_score': calculate_quality_score(metrics, artifacts)
    }

def calculate_quality_score(metrics, artifacts):
    """Calculate overall quality score (0-100)."""
    score = 100
    
    # Penalize artifacts
    if artifacts['low_contrast']:
        score -= 20
    if artifacts['overexposed'] or artifacts['underexposed']:
        score -= 15
    if artifacts['low_sharpness']:
        score -= 10
    
    # Bonus for good metrics
    if metrics['contrast'] > 50:
        score += 5
    if 30 < metrics['mean_intensity'] < 200:
        score += 5
    if metrics['sharpness'] > 500:
        score += 5
    
    return max(0, min(100, score))

def analyze_track_appearance(track_path):
    """Analyze track appearance image."""
    if not os.path.exists(track_path):
        return None
    
    img = cv2.imread(track_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Basic metrics
    metrics = {
        'resolution': img.shape[:2],
        'mean_intensity': np.mean(gray),
        'contrast': np.std(gray),
        'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
        'object_coverage': calculate_object_coverage(gray)
    }
    
    return {
        'path': track_path,
        'metrics': metrics
    }

def calculate_object_coverage(gray):
    """Estimate how much of the image contains the tracked object."""
    # Simple threshold-based estimation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coverage = np.sum(binary > 0) / binary.size
    return coverage

def analyze_pipeline_results(output_dir):
    """Analyze all pipeline output files."""
    output_path = Path(output_dir)
    
    # Find all background and track files
    bg_files = list(output_path.glob("*_background.png"))
    track_files = list(output_path.glob("*_track_*_appearance.png"))
    result_files = list(output_path.glob("*_final_results.json"))
    
    analysis = {
        'backgrounds': [],
        'tracks': [],
        'summary': {}
    }
    
    print(f"Found {len(bg_files)} background files")
    print(f"Found {len(track_files)} track appearance files")
    print(f"Found {len(result_files)} result files")
    
    # Analyze backgrounds
    for bg_file in bg_files:
        bg_analysis = analyze_background_quality(str(bg_file))
        if bg_analysis:
            analysis['backgrounds'].append(bg_analysis)
            print(f"\nBackground: {bg_file.name}")
            print(f"  Quality Score: {bg_analysis['quality_score']:.1f}/100")
            print(f"  Resolution: {bg_analysis['metrics']['resolution']}")
            print(f"  Contrast: {bg_analysis['metrics']['contrast']:.1f}")
            print(f"  Sharpness: {bg_analysis['metrics']['sharpness']:.1f}")
            
            if any(bg_analysis['artifacts'].values()):
                print(f"  Artifacts: {[k for k, v in bg_analysis['artifacts'].items() if v]}")
    
    # Analyze track appearances
    for track_file in track_files:
        track_analysis = analyze_track_appearance(str(track_file))
        if track_analysis:
            analysis['tracks'].append(track_analysis)
            print(f"\nTrack: {track_file.name}")
            print(f"  Resolution: {track_analysis['metrics']['resolution']}")
            print(f"  Object Coverage: {track_analysis['metrics']['object_coverage']:.2f}")
            print(f"  Contrast: {track_analysis['metrics']['contrast']:.1f}")
    
    # Analyze JSON results
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            print(f"\nResults from {result_file.name}:")
            if 'scenes' in results:
                for scene_id, scene_data in results['scenes'].items():
                    print(f"  Scene {scene_id}:")
                    if 'tracks' in scene_data:
                        print(f"    Tracks: {len(scene_data['tracks'])}")
                        
                        # Calculate average tracking quality
                        qualities = []
                        for track in scene_data['tracks'].values():
                            if 'quality_score' in track:
                                qualities.append(track['quality_score'])
                        
                        if qualities:
                            avg_quality = np.mean(qualities)
                            print(f"    Average Quality: {avg_quality:.3f}")
                            analysis['summary'][f'scene_{scene_id}_avg_quality'] = avg_quality
        
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    # Calculate overall summary
    if analysis['backgrounds']:
        bg_scores = [bg['quality_score'] for bg in analysis['backgrounds']]
        analysis['summary']['avg_background_quality'] = np.mean(bg_scores)
        analysis['summary']['min_background_quality'] = np.min(bg_scores)
    
    if analysis['tracks']:
        track_coverages = [t['metrics']['object_coverage'] for t in analysis['tracks']]
        analysis['summary']['avg_object_coverage'] = np.mean(track_coverages)
    
    return analysis

def create_quality_report(analysis, output_file):
    """Create a detailed quality report."""
    with open(output_file, 'w') as f:
        f.write("# PointStream Pipeline Quality Analysis Report\n\n")
        
        # Summary
        f.write("## Summary\n")
        if 'avg_background_quality' in analysis['summary']:
            f.write(f"- Average Background Quality: {analysis['summary']['avg_background_quality']:.1f}/100\n")
        if 'avg_object_coverage' in analysis['summary']:
            f.write(f"- Average Object Coverage: {analysis['summary']['avg_object_coverage']:.2f}\n")
        f.write(f"- Total Backgrounds Generated: {len(analysis['backgrounds'])}\n")
        f.write(f"- Total Track Appearances: {len(analysis['tracks'])}\n\n")
        
        # Detailed background analysis
        f.write("## Background Analysis\n")
        for bg in analysis['backgrounds']:
            f.write(f"### {Path(bg['path']).name}\n")
            f.write(f"- Quality Score: {bg['quality_score']:.1f}/100\n")
            f.write(f"- Resolution: {bg['metrics']['resolution']}\n")
            f.write(f"- Contrast: {bg['metrics']['contrast']:.1f}\n")
            f.write(f"- Sharpness: {bg['metrics']['sharpness']:.1f}\n")
            
            artifacts = [k for k, v in bg['artifacts'].items() if v]
            if artifacts:
                f.write(f"- Artifacts: {', '.join(artifacts)}\n")
            f.write("\n")
        
        # Track analysis
        f.write("## Track Appearance Analysis\n")
        for track in analysis['tracks']:
            f.write(f"### {Path(track['path']).name}\n")
            f.write(f"- Resolution: {track['metrics']['resolution']}\n")
            f.write(f"- Object Coverage: {track['metrics']['object_coverage']:.2f}\n")
            f.write(f"- Contrast: {track['metrics']['contrast']:.1f}\n\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze PointStream pipeline visual quality')
    parser.add_argument('--output-dir', default='/home/itec/emanuele/PointStream/artifacts/pipeline_output',
                       help='Directory containing pipeline output files')
    parser.add_argument('--report-file', default='/home/itec/emanuele/PointStream/quality_analysis_report.md',
                       help='Output file for quality report')
    
    args = parser.parse_args()
    
    print("Analyzing PointStream Pipeline Visual Quality...")
    print(f"Output directory: {args.output_dir}")
    
    analysis = analyze_pipeline_results(args.output_dir)
    
    print("\n" + "="*50)
    print("OVERALL SUMMARY")
    print("="*50)
    
    for key, value in analysis['summary'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Create detailed report
    create_quality_report(analysis, args.report_file)
    print(f"\nDetailed report saved to: {args.report_file}")

if __name__ == "__main__":
    main()
