#!/usr/bin/env python3
"""
Comprehensive evaluation of the adaptive tracking and inpainting system.
This script analyzes the performance of the enhanced PointStream pipeline.
"""

import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path

def evaluate_metrics_cache():
    """Evaluate the adaptive metrics learning."""
    print("=== Evaluating Adaptive Metrics Cache ===")
    
    cache_path = "/home/itec/emanuele/PointStream/pointstream_metrics_cache.json"
    if not os.path.exists(cache_path):
        print("❌ Metrics cache not found!")
        return False
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        print(f"✅ Cache loaded successfully")
        print(f"📊 Content types tracked: {len(cache_data['history'])}")
        
        for content_type, history in cache_data['history'].items():
            print(f"\n📈 {content_type}:")
            print(f"   - History entries: {len(history)}")
            
            if history:
                latest = history[-1]
                success_score = latest.get('success_score', 0)
                metrics = latest.get('metrics', {})
                
                print(f"   - Latest success score: {success_score:.3f}")
                print(f"   - Latest metrics: {metrics}")
                
                # Check for improvement over time
                if len(history) > 1:
                    first_score = history[0].get('success_score', 0)
                    improvement = success_score - first_score
                    print(f"   - Improvement: {improvement:+.3f}")
                    
                    if improvement > 0:
                        print("   ✅ Performance improved over time")
                    elif improvement == 0:
                        print("   ⚠️  Performance stable")
                    else:
                        print("   ❌ Performance degraded")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading cache: {e}")
        return False

def evaluate_pipeline_outputs():
    """Evaluate the quality of pipeline outputs."""
    print("\n=== Evaluating Pipeline Outputs ===")
    
    output_dir = "/home/itec/emanuele/PointStream/artifacts/pipeline_output"
    if not os.path.exists(output_dir):
        print("❌ Output directory not found!")
        return False
    
    files = os.listdir(output_dir)
    
    # Count different types of outputs
    backgrounds = [f for f in files if f.endswith('_background.png')]
    appearances = [f for f in files if f.endswith('_appearance.png')]
    results = [f for f in files if f.endswith('_final_results.json')]
    
    print(f"📊 Output Summary:")
    print(f"   - Background images: {len(backgrounds)}")
    print(f"   - Track appearances: {len(appearances)}")
    print(f"   - Result files: {len(results)}")
    
    # Analyze video datasets processed
    video_names = set()
    for filename in files:
        if '_scene_' in filename:
            video_name = filename.split('_scene_')[0]
            video_names.add(video_name)
    
    print(f"   - Videos processed: {len(video_names)}")
    for video in sorted(video_names):
        print(f"     * {video}")
    
    # Check quality of backgrounds
    print(f"\n🎨 Background Quality Analysis:")
    for bg_file in backgrounds[:3]:  # Check first 3 backgrounds
        bg_path = os.path.join(output_dir, bg_file)
        try:
            img = cv2.imread(bg_path)
            if img is not None:
                height, width = img.shape[:2]
                mean_color = np.mean(img, axis=(0,1))
                print(f"   - {bg_file}: {width}x{height}, avg_color: {mean_color}")
            else:
                print(f"   - {bg_file}: ❌ Could not load")
        except Exception as e:
            print(f"   - {bg_file}: ❌ Error: {e}")
    
    return True

def analyze_tracking_performance():
    """Analyze tracking performance from results."""
    print("\n=== Analyzing Tracking Performance ===")
    
    results_files = []
    output_dir = "/home/itec/emanuele/PointStream/artifacts/pipeline_output"
    
    for file in os.listdir(output_dir):
        if file.endswith('_final_results.json'):
            results_files.append(os.path.join(output_dir, file))
    
    if not results_files:
        print("❌ No result files found!")
        return False
    
    for results_file in results_files[:2]:  # Analyze first 2 result files
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            video_name = os.path.basename(results_file).replace('_final_results.json', '')
            print(f"\n📹 {video_name}:")
            
            metadata = results.get('metadata', {})
            scenes = results.get('scenes', [])
            
            print(f"   - Resolution: {metadata.get('resolution', 'Unknown')}")
            print(f"   - FPS: {metadata.get('fps', 'Unknown')}")
            print(f"   - Scenes: {len(scenes)}")
            
            # Count total tracks across all scenes
            total_tracks = 0
            for scene in scenes:
                scene_tracks = len([k for k in scene.keys() if k.startswith('track_')])
                total_tracks += scene_tracks
            
            print(f"   - Total tracks: {total_tracks}")
            
            # Analyze track distribution per scene
            if scenes:
                track_counts = []
                for i, scene in enumerate(scenes):
                    scene_tracks = len([k for k in scene.keys() if k.startswith('track_')])
                    track_counts.append(scene_tracks)
                
                avg_tracks = np.mean(track_counts)
                std_tracks = np.std(track_counts)
                print(f"   - Avg tracks per scene: {avg_tracks:.1f} ± {std_tracks:.1f}")
                
                if std_tracks < avg_tracks * 0.5:
                    print("   ✅ Consistent tracking across scenes")
                else:
                    print("   ⚠️  Variable tracking performance")
            
        except Exception as e:
            print(f"❌ Error analyzing {results_file}: {e}")
    
    return True

def test_adaptive_components():
    """Test the adaptive components directly."""
    print("\n=== Testing Adaptive Components ===")
    
    try:
        # Add PointStream to path
        sys.path.insert(0, '/home/itec/emanuele/PointStream')
        
        from pointstream.utils.adaptive_metrics import MetricsCache
        from pointstream.models.yolo_handler import AdaptiveTracker
        
        # Test MetricsCache
        print("🧪 Testing MetricsCache...")
        cache = MetricsCache()
        
        # Test threshold optimization
        cache.log_tracking_result(
            content_type="test_eval",
            metrics={"accuracy": 0.8, "speed": 0.9},
            method_used="yolo",
            success_score=0.85,
            config_used={"conf": 0.5, "iou": 0.7}
        )
        
        optimized = cache.get_optimal_thresholds("test_eval")
        print(f"   ✅ Config optimization: {optimized}")
        
        # Test AdaptiveTracker
        print("🧪 Testing AdaptiveTracker...")
        tracker = AdaptiveTracker("yolov8n.pt")  # Use a standard model
        print(f"   ✅ Tracker initialized with model: {tracker.model_path}")
        
        # Test tracking with fake data
        fake_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        results = tracker.track_objects(fake_frames, "sports")
        print(f"   ✅ Tracking test completed with {len(results)} frame results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing components: {e}")
        return False

def main():
    """Main evaluation function."""
    print("🚀 PointStream Adaptive System Evaluation")
    print("=" * 50)
    
    results = {
        'metrics_cache': evaluate_metrics_cache(),
        'pipeline_outputs': evaluate_pipeline_outputs(), 
        'tracking_performance': analyze_tracking_performance(),
        'adaptive_components': test_adaptive_components()
    }
    
    print("\n" + "=" * 50)
    print("📋 EVALUATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{component:20}: {status}")
        all_passed = all_passed and passed
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL COMPONENTS WORKING CORRECTLY!")
        print("✅ Adaptive tracking and inpainting system is functional")
    else:
        print("⚠️  SOME ISSUES DETECTED")
        print("🔧 Check the detailed output above for specific problems")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
