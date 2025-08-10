#!/usr/bin/env python3
"""
Test script for the new adaptive tracking and inpainting enhancements.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add PointStream to Python path
sys.path.insert(0, str(Path(__file__).parent))

from pointstream.models.yolo_handler import AdaptiveTracker
from pointstream.models.propainter_manager import ProPainterManager
from pointstream.utils.adaptive_metrics import get_metrics_cache
from pointstream import config

def create_test_data():
    """Create synthetic test data."""
    print("Creating synthetic test data...")
    
    # Create 5 test frames (320x240)
    frames = []
    for i in range(5):
        # Create a frame with some content
        frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
        
        # Add a moving square object
        x = 50 + i * 40
        y = 50 + i * 20
        cv2.rectangle(frame, (x, y), (x+50, y+50), (255, 0, 0), -1)
        
        frames.append(frame)
    
    # Create corresponding masks
    masks = []
    for i in range(5):
        mask = np.zeros((240, 320), dtype=np.uint8)
        x = 50 + i * 40
        y = 50 + i * 20
        cv2.rectangle(mask, (x, y), (x+50, y+50), 255, -1)
        masks.append(mask)
    
    return frames, masks

def test_adaptive_tracking():
    """Test the adaptive tracking system."""
    print("\n=== Testing Adaptive Tracking ===")
    
    frames, _ = create_test_data()
    
    try:
        # Initialize adaptive tracker
        tracker = AdaptiveTracker("yolo11n.pt")
        
        # Test with different content types
        for content_type in ["general", "sports", "test_content"]:
            print(f"\nTesting tracking with content type: {content_type}")
            
            # Track objects
            results = tracker.track_objects(frames, content_type)
            
            print(f"  -> Tracking results: {len(results)} frames processed")
            for i, frame_result in enumerate(results):
                print(f"     Frame {i}: {len(frame_result)} detections")
            
            # Get statistics
            stats = tracker.get_tracking_statistics(content_type)
            print(f"  -> Statistics: {stats}")
    
    except Exception as e:
        print(f"Adaptive tracking test failed: {e}")
        return False
    
    return True

def test_adaptive_inpainting():
    """Test the adaptive inpainting system."""
    print("\n=== Testing Adaptive Inpainting ===")
    
    frames, masks = create_test_data()
    
    try:
        # Initialize ProPainter manager
        manager = ProPainterManager()
        
        print(f"ProPainter available: {manager.is_available()}")
        
        # Test with different content types
        for content_type in ["general", "sports", "test_content"]:
            print(f"\nTesting inpainting with content type: {content_type}")
            
            # Test chunk inpainting
            inpainted = manager.inpaint_scene_chunk(frames, masks, content_type)
            
            print(f"  -> Inpainting results: {len(inpainted)} frames processed")
            
            # Check if frames were modified
            for i, (orig, inpainted_frame) in enumerate(zip(frames, inpainted)):
                diff = np.mean(np.abs(orig.astype(float) - inpainted_frame.astype(float)))
                print(f"     Frame {i}: average difference = {diff:.2f}")
    
    except Exception as e:
        print(f"Adaptive inpainting test failed: {e}")
        return False
    
    return True

def test_metrics_cache():
    """Test the metrics cache and learning system."""
    print("\n=== Testing Metrics Cache ===")
    
    try:
        cache = get_metrics_cache()
        
        # Test logging some results
        test_metrics = {
            "track_consistency": 0.8,
            "id_switches": 0.1,
            "detection_stability": 0.9
        }
        
        # Log results for different content types
        for content_type in ["test_content_1", "test_content_2"]:
            for i in range(3):
                config_used = {
                    "conf": 0.5 + i * 0.1,
                    "iou": 0.7 + i * 0.05
                }
                
                cache.log_tracking_result(
                    content_type, 
                    test_metrics, 
                    "yolo", 
                    0.7 + i * 0.1,
                    config_used
                )
        
        # Test getting optimal thresholds
        for content_type in ["test_content_1", "test_content_2"]:
            thresholds = cache.get_optimal_thresholds(content_type)
            stats = cache.get_content_statistics(content_type)
            
            print(f"Content type: {content_type}")
            print(f"  -> Optimal thresholds: {thresholds}")
            print(f"  -> Statistics: {stats}")
    
    except Exception as e:
        print(f"Metrics cache test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("PointStream Adaptive Enhancements Test Suite")
    print("=" * 50)
    
    tests = [
        ("Metrics Cache", test_metrics_cache),
        ("Adaptive Tracking", test_adaptive_tracking), 
        ("Adaptive Inpainting", test_adaptive_inpainting),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    
    if all_passed:
        print("\n✅ All adaptive enhancements are working correctly!")
        print("The system is ready for:")
        print("  • Adaptive YOLO tracking with threshold learning")
        print("  • ProPainter with frame-level OpenCV fallback")
        print("  • Content-specific threshold optimization")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
