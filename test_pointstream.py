#!/usr/bin/env python3
"""
Test script for PointStream pipeline
"""

import sys
import logging
from pathlib import Path

# Add the PointStream directory to the path
pointstream_dir = Path(__file__).parent
sys.path.insert(0, str(pointstream_dir))

from pointstream import (
    split_video_into_scenes,
    segment_objects_in_scene,
    stitch_scene_panorama,
    classify_objects,
    extract_keypoints,
    process_video_pipeline
)


def test_individual_components(video_path: str):
    """Test each component individually."""
    print("Testing individual components...")
    
    # Test splitter
    print("\n1. Testing video splitter...")
    scenes = split_video_into_scenes(video_path)
    print(f"   Found {len(scenes)} scenes")
    
    if scenes:
        # Test segmenter on first scene
        print("\n2. Testing segmenter...")
        first_scene = scenes[0]
        segmentation_result = segment_objects_in_scene(first_scene)
        print(f"   Found {segmentation_result['total_objects']} objects")
        
        # Test stitcher
        print("\n3. Testing stitcher...")
        stitching_result = stitch_scene_panorama(first_scene, segmentation_result)
        print(f"   Created {stitching_result['scene_type']} panorama")
        
        # Test classifier
        print("\n4. Testing classifier...")
        classification_result = classify_objects(segmentation_result)
        print(f"   Classification stats: {classification_result['classification_stats']}")
        
        # Test keypointer
        print("\n5. Testing keypointer...")
        keypoint_result = extract_keypoints(classification_result)
        print(f"   Extracted keypoints for {keypoint_result['objects_with_keypoints']}/{keypoint_result['total_objects']} objects")


def test_full_pipeline(video_path: str, output_dir: str = None):
    """Test the complete pipeline."""
    print("\nTesting full pipeline...")
    
    results = process_video_pipeline(video_path, output_dir)
    
    print(f"\nPipeline Results:")
    print(f"Total scenes: {results['total_scenes']}")
    print(f"Total objects: {results['total_objects']}")
    print(f"Objects with keypoints: {results['total_objects_with_keypoints']}")
    print(f"Processing time: {results['total_processing_time']:.2f}s")
    
    if output_dir:
        print(f"Results saved to: {output_dir}")


def main():
    """Main test function."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Check for test video
    test_video = pointstream_dir / "room.mp4"
    if not test_video.exists():
        print(f"Test video not found: {test_video}")
        print("Please ensure room.mp4 exists in the PointStream directory")
        return
    
    try:
        # Test individual components
        test_individual_components(str(test_video))
        
        # Test full pipeline
        output_dir = pointstream_dir / "test_output"
        test_full_pipeline(str(test_video), str(output_dir))
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()