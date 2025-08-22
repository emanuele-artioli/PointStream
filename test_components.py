#!/usr/bin/env python3
"""
Test script to verify the new semantic classification and duplicate filtering functionality.
"""

import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.semantic_classifier import SemanticClassifier
from scripts.duplicate_filter import DuplicateFilter
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_semantic_classifier():
    """Test the semantic classifier with sample class names."""
    print("\n=== Testing Semantic Classifier ===")
    
    try:
        classifier = SemanticClassifier()
        
        # Test class names
        test_classes = [
            'person', 'dog', 'cat', 'car', 'bicycle', 'horse', 
            'bird', 'cow', 'sheep', 'airplane', 'train', 'boat',
            'human', 'animal', 'vehicle', 'bear', 'elephant'
        ]
        
        results = []
        for class_name in test_classes:
            result = classifier.classify_class_name(class_name)
            results.append(result)
            print(f"{class_name:10} -> {result['semantic_category']:8} (conf: {result['confidence']:.3f}, method: {result['method']})")
        
        # Test with fake objects
        fake_objects = [
            {'class_name': 'person', 'confidence': 0.9},
            {'class_name': 'dog', 'confidence': 0.8},
            {'class_name': 'car', 'confidence': 0.7},
            {'class_name': 'bird', 'confidence': 0.85}
        ]
        
        classified_objects = classifier.classify_objects(fake_objects)
        print(f"\nClassified {len(classified_objects)} objects:")
        for obj in classified_objects:
            print(f"  {obj['original_class_name']} -> {obj['semantic_category']} (conf: {obj['semantic_confidence']:.3f})")
        
        # Get statistics
        stats = classifier.get_classification_statistics(classified_objects)
        print(f"\nClassification Statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"Error testing semantic classifier: {e}")
        return False

def test_duplicate_filter():
    """Test the duplicate filter with sample objects."""
    print("\n=== Testing Duplicate Filter ===")
    
    try:
        duplicate_filter = DuplicateFilter()
        
        # Create test objects with overlapping bounding boxes
        test_objects = [
            {
                'class_name': 'person', 
                'confidence': 0.9,
                'bbox': [10, 10, 50, 50],
                'area': 1600,
                'mask_area': 1500,
                'track_id': 1
            },
            {
                'class_name': 'human',  # Should be considered duplicate of above
                'confidence': 0.7,
                'bbox': [15, 15, 55, 55],
                'area': 1600,
                'mask_area': 1400,
                'track_id': None
            },
            {
                'class_name': 'dog',
                'confidence': 0.8,
                'bbox': [100, 100, 140, 140],
                'area': 1600,
                'mask_area': 1500,
                'track_id': 2
            },
            {
                'class_name': 'car',
                'confidence': 0.85,
                'bbox': [200, 200, 250, 250],
                'area': 2500,
                'mask_area': 2300,
                'track_id': 3
            }
        ]
        
        # Add masks (simple binary masks for testing)
        for i, obj in enumerate(test_objects):
            bbox = obj['bbox']
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            mask = np.ones((h, w), dtype=np.uint8)
            obj['mask'] = mask
        
        print(f"Original objects: {len(test_objects)}")
        for obj in test_objects:
            print(f"  {obj['class_name']} (conf: {obj['confidence']}, bbox: {obj['bbox']}, track: {obj.get('track_id')})")
        
        # Apply duplicate filtering
        result = duplicate_filter.filter_duplicates(test_objects)
        
        print(f"\nFiltered objects: {result['filtered_count']}")
        print(f"Removed duplicates: {result['removed_duplicates']}")
        
        for obj in result['objects']:
            print(f"  {obj['class_name']} (conf: {obj['confidence']}, bbox: {obj['bbox']}, track: {obj.get('track_id')})")
        
        return True
        
    except Exception as e:
        print(f"Error testing duplicate filter: {e}")
        return False

def main():
    """Run all tests."""
    print("PointStream Component Tests")
    print("=" * 50)
    
    semantic_ok = test_semantic_classifier()
    duplicate_ok = test_duplicate_filter()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Semantic Classifier: {'✅ PASS' if semantic_ok else '❌ FAIL'}")
    print(f"Duplicate Filter:    {'✅ PASS' if duplicate_ok else '❌ FAIL'}")
    
    if semantic_ok and duplicate_ok:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
