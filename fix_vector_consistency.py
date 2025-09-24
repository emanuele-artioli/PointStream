#!/usr/bin/env python3
"""
Vector Dimension Consistency Fix for PointStream
==================================================

This script fixes the vector dimension mismatches by ensuring consistent
pose vector generation across server and client components.

The root cause is that:
1. Server (keypointer.py) creates vectors one way
2. Client (object_generator.py) processes them differently for temporal context
3. Different categories have different vector size calculations

Expected vector sizes:
- Human: 2048 (appearance) + 153 (17*3*2 + 17*2) = 2201
- Animal: 2048 (appearance) + 108 (12*3*2 + 12*2) = 2156  
- Other: 2048 (appearance) + 216 (24*3*2 + 24*2) = 2264

Current issues:
- Other objects: Getting 2196 instead of 2264
- Human objects: Sometimes getting 2243 instead of 2201

Solution:
Create a centralized vector calculation utility that both server and client use.
"""

import sys
import os
import numpy as np

def calculate_expected_pose_vector_size(category: str, temporal_frames: int = 2, include_confidence: bool = True):
    """Calculate the expected pose vector size for a category."""
    
    # Keypoint counts per category
    keypoint_counts = {
        'human': 17,
        'animal': 12, 
        'other': 24
    }
    
    keypoint_count = keypoint_counts.get(category, 24)
    
    # Base size: keypoints * dimensions_per_keypoint
    coords_per_kp = 3 if include_confidence else 2
    base_size = keypoint_count * coords_per_kp
    
    # Add temporal context (duplicate current frame for each temporal frame)
    total_size = base_size + (base_size * temporal_frames)
    
    return total_size

def generate_vector_calculation_utility():
    """Generate a utility module for consistent vector calculations."""
    
    utility_code = '''"""
Vector Calculation Utility for PointStream
==========================================

This module provides centralized vector calculation functions to ensure
dimensional consistency between server and client components.
"""

def calculate_pose_vector_size(category: str, temporal_frames: int = 2, include_confidence: bool = True) -> int:
    """
    Calculate the expected pose vector size for a category.
    
    Args:
        category: Object category ('human', 'animal', 'other')
        temporal_frames: Number of temporal frames to include
        include_confidence: Whether to include confidence values
        
    Returns:
        Expected pose vector size
    """
    # Keypoint counts per category
    keypoint_counts = {
        'human': 17,
        'animal': 12,
        'other': 24
    }
    
    keypoint_count = keypoint_counts.get(category, 24)
    
    # Base size: keypoints * dimensions_per_keypoint  
    coords_per_kp = 3 if include_confidence else 2
    base_size = keypoint_count * coords_per_kp
    
    # Add temporal context (duplicate current frame for each temporal frame)
    total_size = base_size + (base_size * temporal_frames)
    
    return total_size


def build_pose_vector(keypoints_data, category: str, temporal_frames: int = 2, 
                     include_confidence: bool = True, temporal_context_data=None) -> list:
    """
    Build a pose vector with consistent dimensions.
    
    Args:
        keypoints_data: Keypoint data (list of [x, y, confidence] points)
        category: Object category ('human', 'animal', 'other')
        temporal_frames: Number of temporal frames to include
        include_confidence: Whether to include confidence values
        temporal_context_data: Previous frame keypoints for temporal context
        
    Returns:
        Flattened pose vector with consistent dimensions
    """
    # Keypoint counts per category
    keypoint_counts = {
        'human': 17,
        'animal': 12,
        'other': 24
    }
    
    expected_keypoints = keypoint_counts.get(category, 24)
    coords_per_kp = 3 if include_confidence else 2
    
    # Build current frame vector
    current_vector = []
    
    for i in range(expected_keypoints):
        if i < len(keypoints_data) and len(keypoints_data[i]) >= 2:
            kp = keypoints_data[i]
            current_vector.extend([kp[0], kp[1]])
            if include_confidence:
                conf = kp[2] if len(kp) >= 3 else 1.0
                current_vector.append(conf)
        else:
            # Pad with zeros
            current_vector.extend([0.0, 0.0])
            if include_confidence:
                current_vector.append(0.0)
    
    # Add temporal context
    final_vector = current_vector[:]
    
    if temporal_frames > 0:
        if temporal_context_data and len(temporal_context_data) > 0:
            # Use provided temporal context
            for frame_data in temporal_context_data[:temporal_frames]:
                if isinstance(frame_data, list) and len(frame_data) >= len(current_vector):
                    final_vector.extend(frame_data[:len(current_vector)])
                else:
                    # Fallback to duplicating current frame
                    final_vector.extend(current_vector)
        else:
            # Duplicate current frame for temporal context
            for _ in range(temporal_frames):
                final_vector.extend(current_vector)
    
    return final_vector


def validate_vector_dimensions(vector, category: str, temporal_frames: int = 2, 
                             include_confidence: bool = True) -> tuple:
    """
    Validate that a vector has the correct dimensions.
    
    Args:
        vector: The vector to validate
        category: Expected category
        temporal_frames: Expected temporal frames
        include_confidence: Expected confidence inclusion
        
    Returns:
        Tuple of (is_valid, expected_size, actual_size, error_message)
    """
    expected_size = calculate_pose_vector_size(category, temporal_frames, include_confidence)
    actual_size = len(vector) if vector else 0
    
    if actual_size == expected_size:
        return True, expected_size, actual_size, None
    else:
        error_msg = f"Vector dimension mismatch: expected {expected_size}, got {actual_size} for category '{category}'"
        return False, expected_size, actual_size, error_msg
'''

    return utility_code

def main():
    """Main function to create the vector utility and show analysis."""
    
    print("🔧 PointStream Vector Dimension Consistency Fix")
    print("=" * 55)
    
    # Show current expected sizes
    print("\\n📐 Expected Vector Sizes (with 2 temporal frames + confidence):")
    for category in ['human', 'animal', 'other']:
        size = calculate_expected_pose_vector_size(category, 2, True)
        appearance_size = 2048
        total_size = appearance_size + size
        print(f"   {category.capitalize()}: {size} pose + {appearance_size} appearance = {total_size} total")
    
    print("\\n🔍 Current Issues Found in Logs:")
    print("   ❌ Other objects: Input 1x2196, Expected 1x2264 (diff: -68)")
    print("   ❌ Human objects: Input 1x2243, Expected 1x2201 (diff: +42)")
    print("   ✅ Human objects: Input 1x2201 working correctly")
    
    # Generate utility file
    print("\\n📝 Generating vector calculation utility...")
    utility_code = generate_vector_calculation_utility()
    
    with open('utils/vector_utils.py', 'w') as f:
        f.write(utility_code)
    
    print("   ✅ Created utils/vector_utils.py")
    
    # Show the fixes needed
    print("\\n🛠️ Fixes Required:")
    print("   1. Import vector_utils in keypointer.py and object_generator.py")
    print("   2. Replace ad-hoc vector calculations with centralized functions")
    print("   3. Ensure temporal context extraction uses same logic")
    print("   4. Validate vectors before model input")
    
    print("\\n🎯 Next Steps:")
    print("   1. Run this script to create the utility")
    print("   2. Update keypointer.py to use vector_utils.build_pose_vector()")
    print("   3. Update object_generator.py to use vector_utils.validate_vector_dimensions()")
    print("   4. Test with a small video to verify fix")

if __name__ == '__main__':
    main()