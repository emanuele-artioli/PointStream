"""
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
