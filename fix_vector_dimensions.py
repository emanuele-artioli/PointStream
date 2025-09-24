#!/usr/bin/env python3
"""
PointStream Vector Dimension Fix
Ensures all models and vector processing use compatible dimensions.
"""

import torch
import sys
from pathlib import Path

def analyze_model_requirements():
    """Analyze the trained human model to understand its requirements."""
    
    model_path = Path("models/human_cgan.pth")
    if not model_path.exists():
        print("❌ Human model not found")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        metadata = checkpoint.get('model_metadata', {})
        
        print("📊 Human Model Analysis:")
        print(f"   Vector input size: {metadata.get('vector_input_size', 'Unknown')}")
        print(f"   Keypoint channels: {metadata.get('keypoint_channels', 'Unknown')}")
        print(f"   Temporal frames: {metadata.get('temporal_frames', 'Unknown')}")
        print(f"   Include confidence: {metadata.get('include_confidence', 'Unknown')}")
        
        expected_pose_size = metadata.get('discriminator_metadata', {}).get('pose_vector_size', 'Unknown')
        print(f"   Expected pose vector size: {expected_pose_size}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        return None

def check_config_consistency():
    """Check if config parameters match model requirements."""
    
    sys.path.append('.')
    from utils.config import config
    
    print("\n🔧 Configuration Analysis:")
    
    # Use the custom config parser that handles inline comments
    try:
        human_keypoints = config.get_int('keypoints', 'human_num_keypoints', 17)
        animal_keypoints = config.get_int('keypoints', 'animal_num_keypoints', 12)
        other_keypoints = config.get_int('keypoints', 'other_num_keypoints', 24)
        
        temporal_frames = config.get_int('keypoints', 'temporal_frames', 2)
        include_confidence = config.get_bool('keypoints', 'include_confidence_in_vectors', True)
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        # Fallback values
        human_keypoints = 17
        animal_keypoints = 12
        other_keypoints = 24
        temporal_frames = 2
        include_confidence = True
        print("   Using fallback values")
    
    print(f"   Human keypoints: {human_keypoints}")
    print(f"   Animal keypoints: {animal_keypoints}")  
    print(f"   Other keypoints: {other_keypoints}")
    print(f"   Temporal frames: {temporal_frames}")
    print(f"   Include confidence: {include_confidence}")
    
    # Calculate expected vector sizes
    def calc_vector_size(keypoints, temporal, confidence):
        values_per_kp = 3 if confidence else 2
        pose_size = keypoints * values_per_kp * (1 + temporal)
        return 2048 + pose_size  # appearance + pose
    
    human_vector_size = calc_vector_size(human_keypoints, temporal_frames, include_confidence)
    animal_vector_size = calc_vector_size(animal_keypoints, temporal_frames, include_confidence)
    other_vector_size = calc_vector_size(other_keypoints, temporal_frames, include_confidence)
    
    print(f"\n📐 Expected Vector Sizes:")
    print(f"   Human: {human_vector_size} (appearance: 2048 + pose: {human_vector_size-2048})")
    print(f"   Animal: {animal_vector_size} (appearance: 2048 + pose: {animal_vector_size-2048})")
    print(f"   Other: {other_vector_size} (appearance: 2048 + pose: {other_vector_size-2048})")
    
    return {
        'human': human_vector_size,
        'animal': animal_vector_size, 
        'other': other_vector_size
    }

def main():
    print("🔧 PointStream Vector Dimension Analysis")
    print("="*50)
    
    # Analyze the trained model
    model_metadata = analyze_model_requirements()
    
    # Check configuration
    expected_sizes = check_config_consistency()
    
    if model_metadata:
        trained_size = model_metadata.get('vector_input_size')
        expected_human_size = expected_sizes.get('human')
        
        print(f"\n✅ Compatibility Check:")
        if trained_size == expected_human_size:
            print(f"   Human model: ✅ Compatible ({trained_size} = {expected_human_size})")
        else:
            print(f"   Human model: ❌ Incompatible ({trained_size} ≠ {expected_human_size})")
            
        # Check missing models
        animal_model = Path("models/animal_cgan.pth")
        other_model = Path("models/other_cgan.pth")
        
        print(f"   Animal model: {'✅ Present' if animal_model.exists() else '❌ Missing'}")
        print(f"   Other model: {'✅ Present' if other_model.exists() else '❌ Missing'}")
        
    print(f"\n🎯 Recommendations:")
    if model_metadata:
        trained_size = model_metadata.get('vector_input_size', 0)
        if trained_size != expected_sizes.get('human', 0):
            print("   1. Retrain human model with current config OR update config to match trained model")
    
    if not Path("models/animal_cgan.pth").exists():
        print("   2. Train animal model or disable animal object processing")
        
    if not Path("models/other_cgan.pth").exists():
        print("   3. Train other model or use simpler fallback for 'other' objects")
        
    print("   4. Ensure vector processing matches model expectations exactly")

if __name__ == "__main__":
    main()