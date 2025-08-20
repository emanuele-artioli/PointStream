#!/usr/bin/env python3
"""
Test script to compare old vs new black area inpainting approaches.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def old_inpainting(panorama, black_threshold=10):
    """Old approach: inpaint all black areas including borders."""
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    black_mask = (gray < black_threshold).astype(np.uint8)
    
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    
    inpainted = cv2.inpaint(panorama, black_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return inpainted, black_mask

def new_smart_inpainting(panorama, black_threshold=10, border_width=10):
    """New approach: exclude border areas from inpainting."""
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    black_mask = (gray < black_threshold).astype(np.uint8)
    
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    
    h, w = black_mask.shape
    border_mask = np.zeros_like(black_mask)
    
    # Create border region
    border_mask[:border_width, :] = 1    # Top
    border_mask[-border_width:, :] = 1   # Bottom 
    border_mask[:, :border_width] = 1    # Left
    border_mask[:, -border_width:] = 1   # Right
    
    # Find interior black areas only
    num_labels, labels = cv2.connectedComponents(black_mask)
    interior_black_mask = np.zeros_like(black_mask)
    
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        if not np.any(component_mask & border_mask):
            interior_black_mask |= component_mask
    
    inpainted = cv2.inpaint(panorama, interior_black_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return inpainted, interior_black_mask, black_mask

if __name__ == "__main__":
    # Test with a sample panorama (if available)
    import glob
    
    # Look for a recent panorama
    panorama_files = glob.glob("./output*/panoramas/*.jpg")
    
    if panorama_files:
        print(f"Testing with: {panorama_files[0]}")
        panorama = cv2.imread(panorama_files[0])
        
        if panorama is not None:
            # Apply old method
            old_result, old_mask = old_inpainting(panorama)
            
            # Apply new method  
            new_result, new_mask, original_mask = new_smart_inpainting(panorama)
            
            print(f"Original black pixels: {np.sum(original_mask)}")
            print(f"Old method inpainted: {np.sum(old_mask)} pixels")
            print(f"New method inpainted: {np.sum(new_mask)} pixels")
            print(f"Border pixels excluded: {np.sum(original_mask) - np.sum(new_mask)}")
            
            # Save comparison
            cv2.imwrite("./comparison_original.jpg", panorama)
            cv2.imwrite("./comparison_old_method.jpg", old_result)
            cv2.imwrite("./comparison_new_method.jpg", new_result)
            cv2.imwrite("./comparison_old_mask.jpg", old_mask * 255)
            cv2.imwrite("./comparison_new_mask.jpg", new_mask * 255)
            
            print("Comparison images saved!")
        else:
            print("Failed to load panorama")
    else:
        print("No panorama files found. Run the pipeline first.")
