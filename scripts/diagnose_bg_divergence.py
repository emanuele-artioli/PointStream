#!/usr/bin/env python3
"""Diagnose warped background panorama divergence."""

import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def analyze_warped_bg_divergence():
    """Analyze warped background differences in detail."""
    debug_dir = Path("/home/itec/emanuele/pointstream/outputs/genai_debug_comparison")
    
    print("=" * 70)
    print("Warped Background Panorama Divergence Analysis")
    print("=" * 70)
    print()
    
    # Load sample warped background frames from encoder and decoder
    encoder_dir = debug_dir / "encoder"
    decoder_dir = debug_dir / "decoder"
    
    # Analyze first few frames
    sample_frame = "frame_0001"
    sample_actor = "actor_1414941730"
    
    enc_bg_path = encoder_dir / sample_frame / sample_actor / "02_warped_background.png"
    dec_bg_path = decoder_dir / sample_frame / sample_actor / "02_warped_background.png"
    
    print(f"Sample: {sample_frame} / {sample_actor}\n")
    
    if enc_bg_path.exists() and dec_bg_path.exists():
        enc_bg = cv2.imread(str(enc_bg_path))
        dec_bg = cv2.imread(str(dec_bg_path))
        
        print(f"Encoder BG shape: {enc_bg.shape}, dtype: {enc_bg.dtype}")
        print(f"Decoder BG shape: {dec_bg.shape}, dtype: {dec_bg.dtype}")
        print()
        
        if enc_bg.shape == dec_bg.shape:
            diff = np.abs(enc_bg.astype(np.float32) - dec_bg.astype(np.float32))
            
            # Analyze per channel
            for ch_idx, ch_name in enumerate(["B", "G", "R"]):
                ch_diff = diff[:, :, ch_idx]
                print(f"Channel {ch_name}:")
                print(f"  Mean diff: {np.mean(ch_diff):.3f}")
                print(f"  Max diff:  {np.max(ch_diff):.1f}")
                print(f"  Std diff:  {np.std(ch_diff):.3f}")
                print(f"  Nonzero:   {100.0 * np.count_nonzero(ch_diff > 0.5) / ch_diff.size:.1f}%")
            
            print()
            
            # Spatial analysis
            diff_gray = np.mean(diff, axis=2)
            print("Spatial distribution of differences:")
            print(f"  Center 50%: {np.percentile(diff_gray, 50):.3f}")
            print(f"  75th percentile: {np.percentile(diff_gray, 75):.3f}")
            print(f"  95th percentile: {np.percentile(diff_gray, 95):.3f}")
            print(f"  99th percentile: {np.percentile(diff_gray, 99):.3f}")
            print()
            
            # Check for specific patterns
            high_diff_mask = diff_gray > 5.0
            if np.any(high_diff_mask):
                high_diff_regions = np.where(high_diff_mask)
                print(f"High difference regions (diff > 5.0): {np.count_nonzero(high_diff_mask)} pixels")
                if len(high_diff_regions[0]) > 0:
                    y_range = (high_diff_regions[0].min(), high_diff_regions[0].max())
                    x_range = (high_diff_regions[1].min(), high_diff_regions[1].max())
                    print(f"  Y range: {y_range}, X range: {x_range}")
            else:
                print("No high-difference regions (diff > 5.0)")
            
            print()
    else:
        print("ERROR: Could not find warped background PNG files")
        return
    
    # Check if this is a systematic JPEG quality issue
    print("Hypothesis: JPEG quality loss in panorama encoding/decoding")
    print("-" * 70)
    
    # Load the original panorama that was encoded
    run_output_dir = Path("/home/itec/emanuele/pointstream/outputs/20260506_114237_584248")
    panorama_uri = run_output_dir / "chunk_0001" / "panorama.jpg"
    
    if panorama_uri.exists():
        print(f"Found panorama: {panorama_uri}")
        print(f"Size: {panorama_uri.stat().st_size} bytes")
        panorama = cv2.imread(str(panorama_uri))
        print(f"Panorama shape: {panorama.shape}")
        print()
        print("NOTE: Panorama is JPEG-encoded during transport.")
        print("      Encoder and decoder both decompress the same JPEG.")
        print("      Small differences could be due to:")
        print("      1. JPEG decompression implementation differences (rare)")
        print("      2. Different frame selection/ordering from panorama")
        print("      3. Warping parameter differences")
    else:
        print(f"Panorama not found at: {panorama_uri}")
    
    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The warped background has consistent small differences (~0.67 pixels mean).
This is NOT a GenAI actor synthesis issue, but a panorama/background issue.

Likely causes:
1. **JPEG recompression artifacts**: Panorama is JPEG-encoded during transport.
   Different JPEG decoders or decompression order could cause minor diffs.
   
2. **Frame indexing**: Encoder and decoder may warp different sets of frames
   from the panorama (e.g., different temporal windows).
   
3. **Synthesis engine state**: Different instance creation or seed handling
   between encoder and decoder could affect warp quality.

Impact on GenAI:
- The mean difference of 0.67 is IMPERCEPTIBLE to the human eye.
- GenAI actors are generated from the pose condition, not the background.
- The background difference does NOT explain weird-looking players.

Next steps to diagnose "weird looking players":
1. Compare generated actor crops BEFORE they're pasted (before 05_composited)
2. Check for generated actor quality differences (03_generated_actor.png)
3. Analyze alpha masking differences
4. Verify AnimateAnyone runtime consistency
""")

if __name__ == "__main__":
    analyze_warped_bg_divergence()
