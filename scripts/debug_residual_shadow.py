#!/usr/bin/env python
"""Debug script to trace the residual shadow artifact through the pipeline.

This script captures the residual frames at multiple stages:
1. Raw computed residual (before encoding)
2. Post-FFmpeg encoded/decoded residual
3. Residual when applied back to predicted frame
4. Final reconstructed output

Outputs debug images and video for inspection.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from src.decoder.compositor import ResidualCompositor
from src.decoder.mock_renderer import DecoderRenderer
from src.encoder.orchestrator import EncoderPipeline
from src.encoder.residual_calculator import ResidualCalculator, BinaryActorImportanceMapper
from src.encoder.video_io import encode_video_frames_ffmpeg, iter_video_frames_ffmpeg, probe_video_metadata
from src.shared.schemas import VideoChunk, ResidualMode
from src.transport.disk import DiskTransport


def debug_residual_pipeline(
    input_video: Path,
    output_dir: Path,
    num_frames: int = 24,
) -> None:
    """Trace residual frames through encode/decode pipeline with debug outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DEBUG] Input: {input_video}")
    print(f"[DEBUG] Output: {output_dir}")
    
    # ============================================================================
    # STAGE 1: Encode with GenAI enabled to capture raw residual differences
    # ============================================================================
    print("\n[STAGE 1] Encoding chunk with GenAI enabled...")
    os.environ["POINTSTREAM_ENABLE_GENAI"] = "1"
    os.environ["POINTSTREAM_GENAI_BACKEND"] = "animate-anyone"
    
    from src.encoder.mock_extractors import MockActorExtractor
    encoder_pipeline = EncoderPipeline(actor_extractor=MockActorExtractor())
    
    try:
        payload, _decoded, frame_states = encoder_pipeline.encode_video_file_with_states(
            video_path=input_video,
            chunk_id="debug_residual_0001",
            max_frames=num_frames,
        )
        print(f"[STAGE 1] Payload created: {payload.chunk.num_frames} frames")
    finally:
        encoder_pipeline.shutdown()
    
    # ============================================================================
    # STAGE 2: Capture raw residual frames before FFmpeg encoding
    # ============================================================================
    print("\n[STAGE 2] Computing raw residual frames (before FFmpeg encoding)...")
    
    from src.shared.synthesis_engine import SynthesisEngine
    
    synthesis_engine = SynthesisEngine(seed=1337, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Synthesize predicted frame
    predicted_frames = synthesis_engine.synthesize(payload, include_guidance_overlays=False).frames_bgr
    
    # Apply GenAI for residual calculation
    compositor = synthesis_engine.get_genai_compositor()
    if hasattr(compositor, 'uses_temporal_pose_sequence') and compositor.uses_temporal_pose_sequence():
        print("[STAGE 2] Using Animate-Anyone GenAI compositor for predictions")
    
    # Manually compute raw residuals to dump them
    from src.encoder.video_io import iter_video_frames_ffmpeg
    
    chunk = payload.chunk
    source_metadata = probe_video_metadata(chunk.source_uri)
    source_iter = iter_video_frames_ffmpeg(
        chunk.source_uri,
        width=int(chunk.width),
        height=int(chunk.height),
    )
    
    raw_residual_frames: list[np.ndarray] = []
    raw_predicted_frames: list[np.ndarray] = []
    raw_source_frames: list[np.ndarray] = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for frame_idx in range(int(chunk.num_frames)):
        try:
            original_np = next(source_iter)
        except StopIteration:
            break
        
        original_tensor = (
            torch.from_numpy(np.asarray(original_np, dtype=np.uint8))
            .permute(2, 0, 1)
            .to(device, dtype=torch.float32)
        )
        predicted_tensor = predicted_frames[frame_idx].to(device, dtype=torch.float32)
        
        # Compute raw difference
        raw_diff = original_tensor - predicted_tensor
        
        # Encode as signed residual around 128
        encoded_residual = torch.clamp(raw_diff + 128.0, 0.0, 255.0).to(torch.uint8)
        
        # Store for inspection
        raw_residual_frames.append(encoded_residual.permute(1, 2, 0).cpu().numpy())
        raw_predicted_frames.append(predicted_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        raw_source_frames.append(original_np)
    
    print(f"[STAGE 2] Captured {len(raw_residual_frames)} raw residual frames")
    
    # Save first few residual frames as debug images
    for idx in range(min(3, len(raw_residual_frames))):
        residual_img = raw_residual_frames[idx]
        source_img = raw_source_frames[idx]
        predicted_img = raw_predicted_frames[idx]
        
        # Visualize residuals: clamp to visible range around 128
        residual_vis = residual_img.astype(np.float32)
        residual_vis = ((residual_vis - 128.0) * 2.0 + 128.0).clip(0, 255).astype(np.uint8)
        
        cv2.imwrite(str(output_dir / f"01_raw_residual_frame_{idx:02d}.jpg"), residual_vis)
        cv2.imwrite(str(output_dir / f"01_source_frame_{idx:02d}.jpg"), source_img)
        cv2.imwrite(str(output_dir / f"01_predicted_frame_{idx:02d}.jpg"), predicted_img)
    
    # Save raw residuals as video
    raw_residual_video = output_dir / "02_raw_residual.mp4"
    encode_video_frames_ffmpeg(
        output_path=raw_residual_video,
        frames_bgr=iter(raw_residual_frames),
        fps=float(chunk.fps),
        width=int(chunk.width),
        height=int(chunk.height),
        codec="libx265",
        pix_fmt="yuv420p",
        crf=28,
        preset="medium",
    )
    print(f"[STAGE 2] Saved raw residuals video: {raw_residual_video}")
    
    # ============================================================================
    # STAGE 3: Encode residuals through FFmpeg and decode back
    # ============================================================================
    print("\n[STAGE 3] Encoding residuals through FFmpeg (libx265) and decoding back...")
    
    encoded_residual_video = output_dir / "03_encoded_residual.mp4"
    encode_video_frames_ffmpeg(
        output_path=encoded_residual_video,
        frames_bgr=iter(raw_residual_frames),
        fps=float(chunk.fps),
        width=int(chunk.width),
        height=int(chunk.height),
        codec="libx265",
        pix_fmt="yuv420p",
        crf=28,
        preset="medium",
    )
    print(f"[STAGE 3] Encoded residuals: {encoded_residual_video}")
    
    # Decode them back
    decoded_residual_frames = list(
        iter_video_frames_ffmpeg(
            encoded_residual_video,
            width=int(chunk.width),
            height=int(chunk.height),
        )
    )
    print(f"[STAGE 3] Decoded {len(decoded_residual_frames)} residual frames")
    
    # Compare raw vs. decoded residuals frame-by-frame
    for idx in range(min(3, len(raw_residual_frames))):
        raw_res = raw_residual_frames[idx]
        decoded_res = decoded_residual_frames[idx]
        
        # Compute differences
        diff = np.abs(raw_res.astype(np.int16) - decoded_res.astype(np.int16))
        mae = float(np.mean(diff))
        
        print(f"[STAGE 3]   Frame {idx}: MAE between raw and decoded residual = {mae:.2f}")
        
        # Visualize differences
        diff_vis = np.clip(diff.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
        
        cv2.imwrite(str(output_dir / f"03_residual_raw_frame_{idx:02d}.jpg"), raw_res)
        cv2.imwrite(str(output_dir / f"03_residual_decoded_frame_{idx:02d}.jpg"), decoded_res)
        cv2.imwrite(str(output_dir / f"03_residual_diff_frame_{idx:02d}.jpg"), diff_vis)
    
    # ============================================================================
    # STAGE 4: Apply residuals back to predicted frames
    # ============================================================================
    print("\n[STAGE 4] Applying residuals back to predicted frames...")
    
    reconstructed_frames: list[np.ndarray] = []
    for frame_idx in range(len(decoded_residual_frames)):
        pred = raw_predicted_frames[frame_idx].astype(np.float32)
        residual = decoded_residual_frames[frame_idx].astype(np.float32)
        
        # Add back: predicted + (residual - 128)
        reconstructed = np.clip(pred + (residual - 128.0), 0.0, 255.0).astype(np.uint8)
        reconstructed_frames.append(reconstructed)
    
    print(f"[STAGE 4] Reconstructed {len(reconstructed_frames)} frames")
    
    # Compare reconstructed to source
    for idx in range(min(3, len(reconstructed_frames))):
        source = raw_source_frames[idx].astype(np.float32)
        recon = reconstructed_frames[idx].astype(np.float32)
        
        mae = float(np.mean(np.abs(source - recon)))
        print(f"[STAGE 4]   Frame {idx}: MAE between source and reconstructed = {mae:.2f}")
        
        cv2.imwrite(str(output_dir / f"04_reconstructed_frame_{idx:02d}.jpg"), reconstructed_frames[idx])
        
        # Visualize error
        error_vis = np.clip(np.abs(source - recon) * 2.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"04_reconstruction_error_frame_{idx:02d}.jpg"), error_vis)
    
    # Save reconstructed video
    reconstructed_video = output_dir / "04_reconstructed_output.mp4"
    encode_video_frames_ffmpeg(
        output_path=reconstructed_video,
        frames_bgr=iter(reconstructed_frames),
        fps=float(chunk.fps),
        width=int(chunk.width),
        height=int(chunk.height),
        codec="libx264",
        pix_fmt="yuv420p",
        crf=18,
        preset="veryfast",
    )
    print(f"[STAGE 4] Saved reconstructed video: {reconstructed_video}")
    
    # ============================================================================
    # STAGE 5: Full pipeline round-trip (encoder -> transport -> decoder)
    # ============================================================================
    print("\n[STAGE 5] Full pipeline round-trip...")
    
    transport_dir = output_dir / "transport"
    transport_dir.mkdir(parents=True, exist_ok=True)
    
    transport = DiskTransport(root_dir=transport_dir)
    transport.send(payload)
    recovered_payload = transport.receive(payload.chunk.chunk_id)
    
    decoder = DecoderRenderer(output_root=output_dir)
    final_output_path = output_dir / "05_final_decoded.mp4"
    decoded_result = decoder.process(recovered_payload, output_path=final_output_path)
    
    print(f"[STAGE 5] Final decoded output: {decoded_result.output_uri}")
    
    # Compare final output to source
    final_frames = list(
        iter_video_frames_ffmpeg(
            final_output_path,
            width=int(chunk.width),
            height=int(chunk.height),
        )
    )
    
    for idx in range(min(3, len(final_frames))):
        source = raw_source_frames[idx].astype(np.float32)
        final = final_frames[idx].astype(np.float32)
        
        mae = float(np.mean(np.abs(source - final)))
        print(f"[STAGE 5]   Frame {idx}: MAE between source and final = {mae:.2f}")
        
        cv2.imwrite(str(output_dir / f"05_final_frame_{idx:02d}.jpg"), final_frames[idx])
        
        # Visualize error
        error_vis = np.clip(np.abs(source - final) * 2.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"05_final_error_frame_{idx:02d}.jpg"), error_vis)
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("DEBUG ARTIFACTS SAVED:")
    print("=" * 80)
    print(f"  Raw residuals:        {output_dir}/01_raw_residual_frame_*.jpg")
    print(f"  Encoded residuals:    {output_dir}/03_encoded_residual.mp4")
    print(f"  Reconstructed:        {output_dir}/04_reconstructed_output.mp4")
    print(f"  Final output:         {final_output_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug residual shadow artifact")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/itec/emanuele/pointstream/assets/real_tennis.mp4"),
        help="Input video path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for debug artifacts (default: outputs/debug_residual_<timestamp>)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=24,
        help="Number of frames to process",
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = Path("outputs") / f"debug_residual_{timestamp}"
    
    debug_residual_pipeline(
        input_video=args.input,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
    )
