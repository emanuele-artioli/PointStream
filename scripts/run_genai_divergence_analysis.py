#!/usr/bin/env python3
"""Run instrumented GenAI pipeline to compare encoder/decoder divergence."""

import os
import sys
import subprocess
from pathlib import Path
import shutil
import json

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def setup_debug_dir():
    """Create and return debug directory."""
    debug_dir = project_root / "outputs" / "genai_debug_comparison"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir

def run_pipeline_with_genai():
    """Run the full pipeline with GenAI enabled and debug exports."""
    debug_dir = setup_debug_dir()
    
    env = os.environ.copy()
    env["POINTSTREAM_DEBUG_GENAI_DIR"] = str(debug_dir)
    env["POINTSTREAM_ENABLE_GENAI"] = "1"
    env["POINTSTREAM_GENAI_BACKEND"] = "animate-anyone"
    
    # Test with real_tennis.mp4
    test_asset = project_root / "assets" / "real_tennis.mp4"
    if not test_asset.exists():
        print(f"ERROR: Test asset not found: {test_asset}")
        return False
    
    cmd = [
        "conda", "run", "-n", "pointstream",
        "python", "-m", "src.main",
        "--input", str(test_asset),
        "--genai-backend", "animate-anyone",
        "--num-frames", "8",
    ]
    
    print("Running pipeline with GenAI enabled...")
    print(f"Debug output: {debug_dir}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, env=env, capture_output=False, timeout=600)
    
    if result.returncode != 0:
        print(f"ERROR: Pipeline failed with return code {result.returncode}")
        return False
    
    print("Pipeline completed successfully!")
    return True

def generate_comparison_report(debug_dir):
    """Generate detailed comparison report."""
    from src.shared.genai_debug import create_debug_report

    report = create_debug_report(str(debug_dir))
    
    report_file = debug_dir / "DIVERGENCE_REPORT.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\nComparison report saved to: {report_file}")
    print("\n" + report)
    
    return report_file

def analyze_frame_directories(debug_dir):
    """Analyze the debug frame structure."""
    encoder_dir = debug_dir / "encoder"
    decoder_dir = debug_dir / "decoder"
    comparison_dir = debug_dir / "comparison"
    
    analysis = {
        "encoder_frames": 0,
        "decoder_frames": 0,
        "comparison_frames": 0,
        "encoder_actors_by_frame": {},
        "decoder_actors_by_frame": {},
    }
    
    if encoder_dir.exists():
        for frame_dir in sorted(encoder_dir.iterdir()):
            if frame_dir.is_dir():
                analysis["encoder_frames"] += 1
                actor_dirs = [d for d in frame_dir.iterdir() if d.is_dir()]
                analysis["encoder_actors_by_frame"][frame_dir.name] = len(actor_dirs)
    
    if decoder_dir.exists():
        for frame_dir in sorted(decoder_dir.iterdir()):
            if frame_dir.is_dir():
                analysis["decoder_frames"] += 1
                actor_dirs = [d for d in frame_dir.iterdir() if d.is_dir()]
                analysis["decoder_actors_by_frame"][frame_dir.name] = len(actor_dirs)
    
    if comparison_dir.exists():
        for frame_dir in sorted(comparison_dir.iterdir()):
            if frame_dir.is_dir():
                analysis["comparison_frames"] += 1
    
    analysis_file = debug_dir / "analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("\nFrame analysis:")
    print(f"  Encoder frames: {analysis['encoder_frames']}")
    print(f"  Decoder frames: {analysis['decoder_frames']}")
    print(f"  Comparison frames: {analysis['comparison_frames']}")
    
    return analysis

def main():
    """Main entry point."""
    print("=" * 70)
    print("GenAI Compositor Divergence Analysis")
    print("=" * 70)
    
    # Setup
    debug_dir = setup_debug_dir()
    print(f"Debug directory: {debug_dir}\n")
    
    # Run pipeline with GenAI and debug exports
    if not run_pipeline_with_genai():
        print("Pipeline run failed. Exiting.")
        return 1
    
    # Analyze results
    print("\nAnalyzing debug outputs...")
    analyze_frame_directories(debug_dir)
    
    # Generate report
    report_file = generate_comparison_report(debug_dir)
    
    print("\n" + "=" * 70)
    print(f"Debug artifacts saved to: {debug_dir}")
    print(f"Report: {report_file}")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
