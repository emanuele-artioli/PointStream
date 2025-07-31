"""
Main entry point for the Pointstream processing pipeline.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import config
# Import all stages
from .pipeline import stage_01_analyzer, stage_02_detector, stage_03_background, stage_04_foreground

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser(description="Pointstream: A Content-Aware Neural Video Codec")
    parser.add_argument("--input-video", type=str, required=True, help="Path to the input video file.")
    args = parser.parse_args()
    video_stem = Path(args.input_video).stem

    print(f"Starting Pointstream pipeline for: {args.input_video}")

    stage1_gen = stage_01_analyzer.run_analysis_pipeline(args.input_video)
    stage2_gen = stage_02_detector.run_detection_pipeline(stage1_gen)
    stage3_gen = stage_03_background.run_background_modeling_pipeline(stage2_gen, video_stem)
    stage4_gen = stage_04_foreground.run_foreground_pipeline(stage3_gen, video_stem)

    all_results = []
    for processed_scene in stage4_gen:
        print(f"--> Pipeline finished processing Scene {processed_scene['scene_index']}\n")
        all_results.append(processed_scene)

    if not all_results:
        print("Pipeline terminated: No scenes were processed.")
        return

    output_path = config.OUTPUT_DIR / f"{video_stem}_final_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print(f"\nPipeline complete. All results saved to: {output_path}")

if __name__ == "__main__":
    main()