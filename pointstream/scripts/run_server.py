"""
Main entry point for the Pointstream server processing pipeline.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from pointstream import config
from pointstream.pipeline import stage_01_analyzer, stage_02_detector, stage_03_background, stage_04_foreground
from pointstream.utils.video_utils import get_video_properties


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser(description="Pointstream: A Content-Aware Neural Video Codec")
    parser.add_argument("--input-video", type=str, required=True, help="Path to the input video file.")
    # FIX: Add a new argument to select the content type
    parser.add_argument(
        "--content-type",
        type=str,
        default="general",
        choices=config.MODEL_REGISTRY.keys(),
        help="Select the model to use based on the video's content."
    )
    args = parser.parse_args()
    
    # --- Select the Model Path ---
    model_path = config.MODEL_REGISTRY[args.content_type]
    print(f"Using '{args.content_type}' model: {model_path}")

    video_path = args.input_video
    video_stem = Path(video_path).stem

    print(f"Starting Pointstream pipeline for: {video_path}")

    # --- Get Video Metadata at the Start ---
    props = get_video_properties(video_path)
    if not props:
        print("Could not read video properties. Aborting.")
        return
    _, fps, width, height = props
    
    video_metadata = {
        "fps": fps,
        "resolution": [width, height]
    }

    # --- Chain all pipeline stages together ---
    stage1_gen = stage_01_analyzer.run_analysis_pipeline(args.input_video)
    # FIX: Pass the selected model_path to the detection stage
    stage2_gen = stage_02_detector.run_detection_pipeline(stage1_gen, model_path)
    stage3_gen = stage_03_background.run_background_modeling_pipeline(stage2_gen, Path(args.input_video).stem)
    stage4_gen = stage_04_foreground.run_foreground_pipeline(stage3_gen, args.input_video)

    # --- Structure the Final Output ---
    scenes_list = list(stage4_gen)
    
    # DEBUG: Print scene structure
    print(f"\n--> DEBUG: Total scenes processed: {len(scenes_list)}")
    for i, scene in enumerate(scenes_list):
        print(f"    Scene {i}: keys = {list(scene.keys())}")
        if 'foreground_objects' in scene:
            print(f"    Scene {i}: foreground_objects count = {len(scene['foreground_objects'])}")
            for j, obj in enumerate(scene['foreground_objects']):
                print(f"      Object {j}: track_id={obj.get('track_id')}, class={obj.get('class_name')}, keypoint_type={obj.get('keypoint_type')}")
                if 'keypoints' in obj:
                    print(f"      Object {j}: keypoints shape = {[kp.shape if hasattr(kp, 'shape') else type(kp) for kp in obj['keypoints']]}")
    
    final_output = {
        "metadata": video_metadata,
        "scenes": scenes_list
    }

    print(f"\n--> Pipeline finished processing all scenes.")

    output_path = config.OUTPUT_DIR / f"{video_stem}_final_results.json"
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2, cls=NumpyEncoder)

    print(f"\nPipeline complete. All results saved to: {output_path}")

if __name__ == "__main__":
    main()