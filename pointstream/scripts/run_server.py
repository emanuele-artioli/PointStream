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
from pointstream.codecs import encode_complex_scene_av1


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

    # --- Chain all pipeline stages together with content type ---
    stage1_gen = stage_01_analyzer.run_analysis_pipeline(args.input_video)
    # FIX: Pass the selected model_path and content_type to the detection stage
    stage2_gen = stage_02_detector.run_detection_pipeline(stage1_gen, model_path, args.content_type)
    stage3_gen = stage_03_background.run_background_modeling_pipeline(stage2_gen, Path(args.input_video).stem, args.content_type)
    stage4_gen = stage_04_foreground.run_foreground_pipeline(stage3_gen, args.input_video)

    # --- Structure the Final Output ---
    scenes_list = []
    
    for scene in stage4_gen:
        # Handle complex scenes with AV1 encoding
        if scene['motion_type'] == 'COMPLEX':
            print(f"\n--> Encoding complex scene {scene['scene_index']} with AV1...")
            
            # Get original frames back for AV1 encoding
            if 'frames' not in scene:
                print(f"    -> Warning: No frames available for complex scene {scene['scene_index']}")
                scene['av1_encoded_path'] = None
                scene['av1_file_size'] = 0
            else:
                av1_output_path = config.OUTPUT_DIR / f"{video_stem}_scene_{scene['scene_index']}_complex.webm"
                av1_path = encode_complex_scene_av1(
                    scene['frames'], 
                    str(av1_output_path), 
                    fps=video_metadata['fps']
                )
                scene['av1_encoded_path'] = av1_path
                scene['av1_file_size'] = Path(av1_path).stat().st_size if av1_path else 0
                
                # Remove frames to save memory after encoding
                del scene['frames']
        
        scenes_list.append(scene)
    
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