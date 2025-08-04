"""
Tests for the client-side Reconstructor.
"""
import pytest
from pathlib import Path
import json
import cv2
import numpy as np
from pointstream.client.reconstructor import Reconstructor
from pointstream.pipeline import stage_01_analyzer, stage_02_detector, stage_03_background, stage_04_foreground
from pointstream.utils.video_utils import get_video_properties

@pytest.fixture(scope="module")
def processed_data(tmp_path_factory):
    """Runs the entire server pipeline once to generate test data."""
    output_dir = tmp_path_factory.mktemp("test_output")
    video_path = Path(__file__).parent / "data" / "DAVIS_stitched.mp4"
    if not video_path.exists(): pytest.skip(f"Test video not found at {video_path}")
    
    video_stem = video_path.stem
    json_output_path = output_dir / f"{video_stem}_final_results.json"

    props = get_video_properties(str(video_path))
    video_metadata = {"fps": props[1], "resolution": [props[2], props[3]]}

    stage1_gen = stage_01_analyzer.run_analysis_pipeline(str(video_path))
    stage2_gen = stage_02_detector.run_detection_pipeline(stage1_gen)
    stage3_gen = stage_03_background.run_background_modeling_pipeline(stage2_gen, video_stem)
    stage4_gen = stage_04_foreground.run_foreground_pipeline(stage3_gen, str(video_path))
    
    final_output = {"metadata": video_metadata, "scenes": list(stage4_gen)}

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(json_output_path, 'w') as f:
        json.dump(final_output, f, indent=2, cls=NumpyEncoder)
        
    return str(json_output_path)

def test_reconstructor_runs_and_creates_video(processed_data, tmp_path):
    """Tests if the Reconstructor can be initialized and run without errors."""
    json_path = processed_data
    output_video_path = tmp_path / "reconstructed.mp4"

    # The Reconstructor is now fully independent.
    reconstructor = Reconstructor(data_path=json_path)
    reconstructor.run(str(output_video_path))

    assert output_video_path.exists(), "Reconstructed video file was not created."
    assert output_video_path.stat().st_size > 0, "Reconstructed video file is empty."

    cap = cv2.VideoCapture(str(output_video_path))
    assert cap.isOpened(), "Could not open the reconstructed video with OpenCV."
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0, "Reconstructed video has no frames."
    cap.release()