import argparse
import logging
import subprocess
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(args):
    """Orchestrates the entire PointStream pipeline from a single command."""
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.info("Step 1: Starting Segmenter process...")
    segmenter_cmd = [
        'python', 'scripts/segmenter.py',
        '--input-video', args.input_video,
        '--output-dir', args.output_dir
    ]
    # In a real system, these would be managed as long-running, concurrent processes.
    # For this skeleton, we run them sequentially for simplicity.
    subprocess.run(segmenter_cmd, check=True)
    logging.info("Segmenter finished.")

    logging.info("Step 2: Starting Extractor process...")
    extractor_cmd = [
        'python', 'scripts/extractor.py',
        '--input-dir', args.output_dir,
        '--quality-levels', args.quality_levels
    ]
    subprocess.run(extractor_cmd, check=True)
    logging.info("Extractor finished.")

    logging.info("Step 3: Simulating Client playback...")
    # The client would now be started with the master manifest.
    # We'll use the first quality level specified for playback simulation.
    playback_quality = args.quality_levels.split(',')[0]
    client_cmd = [
        'python', 'scripts/client.py',
        '--manifest', os.path.join(args.output_dir, 'master_manifest.json'),
        '--quality', playback_quality
    ]
    subprocess.run(client_cmd, check=True)
    logging.info("Pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full PointStream end-to-end pipeline.")
    parser.add_argument("--input-video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for all intermediate and final output.")
    parser.add_argument("--quality-levels", type=str, default="low,medium,high", help="Comma-separated list of quality levels to generate (e.g., 'medium' or 'low,high').")
    args = parser.parse_args()
    run_pipeline(args)