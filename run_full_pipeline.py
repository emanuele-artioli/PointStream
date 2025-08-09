#!/usr/bin/env python3
"""
PointStream Full Pipeline with Evaluation

Runs the complete PointStream pipeline (server → client → evaluation) 
and generates comprehensive results for publication.
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description, check=True):
    """Run a command and log the output."""
    logger.info(f"🚀 {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0 and check:
        logger.error(f"❌ {description} failed!")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
        sys.exit(1)
    
    logger.info(f"✅ {description} completed in {end_time - start_time:.2f}s")
    if result.stdout:
        logger.info(f"Output: {result.stdout.strip()}")
    
    return result, end_time - start_time


def main():
    parser = argparse.ArgumentParser(
        description="Run complete PointStream pipeline with evaluation"
    )
    parser.add_argument(
        "--input-video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--content-type",
        type=str,
        default="general",
        help="Content type for model selection"
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip server pipeline (use existing results)"
    )
    parser.add_argument(
        "--skip-client",
        action="store_true",
        help="Skip client reconstruction (use existing results)"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation phase"
    )
    parser.add_argument(
        "--evaluation-output",
        type=str,
        default="evaluation_results",
        help="Directory for evaluation results"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames for quality metrics"
    )
    parser.add_argument(
        "--skip-fvd",
        action="store_true",
        help="Skip FVD computation"
    )

    args = parser.parse_args()

    # Validate input
    input_video = Path(args.input_video)
    if not input_video.exists():
        logger.error(f"Input video not found: {input_video}")
        sys.exit(1)

    video_stem = input_video.stem
    
    logger.info("=" * 80)
    logger.info("POINTSTREAM FULL PIPELINE WITH EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Input video: {input_video}")
    logger.info(f"Content type: {args.content_type}")
    logger.info(f"Video stem: {video_stem}")

    total_start_time = time.time()

    # Phase 1: Server Pipeline
    server_time = 0
    json_results_path = Path(f"artifacts/pipeline_output/{video_stem}_final_results.json")
    
    if not args.skip_server:
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 1: SERVER PIPELINE")
        logger.info("=" * 50)
        
        server_cmd = [
            "conda", "run", "-n", "pointstream", "python3", "run_server.py",
            "--input-video", str(input_video),
            "--content-type", args.content_type
        ]
        
        _, server_time = run_command(
            server_cmd, 
            f"Running server pipeline for {video_stem}"
        )
        
        # Check if results were generated
        if not json_results_path.exists():
            logger.error(f"Server pipeline failed - no results at {json_results_path}")
            sys.exit(1)
        
        logger.info(f"✅ Server pipeline completed. Results: {json_results_path}")
    else:
        logger.info("⏭️  Skipping server pipeline")
        if not json_results_path.exists():
            logger.error(f"No existing server results found at {json_results_path}")
            sys.exit(1)

    # Phase 2: Client Pipeline
    client_time = 0
    reconstructed_dir = Path(f"{video_stem}_reconstructed_scenes")
    
    if not args.skip_client:
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 2: CLIENT RECONSTRUCTION")
        logger.info("=" * 50)
        
        client_cmd = [
            "conda", "run", "-n", "pointstream", "python3", "run_client.py",
            "--input-json", str(json_results_path),
            "--output-dir", str(reconstructed_dir)
        ]
        
        _, client_time = run_command(
            client_cmd,
            f"Running client reconstruction for {video_stem}"
        )
        
        # Check if reconstructed videos were generated
        reconstructed_scenes = list(reconstructed_dir.glob("*.mp4"))
        if not reconstructed_scenes:
            logger.error(f"Client reconstruction failed - no videos in {reconstructed_dir}")
            sys.exit(1)
        
        logger.info(f"✅ Client reconstruction completed. {len(reconstructed_scenes)} scenes in {reconstructed_dir}")
    else:
        logger.info("⏭️  Skipping client reconstruction")
        if not reconstructed_dir.exists():
            logger.error(f"No existing reconstructed directory found at {reconstructed_dir}")
            sys.exit(1)

    # Phase 3: Evaluation
    if not args.skip_evaluation:
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 3: COMPREHENSIVE EVALUATION")
        logger.info("=" * 50)
        
        eval_cmd = [
            "conda", "run", "-n", "pointstream", "python3", "run_evaluation.py",
            "--original-video", str(input_video),
            "--json-results", str(json_results_path),
            "--reconstructed-dir", str(reconstructed_dir),
            "--output-dir", args.evaluation_output,
            "--max-frames", str(args.max_frames)
        ]
        
        if args.skip_fvd:
            eval_cmd.append("--skip-fvd")
        
        _, eval_time = run_command(
            eval_cmd,
            f"Running comprehensive evaluation for {video_stem}"
        )
        
        logger.info(f"✅ Evaluation completed. Results in {args.evaluation_output}")
    else:
        logger.info("⏭️  Skipping evaluation")

    # Final Summary
    total_time = time.time() - total_start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"📹 Input video: {input_video}")
    logger.info(f"⚙️  Server processing time: {server_time:.2f}s")
    logger.info(f"🔄 Client reconstruction time: {client_time:.2f}s")
    logger.info(f"⏱️  Total pipeline time: {total_time:.2f}s")
    
    if not args.skip_evaluation:
        logger.info(f"📊 Evaluation results: {args.evaluation_output}")
        logger.info("📋 Check the following files:")
        eval_dir = Path(args.evaluation_output)
        if eval_dir.exists():
            logger.info(f"   • Summary table: {eval_dir / 'summary_table.csv'}")
            logger.info(f"   • LaTeX table: {eval_dir / 'latex_table.tex'}")
            logger.info(f"   • Detailed results: {eval_dir / 'detailed_results.json'}")
            logger.info(f"   • Report: {eval_dir / 'evaluation_report.md'}")
    
    logger.info("\n🎉 PointStream pipeline completed successfully!")
    logger.info("📊 Results are ready for publication!")


if __name__ == "__main__":
    main()
