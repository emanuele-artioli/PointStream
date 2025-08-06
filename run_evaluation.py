#!/usr/bin/env python3
"""
PointStream Pipeline Evaluation Script

Runs comprehensive evaluation after the server and client pipelines,
computing compression metrics, quality metrics, and generating publication-ready reports.
"""

import argparse
import json
import sys
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from pointstream.evaluation.evaluator import Evaluator
    from pointstream.evaluation.metrics import VideoFrameExtractor
except ImportError as e:
    logger.error(f"Failed to import evaluation modules: {e}")
    logger.error("Make sure you're in the PointStream directory and have installed the package")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PointStream pipeline results and generate publication-ready reports"
    )
    parser.add_argument(
        "--original-video",
        type=str,
        required=True,
        help="Path to the original input video file"
    )
    parser.add_argument(
        "--json-results",
        type=str,
        required=True,
        help="Path to the _final_results.json file from the server pipeline"
    )
    parser.add_argument(
        "--reconstructed-dir",
        type=str,
        required=True,
        help="Directory containing reconstructed scene videos from client pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results and reports"
    )
    parser.add_argument(
        "--vmaf-model",
        type=str,
        default="vmaf_v0.6.1",
        help="VMAF model to use for quality evaluation"
    )
    parser.add_argument(
        "--skip-fvd",
        action="store_true",
        help="Skip FVD computation (requires GPU and takes longer)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames to use for quality metrics (for speed)"
    )

    args = parser.parse_args()

    # Validate input paths
    original_video = Path(args.original_video)
    json_results = Path(args.json_results)
    reconstructed_dir = Path(args.reconstructed_dir)
    output_dir = Path(args.output_dir)

    if not original_video.exists():
        logger.error(f"Original video not found: {original_video}")
        sys.exit(1)

    if not json_results.exists():
        logger.error(f"JSON results not found: {json_results}")
        sys.exit(1)

    if not reconstructed_dir.exists():
        logger.error(f"Reconstructed directory not found: {reconstructed_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("POINTSTREAM PIPELINE EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Original video: {original_video}")
    logger.info(f"JSON results: {json_results}")
    logger.info(f"Reconstructed scenes: {reconstructed_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max frames for quality metrics: {args.max_frames}")
    logger.info(f"Skip FVD: {args.skip_fvd}")

    start_time = time.time()

    try:
        # Initialize evaluator
        evaluator = Evaluator(
            original_video_path=str(original_video),
            json_results_path=str(json_results),
            reconstructed_scenes_dir=str(reconstructed_dir),
            vmaf_model=args.vmaf_model
        )

        logger.info("\n" + "-" * 40)
        logger.info("Running comprehensive evaluation...")
        logger.info("-" * 40)

        # Run evaluation
        results = evaluator.evaluate_pipeline(
            max_frames_for_quality=args.max_frames,
            compute_fvd=not args.skip_fvd
        )

        evaluation_time = time.time() - start_time

        logger.info(f"\nEvaluation completed in {evaluation_time:.2f} seconds")

        # Generate reports
        logger.info("\n" + "-" * 40)
        logger.info("Generating reports...")
        logger.info("-" * 40)

        # Save detailed results
        results_file = output_dir / "detailed_results.json"
        evaluator.save_results(results, str(results_file))
        logger.info(f"Detailed results saved to: {results_file}")

        # Generate summary table
        table_file = output_dir / "summary_table.csv"
        evaluator.generate_summary_table(results, str(table_file))
        logger.info(f"Summary table saved to: {table_file}")

        # Generate LaTeX table
        latex_file = output_dir / "latex_table.tex"
        evaluator.generate_latex_table(results, str(latex_file))
        logger.info(f"LaTeX table saved to: {latex_file}")

        # Generate markdown report
        markdown_file = output_dir / "evaluation_report.md"
        evaluator.generate_markdown_report(results, str(markdown_file))
        logger.info(f"Markdown report saved to: {markdown_file}")

        # Print summary to console
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        print(f"\n📁 COMPRESSION METRICS:")
        print(f"   Original size: {results.original_size_mb:.1f} MB")
        print(f"   Compressed size: {results.compressed_size_mb:.1f} MB")
        print(f"   JSON size: {results.json_size_mb:.1f} MB")
        print(f"   Compression ratio: {results.compression_ratio:.1f}x")
        print(f"   Space savings: {results.space_savings_percent:.1f}%")
        print(f"   Bits per pixel: {results.bits_per_pixel:.3f}")

        print(f"\n🎥 QUALITY METRICS:")
        print(f"   Average PSNR: {results.avg_psnr:.2f} dB")
        print(f"   Average SSIM: {results.avg_ssim:.4f}")
        print(f"   Average LPIPS: {results.avg_lpips:.4f}")
        print(f"   VMAF score: {results.vmaf_score:.2f}")
        if not args.skip_fvd:
            print(f"   FVD score: {results.fvd_score:.4f}")

        print(f"\n⚙️  PIPELINE METRICS:")
        print(f"   Number of scenes: {results.num_scenes}")
        print(f"   Number of objects: {results.num_objects}")
        print(f"   Processing time: {results.processing_time:.2f}s")
        print(f"   Reconstruction time: {results.reconstruction_time:.2f}s")

        print(f"\n📊 VIDEO INFO:")
        print(f"   Resolution: {results.original_resolution}")
        print(f"   FPS: {results.original_fps:.1f}")
        print(f"   Total frames: {results.total_frames}")
        print(f"   Duration: {results.duration_seconds:.1f}s")

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"All results saved to: {output_dir}")
        logger.info("Ready for publication!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
