#!/usr/bin/env python3
"""
End-to-End Training Pipeline for PointStream

This script orchestrates the full training pipeline:
1. It takes one or more videos as input.
2. Runs the server-side processing on each video to extract metadata and object images.
3. Aggregates all the extracted data.
4. Runs the client-side model training using the aggregated data.
5. Saves the trained models to a specified directory.

Features:
- Caches processed video data to avoid reprocessing
- Supports incremental dataset updates
- Only processes new videos not in cache
"""

import argparse
import logging
import sys
import tempfile
import subprocess
from pathlib import Path
import shutil

# Add the project root to the Python path to allow importing from client and server
sys.path.insert(0, str(Path(__file__).parent.absolute()))

try:
    from client.client import PointStreamClient, setup_logging as setup_client_logging
    from utils.training_cache import TrainingDataCache
except ImportError as e:
    print(f"Error: Failed to import client components: {e}")
    sys.exit(1)


def setup_pipeline_logging(log_level: str = "INFO"):
    """Setup logging for the training pipeline."""
    logging.getLogger().handlers.clear()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        '🚂 %(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(
        description="PointStream End-to-End Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'input_path',
        type=str,
        help="Path to a single video file or a directory containing video files."
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models',
        help="Directory to save the final trained models."
    )
    parser.add_argument(
        '--metadata-dir',
        type=str,
        default=None,
        help="Directory to store intermediate metadata. If not provided, a temporary directory will be used."
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level for the pipeline.'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help="Directory to store training data cache. If not provided, uses default config location."
    )
    parser.add_argument(
        '--skip-cache',
        action='store_true',
        help="Skip cache and reprocess all videos (useful for testing or cache invalidation)."
    )
    parser.add_argument(
        '--cache-stats',
        action='store_true',
        help="Show cache statistics and exit."
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help="Clear the training cache and exit."
    )
    args = parser.parse_args()

    setup_pipeline_logging(args.log_level)
    logging.info("🚀 Starting End-to-End Training Pipeline...")

    # Initialize training cache
    cache = TrainingDataCache(cache_dir=args.cache_dir)
    
    # Handle cache management commands
    if args.cache_stats:
        stats = cache.get_cache_stats()
        logging.info("📊 Training Cache Statistics:")
        logging.info(f"   Total videos: {stats['total_videos']}")
        logging.info(f"   Total size: {stats['total_size_mb']:.1f} MB")
        logging.info(f"   Total metadata files: {stats['total_metadata_files']}")
        logging.info(f"   Cache directory: {stats['cache_directory']}")
        logging.info(f"   Created: {stats['created']}")
        logging.info(f"   Last updated: {stats['last_updated']}")
        return
    
    if args.clear_cache:
        cache.clear_cache()
        logging.info("🗑️  Training cache cleared")
        return

    # Determine metadata directory
    if args.metadata_dir:
        metadata_path = Path(args.metadata_dir)
        is_temp_metadata = False
    else:
        temp_dir = tempfile.TemporaryDirectory()
        metadata_path = Path(temp_dir.name)
        is_temp_metadata = True

    metadata_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using metadata directory: {metadata_path}")

    try:
        # --- PHASE 1: SERVER PROCESSING ---
        logging.info("--- PHASE 1: SERVER PROCESSING (VIDEO -> METADATA) ---")

        # Discover video files
        input_p = Path(args.input_path)
        if input_p.is_dir():
            video_files = [f for f in input_p.iterdir() if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
        elif input_p.is_file():
            video_files = [input_p]
        else:
            logging.error(f"Input path is not a valid file or directory: {args.input_path}")
            sys.exit(1)

        if not video_files:
            logging.error(f"No video files found in: {args.input_path}")
            sys.exit(1)

        logging.info(f"Found {len(video_files)} video(s) to process.")

        # Check cache for existing videos
        if args.skip_cache:
            videos_to_process = video_files
            logging.info("🔄 Skipping cache - will reprocess all videos")
        else:
            videos_to_process = cache.get_uncached_videos([str(f) for f in video_files])
            videos_to_process = [Path(v) for v in videos_to_process]
            
            cached_count = len(video_files) - len(videos_to_process)
            if cached_count > 0:
                logging.info(f"📋 Found {cached_count} videos already cached")
            if videos_to_process:
                logging.info(f"🎬 Need to process {len(videos_to_process)} new videos")
            else:
                logging.info("✅ All videos already cached - no processing needed")

        # Process uncached videos
        for i, video_file in enumerate(videos_to_process):
            logging.info(f"Processing video {i+1}/{len(videos_to_process)}: {video_file.name}")
            
            # Create temporary directory for this video's metadata
            with tempfile.TemporaryDirectory(prefix=f"pointstream_{video_file.stem}_") as temp_video_dir:
                try:
                    server_command = [
                        sys.executable, 'main.py',
                        str(video_file),
                        '--server-only',
                        '--metadata-dir', temp_video_dir,
                        '--log-level', args.log_level
                    ]
                    subprocess.run(server_command, check=True, capture_output=True, text=True)
                    
                    # Add processed video to cache
                    processing_stats = {
                        'processed_at': logging.Formatter().formatTime(logging.makeLogRecord({})),
                        'processing_successful': True
                    }
                    cache.add_video_to_cache(str(video_file), temp_video_dir, processing_stats)
                    
                    logging.info(f"✅ Successfully processed and cached {video_file.name}")
                    
                except subprocess.CalledProcessError as e:
                    logging.error(f"❌ Server processing failed for {video_file.name}.")
                    logging.error(f"Command: {' '.join(e.cmd)}")
                    logging.error(f"Stderr: {e.stderr}")
                    sys.exit(1)

        # Collect all training data from cache
        logging.info("📦 Collecting training data from cache...")
        training_metadata_dir, collection_stats = cache.collect_all_training_data()
        
        logging.info(f"📊 Training data collection stats:")
        logging.info(f"   Videos: {collection_stats['total_videos']}")
        logging.info(f"   Metadata files: {collection_stats['total_metadata_files']}")
        
        # Update metadata path to use collected data
        metadata_path = Path(training_metadata_dir)
        is_temp_metadata = True  # This is now a temporary collection

        logging.info("--- SERVER PROCESSING COMPLETE ---")

        # --- PHASE 2: MODEL TRAINING ---
        logging.info("--- PHASE 2: MODEL TRAINING (METADATA -> MODELS) ---")

        # Check if models already exist
        models_path = Path("./models")
        animal_model = models_path / "animal_cgan.pth"
        human_model = models_path / "human_cgan.pth" 
        other_model = models_path / "other_cgan.pth"
        
        models_exist = animal_model.exists() or human_model.exists() or other_model.exists()
        
        if models_exist:
            logging.info("✅ Found existing trained models, skipping training phase")
            training_results = {'status': 'skipped', 'reason': 'models_exist'}
            model_results = {}  # Initialize empty dict when skipping training
        else:
            logging.info("🎓 No existing models found, starting training...")
            # We need to setup the client's logging separately if we want to see its output
            # This is a bit of a hack, but it re-initializes the logger for the client's format
            setup_client_logging(args.log_level)

            client = PointStreamClient()
            training_results = client.train_models(metadata_dir=str(metadata_path))

            # Handle case where training_results might not be a dictionary
            if not isinstance(training_results, dict):
                logging.warning(f"Training returned unexpected result type: {type(training_results)}")
                training_results = {}
            
            # Filter out non-model entries (like processing_time added by decorator)
            model_results = {k: v for k, v in training_results.items() 
                            if k not in ['processing_time'] and isinstance(v, dict)}
            
            # Check for training errors
            training_errors = []
            for model_type, result in model_results.items():
                if result.get('error'):
                    training_errors.append(f"{model_type}: {result['error']}")
            
            if training_errors:
                logging.warning("⚠️ Some model training had issues:")
                for error in training_errors:
                    logging.warning(f"   {error}")
                logging.info("Continuing with available models...")
            elif not model_results:
                logging.info("ℹ️ No training data found - will use existing models if available")

        logging.info("✅ Model training completed successfully.")

        # --- PHASE 3: OUTPUT HANDLING ---
        logging.info("--- PHASE 3: SAVING TRAINED MODELS ---")
        output_models_path = Path(args.models_dir)
        output_models_path.mkdir(parents=True, exist_ok=True)

        # The default paths are in the config, let's find them
        from utils import config as pointstream_config
        default_model_dir = Path(pointstream_config.get_str('models', 'model_dir', './models'))

        saved_models = []
        # Use the filtered model_results instead of training_results
        for model_type, result in model_results.items():
            model_path_str = result.get('model_path')
            if model_path_str:
                model_path = Path(model_path_str)
                if model_path.exists():
                    dest_path = output_models_path / model_path.name
                    shutil.move(str(model_path), str(dest_path))
                    logging.info(f"Moved {model_path.name} to {dest_path}")
                    saved_models.append(dest_path)

        logging.info("--- TRAINING COMPLETE ---")

        # --- CLEANUP ---
        if is_temp_metadata:
            logging.info(f"🗑️ Cleaning up temporary metadata collection directory: {metadata_path}")
            import shutil
            shutil.rmtree(metadata_path)

        # Final cache statistics
        logging.info("📊 Final cache statistics:")
        stats = cache.get_cache_stats()
        for key, value in stats.items():
            logging.info(f"   {key.replace('_', ' ').title()}: {value}")

        logging.info("🎉 Training pipeline completed successfully!")

    except Exception as e:
        logging.error(f"❌ Training pipeline failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
