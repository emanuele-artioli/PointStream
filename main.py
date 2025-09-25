#!/usr/bin/env python3
"""
PointStream Main Entry Point

This is the main entry point for the PointStream pipeline.
It coordinates both server and client processing, and provides
quality assessment between original and reconstructed videos.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the current directory to Python path so we can import from subdirectories
sys.path.insert(0, str(Path(__file__).parent))

# Global imports for components (will be set based on needs)
server_main = None
PointStreamPipeline = None
client_main = None
PointStreamClient = None


def import_server_components():
    """Import server components when needed."""
    global server_main, PointStreamPipeline
    try:
        from server.server import main as server_main, PointStreamPipeline
        return True
    except ImportError as e:
        print(f"Error: Cannot import server components: {e}")
        return False


def import_client_components():
    """Import client components when needed."""
    global client_main, PointStreamClient
    try:
        from client.client import main as client_main, PointStreamClient
        return True
    except ImportError as e:
        print(f"Error: Cannot import client components: {e}")
        return False


def setup_logging(log_level: str = "INFO"):
    """Setup unified logging for both server and client."""
    logging.getLogger().handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '🔥 %(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    
    print(f"🚀 PointStream Unified Pipeline Starting - Log Level: {log_level.upper()}")
    print("=" * 80)


def run_model_training(training_videos_dir: str, log_level: str = "INFO") -> bool:
    """Run model training on a collection of training videos."""
    print("\n" + "="*60)
    print("PHASE 2: MODEL TRAINING (TRAINING VIDEOS → TRAINED MODELS)")
    print("="*60)
    
    try:
        import subprocess
        import sys
        
        # Run the training pipeline on the training videos directory
        training_command = [
            sys.executable, 'train_pipeline.py',
            training_videos_dir,
            '--models-dir', './models',
            '--log-level', log_level
        ]
        
        print(f"🎓 Training models on videos from: {training_videos_dir}")
        print(f"📋 Command: {' '.join(training_command)}")
        
        result = subprocess.run(training_command, cwd=Path(__file__).parent, check=True)
        
        print("✅ Model training completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Model training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False


def check_models_exist() -> bool:
    """Check if all trained models already exist."""
    models_path = Path("./models")
    animal_model = models_path / "animal_cgan.pth"
    human_model = models_path / "human_cgan.pth" 
    other_model = models_path / "other_cgan.pth"
    
    return animal_model.exists() and human_model.exists() and other_model.exists()


def get_missing_models() -> list:
    """Get list of missing model names."""
    models_path = Path("./models")
    animal_model = models_path / "animal_cgan.pth"
    human_model = models_path / "human_cgan.pth" 
    other_model = models_path / "other_cgan.pth"
    
    missing = []
    if not human_model.exists():
        missing.append('human')
    if not animal_model.exists():
        missing.append('animal')
    if not other_model.exists():
        missing.append('other')
    
    return missing


def run_server_processing(input_video: str, metadata_dir: str, 
                         server_config: str = None) -> bool:
    """Run server-side processing."""
    print("\n" + "="*60)
    print("PHASE 1: SERVER PROCESSING (VIDEO → METADATA)")
    print("="*60)
    
    try:
        # Initialize server pipeline
        pipeline = PointStreamPipeline(config_file=server_config)
        
        # Process video
        summary = pipeline.process_video(
            input_video=input_video,
            output_dir=metadata_dir,
            enable_saving=True
        )
        
        # Print server summary
        print(f"\n📊 SERVER PROCESSING COMPLETE")
        print(f"Processed scenes: {summary['processed_scenes']}")
        print(f"Simple scenes: {summary['simple_scenes']}")
        print(f"Complex scenes: {summary['complex_scenes']}")
        print(f"Total processing time: {summary['total_processing_time']:.2f}s")
        print(f"Metadata saved to: {metadata_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Server processing failed: {e}")
        return False


def run_client_processing(metadata_dir: str, output_dir: str, 
                         client_config: str = None, 
                         train_models: bool = True) -> bool:
    """Run client-side processing."""
    print("\n" + "="*60)
    print("PHASE 2: CLIENT PROCESSING (METADATA → RECONSTRUCTED VIDEO)")
    print("="*60)
    
    try:
        # Initialize client pipeline
        client = PointStreamClient(config_file=client_config)
        
        # Process metadata
        summary = client.process_metadata(
            metadata_dir=metadata_dir,
            output_dir=output_dir,
            train_models=train_models
        )
        
        # Print client summary
        print(f"\n🔧 CLIENT PROCESSING COMPLETE")
        print(f"Processed scenes: {summary['processed_scenes']}")
        print(f"Successful scenes: {summary['successful_scenes']}")
        print(f"Failed scenes: {summary['failed_scenes']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total reconstruction time: {summary['total_reconstruction_time']:.2f}s")
        print(f"Reconstructed videos saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Client processing failed: {e}")
        return False


def assess_reconstruction_quality(original_video: str, reconstructed_dir: str, 
                                client_config: str = None) -> bool:
    """Assess quality of reconstructed videos against original."""
    print("\n" + "="*60)
    print("PHASE 3: QUALITY ASSESSMENT (ORIGINAL vs RECONSTRUCTED)")
    print("="*60)
    
    try:
        # Initialize client for quality assessment
        client = PointStreamClient(config_file=client_config)
        
        # Find reconstructed videos
        reconstructed_path = Path(reconstructed_dir)
        reconstructed_videos = list(reconstructed_path.glob("*_reconstructed.mp4"))
        
        if not reconstructed_videos:
            print(f"⚠️ No reconstructed videos found in {reconstructed_dir}")
            return False
        
        print(f"📊 Assessing quality for {len(reconstructed_videos)} reconstructed videos")
        
        quality_results = []
        
        for reconstructed_video in reconstructed_videos:
            print(f"\n🔍 Assessing: {reconstructed_video.name}")
            
            quality_result = client.assess_quality(
                original_video=original_video,
                reconstructed_video=str(reconstructed_video)
            )
            
            quality_results.append(quality_result)
            
            # Print individual results
            metrics = quality_result.get('quality_metrics', {})
            if 'ssim' in metrics:
                print(f"   SSIM: {metrics['ssim']['mean']:.4f}")
            if 'psnr' in metrics:
                print(f"   PSNR: {metrics['psnr']['mean']:.2f} dB")
            if 'lpips' in metrics:
                print(f"   LPIPS: {metrics['lpips']['mean']:.4f}")
        
        # Calculate overall statistics
        if quality_results:
            all_ssim = [r.get('quality_metrics', {}).get('ssim', {}).get('mean', 0) 
                       for r in quality_results if 'ssim' in r.get('quality_metrics', {})]
            all_psnr = [r.get('quality_metrics', {}).get('psnr', {}).get('mean', 0) 
                       for r in quality_results if 'psnr' in r.get('quality_metrics', {})]
            
            print(f"\n📈 OVERALL QUALITY METRICS:")
            if all_ssim:
                import numpy as np
                print(f"Average SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
            if all_psnr:
                import numpy as np
                print(f"Average PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality assessment failed: {e}")
        return False


def main():
    """Main entry point for the unified PointStream pipeline."""
    parser = argparse.ArgumentParser(
        description="PointStream Unified Pipeline - Server, Client, and Quality Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with model training (if models don't exist)
    python main.py input.mp4 --full-pipeline --training-videos ./training_dataset
    
    # Full pipeline using existing models only
    python main.py input.mp4 --full-pipeline --no-training
    
    # Full pipeline forcing model retraining
    python main.py input.mp4 --full-pipeline --training-videos ./training_dataset --force-retrain
    
    # Server processing only
    python main.py input.mp4 --server-only --metadata-dir ./metadata
    
    # Client processing only (requires existing metadata and models)
    python main.py --client-only --metadata-dir ./metadata --output-dir ./reconstructed
    
    # Quality assessment only
    python main.py input.mp4 --assess-quality ./reconstructed
    
    # Custom configurations
    python main.py input.mp4 --full-pipeline --training-videos ./training_dataset --server-config ./server/config.ini --client-config ./client/config.ini
        """
    )
    
    parser.add_argument('input_video', nargs='?', help='Path to input video file')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--full-pipeline', action='store_true',
                           help='Run complete pipeline: server → client → quality assessment')
    mode_group.add_argument('--server-only', action='store_true',
                           help='Run server processing only')
    mode_group.add_argument('--client-only', action='store_true',
                           help='Run client processing only')
    mode_group.add_argument('--assess-quality', metavar='RECONSTRUCTED_DIR',
                           help='Assess quality against reconstructed videos in directory')
    
    # Directory options
    parser.add_argument('--metadata-dir', default='./pointstream_metadata',
                       help='Directory for metadata (server output / client input)')
    parser.add_argument('--output-dir', default='./pointstream_reconstructed',
                       help='Output directory for reconstructed videos')
    
    # Configuration files
    parser.add_argument('--server-config', 
                       help='Path to server configuration file')
    parser.add_argument('--client-config',
                       help='Path to client configuration file')
    
    # Training options
    parser.add_argument('--training-videos', metavar='TRAINING_DIR',
                       help='Directory containing training videos for model training (required if models do not exist)')
    parser.add_argument('--no-training', action='store_true',
                       help='Skip model training entirely (use existing models only)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining of models even if they exist')
    
    # Logging
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate arguments
    if (args.full_pipeline or args.server_only or args.assess_quality) and not args.input_video:
        print("Error: Input video is required for server processing and quality assessment")
        sys.exit(1)
    
    if args.input_video and not Path(args.input_video).exists():
        print(f"Error: Input video not found: {args.input_video}")
        sys.exit(1)
    
    if args.client_only and not Path(args.metadata_dir).exists():
        print(f"Error: Metadata directory not found: {args.metadata_dir}")
        sys.exit(1)
    
    success = True
    
    try:
        if args.full_pipeline:
            # Import both server and client components
            if not import_server_components():
                print("❌ Failed to import server components")
                sys.exit(1)
            if not import_client_components():
                print("❌ Failed to import client components")
                sys.exit(1)
                
            # Run complete pipeline
            print("🚀 Starting FULL POINTSTREAM PIPELINE")
            print("This will run: Server Processing → Model Training (if needed) → Client Processing → Quality Assessment")
            
            # Phase 1: Server Processing
            success = run_server_processing(
                args.input_video, 
                args.metadata_dir, 
                args.server_config
            )
            
            if not success:
                print("❌ Server processing failed, stopping pipeline")
                sys.exit(1)
            
            # Phase 2: Model Training (if needed)
            models_exist = check_models_exist()
            missing_models = get_missing_models()
            
            if args.force_retrain or (missing_models and not args.no_training):
                if not args.training_videos:
                    print("❌ Error: Missing models detected and no training videos directory specified.")
                    print(f"   Missing models: {', '.join(missing_models)}")
                    print("   Please provide --training-videos <directory> or use --no-training to skip model training.")
                    sys.exit(1)
                
                if not Path(args.training_videos).exists():
                    print(f"❌ Error: Training videos directory not found: {args.training_videos}")
                    sys.exit(1)

                print("\n" + "="*60)
                print("PHASE 2: MODEL TRAINING (TRAINING MISSING MODELS)")
                print("="*60)
                print(f"🎓 Missing models detected: {', '.join(missing_models)}")
                print("🎓 Training models on videos from:", args.training_videos)
                
                success = run_model_training(args.training_videos, args.log_level)
                
                if not success:
                    print("❌ Model training failed, stopping pipeline")
                    sys.exit(1)
            else:
                if models_exist:
                    print("\n" + "="*60)
                    print("PHASE 2: MODEL TRAINING (SKIPPED - ALL MODELS EXIST)")
                    print("="*60)
                    print("✅ Found all existing trained models, skipping training phase")
                else:
                    print("\n" + "="*60)
                    print("PHASE 2: MODEL TRAINING (SKIPPED - NO-TRAINING FLAG)")
                    print("="*60)
                    print("⏭️ Skipping model training due to --no-training flag")
                    if missing_models:
                        print(f"⚠️  Warning: Missing models: {', '.join(missing_models)}")
                        print("   Client processing may fail for missing model categories")
            
            # Phase 3: Client Processing (reconstruction only, no training)
            success = run_client_processing(
                args.metadata_dir, 
                args.output_dir, 
                args.client_config,
                train_models=False  # Never train during client phase in full pipeline
            )
            
            if not success:
                print("❌ Client processing failed, stopping pipeline")
                sys.exit(1)
            
            # Phase 4: Quality Assessment
            success = assess_reconstruction_quality(
                args.input_video, 
                args.output_dir, 
                args.client_config
            )
            
            if not success:
                print("⚠️ Quality assessment failed, but pipeline completed")
        
        elif args.server_only:
            # Import server components only
            if not import_server_components():
                print("❌ Failed to import server components")
                sys.exit(1)
                
            # Server processing only
            print("🔥 Starting SERVER-ONLY processing")
            success = run_server_processing(
                args.input_video, 
                args.metadata_dir, 
                args.server_config
            )
        
        elif args.client_only:
            # Import client components only
            if not import_client_components():
                print("❌ Failed to import client components")
                sys.exit(1)
                
            # Client processing only
            print("🔧 Starting CLIENT-ONLY processing")
            success = run_client_processing(
                args.metadata_dir, 
                args.output_dir, 
                args.client_config,
                not args.no_training
            )
        
        elif args.assess_quality:
            # Import client components only (needed for quality assessment)
            if not import_client_components():
                print("❌ Failed to import client components")
                sys.exit(1)
                
            # Quality assessment only
            print("📊 Starting QUALITY ASSESSMENT")
            success = assess_reconstruction_quality(
                args.input_video, 
                args.assess_quality, 
                args.client_config
            )
        
        # Final summary
        if success:
            print(f"\n{'='*80}")
            print("🎉 POINTSTREAM PIPELINE COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print("❌ POINTSTREAM PIPELINE FAILED")
            print(f"{'='*80}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
