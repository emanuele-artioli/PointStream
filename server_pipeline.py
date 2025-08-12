#!/usr/bin/env python3
"""
Video Processing Pipeline

This script provides a complete video processing pipeline that:
1. Video Scene Splitter - segments video into scenes and encodes them
2. Object Segmentation Inpainter - detects objects, extracts them, and inpaints backgrounds
3. Saves all processed outputs for further use

This is a production-ready pipeline for real-time streaming scenarios
with comprehensive object detection and background separation capabilities.
"""

import sys
import logging
import argparse
import time
from pathlib import Path

try:
    from video_scene_splitter import VideoSceneSplitter
    from object_segmentation_inpainter import ObjectSegmentationInpainter
    import config
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure video_scene_splitter.py and object_segmentation_inpainter.py are in the same directory")
    sys.exit(1)


def setup_logging(output_dir: Path = None, log_level: str = "INFO"):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Setup basic logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    
    # Add file handler if output directory is specified
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "pipeline_processing.log")
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def process_video_pipeline(input_video: str, output_dir: str = None, 
                         config_file: str = None, enable_saving: bool = True,
                         batch_size: int = None, yolo_model: str = None):
    """
    Process video through the complete pipeline.
    
    Args:
        input_video: Path to input video
        output_dir: Output directory for saved files
        config_file: Configuration file path
        enable_saving: Whether to save intermediate files (default: True)
        batch_size: Batch size for processing
        yolo_model: YOLO model to use
    """
    
    # Load configuration
    if config_file:
        config.load_config(config_file)
    
    output_path = Path(output_dir) if output_dir else None
    
    # Setup output directories if saving is enabled
    if enable_saving and output_path:
        output_path.mkdir(exist_ok=True)
        scene_output_dir = output_path / "scenes" 
        object_output_dir = output_path / "objects"
        scene_output_dir.mkdir(exist_ok=True)
        object_output_dir.mkdir(exist_ok=True)
    else:
        scene_output_dir = None
        object_output_dir = None
    
    # Initialize pipeline stages
    logging.info("Initializing video processing pipeline...")
    
    # Stage 1: Video Scene Splitter (enable encoding to save scene videos)
    splitter = VideoSceneSplitter(
        input_video=input_video,
        output_dir=str(scene_output_dir) if scene_output_dir else None,
        batch_size=batch_size,
        config_file=config_file,
        enable_encoding=enable_saving  # Save scene videos when saving is enabled
    )
    
    # Stage 2: Object Segmentation Inpainter (enable saving for objects and backgrounds)
    processor = ObjectSegmentationInpainter(
        output_dir=str(object_output_dir) if object_output_dir else None,
        config_file=config_file,
        enable_saving=enable_saving,  # Save object images and backgrounds
        model_name=yolo_model
    )
    
    # Pipeline statistics
    pipeline_start_time = time.time()
    scene_count = 0
    total_objects = 0
    processing_times = {'scenes': [], 'objects': []}
    
    print(f"Starting video processing pipeline...")
    print(f"Input video: {input_video}")
    print(f"Saving outputs: {enable_saving}")
    if enable_saving:
        print(f"Output directory: {output_path}")
    print(f"Processing mode: Real-time simulation")
    print("-" * 60)
    
    try:
        # Get scene generator from splitter
        scene_generator = splitter.process_video_realtime_generator()
        
        # Process scenes through object segmentation
        for processed_data in processor.process_scene_generator(scene_generator):
            
            # Handle completion status
            if processed_data.get('status') == 'complete':
                pipeline_end_time = time.time()
                pipeline_duration = pipeline_end_time - pipeline_start_time
                
                # Extract final summary
                summary = processed_data['summary']
                obj_summary = summary.get('object_processing', {})
                
                print(f"\n{'='*60}")
                print(f"PIPELINE PROCESSING COMPLETE")
                print(f"{'='*60}")
                print(f"Total scenes processed: {scene_count}")
                print(f"Total objects detected: {total_objects}")
                print(f"Pipeline duration: {pipeline_duration:.3f}s")
                print()
                print("PERFORMANCE BREAKDOWN:")
                # Calculate TRUE production time including object processing
                scene_production_time = summary.get('production_processing_time', summary.get('total_processing_time', 0))
                object_processing_time = obj_summary.get('total_processing_time', 0)
                true_production_time = scene_production_time + object_processing_time
                
                core_detection_time = summary.get('core_scene_detection_time', 0)
                frame_extraction_time = summary.get('frame_extraction_time', 0)
                stats_time = summary.get('stats_collection_time', 0)
                
                print(f"  TRUE PRODUCTION time (scene + object): {true_production_time:.3f}s")
                print(f"    - Scene processing: {scene_production_time:.3f}s")
                print(f"      - Core scene detection: {core_detection_time:.3f}s")
                print(f"      - Frame extraction: {frame_extraction_time:.3f}s")
                print(f"    - Object processing: {object_processing_time:.3f}s")
                if stats_time > 0:
                    print(f"  Optional stats collection: {stats_time:.3f}s")
                
                if enable_saving:
                    scene_encoding_time = summary.get('total_encoding_time', 0)
                    object_saving_time = obj_summary.get('total_saving_time', 0)
                    print(f"  Optional scene encoding: {scene_encoding_time:.3f}s")
                    print(f"  Optional object saving: {object_saving_time:.3f}s")
                
                print()
                print("THROUGHPUT (Production metrics - what matters for real-time):")
                production_fps = summary.get('production_frames_per_second', summary.get('frames_per_second_processing', 0))
                production_rtf = summary.get('production_real_time_factor', summary.get('real_time_factor', 0))
                obj_fps = obj_summary.get('processing_fps', 0)
                
                print(f"  Scene processing: {production_fps:.1f} frames/second")
                print(f"  Object processing: {obj_fps:.1f} scenes/second") 
                print(f"  PRODUCTION real-time factor: {production_rtf:.2f}x")
                
                if production_rtf >= 1.0:
                    print(f"  ✅ Pipeline is faster than real-time for PRODUCTION use!")
                else:
                    print(f"  ⚠️  Pipeline is slower than real-time for production use")
                    
                if stats_time > 0 or (enable_saving and summary.get('total_encoding_time', 0) > 0):
                    print(f"  ℹ️  Optional operations (stats/encoding) add overhead - disable for production")
                
                # Show file locations if saving
                if enable_saving and output_path:
                    print()
                    print("OUTPUT FILES:")
                    print(f"  Scene videos: {scene_output_dir}")
                    print(f"  Object images: {object_output_dir}")
                    print(f"  Processing logs: {output_path}")
                
                break
            
            # Handle processing errors
            elif processed_data.get('status') == 'error':
                scene_num = processed_data.get('scene_number', '?')
                error_msg = processed_data.get('error', 'Unknown error')
                print(f"❌ Error processing scene {scene_num}: {error_msg}")
                continue
            
            # Handle regular processed scenes
            else:
                scene_count += 1
                scene_num = processed_data.get('scene_number', scene_count)
                num_objects = processed_data.get('total_objects_detected', 0)
                total_objects += num_objects
                
                # Get timing information
                scene_proc_time = processed_data.get('processing_time', 0)
                original_data = processed_data.get('original_scene_data', {})
                scene_duration = original_data.get('duration', 0)
                
                print(f"Scene {scene_num:3d}: {scene_duration:5.2f}s duration, "
                      f"{num_objects} objects, {scene_proc_time:.3f}s processing")
                
                # Show detected objects
                if num_objects > 0:
                    # Extract objects from frame data (objects are in individual frames)
                    frame_data_list = processed_data.get('frames_data', [])
                    all_objects = []
                    for frame_data in frame_data_list:
                        frame_objects = frame_data.get('objects', [])
                        all_objects.extend(frame_objects)
                    
                    # Show a sample of unique objects (by class name)
                    if all_objects:
                        # Get unique objects by class name and pick highest confidence
                        unique_objects = {}
                        for obj in all_objects:
                            class_name = obj['class_name']
                            if class_name not in unique_objects or obj['confidence'] > unique_objects[class_name]['confidence']:
                                unique_objects[class_name] = obj
                        
                        obj_info = []
                        for obj in unique_objects.values():
                            name = obj['class_name']
                            conf = obj['confidence']
                            obj_info.append(f"{name}({conf:.2f})")
                        print(f"           Objects: {', '.join(obj_info)}")
                
                # In a real application, you would use the processed data here:
                # - processed_data['object_images']: Individual object images
                # - processed_data['inpainted_background']: Clean background
                # - processed_data['combined_mask']: Object location mask
                # - processed_data['objects']: Object metadata
                
                # Example: Pass to next pipeline stage
                # next_stage_processor.process(processed_data)
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        logging.info("Pipeline processing interrupted by user")
    
    except Exception as e:
        print(f"\n\nPipeline error: {e}")
        logging.error(f"Pipeline processing error: {e}")
        raise
    
    finally:
        # Clean up
        splitter.close()
        logging.info("Pipeline processing finished")


def main():
    parser = argparse.ArgumentParser(
        description="Video Processing Pipeline - Complete scene splitting and object segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full processing with all outputs saved
    python video_processing_pipeline.py input.mp4
    
    # Custom output directory
    python video_processing_pipeline.py input.mp4 --output-dir ./my_output
    
    # Process without saving files (data only)
    python video_processing_pipeline.py input.mp4 --no-saving
    
    # Custom batch size and YOLO model
    python video_processing_pipeline.py input.mp4 --batch-size 200 --yolo-model yolov8s-seg.pt
    
    # Use custom configuration
    python video_processing_pipeline.py input.mp4 --config config_custom.ini
        """
    )
    
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output-dir', default='./processing_output',
                       help='Output directory for all generated files (default: ./processing_output)')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--no-saving', action='store_true',
                       help='Disable saving files (process data only)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for scene processing')
    parser.add_argument('--yolo-model', 
                       help='YOLO model to use (e.g., yolov8n-seg.pt, yolov8s-seg.pt)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir) if args.output_dir else None
    setup_logging(output_dir, args.log_level)
    
    # Validate input
    if not Path(args.input_video).exists():
        print(f"Error: Input video not found: {args.input_video}")
        sys.exit(1)
    
    # Run pipeline
    try:
        process_video_pipeline(
            input_video=args.input_video,
            output_dir=args.output_dir,
            config_file=args.config,
            enable_saving=not args.no_saving,  # Save by default, disable with --no-saving
            batch_size=args.batch_size,
            yolo_model=args.yolo_model
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
