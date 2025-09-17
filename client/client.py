#!/usr/bin/env python3
"""
PointStream Client Pipeline - Video Reconstruction using Generative Models

This module implements the client-side processing pipeline for the PointStream system.
It reconstructs videos from metadata, panoramas, and keypoints using generative models.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np
import cv2
from PIL import Image

# Suppress warnings before importing other modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Import PointStream utilities
try:
    from utils.decorators import track_performance
    from utils import config
    from client.scripts.background_reconstructor import BackgroundReconstructor
    from client.scripts.object_generator import ObjectGenerator
    from client.scripts.frame_composer import FrameComposer
    from client.scripts.video_assembler import VideoAssembler
    from client.scripts.quality_assessor import QualityAssessor
    from client.scripts.model_trainer import ModelTrainer
    from client.scripts.demuxer import MetadataDemuxer
except ImportError as e:
    logging.error(f"Failed to import PointStream client components: {e}")
    print("Error: Cannot import required PointStream client components")
    sys.exit(1)


class PointStreamClient:
    """
    PointStream client pipeline for video reconstruction using generative models.
    
    The client reconstructs videos from metadata without accessing the original video,
    using panoramas, homographies, keypoints, and generative models.
    """
    
    def __init__(self, config_file: str = None):
        """Initialize the client pipeline with configuration."""
        # Load configuration
        if config_file:
            config.load_config(config_file)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"🔥 Client initialized on device: {self.device}")
        
        # Initialize components
        logging.info("🚀 Initializing PointStream client components...")
        self._initialize_components()
        
        # Statistics tracking
        self.processed_scenes = 0
        self.reconstruction_times = []
        self.quality_scores = []
        
        logging.info("✅ PointStream Client initialized")
        logging.info("🎯 Workflow: Background Reconstruction → Object Generation → Frame Composition → Video Assembly")
    
    def _initialize_components(self):
        """Initialize all client processing components."""
        try:
            logging.info("🔧 Loading client components...")
            
            # Initialize components
            logging.info("   🖼️  Initializing Background Reconstructor...")
            self.background_reconstructor = BackgroundReconstructor()
            
            logging.info("   🤖 Initializing Object Generator...")
            self.object_generator = ObjectGenerator(device=self.device)
            
            logging.info("   🎨 Initializing Frame Composer...")
            self.frame_composer = FrameComposer()
            
            logging.info("   🎬 Initializing Video Assembler...")
            self.video_assembler = VideoAssembler()
            
            logging.info("   📊 Initializing Quality Assessor...")
            self.quality_assessor = QualityAssessor()
            
            logging.info("   🎓 Initializing Model Trainer...")
            self.model_trainer = ModelTrainer(device=self.device)
            
            logging.info("   📦 Initializing Metadata Demuxer...")
            self.demuxer = MetadataDemuxer()
            
            logging.info("✅ All client components loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize client components: {e}")
            raise
    
    @track_performance
    def train_models(self, metadata_dir: str, training_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train generative models using extracted object data.
        
        Args:
            metadata_dir: Directory containing scene metadata and object data
            training_config: Optional training configuration override
            
        Returns:
            Training results summary
        """
        logging.info("🎓 Starting model training phase...")
        
        # Load training data from metadata
        training_data = self._load_training_data(metadata_dir)
        
        # Train models for each object type
        training_results = {}
        
        # Train human model
        if training_data.get('human_objects'):
            logging.info("👤 Training human generation model...")
            human_result = self.model_trainer.train_human_model(
                training_data['human_objects'], 
                config_override=training_config
            )
            training_results['human'] = human_result
        
        # Train animal model
        if training_data.get('animal_objects'):
            logging.info("🐾 Training animal generation model...")
            animal_result = self.model_trainer.train_animal_model(
                training_data['animal_objects'], 
                config_override=training_config
            )
            training_results['animal'] = animal_result
        
        # Train other objects model
        if training_data.get('other_objects'):
            logging.info("📦 Training other objects generation model...")
            other_result = self.model_trainer.train_other_model(
                training_data['other_objects'], 
                config_override=training_config
            )
            training_results['other'] = other_result
        
        logging.info("✅ Model training completed")
        return training_results
    
    def _load_training_data(self, metadata_dir: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Load and organize training data by object type, grouped by object ID."""
        metadata_path = Path(metadata_dir)
        
        # Organize objects by semantic category and then by object ID
        human_objects = {}
        animal_objects = {}
        other_objects = {}
        
        # Scan for .pzm files first, fallback to .json
        metadata_files = list(metadata_path.glob("scene_*_metadata.pzm"))
        if not metadata_files:
            metadata_files = list(metadata_path.glob("scene_*_metadata.json"))

        for metadata_file in metadata_files:
            scene_data = self._load_scene_metadata(metadata_file)
            scene_number = scene_data.get('scene_number', 1)
            objects_dir = metadata_path / "objects" / f"scene_{scene_number:04d}"
            
            # Extract objects by category - check multiple possible locations
            objects = []
            if 'objects' in scene_data:
                objects = scene_data['objects']
            elif 'keypoint_result' in scene_data and 'objects' in scene_data['keypoint_result']:
                objects = scene_data['keypoint_result']['objects']
            
            for obj in objects:
                category = obj.get('semantic_category', 'other')
                track_id = obj.get('track_id')
                frame_index = obj.get('frame_index', 0)

                if track_id is None:
                    continue # Skip objects without a track ID

                # Create object ID based on track_id for grouping
                object_id = f"track_{track_id}"
                
                # Enhance object data with file paths and computed data
                enhanced_obj = obj.copy()
                
                # Find the corresponding image file
                image_filename = f"{category}_track_{track_id}_frame_{frame_index:04d}.png"
                image_path = objects_dir / image_filename
                
                if image_path.exists():
                    enhanced_obj['cropped_image'] = str(image_path)
                    
                    # Generate appearance vector (simplified - could be enhanced with actual feature extraction)
                    enhanced_obj['v_appearance'] = self._generate_appearance_vector(str(image_path))
                    
                    # Convert keypoints to the expected format for pose
                    if 'keypoints' in obj and 'points' in obj['keypoints']:
                        enhanced_obj['p_pose'] = {
                            'points': obj['keypoints']['points'],
                            'confidence_scores': obj['keypoints'].get('confidence_scores', [])
                        }

                    if category == 'human':
                        human_objects.setdefault(object_id, []).append(enhanced_obj)
                    elif category == 'animal':
                        animal_objects.setdefault(object_id, []).append(enhanced_obj)
                    else:
                        other_objects.setdefault(object_id, []).append(enhanced_obj)

        # Sort object appearances by frame number to ensure consistent ordering
        for group in [human_objects, animal_objects, other_objects]:
            for object_id in group:
                # Sort by frame index
                group[object_id].sort(key=lambda o: o.get('frame_index', 0))
        
        total_human_instances = sum(len(v) for v in human_objects.values())
        total_animal_instances = sum(len(v) for v in animal_objects.values())
        total_other_instances = sum(len(v) for v in other_objects.values())

        logging.info(f"📚 Training data loaded and grouped by object ID:")
        logging.info(f"   👤 Human objects: {len(human_objects)} unique, {total_human_instances} instances")
        logging.info(f"   🐾 Animal objects: {len(animal_objects)} unique, {total_animal_instances} instances")
        logging.info(f"   📦 Other objects: {len(other_objects)} unique, {total_other_instances} instances")
        
        return {
            'human_objects': human_objects,
            'animal_objects': animal_objects,
            'other_objects': other_objects
        }
    
    def _generate_appearance_vector(self, image_path: str) -> np.ndarray:
        """
        Generate a simple appearance vector from an image.
        This is a placeholder - could be enhanced with proper feature extraction.
        """
        import cv2
        import numpy as np
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return np.random.randn(2048).astype(np.float32)  # Fallback to random
            
            # Simple appearance features: resize, flatten, normalize
            img_resized = cv2.resize(img, (64, 64))  
            features = img_resized.flatten().astype(np.float32)
            
            # Pad or truncate to 2048 dimensions
            if len(features) > 2048:
                features = features[:2048]
            elif len(features) < 2048:
                padding = np.zeros(2048 - len(features), dtype=np.float32)
                features = np.concatenate([features, padding])
                
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            logging.warning(f"Failed to generate appearance vector for {image_path}: {e}")
            return np.random.randn(2048).astype(np.float32)  # Fallback to random
    
    @track_performance
    def reconstruct_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct a complete scene from metadata.
        
        Args:
            scene_data: Scene metadata including panorama path, homographies, objects
            
        Returns:
            Reconstructed scene data with generated frames
        """
        scene_number = scene_data.get('scene_number', 0)
        logging.info(f"🎬 Reconstructing scene {scene_number}")
        
        try:
            # STEP 1: Reconstruct backgrounds for each frame
            logging.info(f"🖼️  Scene {scene_number}: Step 1/4 - Background reconstruction...")
            step_start = time.time()
            background_result = self.background_reconstructor.reconstruct_backgrounds(scene_data)
            step_time = time.time() - step_start
            logging.info(f"   ✅ Background reconstruction completed in {step_time:.1f}s")
            
            # STEP 2: Generate objects for each frame
            logging.info(f"🤖 Scene {scene_number}: Step 2/4 - Object generation...")
            step_start = time.time()
            object_result = self.object_generator.generate_objects(scene_data)
            step_time = time.time() - step_start
            logging.info(f"   ✅ Object generation completed in {step_time:.1f}s")
            
            # Get video properties from metadata
            video_properties = scene_data.get('video_properties', {})
            fps = video_properties.get('fps', 25.0)
            resolution = video_properties.get('resolution')

            # STEP 3: Compose frames by overlaying objects on backgrounds
            logging.info(f"🎨 Scene {scene_number}: Step 3/4 - Frame composition...")
            step_start = time.time()
            frame_result = self.frame_composer.compose_frames(
                background_result['backgrounds'],
                object_result['generated_objects'],
                output_resolution=resolution
            )
            step_time = time.time() - step_start
            logging.info(f"   ✅ Frame composition completed in {step_time:.1f}s")

            # STEP 4: Assemble frames into video
            logging.info(f"🎬 Scene {scene_number}: Step 4/4 - Video assembly...")
            step_start = time.time()
            video_result = self.video_assembler.assemble_video(
                frame_result['composed_frames'],
                fps,
                scene_number
            )
            step_time = time.time() - step_start
            logging.info(f"   ✅ Video assembly completed in {step_time:.1f}s")
            
            # Combine all results
            reconstruction_result = {
                'scene_number': scene_number,
                'status': 'success',
                'background_result': background_result,
                'object_result': object_result,
                'frame_result': frame_result,
                'video_result': video_result,
                'reconstructed_video_path': video_result.get('video_path'),
                'frame_count': len(frame_result.get('composed_frames', [])),
                'processing_time': sum([
                    background_result.get('processing_time', 0),
                    object_result.get('processing_time', 0),
                    frame_result.get('processing_time', 0),
                    video_result.get('processing_time', 0)
                ])
            }
            
            logging.info(f"🎉 Scene {scene_number} reconstruction completed successfully")
            return reconstruction_result
            
        except Exception as e:
            logging.error(f"Scene {scene_number} reconstruction failed: {e}")
            return {
                'scene_number': scene_number,
                'status': 'error',
                'error': str(e)
            }
    
    @track_performance
    def process_metadata(self, metadata_dir: str, output_dir: str = None, 
                        train_models: bool = True) -> Dict[str, Any]:
        """
        Process complete metadata directory to reconstruct videos.
        
        Args:
            metadata_dir: Directory containing scene metadata and object data
            output_dir: Output directory for reconstructed videos
            train_models: Whether to train generative models first
            
        Returns:
            Processing summary
        """
        logging.info(f"🎬 Starting PointStream client processing: {metadata_dir}")
        
        # Setup output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            logging.info(f"📁 Output directory: {output_path}")
        
        metadata_path = Path(metadata_dir)
        
        try:
            # PHASE 1: Model Training (if enabled)
            if train_models:
                logging.info("🎓 Phase 1: Training generative models...")
                training_results = self.train_models(metadata_dir)
                logging.info("✅ Model training phase completed")
            else:
                logging.info("⏭️  Skipping model training - using existing models")
                training_results = {}
            
            # PHASE 2: Scene Reconstruction
            logging.info("🎬 Phase 2: Scene reconstruction...")
            
            # Find all scene metadata files (.pzm preferred)
            scene_files = sorted(metadata_path.glob("scene_*_metadata.pzm"))
            if not scene_files:
                scene_files = sorted(metadata_path.glob("scene_*_metadata.json"))
            logging.info(f"📚 Found {len(scene_files)} scenes to reconstruct")
            
            reconstruction_results = []
            
            for scene_file in scene_files:
                # Load scene metadata
                scene_data = self._load_scene_metadata(scene_file)
                
                scene_number = scene_data.get('scene_number', 0)
                processing_start = time.time()
                
                logging.info(f"📺 Processing scene {scene_number} (#{self.processed_scenes + 1})")
                
                # Add panorama and object paths to scene data
                scene_data = self._prepare_scene_data(scene_data, metadata_path)
                
                # Reconstruct scene
                result = self.reconstruct_scene(scene_data)
                
                processing_time = time.time() - processing_start
                self.reconstruction_times.append(processing_time)
                self.processed_scenes += 1
                
                logging.info(f"⏱️  Scene {scene_number} completed in {processing_time:.1f}s")
                
                # Save reconstructed video if successful
                if result.get('status') == 'success' and output_dir:
                    reconstructed_path = result.get('reconstructed_video_path')
                    if reconstructed_path:
                        # Move to output directory (handle cross-device link)
                        final_path = output_path / f"scene_{scene_number:04d}_reconstructed.mp4"
                        if Path(reconstructed_path).exists():
                            try:
                                Path(reconstructed_path).rename(final_path)
                            except OSError:
                                # Handle cross-device link error
                                shutil.copy2(reconstructed_path, final_path)
                                Path(reconstructed_path).unlink()
                            result['final_video_path'] = str(final_path)
                            logging.info(f"💾 Scene {scene_number} saved: {final_path}")
                
                reconstruction_results.append(result)
                
                # Progress summary (every 5 scenes)
                if self.processed_scenes % 5 == 0:
                    avg_time = sum(self.reconstruction_times) / len(self.reconstruction_times)
                    success_rate = sum(1 for r in reconstruction_results if r.get('status') == 'success') / len(reconstruction_results)
                    logging.info(f"📊 Progress: {self.processed_scenes} scenes | Avg: {avg_time:.1f}s/scene | Success: {success_rate:.1%}")
                    logging.info("-" * 80)
            
            # Generate summary
            summary = self._generate_reconstruction_summary(training_results, reconstruction_results)
            
            logging.info("PointStream client processing complete")
            return summary
            
        except Exception as e:
            logging.error(f"Client processing failed: {e}")
            raise
    
    def _prepare_scene_data(self, scene_data: Dict[str, Any], metadata_path: Path) -> Dict[str, Any]:
        """Prepare scene data by adding file paths."""
        scene_number = scene_data.get('scene_number', 0)
        
        # Add panorama path
        panorama_path = metadata_path / "panoramas" / f"scene_{scene_number:04d}_panorama.jpg"
        if panorama_path.exists():
            scene_data['panorama_path'] = str(panorama_path)
        
        # Add object images paths
        objects_dir = metadata_path / "objects" / f"scene_{scene_number:04d}"
        if objects_dir.exists():
            scene_data['objects_dir'] = str(objects_dir)
        
        # Extract frame dimensions from video_properties for background reconstructor
        video_properties = scene_data.get('video_properties', {})
        resolution = video_properties.get('resolution', {})
        if isinstance(resolution, dict):
            scene_data['frame_width'] = resolution.get('width', 1920)
            scene_data['frame_height'] = resolution.get('height', 1080)
        elif isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            scene_data['frame_width'] = resolution[0]
            scene_data['frame_height'] = resolution[1]
        
        # The demuxer already extracted homographies to the top level, so no need to do it here
        if 'homographies' in scene_data:
            logging.info(f"📊 Found {len(scene_data['homographies'])} homography matrices for scene {scene_number}")
        
        return scene_data
    
    def _generate_reconstruction_summary(self, training_results: Dict[str, Any], 
                                       reconstruction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final reconstruction summary."""
        total_time = sum(self.reconstruction_times)
        successful_scenes = [r for r in reconstruction_results if r.get('status') == 'success']
        
        return {
            'training_results': training_results,
            'processed_scenes': self.processed_scenes,
            'successful_scenes': len(successful_scenes),
            'failed_scenes': self.processed_scenes - len(successful_scenes),
            'success_rate': len(successful_scenes) / max(self.processed_scenes, 1),
            'total_reconstruction_time': total_time,
            'average_reconstruction_time': total_time / max(self.processed_scenes, 1),
            'throughput': self.processed_scenes / total_time if total_time > 0 else 0,
            'reconstruction_results': reconstruction_results
        }
    
    @track_performance
    def assess_quality(self, original_video: str, reconstructed_video: str) -> Dict[str, Any]:
        """
        Assess quality of reconstructed video compared to original.
        
        Args:
            original_video: Path to original video
            reconstructed_video: Path to reconstructed video
            
        Returns:
            Quality assessment metrics
        """
        logging.info(f"📊 Assessing reconstruction quality...")
        
        return self.quality_assessor.assess_video_quality(
            original_video, 
            reconstructed_video
        )

    def _load_scene_metadata(self, metadata_file: Path) -> Dict[str, Any]:
        """
        Load scene metadata from file, handling both compressed and uncompressed formats.
        
        Args:
            metadata_file: Path to metadata file (.json, .pzm, or .json.gz)
            
        Returns:
            Scene metadata dictionary
        """
        try:
            # Use demuxer to handle all metadata file types
            return self.demuxer.get_scene_metadata(metadata_file)
        except Exception as e:
            # Fallback to direct JSON loading for compatibility
            logging.warning(f"Demuxer failed for {metadata_file.name}, trying direct JSON load: {e}")
            with open(metadata_file, 'r') as f:
                return json.load(f)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Format for client output
    formatter = logging.Formatter(
        '🔧 %(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    
    print(f"🔧 PointStream Client Starting - Log Level: {log_level.upper()}")
    print("=" * 80)


def main():
    """Main entry point for the PointStream client."""
    parser = argparse.ArgumentParser(
        description="PointStream Client - Video Reconstruction using Generative Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Reconstruct videos from metadata
    python client.py ./metadata_dir
    
    # Custom output directory
    python client.py ./metadata_dir --output-dir ./reconstructed
    
    # Skip model training (use existing models)
    python client.py ./metadata_dir --no-training
    
    # Assess quality against original video
    python client.py ./metadata_dir --assess-quality ./original.mp4
        """
    )
    
    parser.add_argument('metadata_dir', help='Path to metadata directory')
    parser.add_argument('--output-dir', default='./reconstructed_videos',
                       help='Output directory for reconstructed videos')
    parser.add_argument('--config', help='Path to client configuration file')
    parser.add_argument('--no-training', action='store_true',
                       help='Skip model training and use existing models')
    parser.add_argument('--assess-quality', metavar='ORIGINAL_VIDEO',
                       help='Assess quality against original video')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate input
    if not Path(args.metadata_dir).exists():
        print(f"Error: Metadata directory not found: {args.metadata_dir}")
        sys.exit(1)
    
    try:
        # Initialize client
        client = PointStreamClient(config_file=args.config)
        
        # Process metadata
        summary = client.process_metadata(
            metadata_dir=args.metadata_dir,
            output_dir=args.output_dir,
            train_models=not args.no_training
        )
        
        # Assess quality if requested
        if args.assess_quality:
            # Find reconstructed videos and assess quality
            output_path = Path(args.output_dir)
            quality_results = []
            
            for reconstructed_video in output_path.glob("*_reconstructed.mp4"):
                quality_result = client.assess_quality(
                    args.assess_quality, 
                    str(reconstructed_video)
                )
                quality_results.append(quality_result)
            
            summary['quality_assessment'] = quality_results
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"POINTSTREAM CLIENT PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Processed scenes: {summary['processed_scenes']}")
        print(f"Successful scenes: {summary['successful_scenes']}")
        print(f"Failed scenes: {summary['failed_scenes']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total reconstruction time: {summary['total_reconstruction_time']:.3f}s")
        print(f"Average time per scene: {summary['average_reconstruction_time']:.3f}s")
        print(f"Throughput: {summary['throughput']:.2f} scenes/second")
        print(f"Output directory: {args.output_dir}")
        
        if args.assess_quality:
            quality_results = summary.get('quality_assessment', [])
            if quality_results:
                avg_ssim = np.mean([q.get('ssim', 0) for q in quality_results])
                avg_psnr = np.mean([q.get('psnr', 0) for q in quality_results])
                print(f"Average SSIM: {avg_ssim:.3f}")
                print(f"Average PSNR: {avg_psnr:.2f} dB")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Client pipeline failed: {e}")
        sys.exit(1)




if __name__ == "__main__":
    main()
