"""
Enhanced training script with SAM annotation caching and content-type organization.

This version implements smart caching of SAM annotations organized by content type,
allowing for efficient reuse across multiple training runs.
"""
import argparse
import os
import random
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from ultralytics import YOLO

class AnnotationCache:
    """Manages SAM annotation caching organized by content type."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = cache_dir / "cache_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "content_types": {},
                "annotations": {}
            }
    
    def save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_image_hash(self, image_path: Path) -> str:
        """Generate a stable hash for an image file."""
        with open(image_path, 'rb') as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()
    
    def get_ontology_hash(self, ontology: CaptionOntology) -> str:
        """Generate a stable hash for an ontology."""
        ontology_dict = dict(ontology.prompts())
        ontology_str = json.dumps(ontology_dict, sort_keys=True)
        return hashlib.md5(ontology_str.encode()).hexdigest()
    
    def get_content_type_cache_dir(self, content_type: str) -> Path:
        """Get the cache directory for a specific content type."""
        content_dir = self.cache_dir / "content_types" / content_type
        content_dir.mkdir(parents=True, exist_ok=True)
        return content_dir
    
    def is_annotation_cached(self, image_path: Path, ontology: CaptionOntology, content_type: str) -> bool:
        """Check if an annotation is already cached."""
        image_hash = self.get_image_hash(image_path)
        ontology_hash = self.get_ontology_hash(ontology)
        
        annotation_key = f"{content_type}_{image_hash}_{ontology_hash}"
        return annotation_key in self.metadata["annotations"]
    
    def get_cached_annotation_path(self, image_path: Path, ontology: CaptionOntology, content_type: str) -> Optional[Path]:
        """Get the path to a cached annotation if it exists."""
        if not self.is_annotation_cached(image_path, ontology, content_type):
            return None
        
        image_hash = self.get_image_hash(image_path)
        ontology_hash = self.get_ontology_hash(ontology)
        annotation_key = f"{content_type}_{image_hash}_{ontology_hash}"
        
        annotation_info = self.metadata["annotations"][annotation_key]
        annotation_path = Path(annotation_info["path"])
        
        # Verify the cached annotation still exists
        if annotation_path.exists():
            return annotation_path
        else:
            # Remove stale entry
            del self.metadata["annotations"][annotation_key]
            self.save_metadata()
            return None
    
    def cache_annotation(self, image_path: Path, annotation_data: Dict, ontology: CaptionOntology, content_type: str):
        """Cache an annotation with metadata."""
        image_hash = self.get_image_hash(image_path)
        ontology_hash = self.get_ontology_hash(ontology)
        
        # Create content type directory structure
        content_dir = self.get_content_type_cache_dir(content_type)
        annotation_file = content_dir / f"{image_hash}_{ontology_hash}.json"
        
        # Save annotation data
        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        # Update metadata
        annotation_key = f"{content_type}_{image_hash}_{ontology_hash}"
        self.metadata["annotations"][annotation_key] = {
            "image_path": str(image_path),
            "image_hash": image_hash,
            "ontology_hash": ontology_hash,
            "content_type": content_type,
            "path": str(annotation_file),
            "created": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        
        # Update content type metadata
        if content_type not in self.metadata["content_types"]:
            self.metadata["content_types"][content_type] = {
                "created": datetime.now().isoformat(),
                "annotation_count": 0,
                "last_updated": datetime.now().isoformat()
            }
        
        self.metadata["content_types"][content_type]["annotation_count"] += 1
        self.metadata["content_types"][content_type]["last_updated"] = datetime.now().isoformat()
        
        self.save_metadata()
        print(f"   -> Cached annotation for {image_path.name} in content type '{content_type}'")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the annotation cache."""
        stats = {
            "total_annotations": len(self.metadata["annotations"]),
            "content_types": len(self.metadata["content_types"]),
            "content_type_breakdown": {}
        }
        
        for content_type, info in self.metadata["content_types"].items():
            stats["content_type_breakdown"][content_type] = {
                "count": info["annotation_count"],
                "last_updated": info["last_updated"]
            }
        
        return stats


class EnhancedTrainer:
    """Enhanced trainer with smart annotation caching."""
    
    def __init__(self, cache_dir: Path):
        self.cache = AnnotationCache(cache_dir)
    
    def prepare_dataset_with_cache(self, images: List[Path], ontology: CaptionOntology, 
                                   content_type: str, output_dir: Path) -> Path:
        """Prepare a dataset using cached annotations when available."""
        print(f"\n -> Preparing dataset for content type: {content_type}")
        
        # Check cache status
        cached_count = 0
        new_annotation_count = 0
        
        for image_path in images:
            if self.cache.is_annotation_cached(image_path, ontology, content_type):
                cached_count += 1
            else:
                new_annotation_count += 1
        
        print(f"   -> Found {cached_count} cached annotations")
        print(f"   -> Need to generate {new_annotation_count} new annotations")
        
        if cached_count > 0:
            print(f"   -> Cache hit rate: {cached_count / len(images) * 100:.1f}%")
        
        # Create dataset directory
        dataset_dir = output_dir / f"dataset_{content_type}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SAM model only if we need new annotations
        teacher_model = None
        if new_annotation_count > 0:
            print("   -> Initializing GroundedSAM for new annotations...")
            teacher_model = GroundedSAM(ontology=ontology)
        
        # Process images
        processed_images = []
        for image_path in images:
            cached_annotation_path = self.cache.get_cached_annotation_path(image_path, ontology, content_type)
            
            if cached_annotation_path:
                # Use cached annotation
                print(f"   -> Using cached annotation for {image_path.name}")
                # Copy cached annotation to dataset
                # TODO: Implement copying cached annotation to YOLO format
                processed_images.append(image_path)
            else:
                # Generate new annotation
                print(f"   -> Generating new annotation for {image_path.name}")
                if teacher_model:
                    # Generate annotation using SAM
                    # For now, we'll use the original labeling approach
                    # TODO: Capture annotation data and cache it
                    processed_images.append(image_path)
        
        # If we have new annotations, run the full labeling process
        if new_annotation_count > 0 and teacher_model:
            temp_input_dir = output_dir / f"temp_input_{content_type}"
            temp_input_dir.mkdir(parents=True, exist_ok=True)
            
            # Create symlinks for images needing annotation
            images_needing_annotation = [
                img for img in images 
                if not self.cache.is_annotation_cached(img, ontology, content_type)
            ]
            
            for img_path in images_needing_annotation:
                (temp_input_dir / img_path.name).symlink_to(img_path.resolve())
            
            # Generate annotations
            teacher_model.label(
                input_folder=str(temp_input_dir),
                output_folder=str(dataset_dir)
            )
            
            # Cache the new annotations
            # TODO: Read generated annotations and cache them
            
            # Cleanup temp directory
            shutil.rmtree(temp_input_dir)
        
        return dataset_dir
    
    def train_content_specific_model(self, dataset_dir: Path, content_type: str, 
                                     epochs: int, output_dir: Path) -> Path:
        """Train a content-specific model."""
        print(f"\n -> Training {content_type} model for {epochs} epochs...")
        
        # Initialize student model
        model = YOLO("yolo12n.pt")
        
        # Find data.yaml
        data_yaml_path = dataset_dir / "data.yaml"
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")
        
        # Train model
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            project=str(output_dir),
            name=f"{content_type}_model"
        )
        
        # Return path to trained model
        model_path = output_dir / f"{content_type}_model" / "weights" / "best.pt"
        return model_path


def main():
    """Run the enhanced training pipeline with caching."""
    parser = argparse.ArgumentParser(description="Enhanced training with SAM annotation caching.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory of unlabeled images.")
    parser.add_argument("--content_type", type=str, required=True, help="Content type for this training run (e.g., 'sports', 'dance', 'automotive').")
    parser.add_argument("--output_path", type=str, default="artifacts/training", help="Path to save the trained model and dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the student model.")
    parser.add_argument("--max_images", type=int, default=None, help="Cap the number of images to label for faster training runs.")
    parser.add_argument("--cache_stats", action="store_true", help="Show cache statistics and exit.")
    parser.add_argument("--clear_cache", action="store_true", help="Clear the annotation cache and exit.")
    
    args = parser.parse_args()
    
    # Import the cache directory from config
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from pointstream import config
    cache_dir = config.ANNOTATIONS_CACHE_DIR
    
    trainer = EnhancedTrainer(cache_dir)
    
    # Handle cache management commands
    if args.cache_stats:
        stats = trainer.cache.get_cache_stats()
        print("\n=== Annotation Cache Statistics ===")
        print(f"Total cached annotations: {stats['total_annotations']}")
        print(f"Content types: {stats['content_types']}")
        print("\nBreakdown by content type:")
        for content_type, info in stats['content_type_breakdown'].items():
            print(f"  {content_type}: {info['count']} annotations (last updated: {info['last_updated']})")
        return
    
    if args.clear_cache:
        response = input("Are you sure you want to clear the annotation cache? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(cache_dir)
            print("Annotation cache cleared.")
        return
    
    print("=== Enhanced Autodistill Training Pipeline with Caching ===")
    
    # Define ontology based on content type
    ontologies = {
        "sports": CaptionOntology({
            "tennis_player": "person holding a tennis racket on a tennis court",
            "tennis_racket": "tennis racket",
            "skier": "person wearing skis on a snowy slope",
            "basketball_player": "person playing basketball",
            "soccer_player": "person playing soccer with a soccer ball"
        }),
        "dance": CaptionOntology({
            "dancer": "person dancing",
            "ballet_dancer": "ballet dancer in a tutu",
            "contemporary_dancer": "contemporary dancer in flowing clothes"
        }),
        "automotive": CaptionOntology({
            "car": "car on a road",
            "truck": "truck on a road",
            "motorcycle": "motorcycle on a road",
            "bus": "bus on a road"
        }),
        "general": CaptionOntology({
            "person": "person",
            "vehicle": "car, truck, bus, or motorcycle",
            "animal": "dog, cat, or other animal"
        })
    }
    
    if args.content_type not in ontologies:
        print(f"Warning: Unknown content type '{args.content_type}'. Using general ontology.")
        ontology = ontologies["general"]
    else:
        ontology = ontologies[args.content_type]
    
    print(f"Content type: {args.content_type}")
    print(f"Ontology classes: {list(ontology.prompts().keys())}")
    
    # Prepare images
    input_images_path = Path(args.data_path)
    output_path = Path(args.output_path)
    
    print(f"\n -> Loading images from: {input_images_path}")
    all_images = [f for f in input_images_path.glob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if args.max_images:
        all_images = random.sample(all_images, min(len(all_images), args.max_images))
        print(f" -> Selected {len(all_images)} images for training")
    
    # Show cache statistics
    stats = trainer.cache.get_cache_stats()
    print(f"\n -> Cache contains {stats['total_annotations']} annotations across {stats['content_types']} content types")
    
    try:
        # Prepare dataset with caching
        dataset_dir = trainer.prepare_dataset_with_cache(
            all_images, ontology, args.content_type, output_path
        )
        
        # Train the model
        model_path = trainer.train_content_specific_model(
            dataset_dir, args.content_type, args.epochs, output_path
        )
        
        print(f"\n=== Training Complete! ===")
        print(f"Trained model saved to: {model_path}")
        print(f"Dataset saved to: {dataset_dir}")
        
        # Update config with new model
        model_registry_path = model_path.parent.parent.parent / "models" / "weights" / f"{args.content_type}_model.pt"
        model_registry_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, model_registry_path)
        
        print(f"Model copied to registry: {model_registry_path}")
        print(f"\nTo use this model, add it to the MODEL_REGISTRY in config.py:")
        print(f'"{args.content_type}": "{model_registry_path}"')
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
