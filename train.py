"""
Offline training script for creating custom "student" object detection models.

This version uses autodistill for auto-labeling (the "teacher") and the
standard ultralytics library for fine-tuning a pre-trained YOLO model.
"""
import argparse
import os
import random
import shutil
from pathlib import Path
from datetime import datetime
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from ultralytics import YOLO

def main():
    """Runs the full distillation pipeline with smart annotation caching."""
    parser = argparse.ArgumentParser(description="Train a student model with autodistill.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory of unlabeled images.")
    parser.add_argument("--output_path", type=str, default="artifacts/training", help="Path to save the trained model and dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the student model.")
    parser.add_argument("--max_images", type=int, default=None, help="Cap the number of images to label for faster training runs.")
    parser.add_argument("--content_type", type=str, default="sports", help="Content type for organizing annotations (sports, general, etc.)")
    parser.add_argument("--force_relabel", action="store_true", help="Force re-labeling even if annotations exist")
    args = parser.parse_args()

    print("--- Starting Autodistill Training Pipeline with Smart Caching ---")

    # --- 1. Define Content-Specific Ontologies ---
    ontologies = {
        "sports": CaptionOntology({
            "tennis_player": "person holding a tennis racket on a tennis court",
            "skier": "person wearing skis on a snowy slope", 
            "tennis_racket": "tennis racket",
            "sports_ball": "tennis ball or sports ball",
            "athlete": "athlete or sports person in action"
        }),
        "general": CaptionOntology({
            "person": "person or human being",
            "vehicle": "car, truck, bicycle, or any vehicle",
            "animal": "dog, cat, bird, or any animal"
        }),
        "dance": CaptionOntology({
            "dancer": "person dancing or in dance pose",
            "performer": "stage performer or entertainer"
        })
    }
    
    # Select ontology based on content type
    ontology = ontologies.get(args.content_type, ontologies["sports"])
    # Get the classes from the CaptionOntology
    try:
        ontology_classes = ontology.classes()
    except:
        ontology_classes = ["tennis_player", "tennis_racket", "sports_ball", "athlete"]  # fallback
    
    print(f" -> Using {args.content_type} ontology with classes: {ontology_classes}")

    # --- 2. Smart Annotation Caching ---
    # Create content-specific annotation cache directory
    annotations_cache_dir = Path("artifacts/annotations_cache") / args.content_type
    annotations_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have annotations for this content type
    existing_annotations = list(annotations_cache_dir.glob("*/"))
    use_cached = not args.force_relabel and len(existing_annotations) > 0
    
    if use_cached:
        print(f" -> Found existing {args.content_type} annotations in cache: {annotations_cache_dir}")
        print(f" -> Found {len(existing_annotations)} cached annotation sets")
        print(" -> Skipping annotation step, using cached data")
        
        # Use the most recent annotation set (or we could let user choose)
        dataset_path = max(existing_annotations, key=lambda p: p.stat().st_mtime)
        print(f" -> Using cached annotations from: {dataset_path}")
    else:
        print(f" -> No cached annotations found or force_relabel=True")
        print(f" -> Will create new annotations in: {annotations_cache_dir}")
        
        # Create timestamped subdirectory for this annotation run
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = annotations_cache_dir / f"annotated_{timestamp}"
        dataset_path.mkdir(parents=True, exist_ok=True)

        # --- 3. Prepare Image Subset for Labeling (only if not using cache) ---
        input_images_path = Path(args.data_path)
        temp_subset_path = Path(args.output_path) / "temp_image_subset"
        labeling_path = input_images_path

        if args.max_images:
            print(f" -> Selecting a random subset of {args.max_images} images...")
            all_images = [f for f in input_images_path.glob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            if len(all_images) == 0:
                print(f" -> ERROR: No images found in {input_images_path}")
                return
            
            selected_images = random.sample(all_images, min(len(all_images), args.max_images))
            
            temp_subset_path.mkdir(parents=True, exist_ok=True)
            # Clear any existing symlinks first
            for existing_file in temp_subset_path.glob("*"):
                if existing_file.is_symlink() or existing_file.is_file():
                    existing_file.unlink()
            
            for img_path in selected_images:
                # Use resolve() to get the absolute path for the symlink
                (temp_subset_path / img_path.name).symlink_to(img_path.resolve())
            
            labeling_path = temp_subset_path
            print(f" -> Using temporary subset for labeling: {labeling_path}")

        # --- 4. Initialize the Teacher Model and Auto-Label (only if not using cache) ---
        print(" -> Initializing teacher model (GroundedSAM)...")
        teacher_model = GroundedSAM(ontology=ontology)

        print(f" -> Starting auto-labeling process for {args.content_type} content...")
        print(f" -> Input images: {labeling_path}")
        print(f" -> Output annotations: {dataset_path}")
        
        teacher_model.label(
            input_folder=str(labeling_path),
            output_folder=str(dataset_path)
        )
        print(f" -> Auto-labeling complete. Annotations cached in: {dataset_path}")

        # Cleanup temporary subset
        if temp_subset_path.exists():
            print(f" -> Cleaning up temporary directory: {temp_subset_path}")
            shutil.rmtree(temp_subset_path)
    
    # --- 5. Train Student Model using Cached Annotations ---
    print(f" -> Training student model using annotations from: {dataset_path}")
    model = YOLO("yolo11n.pt")  # Start from the pre-trained YOLO model

    data_yaml_path = dataset_path / "data.yaml"
    if not data_yaml_path.exists():
        print(f" -> ERROR: data.yaml not found at {data_yaml_path}")
        print(" -> Available files in dataset:")
        for f in dataset_path.iterdir():
            print(f"    {f}")
        return

    print(f" -> Starting training for {args.epochs} epochs...")
    print(f" -> Using dataset: {data_yaml_path}")
    
    # Create content-specific training output directory
    training_output = Path(args.output_path) / f"training_{args.content_type}"
    training_output.mkdir(parents=True, exist_ok=True)
    
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        project=str(training_output),
        name=f"run_{args.content_type}",
        imgsz=640,
        batch=16,
        workers=8
    )
    
    # Copy the best model to a standard location with content type in name
    best_model_path = training_output / f"run_{args.content_type}" / "weights" / "best.pt"
    final_model_path = Path(args.output_path) / f"{args.content_type}_model.pt"
    
    if best_model_path.exists():
        shutil.copy2(best_model_path, final_model_path)
        print(f" -> Training complete! Model saved as: {final_model_path}")
        print(f" -> Annotations cached for reuse in: {dataset_path}")
    else:
        print(" -> Warning: Training completed but best.pt not found at expected location")
        print(f" -> Check training output in: {training_output}")

    print(f" -> {args.content_type.capitalize()} model training pipeline completed successfully!")
    print(f" -> Next time, run with the same --content_type to reuse annotations automatically")

if __name__ == "__main__":
    main()
