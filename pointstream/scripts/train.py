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
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from ultralytics import YOLO

def main():
    """Runs the full distillation pipeline."""
    parser = argparse.ArgumentParser(description="Train a student model with autodistill.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory of unlabeled images.")
    parser.add_argument("--output_path", type=str, default="artifacts/training", help="Path to save the trained model and dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the student model.")
    parser.add_argument("--max_images", type=int, default=None, help="Cap the number of images to label for faster training runs.")
    args = parser.parse_args()

    print("--- Starting Autodistill Training Pipeline ---")

    # --- 1. Define the Ontology ---
    # This is where you define the classes for your specialized model.
    # The key is the class name your model will learn.
    # The value is the detailed prompt for the teacher model.
    ontology = CaptionOntology({
        "tennis_player": "person holding a tennis racket on a tennis court",
        "skier": "person wearing skis on a snowy slope",
        "tennis_racket": "tennis racket"
        # Add other classes for your "sports" model here
    })

    # --- 2. Prepare Image Subset for Labeling ---
    input_images_path = Path(args.data_path)
    temp_subset_path = Path(args.output_path) / "temp_image_subset"
    labeling_path = input_images_path

    try:
        if args.max_images:
            print(f" -> Selecting a random subset of {args.max_images} images...")
            all_images = [f for f in input_images_path.glob('*') if f.is_file()]
            selected_images = random.sample(all_images, min(len(all_images), args.max_images))
            
            temp_subset_path.mkdir(parents=True, exist_ok=True)
            for img_path in selected_images:
                # Use resolve() to get the absolute path for the symlink
                (temp_subset_path / img_path.name).symlink_to(img_path.resolve())
            
            labeling_path = temp_subset_path
            print(f" -> Using temporary subset for labeling: {labeling_path}")

        # --- 3. Initialize the Teacher Model and Auto-Label ---
        print(" -> Initializing teacher model (GroundedSAM)...")
        teacher_model = GroundedSAM(ontology=ontology)

        print(f" -> Starting auto-labeling process for images in: {labeling_path}")
        dataset_path = Path(args.output_path) / "dataset"
        teacher_model.label(
            input_folder=str(labeling_path),
            output_folder=str(dataset_path)
        )
        print(f" -> Auto-labeling complete. Dataset saved to: {dataset_path}")

        # --- 4. Fine-Tune the Student Model using Ultralytics Directly ---
        print(" -> Initializing student model (YOLOv12)...")
        model = YOLO("yolo12n.pt") # Start from the pre-trained YOLOv12 model

        print(f" -> Starting training for {args.epochs} epochs...")
        data_yaml_path = dataset_path / "data.yaml"
        
        model.train(
            data=str(data_yaml_path),
            epochs=args.epochs,
            project=args.output_path
        )
        print(" -> Training complete!")
        print(f" -> Trained model weights saved in: {args.output_path}/train/weights/best.pt")

    finally:
        # --- 5. Cleanup ---
        if temp_subset_path.exists():
            print(f" -> Cleaning up temporary directory: {temp_subset_path}")
            shutil.rmtree(temp_subset_path)

if __name__ == "__main__":
    main()
