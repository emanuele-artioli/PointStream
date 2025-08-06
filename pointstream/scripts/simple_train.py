"""
Simplified training script for demonstration purposes.
Creates a mock trained model for testing the PointStream pipeline.
"""
import argparse
import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

def create_mock_dataset(data_path, output_path, max_images=10):
    """Create a mock YOLO dataset structure for testing."""
    dataset_path = Path(output_path) / "dataset"
    
    # Create dataset directory structure
    for split in ['train', 'val']:
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy some images to train and val
    input_images = list(Path(data_path).glob('*'))[:max_images]
    train_images = input_images[:int(len(input_images) * 0.8)]
    val_images = input_images[int(len(input_images) * 0.8):]
    
    # Copy training images and create mock labels
    for i, img_path in enumerate(train_images):
        if img_path.is_file():
            dest_img = dataset_path / 'train' / 'images' / f'train_{i}.jpg'
            shutil.copy2(img_path, dest_img)
            
            # Create a mock label file (empty - no objects detected for simplicity)
            label_file = dataset_path / 'train' / 'labels' / f'train_{i}.txt'
            label_file.write_text("")
    
    # Copy validation images and create mock labels
    for i, img_path in enumerate(val_images):
        if img_path.is_file():
            dest_img = dataset_path / 'val' / 'images' / f'val_{i}.jpg'
            shutil.copy2(img_path, dest_img)
            
            # Create a mock label file
            label_file = dataset_path / 'val' / 'labels' / f'val_{i}.txt'
            label_file.write_text("")
    
    # Create data.yaml file
    data_yaml = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'tennis_player',
            1: 'tennis_racket',
            2: 'sports_equipment'
        }
    }
    
    with open(dataset_path / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return dataset_path / 'data.yaml'

def main():
    """Runs a simplified training pipeline."""
    parser = argparse.ArgumentParser(description="Train a simplified sports model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory of images.")
    parser.add_argument("--output_path", type=str, default="artifacts/training", help="Path to save the trained model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--max_images", type=int, default=20, help="Maximum number of images to use.")
    parser.add_argument("--model_name", type=str, default="sports_model", help="Name for the trained model.")
    args = parser.parse_args()

    print("--- Starting Simplified Sports Model Training ---")
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create mock dataset
    print(f" -> Creating mock dataset from {args.data_path}...")
    data_yaml_path = create_mock_dataset(args.data_path, args.output_path, args.max_images)
    print(f" -> Mock dataset created at: {data_yaml_path}")

    # Train YOLO model
    print(" -> Initializing YOLO model...")
    model = YOLO("yolo11n.pt")  # Use YOLOv11 nano for fast training
    
    print(f" -> Starting training for {args.epochs} epochs...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=640,
        batch=4,
        project=str(output_path),
        name="sports_train",
        verbose=True
    )
    
    # Copy the best model to a standard location
    best_model_path = output_path / "sports_train" / "weights" / "best.pt"
    final_model_path = output_path / f"{args.model_name}.pt"
    
    if best_model_path.exists():
        shutil.copy2(best_model_path, final_model_path)
        print(f" -> Training complete! Model saved as: {final_model_path}")
    else:
        print(" -> Warning: Training completed but best.pt not found at expected location")
        print(f" -> Check training output in: {output_path}")

    print(" -> Sports model training pipeline completed successfully!")
    return str(final_model_path)

if __name__ == "__main__":
    main()
