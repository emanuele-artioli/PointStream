#!/usr/bin/env python3
"""
PointStream - Streamlined Video Processing Pipeline

This script consolidates the core server components into a single pipeline:
- splitter: Takes video and returns scenes split at cuts
- segmenter: Uses YOLO to segment objects in scenes  
- stitcher: Inpaints objects and builds panoramas
- classifier: Classifies objects as human, animal, or other
- keypointer: Extracts keypoints using mmpose (human/animal) or canny (other)

Each component is implemented as a single function with minimal dependencies.
"""

import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Core dependencies
    from scenedetect import detect, ContentDetector
    from ultralytics import YOLO
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from mmpose.apis import MMPoseInferencer
except ImportError as e:
    logging.error(f"Missing required dependency: {e}")
    logging.error("Please install: pip install scenedetect ultralytics torch sentence-transformers scikit-learn mmpose")
    raise


def split_video_into_scenes(video_path: str, threshold: float = 30.0, min_scene_len: int = 15) -> List[Dict[str, Any]]:
    """
    Split video into scenes using PySceneDetect.
    
    Args:
        video_path: Path to input video
        threshold: Scene change threshold (higher = fewer cuts)
        min_scene_len: Minimum scene length in frames
        
    Returns:
        List of scene dictionaries with frames, timestamps, and metadata
    """
    logging.info(f"Splitting video: {video_path}")
    
    # Detect scene changes
    scene_list = detect(video_path, ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    
    if not scene_list:
        logging.warning("No scenes detected, using entire video as one scene")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scene_list = [(0.0, frame_count / fps)]
        cap.release()
    
    # Extract frames for each scene
    scenes = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = int(start_time.get_seconds() * fps) if hasattr(start_time, 'get_seconds') else int(start_time * fps)
        end_frame = int(end_time.get_seconds() * fps) if hasattr(end_time, 'get_seconds') else int(end_time * fps)
        
        # Extract frames
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_num in range(start_frame, min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
        
        if frames:
            scene_data = {
                'scene_number': i + 1,
                'start_time': start_time.get_seconds() if hasattr(start_time, 'get_seconds') else start_time,
                'end_time': end_time.get_seconds() if hasattr(end_time, 'get_seconds') else end_time,
                'frames': frames,
                'frame_count': len(frames),
                'fps': fps
            }
            scenes.append(scene_data)
    
    cap.release()
    logging.info(f"Split video into {len(scenes)} scenes")
    return scenes


def segment_objects_in_scene(scene: Dict[str, Any], model_path: str = 'yolov8n-seg.pt', 
                           confidence: float = 0.25, iou: float = 0.7) -> Dict[str, Any]:
    """
    Segment objects in scene frames using YOLO.
    
    Args:
        scene: Scene data from splitter
        model_path: Path to YOLO segmentation model
        confidence: Confidence threshold for detections
        iou: IoU threshold for NMS
        
    Returns:
        Dictionary with segmented objects data
    """
    logging.info(f"Segmenting scene {scene['scene_number']} with {len(scene['frames'])} frames")
    
    # Load YOLO model
    model = YOLO(model_path)
    
    # Process frames
    all_objects = []
    frames = scene['frames']
    
    for frame_idx, frame in enumerate(frames):
        # Run YOLO detection with tracking
        results = model.track(frame, conf=confidence, iou=iou, persist=True)
        
        frame_objects = []
        for result in results:
            if result.boxes is not None and result.masks is not None:
                boxes = result.boxes.xywh.cpu().numpy()  # x, y, w, h format
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                masks = result.masks.data.cpu().numpy()
                
                # Get track IDs if available
                track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None
                
                for i, (box, conf, class_id, mask) in enumerate(zip(boxes, confidences, class_ids, masks)):
                    x, y, w, h = box
                    
                    # Convert mask to frame size and get binary mask
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    binary_mask = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Crop object image using bounding box
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    
                    # Ensure coordinates are within frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        cropped_image = frame[y1:y2, x1:x2]
                        
                        obj_data = {
                            'frame_index': frame_idx,
                            'object_id': f"scene_{scene['scene_number']}_frame_{frame_idx}_obj_{i}",
                            'track_id': track_ids[i] if track_ids is not None else None,
                            'class_id': class_id,
                            'class_name': model.names[class_id],
                            'confidence': float(conf),
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # x, y, w, h
                            'mask': binary_mask,
                            'cropped_image': cropped_image,
                            'crop_size': [cropped_image.shape[1], cropped_image.shape[0]]
                        }
                        frame_objects.append(obj_data)
        
        all_objects.extend(frame_objects)
    
    logging.info(f"Found {len(all_objects)} objects in scene {scene['scene_number']}")
    return {
        'scene_number': scene['scene_number'],
        'objects': all_objects,
        'total_objects': len(all_objects)
    }


def stitch_scene_panorama(scene: Dict[str, Any], segmentation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create panorama from scene frames with object inpainting.
    
    Args:
        scene: Scene data from splitter
        segmentation_result: Objects data from segmenter
        
    Returns:
        Dictionary with panorama and stitching info
    """
    logging.info(f"Stitching panorama for scene {scene['scene_number']}")
    
    frames = scene['frames']
    if len(frames) <= 1:
        return {
            'scene_number': scene['scene_number'],
            'panorama': frames[0] if frames else None,
            'scene_type': 'Static'
        }
    
    # Create masks for inpainting (black out detected objects)
    masked_frames = []
    objects_by_frame = {}
    
    # Group objects by frame
    for obj in segmentation_result['objects']:
        frame_idx = obj['frame_index']
        if frame_idx not in objects_by_frame:
            objects_by_frame[frame_idx] = []
        objects_by_frame[frame_idx].append(obj)
    
    # Create masked frames with objects blacked out
    for i, frame in enumerate(frames):
        masked_frame = frame.copy()
        if i in objects_by_frame:
            for obj in objects_by_frame[i]:
                mask = obj['mask']
                masked_frame[mask > 0] = [0, 0, 0]  # Black out objects
        masked_frames.append(masked_frame)
    
    # Simple panorama creation using OpenCV stitcher
    try:
        stitcher = cv2.Stitcher.create()
        status, panorama = stitcher.stitch(masked_frames)
        
        if status == cv2.Stitcher_OK:
            logging.info(f"Successfully created panorama for scene {scene['scene_number']}")
            return {
                'scene_number': scene['scene_number'],
                'panorama': panorama,
                'scene_type': 'Panoramic'
            }
        else:
            logging.warning(f"Stitching failed for scene {scene['scene_number']}, using first frame")
            return {
                'scene_number': scene['scene_number'],
                'panorama': frames[0],
                'scene_type': 'Complex'
            }
    except Exception as e:
        logging.error(f"Error during stitching: {e}")
        return {
            'scene_number': scene['scene_number'],
            'panorama': frames[0],
            'scene_type': 'Complex'
        }


def classify_objects(segmentation_result: Dict[str, Any], 
                    human_threshold: float = 0.6, animal_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Classify object class names into human, animal, or other categories.
    
    Args:
        segmentation_result: Objects data from segmenter
        human_threshold: Similarity threshold for human classification
        animal_threshold: Similarity threshold for animal classification
        
    Returns:
        Dictionary with classified objects
    """
    logging.info(f"Classifying {len(segmentation_result['objects'])} objects")
    
    # Initialize sentence transformer model for semantic similarity
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Reference terms for classification
    human_terms = ['person', 'human', 'man', 'woman', 'child', 'people', 'player', 'athlete']
    animal_terms = ['dog', 'cat', 'horse', 'cow', 'bird', 'animal', 'pet', 'wildlife', 'mammal']
    
    # Get embeddings for reference terms
    human_embeddings = model.encode(human_terms)
    animal_embeddings = model.encode(animal_terms)
    
    classified_objects = []
    for obj in segmentation_result['objects']:
        class_name = obj['class_name']
        
        # Get embedding for this class name
        class_embedding = model.encode([class_name])
        
        # Calculate similarities
        human_similarities = cosine_similarity(class_embedding, human_embeddings).max()
        animal_similarities = cosine_similarity(class_embedding, animal_embeddings).max()
        
        # Classify based on highest similarity above threshold
        if human_similarities >= human_threshold:
            semantic_category = 'human'
            confidence = float(human_similarities)
        elif animal_similarities >= animal_threshold:
            semantic_category = 'animal'
            confidence = float(animal_similarities)
        else:
            semantic_category = 'other'
            confidence = max(float(human_similarities), float(animal_similarities))
        
        # Add classification info to object
        obj['semantic_category'] = semantic_category
        obj['semantic_confidence'] = confidence
        classified_objects.append(obj)
    
    # Count classifications
    stats = {
        'human': sum(1 for obj in classified_objects if obj['semantic_category'] == 'human'),
        'animal': sum(1 for obj in classified_objects if obj['semantic_category'] == 'animal'),
        'other': sum(1 for obj in classified_objects if obj['semantic_category'] == 'other')
    }
    
    logging.info(f"Classification results: {stats}")
    return {
        'scene_number': segmentation_result['scene_number'],
        'objects': classified_objects,
        'classification_stats': stats
    }


def extract_keypoints(classification_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract keypoints from classified objects using mmpose or canny edge detection.
    
    Args:
        classification_result: Classified objects from classifier
        
    Returns:
        Dictionary with objects containing keypoints
    """
    logging.info(f"Extracting keypoints from {len(classification_result['objects'])} objects")
    
    # Initialize MMPose inferencers
    human_inferencer = MMPoseInferencer('human')
    animal_inferencer = MMPoseInferencer('animal')
    
    objects_with_keypoints = []
    for obj in classification_result['objects']:
        category = obj['semantic_category']
        cropped_image = obj['cropped_image']
        
        keypoints = []
        
        if category == 'human':
            # Use MMPose human model
            try:
                results = human_inferencer(cropped_image, show=False)
                if results and 'predictions' in results:
                    for pred in results['predictions']:
                        if 'keypoints' in pred:
                            kpts = pred['keypoints']
                            # Normalize keypoints to [0,1] range
                            h, w = cropped_image.shape[:2]
                            normalized_kpts = []
                            for kpt in kpts:
                                x, y, conf = kpt[:3]
                                normalized_kpts.append([x/w, y/h, conf])
                            keypoints = normalized_kpts
                            break
            except Exception as e:
                logging.warning(f"Failed to extract human keypoints: {e}")
        
        elif category == 'animal':
            # Use MMPose animal model
            try:
                results = animal_inferencer(cropped_image, show=False)
                if results and 'predictions' in results:
                    for pred in results['predictions']:
                        if 'keypoints' in pred:
                            kpts = pred['keypoints']
                            # Normalize keypoints to [0,1] range  
                            h, w = cropped_image.shape[:2]
                            normalized_kpts = []
                            for kpt in kpts:
                                x, y, conf = kpt[:3]
                                normalized_kpts.append([x/w, y/h, conf])
                            keypoints = normalized_kpts
                            break
            except Exception as e:
                logging.warning(f"Failed to extract animal keypoints: {e}")
        
        else:  # category == 'other'
            # Use Canny edge detection for other objects
            try:
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Find contours from edges
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get largest contour and sample keypoints
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Sample points along the contour
                    h, w = cropped_image.shape[:2]
                    sampled_points = []
                    
                    if len(largest_contour) >= 8:
                        # Sample evenly spaced points
                        step = max(1, len(largest_contour) // 8)
                        for i in range(0, len(largest_contour), step):
                            point = largest_contour[i][0]  # Extract x, y from contour format
                            x, y = point
                            # Normalize to [0,1] range and add confidence of 1.0
                            sampled_points.append([x/w, y/h, 1.0])
                    
                    keypoints = sampled_points[:8]  # Limit to 8 keypoints
            except Exception as e:
                logging.warning(f"Failed to extract canny keypoints: {e}")
        
        # Add keypoints to object
        obj['keypoints'] = keypoints
        obj['has_keypoints'] = len(keypoints) > 0
        objects_with_keypoints.append(obj)
    
    # Count objects with keypoints
    with_keypoints = sum(1 for obj in objects_with_keypoints if obj['has_keypoints'])
    logging.info(f"Extracted keypoints for {with_keypoints}/{len(objects_with_keypoints)} objects")
    
    return {
        'scene_number': classification_result['scene_number'],
        'objects': objects_with_keypoints,
        'objects_with_keypoints': with_keypoints,
        'total_objects': len(objects_with_keypoints)
    }


def process_video_pipeline(video_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Process a video through the complete PointStream pipeline.
    
    Args:
        video_path: Path to input video
        output_dir: Optional output directory for saving results
        
    Returns:
        Complete processing results
    """
    start_time = time.time()
    logging.info(f"Starting PointStream pipeline for: {video_path}")
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    
    # STEP 1: Split video into scenes
    scenes = split_video_into_scenes(video_path)
    
    processed_scenes = []
    
    for scene in scenes:
        scene_start = time.time()
        logging.info(f"Processing scene {scene['scene_number']}")
        
        # STEP 2: Segment objects in scene
        segmentation_result = segment_objects_in_scene(scene)
        
        # STEP 3: Create panorama with object inpainting
        stitching_result = stitch_scene_panorama(scene, segmentation_result)
        
        # STEP 4: Classify objects semantically
        classification_result = classify_objects(segmentation_result)
        
        # STEP 5: Extract keypoints
        keypoint_result = extract_keypoints(classification_result)
        
        # Combine results
        scene_result = {
            'scene_number': scene['scene_number'],
            'start_time': scene['start_time'],
            'end_time': scene['end_time'],
            'frame_count': scene['frame_count'],
            'panorama': stitching_result['panorama'],
            'scene_type': stitching_result['scene_type'],
            'objects': keypoint_result['objects'],
            'total_objects': keypoint_result['total_objects'],
            'objects_with_keypoints': keypoint_result['objects_with_keypoints'],
            'classification_stats': classification_result['classification_stats'],
            'processing_time': time.time() - scene_start
        }
        
        processed_scenes.append(scene_result)
        
        # Save scene results if output directory provided
        if output_dir:
            scene_output_dir = output_path / f"scene_{scene['scene_number']:04d}"
            scene_output_dir.mkdir(exist_ok=True)
            
            # Save panorama
            if stitching_result['panorama'] is not None:
                panorama_path = scene_output_dir / "panorama.jpg"
                cv2.imwrite(str(panorama_path), stitching_result['panorama'])
            
            # Save object crops
            objects_dir = scene_output_dir / "objects"
            objects_dir.mkdir(exist_ok=True)
            
            for obj in keypoint_result['objects']:
                obj_filename = f"{obj['object_id']}.jpg"
                obj_path = objects_dir / obj_filename
                cv2.imwrite(str(obj_path), obj['cropped_image'])
            
            # Save metadata
            metadata = {
                'scene_number': scene_result['scene_number'],
                'start_time': scene_result['start_time'],
                'end_time': scene_result['end_time'],
                'frame_count': scene_result['frame_count'],
                'scene_type': scene_result['scene_type'],
                'total_objects': scene_result['total_objects'],
                'objects_with_keypoints': scene_result['objects_with_keypoints'],
                'classification_stats': scene_result['classification_stats'],
                'processing_time': scene_result['processing_time'],
                'objects': [
                    {
                        'object_id': obj['object_id'],
                        'class_name': obj['class_name'],
                        'semantic_category': obj['semantic_category'],
                        'confidence': obj['confidence'],
                        'semantic_confidence': obj['semantic_confidence'],
                        'bbox': obj['bbox'],
                        'keypoints': obj['keypoints'],
                        'has_keypoints': obj['has_keypoints']
                    } for obj in keypoint_result['objects']
                ]
            }
            
            metadata_path = scene_output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Summary statistics
    total_objects = sum(scene['total_objects'] for scene in processed_scenes)
    total_keypoints = sum(scene['objects_with_keypoints'] for scene in processed_scenes)
    
    results = {
        'input_video': video_path,
        'total_scenes': len(processed_scenes),
        'total_objects': total_objects,
        'total_objects_with_keypoints': total_keypoints,
        'total_processing_time': total_time,
        'scenes': processed_scenes
    }
    
    logging.info(f"Pipeline completed in {total_time:.2f}s")
    logging.info(f"Processed {len(processed_scenes)} scenes with {total_objects} objects")
    logging.info(f"Extracted keypoints for {total_keypoints} objects")
    
    return results


def main():
    """Main entry point for the PointStream pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='PointStream - Streamlined Video Processing Pipeline')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output-dir', help='Output directory for saving results')
    parser.add_argument('--scene-threshold', type=float, default=30.0, 
                       help='Scene change detection threshold (default: 30.0)')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='YOLO detection confidence threshold (default: 0.25)')
    parser.add_argument('--human-threshold', type=float, default=0.6,
                       help='Human classification threshold (default: 0.6)')
    parser.add_argument('--animal-threshold', type=float, default=0.5,
                       help='Animal classification threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    try:
        results = process_video_pipeline(
            video_path=args.video_path,
            output_dir=args.output_dir
        )
        
        print(f"\nProcessing complete!")
        print(f"Total scenes: {results['total_scenes']}")
        print(f"Total objects: {results['total_objects']}")
        print(f"Objects with keypoints: {results['total_objects_with_keypoints']}")
        print(f"Processing time: {results['total_processing_time']:.2f}s")
        
        if args.output_dir:
            print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()