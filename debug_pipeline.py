#!/usr/bin/env python3
"""
Debug the segmentation pipeline processing
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

try:
    from server.scripts.segmenter import Segmenter
    
    print("✅ Successfully imported Segmenter")
    
    # Initialize segmenter
    segmenter = Segmenter()
    print(f"✅ Segmenter initialized")
    print(f"   Confidence threshold: {segmenter.confidence_threshold}")
    print(f"   Min object area: {segmenter.min_object_area}")
    print(f"   Max objects: {segmenter.max_objects}")
    print(f"   Selection strategy: {segmenter.selection_strategy}")
    
    # Load a test frame
    video_path = "/home/itec/emanuele/PointStream/stitched.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read frame")
        exit(1)
    
    print(f"📷 Frame loaded: {frame.shape}")
    
    # Test the raw model inference
    print("\n🔍 Testing raw model inference...")
    raw_results = segmenter.model_with_tracking.track(frame, 
                                      conf=segmenter.confidence_threshold,
                                      iou=segmenter.iou_threshold,
                                      device=segmenter.device,
                                      retina_masks=True,
                                      max_det=0,
                                      classes=segmenter.classes,
                                      imgsz=segmenter.yolo_image_size,
                                      half=segmenter.yolo_half_precision,
                                      agnostic_nms=segmenter.agnostic_nms,
                                      verbose=False,
                                      persist=True)
    
    print(f"✅ Raw inference completed")
    
    # Check raw results
    if hasattr(raw_results[0], 'boxes') and raw_results[0].boxes is not None:
        raw_count = len(raw_results[0].boxes)
        print(f"🔍 Raw detections: {raw_count}")
        
        if raw_count > 0:
            print("📊 Raw detection details:")
            for i, box in enumerate(raw_results[0].boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0]) if box.cls is not None else None
                xyxy = box.xyxy[0].cpu().numpy()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                class_name = segmenter.model_with_tracking.names.get(cls, f"class_{cls}")
                print(f"   Detection {i+1}: {class_name} (class={cls}), conf={conf:.3f}, area={area:.0f}")
    else:
        raw_count = 0
        print("❌ No raw detections")
    
    # Test the pipeline processing
    print(f"\n🔧 Testing pipeline processing...")
    processed_data = segmenter._extract_detection_data(raw_results, frame_index=0, with_tracking=True)
    
    processed_count = len(processed_data['objects'])
    print(f"🔍 Processed objects: {processed_count}")
    
    if processed_count > 0:
        print("📊 Processed object details:")
        for i, obj in enumerate(processed_data['objects']):
            print(f"   Object {i+1}: {obj['class_name']} (class={obj['class_id']}), conf={obj['confidence']:.3f}, area={obj['area']:.0f}, mask_area={obj['mask_area']}")
    else:
        print("❌ No processed objects")
    
    # Test the full segmentation method
    print(f"\n🎯 Testing full segmentation method...")
    full_results = segmenter.segment_frames_only([frame])
    
    full_count = len(full_results['frames_data'][0]['objects']) if full_results['frames_data'] else 0
    print(f"🔍 Full method objects: {full_count}")
    
    if full_count > 0:
        print("📊 Full method object details:")
        for i, obj in enumerate(full_results['frames_data'][0]['objects']):
            print(f"   Object {i+1}: {obj['class_name']} (class={obj['class_id']}), conf={obj['confidence']:.3f}, area={obj['area']:.0f}, mask_area={obj['mask_area']}")
    else:
        print("❌ No objects from full method")
    
    # Check the differences
    print(f"\n📊 Summary:")
    print(f"   Raw model detections: {raw_count}")
    print(f"   Processed detections: {processed_count}")
    print(f"   Full method detections: {full_count}")
    
    if raw_count > 0 and (processed_count == 0 or full_count == 0):
        print(f"\n⚠️  ISSUE FOUND: Raw model detects {raw_count} objects but pipeline outputs {full_count}")
        print(f"   This suggests filtering is removing all objects!")
        print(f"   Check min_object_area ({segmenter.min_object_area}) and max_objects ({segmenter.max_objects})")
        
        # Show which step is filtering out objects
        if processed_count == 0:
            print(f"   🔍 Objects filtered out in _extract_detection_data method")
        elif full_count == 0:
            print(f"   🔍 Objects filtered out in segment_frames_only method")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
