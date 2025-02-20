import os
import numpy as np
import cv2
import argparse
from ultralytics import YOLO
import shutil

def calculate_average_movement(box_coordinates, frame_interval=30):
    movements = {}
    for obj_id, coords in box_coordinates.items():
        total_movement = 0
        count = 0
        for i in range(frame_interval, len(coords), frame_interval):
            x1_prev, y1_prev, x2_prev, y2_prev = coords[i-frame_interval]
            x1_curr, y1_curr, x2_curr, y2_curr = coords[i]
            movement = np.sqrt((x1_curr - x1_prev)**2 + (y1_curr - y1_prev)**2 + (x2_curr - x2_prev)**2 + (y2_curr - y2_prev)**2)
            total_movement += movement
            count += 1
        average_movement = total_movement / count if count > 0 else 0
        movements[obj_id] = average_movement
    return movements

def perform_instance_segmentation(video_file, segmented_folder, background_folder, model, N_saved=2):
    results = model.track(
        video_file, 
        conf=0.4, 
        iou=0.5,
        imgsz=960,
        classes=[0], # 0: person, 32: sports ball, 38: tennis racket
        stream=True,
        persist=True,
        retina_masks=True,
        # tracker='/app/custom_botsort.yaml',
    )

    for frame_id, result in enumerate(results):
        background = result.orig_img
        boxes = result.boxes
        masks = result.masks.data
        box_coordinates = {}

        for box, mask in zip(boxes, masks):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_id = int(box.id)

            # Extract bounding box coordinates
            if obj_id not in box_coordinates:
                box_coordinates[obj_id] = []
            box_coordinates[obj_id].append((x1, y1, x2, y2))

            # Save bounding box coordinates in a single csv file
            coordinates_file = os.path.join(segmented_folder, 'coordinates.csv')
            with open(coordinates_file, 'a') as f:
                f.write(f"{frame_id},{obj_id},{x1},{y1},{x2},{y2}\n")

            # Segment object from background
            mask_arr = mask.cpu().numpy().astype(np.uint8)
            output_folder = os.path.join(segmented_folder, f"object_{obj_id}")
            os.makedirs(output_folder, exist_ok=True)
            object_img = cv2.bitwise_and(background, background, mask=mask_arr)
            background = cv2.bitwise_and(background, background, mask=1-mask_arr)

            # Crop image based on bounding box and save
            cropped_img = object_img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(output_folder, f"{frame_id:04d}.png"), cropped_img)

        # Save background image
        cv2.imwrite(os.path.join(background_folder, f"{frame_id:04d}.png"), background)

    # # Filter box coordinates to keep only the top movers
    # movements = calculate_average_movement(box_coordinates)
    # sorted_movements = sorted(movements.items(), key=lambda item: item[1], reverse=True)
    # top_movers = [obj_id for obj_id, _ in sorted_movements[:N_saved]]

    # # Delete objects that are not top movers
    # for obj_id in box_coordinates:
    #     if obj_id not in top_movers:
    #         shutil.rmtree(os.path.join(segmented_folder, f"object_{obj_id}")) 

def stitch_background_images(background_folder, stride=50):
    background_images = []
    for i, filename in enumerate(sorted(os.listdir(background_folder))):
        if filename.endswith(".png") and i % stride == 0:
            img_path = os.path.join(background_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                background_images.append(img)
    
    if not background_images:
        print("No background images found.")
        return None
    
    # Initialize the stitched image with the first background image
    stitched_image = background_images[0].copy()
    
    for img in background_images[1:]:
        # Overlay the images
        mask = (stitched_image == 0)
        stitched_image[mask] = img[mask]
    
    return stitched_image

def main():
    parser = argparse.ArgumentParser(description='Perform instance segmentation on video.')
    parser.add_argument('--video_file', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--segmented_folder', type=str, required=True, help='Folder to save segmented objects.')
    args = parser.parse_args()

    os.makedirs(args.segmented_folder, exist_ok=True)
    background_folder = os.path.join(args.segmented_folder, 'background')
    os.makedirs(background_folder, exist_ok=True)
    
    perform_instance_segmentation(args.video_file, args.segmented_folder, background_folder, model=YOLO('yolo11n-seg.pt'))

    stitched_image = stitch_background_images(background_folder)
    if stitched_image is not None:
        cv2.imwrite(os.path.join(args.segmented_folder, 'full_background.png'), stitched_image)

    # Remove the background folder
    # shutil.rmtree(background_folder)

if __name__ == "__main__":
    main()