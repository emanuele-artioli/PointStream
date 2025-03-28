import os
import time
import subprocess
import cv2
import shutil
import concurrent.futures
from ultralytics import YOLO, FastSAM
import numpy as np
import pandas as pd

def extract_estimations(result, players_ids):
    """Extracts object info (ID, class, confidence, bbox, keypoints) from a YOLO result into class dictionary."""
    boxes = getattr(result, 'boxes', [])
    keypoints = getattr(result, 'keypoints', [])
    people = {}
    rackets = {}
    balls = {}
    for i, box in enumerate(boxes):
        obj_id = int(box.id) if box.id is not None else 9999
        obj = {
            'cls_id': int(box.cls[0]),
            'conf': float(box.conf),
            'bbox': tuple(map(int, box.xyxy[0])),
            'keypoints': keypoints.xy[i].cpu().numpy().astype(np.uint8) if keypoints and i < len(keypoints.data) else None
        }
        if obj['cls_id'] == 0:
            obj['cls_id'] = 'person'
            # if enough players id are already known, and the current person id is not one of them, skip
            if len(players_ids) >= 2 and obj_id not in players_ids:
                continue
            people[obj_id] = obj
        elif obj['cls_id'] == 32:
            obj['cls_id'] = 'ball'
            balls[obj_id] = obj
        elif obj['cls_id'] == 38:
            obj['cls_id'] = 'racket'
            rackets[obj_id] = obj
    return people, rackets, balls

def compute_overlap_area(a, b):
    """Computes the overlapping area between two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)

def find_overlapping_player(racket, people):
    """Finds the person that overlaps the most with a racket."""
    max_overlap = 0
    max_person = None
    for person_id, person in people.items():
        overlap = compute_overlap_area(racket['bbox'], person['bbox'])
        if overlap > max_overlap:
            max_overlap = overlap
            max_person = person_id
    return max_person
     
def fuse_bounding_boxes(box1, box2):
    '''Fuse overlapping bounding boxes into a single one.'''
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return (min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4))

def save_object(object_id, object, obj_img, experiment_folder, frame_id):
    '''Save an object to its subfolder and log its bounding box coordinates.'''
    obj_folder = os.path.join(experiment_folder, f'{object["cls_id"]}_{object_id}')
    os.makedirs(obj_folder, exist_ok=True)
    cv2.imwrite(os.path.join(obj_folder, f'{frame_id}.png'), obj_img)
    
    # Save bounding box in a list
    x1, y1, x2, y2 = object['bbox']
    obj_info = [frame_id, object_id, object['cls_id'], x1, y1, x2, y2] 
    if object['keypoints'] is not None:
        # Add the keypoints to the object info list
        keypoints = object['keypoints']
        keypoints = keypoints.reshape(-1, 2)
        keypoints = keypoints
        obj_info.extend(keypoints.flatten().tolist())
    return obj_info

def stitch_background_images(background_folder, n_samples=50):
    """Combines periodic background frames into a single stitched image."""
    all_images = []
    n_images = len(os.listdir(background_folder))
    for i, name in enumerate(sorted(os.listdir(background_folder))):
        if name.endswith(".png") and i % (n_images // n_samples) == 0:
            img = cv2.imread(os.path.join(background_folder, name))
            if img is not None:
                all_images.append(img)
    if not all_images:
        print("No background images found.")
        return None
    stitched = all_images[0].copy()
    for img in all_images[1:]:
        mask = (stitched == 0)
        stitched[mask] = img[mask]
    return stitched

def generate_mask(img, model, conf=0.01, iou=0.01, imgsz=None, device=None, classes=None):
    '''Use YOLO to segment an object from the background.'''
    results = model.predict(
        source = img,
        conf = conf,
        iou = iou,
        imgsz = imgsz,
        half = 'cuda' in device, # use half precision if on cuda
        device = device,
        batch = 1,
        max_det = 10,
        classes = [0, 32, 38], # person, ball, racket
        retina_masks = True
    )
    # If multiple objects are detected, find the first person and add their mask, then find the first racket and add its mask, then find the first ball and add its mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    person_mask, racket_mask, ball_mask = None, None, None

    # Find the most central person
    img_center = (img.shape[1] / 2, img.shape[0] / 2)
    min_distance = float('inf')
    central_person_idx = None

    for i, box in enumerate(results[0].boxes):
        cls = box.cls[0]
        if cls == 0:  # Person
            bbox = box.xyxy[0]
            person_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            distance = ((person_center[0] - img_center[0]) ** 2 + (person_center[1] - img_center[1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                central_person_idx = i

    # Assign the mask for the most central person
    if central_person_idx is not None:
        person_mask = results[0].masks.data[central_person_idx].cpu().numpy().astype(np.uint8) * 255

        # Find the racket that overlaps the most with the central person
        max_overlap = 0
        for i, box in enumerate(results[0].boxes):
            cls = box.cls[0]
            if cls == 38:  # Racket
                racket_bbox = box.xyxy[0]
                person_bbox = results[0].boxes[central_person_idx].xyxy[0]
                overlap = compute_overlap_area(racket_bbox, person_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    racket_mask = results[0].masks.data[i].cpu().numpy().astype(np.uint8) * 170

    # Assign the mask for the first ball
    for i, box in enumerate(results[0].boxes):
        cls = box.cls[0]
        if cls == 32 and ball_mask is None:  # Ball
            ball_mask = results[0].masks.data[i].cpu().numpy().astype(np.uint8) * 85
            break

    # Combine masks
    if person_mask is not None:
        mask = cv2.bitwise_or(mask, person_mask)
    if racket_mask is not None:
        mask = cv2.bitwise_or(mask, racket_mask)
    if ball_mask is not None:
        mask = cv2.bitwise_or(mask, ball_mask)

    return mask

def expand_bbox(bbox, scale, frame_shape):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    new_w = w * (1 + scale)
    new_h = h * (1 + scale)
    new_x1 = max(int(cx - new_w / 2), 0)
    new_y1 = max(int(cy - new_h / 2), 0)
    new_x2 = min(int(cx + new_w / 2), frame_shape[1])
    new_y2 = min(int(cy + new_h / 2), frame_shape[0])
    return (new_x1, new_y1, new_x2, new_y2)

def main():
    # Start timing the script
    start = time.time()

    # Initialize a list to store timing data
    timing_data = []

    # Get environment variables
    device = os.environ.get("DEVICE", "cuda")
    working_dir = os.environ.get("WORKING_DIR", "/PointStream")
    video_folder = os.environ.get("VIDEO_FOLDER", "/scenes")
    timing_csv_path = os.path.join(working_dir, "experiments/timing_data.csv")
    # If no video_file is provided, use every video in the folder
    video_file = os.environ.get("VIDEO_FILE")
    if not video_file:
        all_videos = [v for v in os.listdir(video_folder) if v.endswith(('.mp4','.mov','.avi'))]
    else:
        all_videos = [video_file]
    # Load the pose estimation model
    estimation_model = os.environ.get("EST_MODEL", None)
    if 'yolo' in estimation_model:
        estimation_model = YOLO(estimation_model)
    else:
        raise ValueError('Model not supported.')

    # Process each video
    for vid in all_videos:
        experiment_folder = f'{working_dir}/experiments/{os.path.basename(vid).split(".")[0]}'
        video_file = os.path.join(video_folder, vid)
        background_folder = os.path.join(experiment_folder, 'background')
        objects_folder = os.path.join(experiment_folder, 'objects')
        csv_file_path = os.path.join(experiment_folder, 'bounding_boxes.csv')
        os.makedirs(objects_folder, exist_ok=True)
        os.makedirs(background_folder, exist_ok=True)

        
        # Keep track of players' IDs across frames
        players_ids = set()
        # Keep track of frame number (to know how many frames are in the video)
        frame_id = 0
        # Keep track of the rows to write to the CSV file
        df_rows = []

        # First, a pass to detect objects of interest in the video
        estimation_start = time.time()
        results = estimation_model.track(
            source = video_file,
            conf = 0.25,
            iou = 0.1,
            imgsz = 640,
            half = 'cuda' in device, # use half precision if on cuda
            device = device,
            batch = 10,
            max_det = 30,
            classes = [0, 32, 38], # person, ball, racket
            retina_masks = True,
            stream = True,
            persist = True,
        )

        for frame_id, result in enumerate(results):
            frame_img = result.orig_img
            people, rackets, balls = extract_estimations(result, players_ids)

            players_with_rackets = set()
            # for each racket find the player they overlap with the most,
            for racket_id, racket in rackets.items():
                player_id = find_overlapping_player(racket, people)
                if player_id is not None:
                    players_ids.add(player_id)
                    # If the racket overlaps with a player, fuse their bounding boxes
                    people[player_id]['bbox'] = fuse_bounding_boxes(people[player_id]['bbox'], racket['bbox'])
                    players_with_rackets.add(player_id)

            # save each ball
            for ball_id, ball in balls.items():
                ball_img = frame_img[ball['bbox'][1]:ball['bbox'][3], ball['bbox'][0]:ball['bbox'][2]]
                df_rows.append(save_object(ball_id, ball, ball_img, objects_folder, frame_id))

            # Save every person (once 2 players are detected, only they will be present in people)
            for person_id, person in people.items():
                # Expand bounding box for recognized players without a racket in this frame
                if person_id in players_ids and person_id not in players_with_rackets:
                    person['bbox'] = expand_bbox(person['bbox'], 0.2, frame_img.shape[:2])
                person_img = frame_img[person['bbox'][1]:person['bbox'][3], person['bbox'][0]:person['bbox'][2]]
                df_rows.append(save_object(person_id, person, person_img, objects_folder, frame_id))

            # Remove every person, racket, and ball from the background
            all_boxes = np.zeros_like(frame_img, dtype=np.uint8)
            for obj in [*people.values(), *rackets.values(), *balls.values()]:
                x1, y1, x2, y2 = obj["bbox"]
                all_boxes[y1:y2, x1:x2] = 1
            frame_img[all_boxes > 0] = 0  # Apply boxes at once

            # Save the background
            cv2.imwrite(os.path.join(background_folder, f'{frame_id}.png'), frame_img)

        estimation_time = time.time() - estimation_start
        estimation_fps = frame_id / estimation_time if estimation_time > 0 else 0
        timing_data.append({"video": vid, "task": "keypoint _extraction", "time_taken": estimation_time, "fps": estimation_fps})

        # Stitch background images
        stitching_start = time.time()
        stitched = stitch_background_images(background_folder)
        if stitched is not None:
            cv2.imwrite(os.path.join(experiment_folder, 'background.png'), stitched)
        shutil.rmtree(background_folder)
        stitching_time = time.time() - stitching_start
        timing_data.append({"video": vid, "task": "background_stitching", "time_taken": stitching_time, "fps": None})

        # Print the total time taken
        print(f'Total time taken for keypoint extraction: {time.time() - start} seconds')

        # Run a second pass of YOLO segmentation on the objects in the subfolders
        start = time.time()

        # Load the segmentation model
        segmentation_model = os.environ.get("SEG_MODEL", None)
        if 'yolo' in segmentation_model:
            segmentation_model = YOLO(segmentation_model)
        else:
            raise ValueError('Model not supported.')

        min_frames = frame_id * 0.9
        segmentation_start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for obj in os.listdir(objects_folder):
                obj_folder = os.path.join(objects_folder, obj)
                if os.path.isdir(obj_folder):
                    # Delete people folders that are missing too many frames (this should be every person besides players)
                    if obj.startswith("person_") and len(os.listdir(obj_folder)) < min_frames:
                        shutil.rmtree(obj_folder)
                    else:
                        # Segment the objects in the subfolders
                        for img in os.listdir(obj_folder):
                            img_path = os.path.join(obj_folder, img)
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                mask = generate_mask(img, segmentation_model, imgsz=img.shape[:2], device=device)
                                cv2.imwrite(img_path.replace('.png', '_mask.png'), mask)
        segmentation_time = time.time() - segmentation_start
        segmentation_fps = frame_id / segmentation_time if segmentation_time > 0 else 0
        timing_data.append({"video": vid, "task": "segmentation", "time_taken": segmentation_time, "fps": segmentation_fps})
        print(f'Total time taken for segmentation: {time.time() - start} seconds')

        # Save all collected data to CSV at once
        df_columns = ['frame_id', 'object_id', 'class_id', 'x1', 'y1', 'x2', 'y2']
        # Add keypoints columns if present
        if df_rows and isinstance(df_rows[0], list) and len(df_rows[0]) > 7:
            keypoints_count = (len(df_rows[0]) - 7) // 2
            for i in range(keypoints_count):
                df_columns.extend([f'keypoint_{i}_x', f'keypoint_{i}_y'])
        # Create a DataFrame and save it to CSV
        df = pd.DataFrame(df_rows, columns=df_columns)
        df.to_csv(csv_file_path, index=False)

        # Zip the experiment folder
        shutil.make_archive(experiment_folder, 'zip', experiment_folder)
        shutil.rmtree(experiment_folder)

    # Save timing data to a CSV file using pandas
    timing_df = pd.DataFrame(timing_data)
    timing_df.to_csv(timing_csv_path, index=False)

if __name__ == "__main__":
    main()