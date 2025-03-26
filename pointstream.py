import os
import time
import subprocess
import cv2
import shutil
import concurrent.futures
from ultralytics import YOLO, FastSAM
import numpy as np
import csv

def extract_detections(result, players_ids):
    """Extracts object info (ID, class, confidence, bbox, mask) from a YOLO result into class dictionary."""
    boxes = getattr(result, 'boxes', [])
    masks = getattr(result, 'masks', None)
    people = {}
    rackets = {}
    balls = {}
    for i, box in enumerate(boxes):
        obj_id = int(box.id) if box.id is not None else 9999
        obj = {
            'cls_id': int(box.cls[0]),
            'conf': float(box.conf),
            'bbox': tuple(map(int, box.xyxy[0])),
            'mask': masks.data[i].cpu().numpy().astype(np.uint8) if masks and i < len(masks.data) else None
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

def save_object(object_id, object, obj_img, experiment_folder, frame_id, csv_writer):
    '''Save an object to its subfolder and log its bounding box coordinates.'''
    obj_folder = os.path.join(experiment_folder, f'{object["cls_id"]}_{object_id}')
    os.makedirs(obj_folder, exist_ok=True)
    cv2.imwrite(os.path.join(obj_folder, f'{frame_id}.png'), obj_img)
    
    # Save bounding box coordinates to CSV
    x1, y1, x2, y2 = object['bbox']
    csv_writer.writerow([frame_id, object_id, object['cls_id'], x1, y1, x2, y2])

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

def segment_with_SAM(img, model, prompt):
    '''Use SAM to segment an object from the background.'''
    results = model(img, texts=prompt)
    mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
    return mask

def segment_with_YOLO(img, model, conf=0.01, iou=0.01, imgsz=None, device=None, classes=None):
    '''Use YOLO to segment an object from the background.'''
    results = model.predict(
                source = img,
                conf = conf,
                iou = iou,
                imgsz = imgsz,
                half = 'cuda' in device, # use half precision if on cuda
                device = device,
                batch = 1,
                max_det = 3,
                classes = [0, 32, 38], # person, ball, racket
                retina_masks = True
            )
    # If multiple objects are detected, find the first person and add their mask, then find the first racket and add its mask, then find the first ball and add its mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, box in enumerate(results[0].boxes):
        if box.cls[0] == 0:
            person_mask = results[0].masks.data[i].cpu().numpy().astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, person_mask)
            break
    for i, box in enumerate(results[0].boxes):
        if box.cls[0] == 38:
            racket_mask = results[0].masks.data[i].cpu().numpy().astype(np.uint8) * 170
            mask = cv2.bitwise_or(mask, racket_mask)
            break
    for i, box in enumerate(results[0].boxes):
        if box.cls[0] == 32:
            ball_mask = results[0].masks.data[i].cpu().numpy().astype(np.uint8) * 85
            mask = cv2.bitwise_or(mask, ball_mask)
            break
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

    # Get environment variables
    device = os.environ.get("DEVICE", "cpu")
    working_dir = os.environ.get("WORKING_DIR", "/PointStream")
    video_folder = os.environ.get("VIDEO_FOLDER", "/scenes")
    video_folder = os.path.join(working_dir, video_folder)
    # If no video_file is provided, use every video in the folder
    video_file = os.environ.get("VIDEO_FILE")
    if not video_file:
        all_videos = [v for v in os.listdir(video_folder) if v.endswith(('.mp4','.mov','.avi'))]
    else:
        all_videos = [video_file]
    # Load the object detection model
    detection_model = os.environ.get("DET_MODEL", None)
    if 'yolo' in detection_model:
        detection_model = YOLO(detection_model)
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

        # Open CSV file for writing bounding box coordinates
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame_id', 'object_id', 'class_id', 'x1', 'y1', 'x2', 'y2'])

            # Keep track of players' IDs across frames
            players_ids = set()

            # First, a pass to detect objects of interest in the video
            results = detection_model.track(
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
                people, rackets, balls = extract_detections(result, players_ids)

                players_with_rackets = set()
                if rackets:
                    # for each racket find the player they overlap with the most,
                    for racket_id, racket in rackets.items():
                        player_id = find_overlapping_player(racket, people)
                        if player_id is not None:
                            players_ids.add(player_id)
                            # If the racket overlaps with a player, fuse their bounding boxes
                            people[player_id]['bbox'] = fuse_bounding_boxes(people[player_id]['bbox'], racket['bbox'])
                            players_with_rackets.add(player_id)

                if balls:
                    # save each ball
                    for ball_id, ball in balls.items():
                        ball_img = frame_img[ball['bbox'][1]:ball['bbox'][3], ball['bbox'][0]:ball['bbox'][2]]
                        save_object(ball_id, ball, ball_img, objects_folder, frame_id, csv_writer)

                # Save every person (once 2 players are detected, only they will be present in people)
                for person_id, person in people.items():
                    # Expand bounding box for recognized players without a racket in this frame
                    if person_id in players_ids and person_id not in players_with_rackets:
                        person['bbox'] = expand_bbox(person['bbox'], 0.5, frame_img.shape[:2])
                    person_img = frame_img[person['bbox'][1]:person['bbox'][3], person['bbox'][0]:person['bbox'][2]]
                    save_object(person_id, person, person_img, objects_folder, frame_id, csv_writer)

                # Remove every person, racket, and ball from the background
                for obj in people.values():
                    x1, y1, x2, y2 = obj['bbox']
                    frame_img[y1:y2, x1:x2] = 0
                for obj in rackets.values():
                    x1, y1, x2, y2 = obj['bbox']
                    frame_img[y1:y2, x1:x2] = 0
                for obj in balls.values():
                    x1, y1, x2, y2 = obj['bbox']
                    frame_img[y1:y2, x1:x2] = 0

                # Save the background
                cv2.imwrite(os.path.join(background_folder, f'{frame_id}.png'), frame_img)

        # Get maximum number of frames from the frame id in the last row of the CSV file
        with open(os.path.join(experiment_folder, 'bounding_boxes.csv')) as f:
            frame_id = int(f.readlines()[-1].split(',')[0])
        min_frames = frame_id * 0.9
        # Delete people folders that are missing too many frames (i.e., everyone besides players)
        for obj in os.listdir(objects_folder):
            if obj.startswith('person_'):
                num_frames = len(os.listdir(os.path.join(objects_folder, obj)))
                if num_frames < min_frames:
                    shutil.rmtree(os.path.join(objects_folder, obj))

        # Stitch background images
        background_folder = os.path.join(experiment_folder, 'background')
        stitched = stitch_background_images(background_folder)
        if stitched is not None:
            cv2.imwrite(os.path.join(experiment_folder, 'background.png'), stitched)
        shutil.rmtree(background_folder)
        
        # Zip the experiment folder
        # shutil.make_archive(experiment_folder, 'zip', experiment_folder)
        # shutil.rmtree(experiment_folder)

        # Print the total time taken
        print(f'Total time taken for detection: {time.time() - start} seconds')

        # Run a second pass of YOLO segmentation on the objects in the subfolders
        start = time.time()

        # Load the object detection model
        segmentation_model = os.environ.get("SEG_MODEL", None)
        if 'yolo' in segmentation_model:
            segmentation_model = YOLO(segmentation_model)
        else:
            raise ValueError('Model not supported.')

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for obj in os.listdir(objects_folder):
                obj_folder = os.path.join(objects_folder, obj)
                if os.path.isdir(obj_folder):
                    for img in os.listdir(obj_folder):
                        img_path = os.path.join(obj_folder, img)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            mask = segment_with_YOLO(img, segmentation_model, imgsz=img.shape[:2], device=device)
                            cv2.imwrite(img_path.replace('.png', '_mask.png'), mask)
        print(f'Total time taken for segmentation: {time.time() - start} seconds')
        

if __name__ == "__main__":
    main()

# TODO: Right now I have only implemented the object detection part of the script. 
# The next steps would be to implement the object segmentation and keypoints (or shape, that thing that Farzad did) estimation.