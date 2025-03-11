import numpy as np
import os
import shutil
import cv2
import argparse
import csv
from ultralytics import YOLO

def extract_detections(result):
    """Extracts object info (ID, class, confidence, bbox, mask) from a YOLO result."""
    detections = {}
    boxes = getattr(result, 'boxes', [])
    masks = getattr(result, 'masks', None)
    for i, box in enumerate(boxes):
        detections[int(box.id)] = {
            'cls_id': int(box.cls[0]),
            'conf': float(box.conf),
            'bbox': tuple(map(int, box.xyxy[0])),
            'mask': masks.data[i].cpu().numpy().astype(np.uint8) if masks and i < len(masks.data) else None
        }
    return detections

def compute_overlap_area(a, b):
    """Computes the overlapping area between two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)

def find_overlapping_player(racket, players):
    """Finds the person that overlaps the most with a racket."""
    max_overlap = 0
    max_player = None
    for player_id, player in players.items():
        overlap = compute_overlap_area(racket['bbox'], player['bbox'])
        if overlap > max_overlap:
            max_overlap = overlap
            max_player = player_id
    return max_player, max_overlap

def find_objects_to_save(people, rackets=None, balls=None):
    """Determines which objects to save based on the number of rackets detected."""
    objects_to_save = {}
    # only save the background if two rackets are detected
    pairings = {}

    # If at least two rackets are detected, save the person that overlaps the most with each racket
    if rackets:
        if len(rackets) >= 2:
            for racket_id, racket in rackets.items():
                player_id, _ = find_overlapping_player(racket, people)
                objects_to_save[player_id] = people[player_id]
                objects_to_save[racket_id] = racket
                pairings[racket_id] = player_id
    # If less than two rackets are detected, save every person and every racket
        else:
            objects_to_save.update(people)
            objects_to_save.update(rackets)
    else:
        objects_to_save.update(people)

    # Add every ball to the list of objects to save
    if balls:
        objects_to_save.update(balls)

    return objects_to_save, pairings   
            
def segment_object(frame_img, obj, frame_id):
    '''Segment an object from the background based on its mask shape.'''
    x1, y1, x2, y2 = obj['bbox']
    if obj['mask'] is not None:
        # Convert to 3-channel by stacking
        rgb_mask = cv2.cvtColor(obj['mask'], cv2.COLOR_GRAY2RGB)
        # TODO: Turn the mask from black into neon pink
        seg = frame_img * rgb_mask
        return seg[y1:y2, x1:x2]

def save_object(object_id, object, obj_img, segmented_folder, frame_id, csv_writer):
    '''Save a segmented object to its subfolder and log its bounding box coordinates.'''
    obj_folder = os.path.join(segmented_folder, f'{object["cls_id"]}_{object_id}')
    os.makedirs(obj_folder, exist_ok=True)
    cv2.imwrite(os.path.join(obj_folder, f'{frame_id}.png'), obj_img)
    
    # Save bounding box coordinates to CSV
    x1, y1, x2, y2 = object['bbox']
    csv_writer.writerow([frame_id, object_id, object['cls_id'], x1, y1, x2, y2])

def stitch_background_images(background_folder, stride=5):
    """Combines periodic background frames into a single stitched image."""
    all_images = []
    for i, name in enumerate(sorted(os.listdir(background_folder))):
        if name.endswith(".png") and i % stride == 0:
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

def main():
    parser = argparse.ArgumentParser(description='Perform instance segmentation on a video.')
    parser.add_argument('--video_file', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--segmented_folder', type=str, required=True, help='Folder to save segmented objects.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    args = parser.parse_args()

    model = YOLO('yolo11l-seg.pt')
    background_folder = os.path.join(args.segmented_folder, 'background')
    segmented_folder = os.path.join(args.segmented_folder, 'objects')
    os.makedirs(segmented_folder, exist_ok=True)
    os.makedirs(background_folder, exist_ok=True)

    # Open CSV file for writing bounding box coordinates
    csv_file_path = os.path.join(args.segmented_folder, 'bounding_boxes.csv')
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_id', 'object_id', 'class_id', 'x1', 'y1', 'x2', 'y2'])

        inf_results = model.track(
            source=args.video_file,
            device=args.device,
            conf=0.25,
            iou=0.2,
            imgsz=1920,
            retina_masks=False,
            stream=True,
            persist=True,
            classes=[0, 32, 38]  # person, ball, racket
        )

        for frame_id, result in enumerate(inf_results):
            frame_img = result.orig_img
            frame_data = extract_detections(result)

            rackets = {}
            people = {}
            balls = {}

            # For each object detected, classify it as a racket, person, or ball
            for obj_id, obj in frame_data.items():
                if obj['cls_id'] == 0:
                    obj['cls_id'] = 'person'
                    people[obj_id] = obj 
                elif obj['cls_id'] == 32:
                    obj['cls_id'] = 'ball'
                    balls[obj_id] = obj
                elif obj['cls_id'] == 38:
                    obj['cls_id'] = 'racket'
                    rackets[obj_id] = obj

            # Save objects based on the number of rackets detected
            objects_to_save, pairings = find_objects_to_save(people, rackets, balls)
            for obj_id, obj in objects_to_save.items():
                obj_img = segment_object(frame_img, obj, frame_id)
                # if saving a racket, add the player ID it is paired with to its file name
                if pairings and obj['cls_id'] == 'racket':
                    player_id = pairings[obj_id]
                    obj_id = f'{obj_id}_{player_id}'
                save_object(obj_id, obj, obj_img, segmented_folder, frame_id, csv_writer)

            if pairings:
                # Remove objects from background
                for obj in objects_to_save.values():
                    x1, y1, x2, y2 = obj['bbox']
                    frame_img[y1:y2, x1:x2] = 0

                # Save background image
                cv2.imwrite(os.path.join(background_folder, f'{frame_id}.png'), frame_img)

        stitched = stitch_background_images(background_folder, stride=5)
        if stitched is not None:
            cv2.imwrite(os.path.join(args.segmented_folder, 'full_background.png'), stitched)

        # Delete person folders that are missing too few frames (should be left with just players)
        min_frames = frame_id * 0.9
        for obj_folder in os.listdir(segmented_folder):
            if obj_folder.startswith('person'):
                num_frames = len(os.listdir(os.path.join(segmented_folder, obj_folder)))
                if num_frames < min_frames:
                    shutil.rmtree(os.path.join(segmented_folder, obj_folder))

if __name__ == "__main__":
    main()