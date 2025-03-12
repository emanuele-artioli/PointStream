import numpy as np
import os
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

def save_object(object_id, object, obj_img, experiment_folder, frame_id, csv_writer):
    '''Save an object to its subfolder and log its bounding box coordinates.'''
    obj_folder = os.path.join(experiment_folder, f'{object["cls_id"]}_{object_id}')
    os.makedirs(obj_folder, exist_ok=True)
    cv2.imwrite(os.path.join(obj_folder, f'{frame_id}.png'), obj_img)
    
    # Save bounding box coordinates to CSV
    x1, y1, x2, y2 = object['bbox']
    csv_writer.writerow([frame_id, object_id, object['cls_id'], x1, y1, x2, y2])
     
def main():
    parser = argparse.ArgumentParser(description='Perform instance segmentation on a video.')
    parser.add_argument('--video_file', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--experiment_folder', type=str, required=True, help='Folder to save objects.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    args = parser.parse_args()

    model = YOLO('yolo11l-seg.pt')
    background_folder = os.path.join(args.experiment_folder, 'background')
    experiment_folder = os.path.join(args.experiment_folder, 'objects')
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(background_folder, exist_ok=True)

    # Open CSV file for writing bounding box coordinates
    csv_file_path = os.path.join(args.experiment_folder, 'bounding_boxes.csv')
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_id', 'object_id', 'class_id', 'x1', 'y1', 'x2', 'y2'])

        inf_results = model.track(
            source = args.video_file,
            device = args.device,
            conf = 0.25,
            iou = 0.2,
            imgsz = 1920,
            # use half precision if on cuda
            half = 'cuda' in args.device,
            retina_masks = True,
            stream = True,
            persist = True,
            classes = [0, 32, 38]  # person, ball, racket
        )

        # Initialize pairings dictionary to keep track of which rackets are paired with which players across frames
        pairings = {}

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
                    # only add the most confident people detections
                    if len(people) < 15:
                        people[obj_id] = obj 
                elif obj['cls_id'] == 32:
                    obj['cls_id'] = 'ball'
                    balls[obj_id] = obj
                elif obj['cls_id'] == 38:
                    obj['cls_id'] = 'racket'
                    # If the racket overlaps a player that has already been paired with a previous racket, this is the same racket
                    if pairings:
                        player_id, overlap = find_overlapping_player(obj, people)
                        if overlap > 0 and player_id in pairings.values():
                            obj_id = next(k for k, v in pairings.items() if v == player_id)
                    rackets[obj_id] = obj

            # Save objects based on the number of rackets detected
            objects_to_save, pairings = find_objects_to_save(people, rackets, balls)
            for obj_id, obj in objects_to_save.items():
                obj_img = segment_object(frame_img, obj, frame_id)
                # if saving a racket, add the player ID it is paired with to its file name
                if pairings and obj['cls_id'] == 'racket':
                    player_id = pairings[obj_id]
                    obj_id = f'{obj_id}_{player_id}'
                save_object(obj_id, obj, obj_img, experiment_folder, frame_id, csv_writer)

            if pairings:
                # Remove objects from background
                for obj in objects_to_save.values():
                    x1, y1, x2, y2 = obj['bbox']
                    frame_img[y1:y2, x1:x2] = 0

            # Save background image
            cv2.imwrite(os.path.join(background_folder, f'{frame_id}.png'), frame_img)

if __name__ == "__main__":
    main()