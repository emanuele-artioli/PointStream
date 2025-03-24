import numpy as np
import os
import cv2
import argparse
import csv
from ultralytics import YOLO

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
            # if enough players id are already known, and the current person id is not one of them, skip
            if len(players_ids) >= 2 and obj_id not in players_ids:
                continue
            people[obj_id] = obj
        elif obj['cls_id'] == 32:
            balls[obj_id] = obj
        elif obj['cls_id'] == 38:
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
      
def segment_object(frame_img, obj, frame_id):
    '''Segment an object from the background based on its mask shape.'''
    x1, y1, x2, y2 = obj['bbox']
    if obj['mask'] is not None:
        # Convert to 3-channel by stacking
        rgb_mask = cv2.cvtColor(obj['mask'], cv2.COLOR_GRAY2RGB)
        # TODO: Turn the mask from black into neon pink
        seg = frame_img * rgb_mask
        return seg[y1:y2, x1:x2]

def save_object(object_id, object, obj_img, experiment_folder, frame_id, csv_writer, frame_img=None):
    '''Save an object to its subfolder and log its bounding box coordinates.'''
    obj_folder = os.path.join(experiment_folder, f'{object["cls_id"]}_{object_id}')
    os.makedirs(obj_folder, exist_ok=True)
    cv2.imwrite(os.path.join(obj_folder, f'{frame_id}.png'), obj_img)
    
    # Save bounding box coordinates to CSV
    x1, y1, x2, y2 = object['bbox']
    csv_writer.writerow([frame_id, object_id, object['cls_id'], x1, y1, x2, y2])

    # Optionally save another copy of the object without segmentation
    if frame_img is not None:
        unsegmented_folder = os.path.join(obj_folder, 'unsegmented')
        os.makedirs(unsegmented_folder, exist_ok=True)
        cv2.imwrite(os.path.join(unsegmented_folder, f'{frame_id}.png'), frame_img[y1:y2, x1:x2])

def main():
    parser = argparse.ArgumentParser(description='Perform instance segmentation on a video.')
    parser.add_argument('--video_file', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--experiment_folder', type=str, required=True, help='Folder to save objects.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--model', type=str, default='models/yolo11l-seg.pt', help='Path to the model file.')
    args = parser.parse_args()

    model = YOLO(args.model)
    background_folder = os.path.join(args.experiment_folder, 'background')
    experiment_folder = os.path.join(args.experiment_folder, 'objects')
    csv_file_path = os.path.join(args.experiment_folder, 'bounding_boxes.csv')
    os.makedirs(experiment_folder, exist_ok=True)
    os.makedirs(background_folder, exist_ok=True)

    # Open CSV file for writing bounding box coordinates
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_id', 'object_id', 'class_id', 'x1', 'y1', 'x2', 'y2'])

        inf_results = model.track(
            source = args.video_file,
            conf = 0.25,
            iou = 0.2,
            imgsz = 3840,
            half = 'cuda' in args.device, # use half precision if on cuda
            device = args.device,
            batch = 1,
            max_det = 30,
            classes = [0, 32, 38],  # person, ball, racket
            retina_masks = True,
            stream = True,
            persist = True,

        )

        # Keep track of players' IDs and balls positions across frames
        players_ids = set()
        previous_balls_boxes = {}
        frame_id = 0

        for frame_id, result in enumerate(inf_results):
            frame_id += 1
            frame_img = result.orig_img
            people, rackets, balls = extract_detections(result, players_ids)

            if rackets:
                # for each racket find the player they overlap with the most,
                for racket_id, racket in rackets.items():
                    player_id = find_overlapping_player(racket, people)
                    if player_id is not None:
                        players_ids.add(player_id)
                    # Segment and save the racket
                    racket_img = segment_object(frame_img, racket, frame_id)
                    save_object(racket_id, racket, racket_img, experiment_folder, frame_id, csv_writer, frame_img)

            if balls:
                # for each ball, check if it has moved enough from the previous frame
                for ball_id, ball in balls.copy().items():
                    if ball_id in previous_balls_boxes.keys():
                        previous_box = previous_balls_boxes[ball_id]
                        current_box = ball['bbox']
                        # If the ball has not moved enough, pop it from the list of balls
                        if compute_overlap_area(previous_box, current_box) > 0.9:
                            balls.pop(ball_id)
                        # Otherwise, segment and save it
                        else:
                            ball_img = segment_object(frame_img, ball, frame_id)
                            save_object(ball_id, ball, ball_img, experiment_folder, frame_id, csv_writer, frame_img)
                    # Save the current ball's bbox for the next frame
                    previous_balls_boxes[ball_id] = ball['bbox']

            # Segment and save every person (once 2 players are detected, only they will be present in people)
            for person_id, person in people.items():
                player_img = segment_object(frame_img, person, frame_id)
                save_object(person_id, person, player_img, experiment_folder, frame_id, csv_writer, frame_img)

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

            # Save background image
            cv2.imwrite(os.path.join(background_folder, f'{frame_id}.png'), frame_img)

if __name__ == "__main__":
    main()