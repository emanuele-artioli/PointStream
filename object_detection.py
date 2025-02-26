import os
import numpy as np
import cv2
import argparse
# import shutil
from ultralytics import YOLO

# Extract bounding box data (obj_id, cls_id, x1, y1, x2, y2) from a tracking result.
def parse_frame(result):
    frame_data = []
    try:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            obj_id = int(box.id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            frame_data.append((obj_id, cls_id, x1, y1, x2, y2))
    except AttributeError:
        # No boxes in this frame
        pass
    return frame_data

# Returns True if there are 2 distinct racket detections (cls_id == 38) and at least one ball (cls_id == 32) in this frame.
def check_for_rackets_and_ball(frame_data):
    rackets = 0
    ball = False
    for obj_id, cls_id, *_ in frame_data:
        if cls_id == 38:
            rackets += 1
        elif cls_id == 32:
            ball = True
    return rackets == 2 and ball

# Identify which objects overlap with the rackets; returns a list of (racket_id, overlapping_obj_id).
def find_racket_player_overlaps(frame_data, racket_ids):
    overlaps = []
    rackets = [d for d in frame_data if d[0] in racket_ids]
    
    for racket in rackets:
        r_id, r_cls, rx1, ry1, rx2, ry2 = racket
        
        max_overlap_area = 0
        best_person_id = None

        # Check all persons in the frame
        for obj_id, obj_cls, x1, y1, x2, y2 in frame_data:
            if obj_cls != 0:  # 0 = person
                continue

            # Compute overlap
            ix1 = max(rx1, x1)
            iy1 = max(ry1, y1)
            ix2 = min(rx2, x2)
            iy2 = min(ry2, y2)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter_area > max_overlap_area:
                max_overlap_area = inter_area
                best_person_id = obj_id

        if best_person_id is not None and max_overlap_area > 0:
            overlaps.append((r_id, best_person_id))

    return overlaps

# Takes two bounding boxes and returns their fused bounding box (min left, max right, etc).
def fuse_boxes(bbox1, bbox2):
    x1 = min(bbox1[0], bbox2[0])
    y1 = min(bbox1[1], bbox2[1])
    x2 = max(bbox1[2], bbox2[2])
    y2 = max(bbox1[3], bbox2[3])
    return (x1, y1, x2, y2)

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

    stitched_image = background_images[0].copy()
    for img in background_images[1:]:
        mask = (stitched_image == 0)
        stitched_image[mask] = img[mask]
    return stitched_image

def perform_object_detection(video_file, detected_folder, background_folder, model):
    os.makedirs(detected_folder, exist_ok=True)
    os.makedirs(background_folder, exist_ok=True)
    frame_buffer = {}

    results = model.track(
        video_file, 
        conf=0.5, 
        iou=0.4, 
        imgsz=3840, 
        stream=True, 
        persist=True, 
        classes=[0, 32, 38]
    )

    for frame_id, result in enumerate(results):
        frame_img = result.orig_img
        frame_data = parse_frame(result)

        # Get people
        people = [d for d in frame_data if d[1] == 0]
        if not people:
            continue
            
        # Get rackets
        rackets = [d for d in frame_data if d[1] == 38]
        # if there are less than 2 rackets, keep this frame and its object data in a buffer, until we find a frame with 2 rackets
        if len(rackets) < 2:
            frame_buffer[frame_id] = [frame_img, frame_data]
            continue

        # for each racket, check which person it overlaps with the most
        overlaps = find_racket_player_overlaps(frame_data, [r[0] for r in rackets])
        # Join the overlapping racket, person pairs' bounding boxes into a player bounding box
        players = {}
        for racket_id, player_id in overlaps:
            player_bbox = [d[2:] for d in people if d[0] == player_id][0]
            racket_bbox = [d[2:] for d in rackets if d[0] == racket_id][0]
            fused_bbox = fuse_boxes(player_bbox, racket_bbox)
            players[player_id] = fused_bbox

        # Get ball
        balls = [d for d in frame_data if d[1] == 32]
        # Check if the ball overlaps with players
        ball_overlaps = [p_id for p_id, p_bbox in players.items() for b in balls if b[2] < p_bbox[2] and b[3] < p_bbox[3]]
        # If a ball does not overlap, save it
        for ball in balls:
            if ball[0] not in ball_overlaps:
                x1, y1, x2, y2 = ball[2:]
                ball_img = frame_img[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(detected_folder, f'ball_{frame_id}_{ball[0]}.png'), ball_img)

        # Save player images
        for player_id, bbox in players.items():
            player_img = frame_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite(os.path.join(detected_folder, f'player_{frame_id}_{player_id}.png'), player_img)
        
        # Substitute players boxes with black boxes in the frame image
        for player_id, bbox in players.items():
            frame_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

        # Save background image
        cv2.imwrite(os.path.join(background_folder, f'background_{frame_id}.png'), frame_img)

        # Now that we know who the players are, we can process the frames in the buffer
        for buffered_frame_id, (buffered_frame_img, buffered_frame_data) in frame_buffer.items():
            for racket_id, player_id in overlaps:
                buffered_players = {}
                
                # If both player and racket are in this frame, fuse their bounding boxes
                if any(d[0] == player_id for d in buffered_frame_data) and any(d[0] == racket_id for d in buffered_frame_data):
                    player_bbox = [d[2:] for d in buffered_frame_data if d[0] == player_id][0]
                    racket_bbox = [d[2:] for d in buffered_frame_data if d[0] == racket_id][0]
                    fused_bbox = fuse_boxes(player_bbox, racket_bbox)
                    buffered_players[player_id] = fused_bbox
                    
                # If only the player is in this frame, use the player's bounding box
                elif any(d[0] == player_id for d in buffered_frame_data):
                    player_bbox = [d[2:] for d in buffered_frame_data if d[0] == player_id][0]
                    buffered_players[player_id] = player_bbox

                # Get ball
                balls = [d for d in buffered_frame_data if d[1] == 32]
                # Check if the ball overlaps with players
                ball_overlaps = [player_id for player_id, player_bbox in buffered_players.items() for b in balls if b[2] < player_bbox[2] and b[3] < player_bbox[3]]
                # If a ball does not overlap, save it
                for ball in balls:
                    if ball[0] not in ball_overlaps:
                        x1, y1, x2, y2 = ball[2:]
                        ball_img = buffered_frame_img[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(detected_folder, f'ball_{buffered_frame_id}_{ball[0]}.png'), ball_img)
                
                # Save player images
                for player_id, bbox in buffered_players.items():
                    player_img = buffered_frame_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    cv2.imwrite(os.path.join(detected_folder, f'player_{buffered_frame_id}_{player_id}.png'), player_img)

                # Substitute players boxes with black boxes in the frame image
                for player_id, bbox in buffered_players.items():
                    buffered_frame_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

            # Save background image
            cv2.imwrite(os.path.join(background_folder, f'background_{buffered_frame_id}.png'), buffered_frame_img)

        # Clear the buffer
        frame_buffer = {}

def main():
    parser = argparse.ArgumentParser(description='Perform instance segmentation on video.')
    parser.add_argument('--video_file', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--detected_folder', type=str, required=True, help='Folder to save segmented objects.')
    args = parser.parse_args()

    os.makedirs(args.detected_folder, exist_ok=True)
    background_folder = os.path.join(args.detected_folder, 'background')
    os.makedirs(background_folder, exist_ok=True)

    # Use our new function
    model = YOLO('yolo11m.pt')
    perform_object_detection(args.video_file, args.detected_folder, background_folder, model)

    # Optionally stitch background
    stitched_image = stitch_background_images(background_folder)
    if stitched_image is not None:
        cv2.imwrite(os.path.join(args.detected_folder, 'full_background.png'), stitched_image)

    # Cleanup
    # shutil.rmtree(background_folder)

if __name__ == "__main__":
    main()