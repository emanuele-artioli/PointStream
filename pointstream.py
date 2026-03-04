import os

import cv2

from utils import load_video_frames, detect_with_yolo

def main():
    video_path = "/home/itec/emanuele/Datasets/federer_djokovic/libsvtav1_crf35_pre5/scene_004.mp4"
    exp_folder = "/home/itec/emanuele/pointstream/experiments/sample_exp"
    
    frames = load_video_frames(video_path)
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(f"{exp_folder}/background", exist_ok=True)
    
    for frame_idx, detection in enumerate(detect_with_yolo(frames, exp_folder)):
        frame = detection["frame"]
        background = frame.copy()

        for (class_name, track_id, bbox) in zip(detection["class_names"], detection["track_ids"], detection["bboxes"]):
            x1, y1, x2, y2 = [int(value.item()) for value in bbox]
            crop = frame[y1:y2, x1:x2]
            background[y1:y2, x1:x2] = 0

            crop_folder = f"{exp_folder}/{class_name}_{int(track_id.item())}"
            os.makedirs(crop_folder, exist_ok=True)
            cv2.imwrite(f"{crop_folder}/{frame_idx:06d}.png", crop)

        cv2.imwrite(f"{exp_folder}/background/{frame_idx:06d}.png", background)
        
        # TODO: sometimes, the same object is detected in multiple frames but with different track IDs. 
        # To rectify this, we could save a csv with track ID, frame number, class names, and bboxes of each detection in each frame.
        # Then, we can load the csv into a dataframe, sort the objects by the number of frames they contain, and if an object with many frames is missing frames, 
        # we can check the dataframe to see if there are other track IDs that have detections in those frames with similar bboxes and class names, and if so, 
        # we can merge the small subfolder into the big one and update the csv. 

if __name__ == "__main__":
    main()