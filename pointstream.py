import os
import cv2
import datetime
from ultralytics import YOLO

from utils import load_video_frames, run_yolo

def main():
    video_path = "/home/itec/emanuele/Datasets/federer_djokovic/libsvtav1_crf35_pre5/scene_004.mp4"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = f"/home/itec/emanuele/pointstream/experiments/{timestamp}"
    classes_to_detect = ["person", "tennis racket"]
    detection_model = YOLO("/home/itec/emanuele/Models/YOLO/yolo26x.pt")
    segmentation_model = YOLO("/home/itec/emanuele/Models/YOLO/yolo26x-seg.pt")
    tracker = "/home/itec/emanuele/Models/YOLO/trackers/botsort.yaml"
    
    frames = load_video_frames(video_path)
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(f"{exp_folder}/background", exist_ok=True)
    
    for frame_idx, detection in enumerate(
        run_yolo(
            frames=frames, 
            exp_folder=exp_folder, 
            classes=classes_to_detect, 
            imgsz=(360, 640),
            model=detection_model, 
            tracker=tracker,
            task="detect",
        )
    ):
        frame = detection["frame"]
        background = frame.copy()

        for (class_name, track_id, bbox) in zip(detection["class_names"], detection["track_ids"], detection["bboxes"]):
            x1, y1, x2, y2 = [int(value.item()) for value in bbox]
            crop = frame[y1:y2, x1:x2]
            track_id_value = int(track_id.item()) if hasattr(track_id, "item") else int(track_id)
            crop_folder = f"{exp_folder}/{class_name}_{track_id_value}"
            bbox_folder = f"{crop_folder}/bbox"
            mask_folder = f"{crop_folder}/mask"
            object_folder = f"{crop_folder}/object"
            os.makedirs(bbox_folder, exist_ok=True)
            os.makedirs(mask_folder, exist_ok=True)
            os.makedirs(object_folder, exist_ok=True)
            cv2.imwrite(f"{bbox_folder}/{frame_idx:06d}.png", crop)
            
            # TODO: sometimes, the same object is detected in multiple frames but with different track IDs. 
            # To rectify this, we could save a csv with track ID, frame number, class names, and bboxes of each detection in each frame.
            # Then, we can load the csv into a dataframe, sort the objects by the number of frames they contain, and if an object with many frames is missing frames, 
            # we can check the dataframe to see if there are other track IDs that have detections in those frames with similar bboxes and class names, and if so, 
            # we can merge the small subfolder into the big one and update the csv. 
            # If there is no other track ID with similar bboxes and class names, we can fallback to averaging the bboxes of the same track ID across its closest available frames to fill in the missing frames,
            # and crop that region from the background. But in this case we need to be careful about occlusions, because if the object is occluded in the missing frames, then we would be cropping the wrong region from the background.

            # Segment the object, save the mask, save the masked crop, and mask the object in the background.
            segmentation = next(
                run_yolo(
                    frames=[crop],
                    exp_folder=exp_folder,
                    classes=[class_name],
                    imgsz=(640, 640),
                    model=segmentation_model,
                    tracker=None,
                    task="segment",
                ),
                None,
            )
            if segmentation is None or segmentation["masks"] is None:
                continue
            mask = (segmentation["masks"][0] > 0.5).detach().cpu().numpy().astype("uint8") * 255
            mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{mask_folder}/{frame_idx:06d}.png", mask)

            masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
            cv2.imwrite(f"{object_folder}/{frame_idx:06d}.png", masked_crop)

            background_crop = background[y1:y2, x1:x2]
            background_crop[mask > 0] = 0
            background[y1:y2, x1:x2] = background_crop

        cv2.imwrite(f"{exp_folder}/background/{frame_idx:06d}.png", background)

if __name__ == "__main__":
    main()