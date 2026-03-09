import os
import cv2
import json
import datetime
from pathlib import Path
from ultralytics import YOLO

from utils import (
    load_video_frames,
    run_yolo,
    stitch_panorama,
    save_dwpose,
    encode_video_libsvtav1,
    resize_and_pad_image,
)

def main():
    video_path = "/home/itec/emanuele/Datasets/federer_djokovic/libsvtav1_crf35_pre5/scene_004.mp4"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = f"/home/itec/emanuele/pointstream/experiments/{timestamp}"
    classes_to_detect = ["person", "tennis racket"]
    detection_model = YOLO("/home/itec/emanuele/Models/YOLO/yolo26x.pt")
    segmentation_model = YOLO("/home/itec/emanuele/Models/YOLO/yolo26x-seg.pt")
    tracker = "/home/itec/emanuele/Models/YOLO/trackers/botsort.yaml"
    save_image_size = (640, 640)
    
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
            if crop.size == 0:
                continue

            resized_crop = resize_and_pad_image(crop, save_image_size)
            track_id_value = int(track_id.item()) if hasattr(track_id, "item") else int(track_id)
            crop_folder = f"{exp_folder}/{class_name}_{track_id_value}"
            bbox_folder = f"{crop_folder}/bbox"
            mask_folder = f"{crop_folder}/mask"
            object_folder = f"{crop_folder}/object"
            os.makedirs(bbox_folder, exist_ok=True)
            os.makedirs(mask_folder, exist_ok=True)
            os.makedirs(object_folder, exist_ok=True)
            cv2.imwrite(f"{bbox_folder}/{frame_idx:06d}.png", resized_crop)
            
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
            resized_mask = resize_and_pad_image(mask, save_image_size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{mask_folder}/{frame_idx:06d}.png", resized_mask)

            masked_crop = cv2.bitwise_and(resized_crop, resized_crop, mask=resized_mask)
            cv2.imwrite(f"{object_folder}/{frame_idx:06d}.png", masked_crop)

            background_crop = background[y1:y2, x1:x2]
            background_crop[mask > 0] = 0
            background[y1:y2, x1:x2] = background_crop

        cv2.imwrite(f"{exp_folder}/background/{frame_idx:06d}.png", background)
        
    # Stitch every 50th of the background frames into a panorama and save it.
    sampled_frame_indices = list(range(0, len(frames), 50))
    sampled_background_frames = []
    for frame_idx in sampled_frame_indices:
        frame = cv2.imread(f"{exp_folder}/background/{frame_idx:06d}.png")
        if frame is not None:
            sampled_background_frames.append(frame)

    panorama, panorama_data = stitch_panorama(sampled_background_frames)
    cv2.imwrite(f"{exp_folder}/panorama.png", panorama)

    # Save panorama metadata
    panorama_metadata_path = Path(exp_folder) / "panorama_data.json"
    panorama_metadata = {
        "sample_stride": 50,
        "sampled_frame_indices": sampled_frame_indices,
        "num_frames": int(panorama_data["num_frames"]),
        "frame_width": int(panorama_data["frame_width"]),
        "frame_height": int(panorama_data["frame_height"]),
        "canvas_width": int(panorama_data["canvas_width"]),
        "canvas_height": int(panorama_data["canvas_height"]),
        "translation": panorama_data["translation"].tolist(),
        "global_homographies": [homography.tolist() for homography in panorama_data["global_homographies"]],
        "composite_homographies": [homography.tolist() for homography in panorama_data["composite_homographies"]],
    }
    with panorama_metadata_path.open("w", encoding="utf-8") as panorama_metadata_file:
        json.dump(panorama_metadata, panorama_metadata_file, indent=2)

    capture = cv2.VideoCapture(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if fps <= 0:
        fps = 25.0

    exp_path = Path(exp_folder)
    dwpose_frames_root = exp_path / "dwpose_frames"
    dwpose_videos_root = exp_path / "dwpose_videos"
    dwpose_frames_root.mkdir(parents=True, exist_ok=True)
    dwpose_videos_root.mkdir(parents=True, exist_ok=True)

    def person_sort_key(path: Path):
        suffix = path.name.rsplit("_", 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, path.name)

    person_dirs = sorted(
        [path for path in exp_path.iterdir() if path.is_dir() and path.name.startswith("person_")],
        key=person_sort_key,
    )

    for person_dir in person_dirs:
        source_frames_dir = person_dir / "object"
        if not source_frames_dir.exists():
            continue

        person_name = person_dir.name
        person_dwpose_frames_dir = dwpose_frames_root / person_name
        save_dwpose(
            frames_folder=str(source_frames_dir),
            output_folder=str(person_dwpose_frames_dir),
        )

        if not any(person_dwpose_frames_dir.glob("*.png")):
            continue

        output_path = dwpose_videos_root / f"{person_name}_dwpose.mp4"
        encode_video_libsvtav1(
            output_path=str(output_path),
            fps=fps,
            crf=25,
            frame_folder=str(person_dwpose_frames_dir),
        )
        print(f"Saved DWPose video for {person_name} at {output_path}")

if __name__ == "__main__":
    main()