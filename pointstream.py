import os
import cv2
import json
import datetime
from pathlib import Path

from utils import (
    load_video_frames,
    run_yolo,
    stitch_panorama,
    animate_panorama,
    extract_dwpose_keypoints,
    save_dwpose_keypoints_csv,
    load_dwpose_keypoints_csv,
    convert_dwpose_keypoints_to_skeleton_frames,
    encode_video_libsvtav1,
    run_animate_anyone_pose2video,
    resize_and_pad_image,
)

EXPERIMENTS_ROOT = Path("/home/itec/emanuele/pointstream/experiments")
DEFAULT_ANIMATE_ANYONE_DIR = Path("/home/itec/emanuele/Moore-AnimateAnyone")
DEFAULT_ANIMATE_ANYONE_CONFIG = DEFAULT_ANIMATE_ANYONE_DIR / "configs" / "prompts" / "run_finetuned.yaml"
ANIMATE_ANYONE_OUTPUT_SIZE = (512, 512)  # (height, width)

def run_server():
    from ultralytics import YOLO

    video_path = "/home/itec/emanuele/Datasets/federer_djokovic/libsvtav1_crf35_pre5/scene_004.mp4"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_path = EXPERIMENTS_ROOT / timestamp
    exp_folder = str(exp_path)
    classes_to_detect = ["person", "tennis racket"]
    # TODO: ablation test: compare with YOLOE, in two modes: one with text prompts (eg. tennis player), one with image (first frame of identified classes).
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
        )):
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
                    max_det=1,
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

    # Save DWPose videos for each detected person.
    capture = cv2.VideoCapture(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()

    exp_path = Path(exp_folder)
    dwpose_keypoints_root = exp_path / "dwpose_keypoints"
    dwpose_frames_root = exp_path / "dwpose_frames"
    dwpose_videos_root = exp_path / "dwpose_videos"
    dwpose_keypoints_root.mkdir(parents=True, exist_ok=True)
    dwpose_frames_root.mkdir(parents=True, exist_ok=True)
    dwpose_videos_root.mkdir(parents=True, exist_ok=True)

    def person_sort_key(path: Path):
        suffix = path.name.rsplit("_", 1)[-1]
        return (0, int(suffix)) if suffix.isdigit() else (1, path.name)

    person_dirs = sorted(
        [path for path in exp_path.iterdir() if path.is_dir() and path.name.startswith("person_")],
        key=person_sort_key,
    )

    dwpose_jobs = []

    for person_dir in person_dirs:
        source_frames_dir = person_dir / "object"
        if not source_frames_dir.exists():
            continue

        reference_candidates = sorted(source_frames_dir.glob("*.png"))
        if not reference_candidates:
            continue
        reference_image_path = reference_candidates[0]

        person_name = person_dir.name
        person_dwpose_keypoints_csv = dwpose_keypoints_root / f"{person_name}_keypoints.csv"
        keypoint_frames = extract_dwpose_keypoints(
            frames_folder=str(source_frames_dir),
        )
        if not keypoint_frames:
            continue

        save_dwpose_keypoints_csv(
            keypoint_frames=keypoint_frames,
            output_csv_path=str(person_dwpose_keypoints_csv),
        )

        dwpose_jobs.append(
            {
                "person_name": person_name,
                "reference_image": str(reference_image_path),
                "keypoints_csv": str(person_dwpose_keypoints_csv),
                "output_video": str(dwpose_videos_root / f"{person_name}_dwpose.mp4"),
                "animate_anyone_output_video": str(exp_path / "animate_anyone_videos" / f"{person_name}_animate_anyone.mp4"),
            }
        )

    metadata_path = exp_path / "metadata.json"
    metadata = {
        "experiment_path": str(exp_path),
        "panorama_image_path": str(exp_path / "panorama.png"),
        "panorama_data_path": str(panorama_metadata_path),
        "fps": fps,
        "frame_size": list(save_image_size),
        "dwpose_keypoints_csv_files": [job["keypoints_csv"] for job in dwpose_jobs],
        "dwpose_jobs": dwpose_jobs,
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def run_client():
    experiment_dirs = [path for path in EXPERIMENTS_ROOT.iterdir() if path.is_dir()]
    if not experiment_dirs:
        print("No experiment folders found under experiments/. Nothing to process on the client.")
        return

    latest_exp_path = max(experiment_dirs, key=lambda path: path.stat().st_mtime)
    metadata_path = latest_exp_path / "metadata.json"
    if not metadata_path.exists():
        print(f"Missing metadata file: {metadata_path}")
        return

    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    panorama_image_path = Path(metadata.get("panorama_image_path", latest_exp_path / "panorama.png"))
    panorama_data_path = Path(metadata.get("panorama_data_path", latest_exp_path / "panorama_data.json"))
    if not panorama_image_path.exists():
        print(f"Missing panorama image file: {panorama_image_path}")
        return
    if not panorama_data_path.exists():
        print(f"Missing panorama data file: {panorama_data_path}")
        return

    panorama = cv2.imread(str(panorama_image_path))
    if panorama is None:
        print(f"Failed to read panorama image: {panorama_image_path}")
        return

    with panorama_data_path.open("r", encoding="utf-8") as panorama_data_file:
        panorama_data = json.load(panorama_data_file)

    reconstructed_background_frames = animate_panorama(
        panorama=panorama,
        panorama_data=panorama_data,
    )

    fps = float(metadata.get("fps", 0.0))
    sample_stride = int(panorama_data.get("sample_stride", 1) or 1)
    background_fps = fps / sample_stride if sample_stride > 0 else fps
    background_output_path = latest_exp_path / "background_from_panorama.mp4"
    if reconstructed_background_frames:
        encode_video_libsvtav1(
            output_path=str(background_output_path),
            fps=background_fps,
            crf=25,
            frames=reconstructed_background_frames,
        )

    frame_size_values = metadata.get("frame_size", [640, 640])
    if len(frame_size_values) != 2:
        frame_size_values = [640, 640]
    save_image_size = (int(frame_size_values[0]), int(frame_size_values[1]))
    client_jobs = metadata.get("dwpose_jobs", [])
    dwpose_frames_root = latest_exp_path / "dwpose_frames"
    animate_anyone_default_dir = Path(
        os.environ.get("POINTSTREAM_ANIMATE_ANYONE_DIR", str(DEFAULT_ANIMATE_ANYONE_DIR))
    ).expanduser()
    animate_anyone_config_value = os.environ.get(
        "POINTSTREAM_ANIMATE_ANYONE_CONFIG",
        str(DEFAULT_ANIMATE_ANYONE_CONFIG),
    )
    animate_anyone_config = Path(animate_anyone_config_value).expanduser()
    if not animate_anyone_config.is_absolute():
        animate_anyone_config = animate_anyone_default_dir / animate_anyone_config
    animate_anyone_output_root = latest_exp_path / "animate_anyone_videos"

    animate_anyone_enabled = True
    if not animate_anyone_default_dir.exists():
        print(f"AnimateAnyone repo not found, skipping pose2video: {animate_anyone_default_dir}")
        animate_anyone_enabled = False
    elif not animate_anyone_config.exists():
        print(f"AnimateAnyone config not found, skipping pose2video: {animate_anyone_config}")
        animate_anyone_enabled = False

    if animate_anyone_enabled:
        animate_anyone_output_root.mkdir(parents=True, exist_ok=True)

    for job in client_jobs:
        person_dwpose_keypoints_csv = Path(job["keypoints_csv"])
        person_name = str(job.get("person_name", person_dwpose_keypoints_csv.stem.replace("_keypoints", "")))
        person_dwpose_frames_dir = dwpose_frames_root / person_name
        output_path = Path(job["output_video"])

        loaded_keypoint_frames = load_dwpose_keypoints_csv(str(person_dwpose_keypoints_csv))
        if not loaded_keypoint_frames:
            continue

        person_dwpose_frames_dir.mkdir(parents=True, exist_ok=True)

        convert_dwpose_keypoints_to_skeleton_frames(
            keypoint_frames=loaded_keypoint_frames,
            output_folder=str(person_dwpose_frames_dir),
            frame_size=save_image_size,
        )

        if not any(person_dwpose_frames_dir.glob("*.png")):
            continue

        encode_video_libsvtav1(
            output_path=str(output_path),
            fps=fps,
            crf=25,
            frame_folder=str(person_dwpose_frames_dir),
        )

        if not animate_anyone_enabled:
            continue

        reference_image_path = Path(job.get("reference_image", "")).expanduser()
        if not reference_image_path.exists():
            fallback_candidates = sorted((latest_exp_path / person_name / "object").glob("*.png"))
            if fallback_candidates:
                reference_image_path = fallback_candidates[0]
            else:
                print(f"Missing reference image for {person_name}, skipping AnimateAnyone inference.")
                continue

        animate_anyone_output_path = Path(
            job.get(
                "animate_anyone_output_video",
                animate_anyone_output_root / f"{person_name}_animate_anyone.mp4",
            )
        ).expanduser()

        try:
            generated_video_path = run_animate_anyone_pose2video(
                ref_image_path=str(reference_image_path),
                skeleton_frames_dir=str(person_dwpose_frames_dir),
                config_path=str(animate_anyone_config),
                animate_anyone_dir=str(animate_anyone_default_dir),
                width=int(ANIMATE_ANYONE_OUTPUT_SIZE[1]),
                height=int(ANIMATE_ANYONE_OUTPUT_SIZE[0]),
                length=None,
                steps=30,
                cfg=3.5,
                seed=42,
                fps=fps if fps > 0 else None,
                save_dir=str(animate_anyone_output_path.parent),
                output_filename=animate_anyone_output_path.name,
                include_input_grid=True,
            )
            print(f"Saved AnimateAnyone video for {person_name}: {generated_video_path}")
        except Exception as exc:
            print(f"AnimateAnyone inference failed for {person_name}: {exc}")


def main():
    run_server()
    run_client()


if __name__ == "__main__":
    main()