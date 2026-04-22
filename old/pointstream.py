import os
import cv2
import json
import datetime
import numpy as np
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
    run_animate_anyone,
    resize_and_pad_image,
    scale_frame_to_bbox,
    overlay_object_on_background_video,
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
    classes_to_detect = ["person"]
    # TODO: ablation test: compare with YOLOE, in two modes: one with text prompts (eg. tennis player), one with image (first frame of identified classes).
    detection_model = YOLO("/home/itec/emanuele/Models/YOLO/yolo26x.pt")
    segmentation_model = YOLO("/home/itec/emanuele/Models/YOLO/yolo26x-seg.pt")
    tracker = os.environ.get("POINTSTREAM_TRACKER", "/home/itec/emanuele/Models/YOLO/trackers/botsort.yaml")
    save_image_size = (640, 640)
    # Use a sampled subset of background frames for panorama stitching by default
    # to reduce server-side processing time.
    panorama_sample_stride = max(1, int(os.environ.get("POINTSTREAM_PANORAMA_SAMPLE_STRIDE", "10")))
    
    frames = load_video_frames(video_path)
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(f"{exp_folder}/background", exist_ok=True)
    tracked_bboxes_by_frame: dict[str, dict[int, list[float]]] = {}
    
    for frame_idx, detection in enumerate(
        run_yolo(
            frames=frames, 
            exp_folder=exp_folder, 
            classes=classes_to_detect, 
            imgsz=(360, 640),
            max_det=2,
            model=detection_model, 
            tracker=tracker,
            task="detect",
        )):
        frame = detection["frame"]
        background = frame.copy()

        frame_class_names = detection.get("class_names") or []
        frame_track_ids = detection.get("track_ids")
        frame_bboxes = detection.get("bboxes")

        if frame_track_ids is None:
            frame_class_names = []
            frame_track_ids = []
            frame_bboxes = []

        for (class_name, track_id, bbox) in zip(frame_class_names, frame_track_ids, frame_bboxes):
            x1, y1, x2, y2 = [int(value.item()) for value in bbox]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            resized_crop = resize_and_pad_image(crop, save_image_size)
            track_id_value = int(track_id.item()) if hasattr(track_id, "item") else int(track_id)
            person_name = f"{class_name}_{track_id_value}"
            tracked_bboxes_by_frame.setdefault(person_name, {})[int(frame_idx)] = [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ]

            crop_folder = f"{exp_folder}/{person_name}"
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
        
    # Stitch background frames into a panorama. By default we use every frame so
    # client-side panorama animation reconstructs the full original length.
    sampled_frame_indices = list(range(0, len(frames), panorama_sample_stride))
    if sampled_frame_indices and sampled_frame_indices[-1] != (len(frames) - 1):
        sampled_frame_indices.append(len(frames) - 1)
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
        "sample_stride": int(panorama_sample_stride),
        "sampled_frame_indices": sampled_frame_indices,
        "num_frames": int(panorama_data["num_frames"]),
        "num_sampled_frames": int(panorama_data["num_frames"]),
        "original_num_frames": int(len(frames)),
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

        source_frame_paths = sorted(source_frames_dir.glob("*.png"))
        if not source_frame_paths:
            continue
        reference_image_path = source_frame_paths[0]

        person_name = person_dir.name
        person_dwpose_keypoints_csv = dwpose_keypoints_root / f"{person_name}_keypoints.csv"
        source_frame_indices: list[int] = []
        for source_frame_index, source_frame_path in enumerate(source_frame_paths):
            frame_stem = source_frame_path.stem
            source_frame_indices.append(int(frame_stem) if frame_stem.isdigit() else int(source_frame_index))

        person_bbox_lookup = tracked_bboxes_by_frame.get(person_name, {})
        person_bboxes = [
            person_bbox_lookup.get(frame_index, [-1.0, -1.0, -1.0, -1.0])
            for frame_index in source_frame_indices
        ]

        keypoint_frames = extract_dwpose_keypoints(
            frames_folder=str(source_frames_dir),
        )
        if not keypoint_frames:
            continue

        if len(keypoint_frames) != len(source_frame_indices):
            min_length = min(len(keypoint_frames), len(source_frame_indices))
            keypoint_frames = keypoint_frames[:min_length]
            source_frame_indices = source_frame_indices[:min_length]
            person_bboxes = person_bboxes[:min_length]

        save_dwpose_keypoints_csv(
            keypoint_frames=keypoint_frames,
            output_csv_path=str(person_dwpose_keypoints_csv),
            frame_indices=source_frame_indices,
            bboxes=person_bboxes,
        )

        dwpose_jobs.append(
            {
                "person_name": person_name,
                "reference_image": str(reference_image_path),
                "keypoints_csv": str(person_dwpose_keypoints_csv),
                "output_video": str(dwpose_videos_root / f"{person_name}_dwpose.mp4"),
                "animate_anyone_output_frames_dir": str(exp_path / "animate_anyone_frames" / person_name),
            }
        )

    metadata_path = exp_path / "metadata.json"
    metadata = {
        "experiment_path": str(exp_path),
        "panorama_image_path": str(exp_path / "panorama.png"),
        "panorama_data_path": str(panorama_metadata_path),
        "fps": fps,
        "num_original_frames": int(len(frames)),
        "panorama_sample_stride": int(panorama_sample_stride),
        "panorama_num_sampled_frames": int(len(sampled_background_frames)),
        "panorama_sampled_frame_indices": sampled_frame_indices,
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
    sample_stride = int(
        metadata.get(
            "panorama_sample_stride",
            panorama_data.get("sample_stride", 1),
        )
        or 1
    )
    sampled_frame_indices = metadata.get(
        "panorama_sampled_frame_indices",
        panorama_data.get("sampled_frame_indices", []),
    )
    if not isinstance(sampled_frame_indices, list):
        sampled_frame_indices = []
    sampled_frame_indices = [int(frame_index) for frame_index in sampled_frame_indices]

    if len(sampled_frame_indices) != len(reconstructed_background_frames):
        sampled_frame_indices = list(range(len(reconstructed_background_frames)))

    background_frame_count = 0
    background_frames_dir = latest_exp_path / "background"
    if background_frames_dir.exists():
        background_frame_count = sum(1 for _ in background_frames_dir.glob("*.png"))

    original_num_frames = int(
        metadata.get(
            "num_original_frames",
            panorama_data.get(
                "original_num_frames",
                background_frame_count if background_frame_count > 0 else len(reconstructed_background_frames),
            ),
        )
        or len(reconstructed_background_frames)
    )
    original_num_frames = max(1, original_num_frames)

    if reconstructed_background_frames and len(reconstructed_background_frames) != original_num_frames:
        sampled_indices_array = np.asarray(sampled_frame_indices, dtype=np.int32)
        expanded_background_frames: list[np.ndarray] = []
        for frame_index in range(original_num_frames):
            nearest_position = int(np.argmin(np.abs(sampled_indices_array - int(frame_index))))
            expanded_background_frames.append(reconstructed_background_frames[nearest_position].copy())
        reconstructed_background_frames = expanded_background_frames
        background_frame_indices = list(range(original_num_frames))
    else:
        background_frame_indices = (
            list(range(original_num_frames))
            if len(reconstructed_background_frames) == original_num_frames
            else sampled_frame_indices
        )

    composited_panorama_frames = [frame.copy() for frame in reconstructed_background_frames]

    reconstructed_is_full_timeline = (
        len(reconstructed_background_frames) == original_num_frames
        and len(background_frame_indices) == original_num_frames
    )
    if reconstructed_is_full_timeline:
        background_fps = fps
    else:
        background_fps = fps / sample_stride if sample_stride > 0 else fps

    if len(background_frame_indices) != len(composited_panorama_frames):
        background_frame_indices = list(range(len(composited_panorama_frames)))

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
    animate_anyone_output_root = latest_exp_path / "animate_anyone_frames"

    animate_anyone_enabled = True
    if not animate_anyone_default_dir.exists():
        print(f"AnimateAnyone repo not found, skipping pose2frames: {animate_anyone_default_dir}")
        animate_anyone_enabled = False
    elif not animate_anyone_config.exists():
        print(f"AnimateAnyone config not found, skipping pose2frames: {animate_anyone_config}")
        animate_anyone_enabled = False

    if animate_anyone_enabled:
        animate_anyone_output_root.mkdir(parents=True, exist_ok=True)

    for job in client_jobs:
        person_dwpose_keypoints_csv = Path(job["keypoints_csv"])
        person_name = str(job.get("person_name", person_dwpose_keypoints_csv.stem.replace("_keypoints", "")))
        person_dwpose_frames_dir = dwpose_frames_root / person_name
        output_path = Path(job["output_video"])

        loaded_keypoint_frames, loaded_bboxes, loaded_frame_indices = load_dwpose_keypoints_csv(
            str(person_dwpose_keypoints_csv),
            return_metadata=True,
        )
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

        animate_anyone_output_dir = Path(
            job.get(
                "animate_anyone_output_frames_dir",
                animate_anyone_output_root / person_name,
            )
        ).expanduser()

        generated_frames_dir = animate_anyone_output_dir

        try:
            generated_frames_dir = Path(run_animate_anyone(
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
                save_video=False,
                save_dir=str(animate_anyone_output_dir),
                filename_prefix="frame",
                transparent_threshold=8,
            ))
            print(f"Saved AnimateAnyone frames for {person_name}: {generated_frames_dir}")
        except Exception as exc:
            print(f"AnimateAnyone frame inference failed for {person_name}: {exc}")
            continue

        if not composited_panorama_frames:
            continue

        if not generated_frames_dir.exists():
            continue

        if not loaded_frame_indices:
            continue

        background_frame_size = (
            int(composited_panorama_frames[0].shape[0]),
            int(composited_panorama_frames[0].shape[1]),
        )
        source_indices = np.asarray(loaded_frame_indices, dtype=np.int32)
        sorted_source_indices = np.sort(source_indices)
        if sorted_source_indices.size > 1:
            index_gaps = np.diff(sorted_source_indices)
            median_gap = int(np.median(index_gaps)) if index_gaps.size > 0 else 1
        else:
            median_gap = 1

        max_alignment_distance = max(1, sample_stride, median_gap * 2)
        aligned_animate_frames: list[str | Path | np.ndarray | None] = []
        aligned_bboxes: list[np.ndarray | list[float] | tuple[float, float, float, float] | None] = []

        for background_frame_index in background_frame_indices:
            nearest_position = int(np.argmin(np.abs(source_indices - int(background_frame_index))))
            nearest_distance = abs(int(source_indices[nearest_position]) - int(background_frame_index))

            if nearest_distance > max_alignment_distance:
                aligned_animate_frames.append(None)
                aligned_bboxes.append(None)
                continue

            aligned_animate_frames.append(generated_frames_dir / f"frame_{nearest_position:06d}.png")
            aligned_bboxes.append(loaded_bboxes[nearest_position])

        scaled_person_overlays = scale_frame_to_bbox(
            animate_anyone_frames=aligned_animate_frames,
            bbox_coordinates=aligned_bboxes,
            output_frame_size=background_frame_size,
        )

        composited_panorama_frames = overlay_object_on_background_video(
            background_frames=composited_panorama_frames,
            object_frames=scaled_person_overlays,
        )

    final_panorama_with_people_path = latest_exp_path / "panorama_with_people.mp4"
    if composited_panorama_frames:
        encode_video_libsvtav1(
            output_path=str(final_panorama_with_people_path),
            fps=background_fps,
            crf=25,
            frames=composited_panorama_frames,
        )
        print(f"Saved panorama with overlaid people: {final_panorama_with_people_path}")


def main():
    run_server()
    run_client()


if __name__ == "__main__":
    main()