"""Export MediaPipe 21-joint hand pose landmarks to compact JSON for web demo."""

from __future__ import annotations

import sqlite3  # noqa: F401 - Host rule: import sqlite3 before torch
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.pipeline.hand_keypoints import extract_video_hand_poses

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLIPS = [
    ("clip_01", Path("demo/outputs/results/clip_01_factory001_worker001_00001/reference_trimmed.mp4")),
    ("clip_02", Path("demo/outputs/results/clip_02_factory001_worker001_00002/reference_trimmed.mp4")),
    ("clip_03", Path("demo/outputs/results/clip_03_factory001_worker001_00000/reference_trimmed.mp4")),
]

OUTPUT_DIR = Path("demo/outputs/pitch")


def export_clip_keypoints(clip_id: str, video_path: Path, output_json: Path) -> None:
    logger.info(f"Extracting keypoints for {clip_id} from {video_path}...")
    poses = extract_video_hand_poses(video_path, max_frames=300, smooth=True)

    frames_data = []
    for p in poses:
        hands_list = []
        for h in p.hands:
            # Round normalized coordinates to 4 decimal places for compact web transfer
            rounded_landmarks = [
                [round(coord, 4) for coord in pt]
                for pt in h.landmarks_norm
            ]
            hands_list.append({
                "side": h.handedness,
                "score": round(h.confidence, 3),
                "bbox": h.bbox,
                "landmarks": rounded_landmarks,
            })
        frames_data.append({
            "frame": p.frame_idx,
            "hands": hands_list,
        })

    payload = {
        "clip_id": clip_id,
        "fps": 30.0,
        "total_frames": len(poses),
        "resolution": [1920, 1080],
        "frames": frames_data,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = output_json.stat().st_size / 1024.0
    logger.info(f"Saved {output_json.name} ({size_kb:.1f} KB, {len(poses)} frames)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for clip_id, video_path in CLIPS:
        if not video_path.exists():
            logger.warning(f"Video {video_path} not found, skipping {clip_id}")
            continue
        out_file = OUTPUT_DIR / f"keypoints_{clip_id}.json"
        export_clip_keypoints(clip_id, video_path, out_file)


if __name__ == "__main__":
    main()

