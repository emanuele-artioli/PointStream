"""Validate real demo ladder media before fleet promotion or publication."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from demo.experiments.clip_identity import CLIP_IDS, file_sha256, load_clip_manifest

EXPECTED_STREAMS = {
    "ref",
    "ps_180",
    "ps_starve",
    "ps_heavy",
    "ps_low",
    "ps_720",
    "ps_1080",
    "av1_180",
    "av1_240",
    "av1_360",
    "av1_540",
    "av1_720",
    "av1_1080",
}


def probe_video(path: Path, ffprobe: str) -> dict:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"expected one readable video stream: {path}")
    return streams[0]


def validate(report: dict, stage: Path, *, ffprobe: str) -> dict:
    clips = report.get("clips", {})
    if report.get("status") != "complete" or set(clips) != set(CLIP_IDS):
        raise ValueError("a complete three-clip export is required")
    for clip_id, row in clips.items():
        if set(row.get("streams", {})) != EXPECTED_STREAMS:
            raise ValueError(f"incomplete ladder for {clip_id}")
        count = row.get("n_frames")
        if type(count) is not int or count < 1:
            raise ValueError("missing actual frame count")
        for identity_key in ("checkpoint_identity", "anchor_identity", "source_identity"):
            identity = row[identity_key]
            if file_sha256(Path(identity["path"]), timeout=90) != identity["sha256"]:
                raise ValueError(f"changed {identity_key}")
        for key, stream in row["streams"].items():
            for metric in ("kbps", "det", "kp"):
                value = stream.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid {metric}: {clip_id}/{key}")
            path = stage / "pitch" / f"web_{clip_id}_{key}.mp4"
            info = probe_video(path, ffprobe)
            if (
                int(info["nb_read_frames"]) != count
                or int(info["width"]) < 1
                or int(info["height"]) < 1
            ):
                raise ValueError(f"invalid decoded media extent: {path}")
            if key.startswith("av1_") and info["codec_name"] != "av1":
                raise ValueError("camera baseline must remain native AV1")
            if not key.startswith("av1_") and info["codec_name"] != "h264":
                raise ValueError("browser composite/reference must be H.264")
        poses = json.loads((stage / "pitch" / f"keypoints_{clip_id}.json").read_text())
        if len(poses) != count:
            raise ValueError("keypoint/media frame mismatch")
    return {
        "complete_ladders": True,
        "native_media_decodes": True,
        "matching_actual_frame_counts": True,
        "unchanged_source_checkpoint_anchor_identities": True,
        "finite_rates": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--ffprobe", default="/opt/local/bin/ffprobe")
    args = parser.parse_args()
    load_clip_manifest(args.source_manifest)
    stage = Path(os.environ["PS_STAGE_DIR"])
    report = json.loads((stage / "report.json").read_text())
    checks = validate(report, stage, ffprobe=args.ffprobe)
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            {
                "passed": True,
                "checks": checks,
                "citable": False,
                "note": "Demo assets and model-agreement diagnostics; not scientific task truth.",
            }
        )
    )


if __name__ == "__main__":
    main()
