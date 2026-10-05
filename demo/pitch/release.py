"""Assemble and verify one complete, immutable public demo release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

from demo.experiments.clip_identity import CLIP_IDS, file_sha256
from demo.experiments.demo_mask_refresh import FAMILIES, RUNGS
from demo.experiments.validate_demo_refresh import EXPECTED_STREAMS, probe_video

NAMES = {
    "clip_01": "Factory 1 · Clip 1",
    "clip_02": "Factory 2 · Clip 1",
    "clip_03": "Factory 1 · Clip 3",
}
LABELS = {
    "dino": "DINOv3 PCA feature preview",
    "rtmpose": "RTMPose hand skeleton",
    "yoloe": "YOLOE hand masks",
    "sam31": "SAM 3.1 hand / tool / workbench",
}


def safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or root.resolve() not in path.resolve().parents
    ):
        raise ValueError("release path escapes its root")
    return path


def verify_release(root: Path, release: dict) -> None:
    if release.get("schema") != "pointstream.demo.release.v1" or set(
        release.get("clips", {})
    ) != set(CLIP_IDS):
        raise ValueError("complete three-clip release required")
    expected = set()
    for clip_id, clip in release["clips"].items():
        if set(clip["streams"]) != EXPECTED_STREAMS or clip["n_frames"] not in (299, 300):
            raise ValueError("incomplete full camera ladder")
        expected.update(f"web_{clip_id}_{key}.mp4" for key in EXPECTED_STREAMS)
        expected.add(f"keypoints_{clip_id}.json")
    entries = release["maps"]["entries"]
    tuples = {(row["clip"], row["family"], row["resolution"]) for row in entries}
    if len(entries) != 72 or tuples != {
        (clip, family, rung) for clip in CLIP_IDS for family in FAMILIES for rung in RUNGS
    }:
        raise ValueError("incomplete or duplicate mask ladders")
    for row in entries:
        if row["n_frames"] != release["clips"][row["clip"]]["n_frames"]:
            raise ValueError("mask/camera frame mismatch")
        expected.add("maps/" + row["overlay_url"])
    if set(release["files"]) != expected:
        raise ValueError("release file inventory differs from its ladders")
    for relative, identity in release["files"].items():
        path = safe_path(root, relative)
        if (
            not path.is_file()
            or path.stat().st_size != identity["bytes"]
            or file_sha256(path) != identity["sha256"]
        ):
            raise ValueError(f"release file changed: {relative}")


def assemble(export: Path, masks: dict[str, Path], out: Path, *, ffprobe: str) -> dict:
    if out.exists() or set(masks) != set(FAMILIES):
        raise ValueError("new release directory and all four mask families required")
    report = json.loads((export / "report.json").read_text())
    if report.get("status") != "complete" or set(report["clips"]) != set(CLIP_IDS):
        raise ValueError("complete camera export required")
    out.mkdir(parents=True)
    release = {
        "schema": "pointstream.demo.release.v1",
        "clips": {},
        "files": {},
        "maps": {"schema": "pointstream.demo.previews.v1", "entries": []},
        "measurements": "Pose-model agreement; not human task truth. Latency and perceptual metrics are not measured for this release.",
        "native_ffmpeg": report.get("native_ffmpeg"),
        "source_manifest_sha256": report["source_manifest"]["sha256"],
        "recipe": "Per-factory trained pix2pix foreground, transmitted DWB2 hand coordinates, six-rung native AV1 background. Composite browser previews are H.264; AV1 baselines and mask previews remain native AV1. Model and anchor setup bytes are separate from streaming rates.",
    }

    def copy_media(source: Path, relative: str, count: int, codec: str) -> dict:
        info = probe_video(source, ffprobe)
        if info["codec_name"] != codec or int(info["nb_read_frames"]) != count:
            raise ValueError("wrong decoded release codec or frame count")
        dest = safe_path(out, relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        identity = {
            "sha256": file_sha256(dest),
            "bytes": dest.stat().st_size,
            "codec": codec,
            "n_frames": count,
            "width": int(info["width"]),
            "height": int(info["height"]),
        }
        release["files"][relative] = identity
        return identity

    for clip_id in CLIP_IDS:
        row = report["clips"][clip_id]
        count = row["n_frames"]
        if count not in (299, 300) or set(row["streams"]) != EXPECTED_STREAMS:
            raise ValueError("full camera export required")
        clip = {
            "name": NAMES[clip_id],
            "keypoints": f"keypoints_{clip_id}.json",
            "n_frames": count,
            "fps": row["fps"],
            "factory": row["factory"],
            "training_scene": row["training_scene"],
            "source_sha256": row["source_identity"]["sha256"],
            "checkpoint_sha256": row["checkpoint_identity"]["sha256"],
            "anchor_sha256": row["anchor_identity"]["sha256"],
            "setup_bytes": row["checkpoint_identity"].get("bytes", 0)
            + row["anchor_identity"].get("bytes", 0),
            "streams": {},
        }
        for key, stream in row["streams"].items():
            relative = f"web_{clip_id}_{key}.mp4"
            identity = copy_media(
                export / "pitch" / relative,
                relative,
                count,
                "av1" if key.startswith("av1_") else "h264",
            )
            values = {
                metric: stream.get(metric)
                for metric in ("kind", "res", "kbps", "mpjpe", "det", "kp")
            }
            for metric in ("kbps", "mpjpe", "det", "kp"):
                value = values[metric]
                if value is not None and (
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                ):
                    if metric == "mpjpe":
                        values[metric] = None
                    else:
                        raise ValueError("invalid camera metric")
            if key == "ref":
                values["kbps"] = round(identity["bytes"] * 8 / (count / clip["fps"]) / 1000, 1)
            clip["streams"][key] = values
        source = export / "pitch" / clip["keypoints"]
        if len(json.loads(source.read_text())) != count:
            raise ValueError("keypoint frame mismatch")
        shutil.copy2(source, out / clip["keypoints"])
        release["files"][clip["keypoints"]] = {
            "sha256": file_sha256(source),
            "bytes": source.stat().st_size,
        }
        release["clips"][clip_id] = clip

    for family, stage in masks.items():
        mask = json.loads((stage / "mask-report.json").read_text())
        if (
            mask.get("status") != "complete"
            or mask.get("family") != family
            or not mask.get("weights")
        ):
            raise ValueError("complete identified mask family required")
        if mask.get("source_manifest_sha256") != release["source_manifest_sha256"]:
            raise ValueError("mask source manifest differs from the camera export")
        for clip_id in CLIP_IDS:
            row = mask["clips"][clip_id]
            clip = release["clips"][clip_id]
            if (
                row["source_identity"]["sha256"] != clip["source_sha256"]
                or row["n_frames"] != clip["n_frames"]
                or set(row["streams"]) != set(RUNGS)
            ):
                raise ValueError("mask source/frame/ladder mismatch")
            for rung, stream in row["streams"].items():
                source = safe_path(stage, stream["relative_path"])
                if file_sha256(source) != stream["sha256"]:
                    raise ValueError("mask export changed")
                relative = f"maps/{clip_id}/{family}_{rung}.mp4"
                identity = copy_media(source, relative, row["n_frames"], "av1")
                release["maps"]["entries"].append(
                    {
                        "clip": clip_id,
                        "family": family,
                        "resolution": rung,
                        "map": family + "_" + rung.removesuffix("p"),
                        "label": LABELS[family] + " " + rung,
                        "status": "ok",
                        "overlay_key": "black",
                        "overlay_url": relative.removeprefix("maps/"),
                        "n_frames": row["n_frames"],
                        "fps": clip["fps"],
                        "kind": "av1_preview",
                        "payload_format": "AV1-encoded visual preview; not native feature/keypoint payload",
                        "preview_kbps": round(
                            identity["bytes"] * 8 / (row["n_frames"] / clip["fps"]) / 1000, 1
                        ),
                        "weights_sha256": [weight["sha256"] for weight in mask["weights"]],
                    }
                )
    release["release_id"] = hashlib.sha256(
        json.dumps(release, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:16]
    verify_release(out, release)
    (out / "demo-release.json").write_text(json.dumps(release, indent=2, allow_nan=False) + "\n")
    (out / "maps/index.json").write_text(json.dumps(release["maps"], indent=2) + "\n")
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument(
        "--mask", action="append", required=True, help="FAMILY=completed stage directory"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffprobe", required=True)
    args = parser.parse_args()
    assemble(
        args.export,
        dict((name, Path(path)) for name, path in (value.split("=", 1) for value in args.mask)),
        args.out,
        ffprobe=args.ffprobe,
    )


if __name__ == "__main__":
    main()
