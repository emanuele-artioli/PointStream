"""Rebuild one website mask family and its full ladder on selected sources."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from demo.experiments.clip_identity import file_sha256, load_clip_manifest
from demo.experiments.validate_demo_refresh import probe_video

FAMILIES = ("yoloe", "sam31", "dino", "rtmpose")
RUNGS = ("180p", "240p", "360p", "540p", "720p", "1080p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dinov3-repo", type=Path)
    parser.add_argument(
        "--inference-python",
        type=Path,
        help="Explicit compatible DINO runtime; never an automatic fallback",
    )
    args = parser.parse_args()
    manifest = load_clip_manifest(args.source_manifest)
    stage = Path(os.environ["PS_STAGE_DIR"])
    if args.validate:
        report = json.loads((stage / "mask-report.json").read_text())
        validate(report, stage, manifest=manifest, family=args.family)
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(
            json.dumps(
                {
                    "passed": True,
                    "checks": {
                        "complete_mask_ladders": True,
                        "real_decodes_and_frame_counts": True,
                        "source_weight_identities": True,
                    },
                    "citable": False,
                }
            )
        )
        return
    os.environ.update(
        PS_NATIVE_THREADS="8",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8",
        FFMPEG="/opt/local/bin/ffmpeg",
    )
    import cv2
    import numpy as np
    import torch
    from demo.experiments.encode_seg_av1_ladder import ladder, render_sam, render_yoloe_hands
    from demo.pipeline.maps.model_paths import require
    from experiments.jobs.monitor import publish_progress

    if not torch.cuda.is_available():
        raise RuntimeError("mask refresh requires the claimed CUDA device")
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    weights = []
    if args.family == "sam31":
        from demo.pipeline.maps.sam31_video import resolve_checkpoint

        checkpoint = resolve_checkpoint()
        if checkpoint is None:
            raise RuntimeError("SAM3.1 checkpoint unavailable")
        weights = [checkpoint]
    elif args.family == "yoloe":
        weights = [require("yoloe26_seg"), require("mobileclip2")]
    elif args.family == "dino":
        if args.dinov3_repo is None or not (args.dinov3_repo / "hubconf.py").is_file():
            raise RuntimeError("explicit local DINOv3 repository required")
        os.environ["DINOV3_REPO"] = str(args.dinov3_repo)
        weights = [require("dinov3_vits")]
    elif args.family == "rtmpose":
        cache = Path("/home/itec/emanuele/.cache/rtmlib/hub/checkpoints")
        weights = [
            cache / "rtmdet_nano_8xb32-300e_hand-267f9c8f.onnx",
            cache / "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx",
        ]
        os.environ.update(PS_RTM_DET=str(weights[0]), PS_RTM_POSE=str(weights[1]))
    identities = [
        {"path": str(path), "sha256": file_sha256(path, timeout=90), "bytes": path.stat().st_size}
        for path in weights
    ]
    report = {
        "status": "preparing",
        "family": args.family,
        "source_manifest_sha256": manifest.sha256,
        "weights": identities,
        "native_tools": {
            name: {
                "path": "/opt/local/bin/" + name,
                "version": subprocess.check_output(
                    ["/opt/local/bin/" + name, "-version"], text=True
                ).splitlines()[0],
            }
            for name in ("ffmpeg", "ffprobe")
        },
        "clips": {},
    }
    if args.family == "dino":
        report["model_code"] = model_code_identity(args.dinov3_repo)
    (stage / "mask-report.json").write_text(json.dumps(report, indent=2))
    for clip in manifest.clips:
        start = time.monotonic()
        clip.verify()
        work = stage / clip.clip_id
        work.mkdir()
        source = work / (clip.clip_id + ".mkv")
        subprocess.run(
            [
                "/opt/local/bin/ffmpeg",
                "-v",
                "error",
                "-i",
                str(clip.path),
                "-frames:v",
                str(args.frames),
                "-an",
                "-c:v",
                "ffv1",
                "-threads",
                "8",
                str(source),
            ],
            check=True,
        )
        count = int(probe_video(source, "/opt/local/bin/ffprobe")["nb_read_frames"])
        png = work / "png"
        if args.family == "yoloe":
            render_yoloe_hands(source, png, max_frames=count)
        elif args.family == "sam31":
            render_sam(
                source,
                png,
                work / "sam",
                max_frames=count,
                prompts_override={"hand": "hand", "tool": "tool", "workbench": "workbench"},
            )
        elif args.family == "dino":
            if args.inference_python is None or not args.inference_python.is_file():
                raise RuntimeError("explicit compatible DINO inference interpreter required")
            subprocess.run(
                [
                    str(args.inference_python),
                    "-m",
                    "demo.pipeline.maps.dinov3_features",
                    "--clip",
                    str(source),
                    "--out",
                    str(work / "dino"),
                    "--max-frames",
                    str(count),
                ],
                check=True,
            )
            png = work / "dino/preview_pca"
        else:
            from demo.evaluation.pose_backends import BACKENDS
            from demo.pipeline.hand_keypoints import render_skeleton_on_canvas

            png.mkdir()
            poses = BACKENDS["rtm_hand"](source, count)
            if len(poses) != count:
                raise RuntimeError("pose frame count mismatch")
            for index, pose in enumerate(poses):
                image = render_skeleton_on_canvas(pose, width=1920, height=1080)
                if not cv2.imwrite(str(png / f"{index:06d}.png"), np.asarray(image)):
                    raise RuntimeError("pose preview write failed")
        paths = sorted(png.glob("*.png"))
        if len(paths) != count:
            raise RuntimeError("mask extraction dropped source frames")
        streams = ladder(png, work / "ladder", args.family, "/opt/local/bin/ffmpeg", count)
        for name, row in streams.items():
            path = Path(row["path"])
            row["relative_path"] = str(path.relative_to(stage))
            row["sha256"] = file_sha256(path)
        clip.verify()
        report["clips"][clip.clip_id] = {
            "source_identity": clip.receipt(),
            "n_frames": count,
            "fps": 30,
            "streams": streams,
            "elapsed_seconds": time.monotonic() - start,
        }
        (stage / "mask-report.json").write_text(json.dumps(report, indent=2))
        publish_progress(os.environ["PS_STAGE"], len(report["clips"]))
    report["status"] = "complete"
    (stage / "mask-report.json").write_text(json.dumps(report, indent=2))
    validate(report, stage, manifest=manifest, family=args.family)


def model_code_identity(path: Path) -> dict:
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True, timeout=90
    )
    if dirty:
        raise ValueError("DINOv3 model repository must be clean")
    return {"path": str(path), "revision": revision}


def validate(report: dict, stage: Path, *, manifest=None, family: str | None = None) -> None:
    if report.get("status") != "complete" or set(report.get("clips", {})) != {
        "clip_01",
        "clip_02",
        "clip_03",
    }:
        raise ValueError("complete three-clip mask report required")
    if report.get("family") not in FAMILIES or not report.get("weights"):
        raise ValueError("identified mask family and weights required")
    if family is not None and report["family"] != family:
        raise ValueError("mask family mismatch")
    if manifest is not None:
        if report.get("source_manifest_sha256") != manifest.sha256:
            raise ValueError("source manifest mismatch")
        for clip in manifest.clips:
            identity = report["clips"][clip.clip_id]["source_identity"]
            if identity["path"] != str(clip.path) or identity["sha256"] != clip.sha256:
                raise ValueError("wrong selected mask source")
    if report["family"] == "dino":
        identity = report["model_code"]
        if model_code_identity(Path(identity["path"])) != identity:
            raise ValueError("DINOv3 model code identity changed")
    for identity in report["weights"]:
        if file_sha256(Path(identity["path"]), timeout=90) != identity["sha256"]:
            raise ValueError("mask checkpoint identity changed")
    for row in report["clips"].values():
        if set(row["streams"]) != set(RUNGS):
            raise ValueError("incomplete mask ladder")
        if not isinstance(row["n_frames"], int) or row["n_frames"] <= 0:
            raise ValueError("positive mask frame count required")
        identity = row["source_identity"]
        if file_sha256(Path(identity["path"])) != identity["sha256"]:
            raise ValueError("mask source identity changed")
        for stream in row["streams"].values():
            path = stage / stream["relative_path"]
            info = probe_video(path, "/opt/local/bin/ffprobe")
            if (
                info["codec_name"] != "av1"
                or int(info["nb_read_frames"]) != row["n_frames"]
                or file_sha256(path) != stream["sha256"]
            ):
                raise ValueError("corrupt or mismatched native mask ladder media")


if __name__ == "__main__":
    main()
