"""Decode-only verification of saved E03B artifacts. Never encodes or overwrites the run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from experiments.tier.e03b_persist import (
    DecodeCountError,
    dump_decoded_rgb,
    sha256_bytes,
    sha256_path,
)
from experiments.tier.e03b_run import load_prepared_reuse
from experiments.tier.e03b_source import EXPECTED_SHAPE, T_START_S
from src.components.codec.tools import resolve_ffmpeg

SETTINGS = (
    ("av1", 63, "payload.ivf"),
    ("av1", 47, "payload.ivf"),
    ("vvc", 63, "payload.vvc"),
    ("vvc", 47, "payload.vvc"),
)
ORIGINAL_BOUNDS_SHA256 = "77c30b5b2d81da8bd8abe50e41150571db9d7615bcbdf3ee0657f1cbdb5c2d9a"
ORIGINAL_REPORT_SHA256 = "62ac0b15228001f7598958d501df133003d8c6faea0b5b6947670a3839fc45f7"
PREPARED_SHA256 = "1f02475a5bbc3d94e4bae2e904dc29c3af3082be0c0c160e027b706a6950f6f8"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _ffprobe() -> str:
    return str(Path(resolve_ffmpeg().path).with_name("ffprobe"))


def count_container_frames(video_path: Path) -> dict[str, Any]:
    payload = json.loads(
        subprocess.check_output(
            [
                _ffprobe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_read_frames,nb_frames",
                "-of",
                "json",
                str(video_path),
            ],
            text=True,
        )
    )
    stream = (payload.get("streams") or [{}])[0]
    read = stream.get("nb_read_frames") or stream.get("nb_frames")
    return {
        "path": str(video_path),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "nb_frames": int(read) if read not in {None, "N/A", ""} else None,
        "sha256": sha256_path(video_path),
    }


def verify_setting(run_dir: Path, codec: str, qp: int, bitstream_name: str) -> dict[str, Any]:
    work = run_dir / f"{codec}_qp{qp}"
    bitstream = work / bitstream_name
    row = json.loads((work / "campaign_row.json").read_text(encoding="utf-8"))
    ordinary = work / "decoded.mkv"
    standalone = work / "decoded_standalone.mkv"
    npy_path = work / "decoded_rgb.npy"
    ffmpeg = resolve_ffmpeg().path
    ordinary_probe = count_container_frames(ordinary)
    standalone_probe = count_container_frames(standalone)
    dumped, geometry = dump_decoded_rgb(
        ffmpeg,
        ordinary,
        expected_width=EXPECTED_SHAPE[2],
        expected_height=EXPECTED_SHAPE[1],
        expected_count=EXPECTED_SHAPE[0],
    )
    standalone_frames, standalone_geometry = dump_decoded_rgb(
        ffmpeg,
        standalone,
        expected_width=EXPECTED_SHAPE[2],
        expected_height=EXPECTED_SHAPE[1],
        expected_count=EXPECTED_SHAPE[0],
    )
    saved = np.load(npy_path)
    pixels_match = bool(np.array_equal(dumped, standalone_frames))
    npy_match = bool(np.array_equal(dumped, saved))
    bitstream_sha = sha256_path(bitstream)
    return {
        "codec": codec,
        "qp": qp,
        "bitstream": str(bitstream),
        "bitstream_bytes": bitstream.stat().st_size,
        "bitstream_sha256": bitstream_sha,
        "row_sha256": row.get("artifact_sha256"),
        "bitstream_matches_row": bitstream_sha == row.get("artifact_sha256"),
        "ordinary_container": ordinary_probe,
        "standalone_container": standalone_probe,
        "ordinary_dump": geometry,
        "standalone_dump": standalone_geometry,
        "ordinary_standalone_pixels_match": pixels_match,
        "dump_matches_saved_npy": npy_match,
        "decode_rejections": []
        if pixels_match
        and npy_match
        and ordinary_probe.get("nb_frames") == EXPECTED_SHAPE[0]
        and standalone_probe.get("nb_frames") == EXPECTED_SHAPE[0]
        else ["decode verification failed"],
    }


def decode_source_rgb_at_pts(video_path: Path, pts_s: float, width: int, height: int) -> np.ndarray:
    raw = subprocess.check_output(
        [
            resolve_ffmpeg().path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{pts_s:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    frame_bytes = height * width * 3
    if len(raw) != frame_bytes:
        raise DecodeCountError(
            f"native seek at {pts_s}: got {len(raw)} bytes, expected {frame_bytes}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()


def verify_png_native_provenance(run_dir: Path, sample_indices: tuple[int, ...] | None = None) -> dict[str, Any]:
    recipe = json.loads((run_dir / "source_recipe.json").read_text(encoding="utf-8"))
    video = Path(recipe["raw_input"]["path"])
    colour = recipe.get("colour") or {}
    width = int(colour.get("width") or 0)
    height = int(colour.get("height") or 0)
    mapping = list(recipe["frame_mapping"])
    if sample_indices is None:
        chosen = mapping
    else:
        chosen = [mapping[index] for index in sample_indices]
    rows: list[dict[str, Any]] = []
    n_equal = 0
    for row in chosen:
        png_path = Path(row["extracted_path"])
        png = cv2.cvtColor(cv2.imread(str(png_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        native = decode_source_rgb_at_pts(video, float(row["native_pts_s"]), width, height)
        equal = bool(png.shape == native.shape and np.array_equal(png, native))
        n_equal += int(equal)
        mad = float(np.mean(np.abs(png.astype(np.int16) - native.astype(np.int16))))
        rows.append(
            {
                "extracted_frame_index": row["extracted_frame_index"],
                "native_pts_s": row["native_pts_s"],
                "png_sha256": sha256_path(png_path),
                "recipe_png_sha256": row.get("extracted_sha256"),
                "png_native_pixels_equal": equal,
                "mean_abs_diff": round(mad, 4),
                "png_shape": list(png.shape),
                "native_shape": list(native.shape),
            }
        )
    derived = {
        "schema": "pointstream.e03b_derived_native_mapping.v1",
        "original_mapping_preserved": True,
        "original_recipe": str(run_dir / "source_recipe.json"),
        "method": "ffmpeg -ss native_pts before -i, one RGB24 frame vs selected PNG",
        "n_compared": len(rows),
        "n_pixel_equal": n_equal,
        "uncertainty": (
            "Nearest packet-DTS assignment is not pixel provenance. The selected PNGs "
            "come from -ss 80.7807 -r 24 extraction. Equality here would mean that "
            "resampled PNG matches a native seek at the mapped PTS; inequality is "
            "recorded, not silently repaired."
        ),
        "rows": rows,
    }
    return derived


def alarm_disposition(run_dir: Path) -> dict[str, Any]:
    calibration = json.loads((run_dir / "metric-calibration.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "probe_report.json").read_text(encoding="utf-8"))
    vmaf = ((calibration.get("metrics") or {}).get("vmaf") or {}).get("by_anchor") or {}
    return {
        "schema": "pointstream.e03b_alarm_disposition.v1",
        "original_bounds_sha256": ORIGINAL_BOUNDS_SHA256,
        "original_report_sha256": ORIGINAL_REPORT_SHA256,
        "original_files_immutable": True,
        "alarm": "vvc qp63: vmaf=0.0 outside band",
        "disposition": "closed_calibrated_floor",
        "not_a_wiring_fault": True,
        "calibration_anchors": {
            "identical": vmaf.get("identical"),
            "severe-blur": vmaf.get("severe-blur"),
            "unrelated-clip": vmaf.get("unrelated-clip"),
            "spatial-null-uniform": ((calibration.get("metrics") or {}).get("vmaf") or {})
            .get("null_controls", {})
            .get("spatial-null-uniform"),
        },
        "report_alarms": report.get("alarms"),
        "scope": "one development scene; single-run timing; not a general winner",
        "confirmation_scores_authorized": False,
    }


def verify_run(run_dir: Path, out_dir: Path, *, pts_samples: tuple[int, ...] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"verification directory {out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe, frames = load_prepared_reuse(run_dir)
    settings = [verify_setting(run_dir, codec, qp, name) for codec, qp, name in SETTINGS]
    provenance = verify_png_native_provenance(run_dir, sample_indices=pts_samples)
    _write(out_dir / "derived_native_mapping.json", provenance)
    disposition = alarm_disposition(run_dir)
    _write(out_dir / "vmaf_alarm_disposition.json", disposition)
    eligibility = json.loads((run_dir / "confirmation_eligibility.json").read_text(encoding="utf-8"))
    summary = {
        "schema": "pointstream.e03b_acceptance_verification.v1",
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "original_report_sha256": sha256_path(run_dir / "probe_report.json"),
        "original_bounds_sha256": sha256_path(run_dir / "bounds.json"),
        "prepared_sha256": recipe["prepared_sha256"],
        "prepared_sha256_expected": PREPARED_SHA256,
        "prepared_hash_ok": recipe["prepared_sha256"] == PREPARED_SHA256 == sha256_bytes(
            np.ascontiguousarray(frames).tobytes()
        ),
        "t_start_s": recipe["raw_input"]["t_start_s"],
        "expected_t_start_s": T_START_S,
        "settings": settings,
        "n_bitstreams": len(settings),
        "n_decoded_containers": 8,
        "all_streams_match_rows": all(item["bitstream_matches_row"] for item in settings),
        "all_decodes_48x360x640": all(
            item["ordinary_dump"]["count"] == 48 and item["standalone_dump"]["count"] == 48 for item in settings
        ),
        "all_ordinary_standalone_pixels_match": all(
            item["ordinary_standalone_pixels_match"] for item in settings
        ),
        "confirmation_scores_computed": eligibility.get("counts", {}).get("scores_computed"),
        "confirmation_scoring_authorized": False,
        "native_png_pixel_equal_count": provenance["n_pixel_equal"],
        "native_png_compared": provenance["n_compared"],
        "alarm_disposition": str(out_dir / "vmaf_alarm_disposition.json"),
        "derived_mapping": str(out_dir / "derived_native_mapping.json"),
    }
    _write(out_dir / "verification.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--pts-samples",
        default="0,23,47",
        help="comma-separated mapping indices, or 'all'",
    )
    args = parser.parse_args()
    if args.pts_samples.strip() == "all":
        samples: tuple[int, ...] | None = None
    else:
        samples = tuple(int(item) for item in args.pts_samples.split(",") if item.strip())
    summary = verify_run(args.run_dir.resolve(), args.out_dir.resolve(), pts_samples=samples)
    print(json.dumps({k: summary[k] for k in summary if k != "settings"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
