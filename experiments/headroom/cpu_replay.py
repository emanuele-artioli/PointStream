"""CPU-only replay of retained BP21 streams; no encoding, models or GPU use.

Scores each independently decoded original/plate stream against BOTH native
luma targets. Equal-frame mean dB replicates the historical PSNR definition;
pooled pixel MSE is reported separately. Source identifiers are bootstrap
clusters, not proof of independent matches. The worker is read-only outside
its unique output directory and pauses when GPU3 receives a GPU workload.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
import scipy

EXPECTED_REPORT = "b6f8b8463c73f7d3a54a918b677bdb5b1e18ae7cf789490c900e023fa1d21240"


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ReadBudget:
    """Cap this worker's cumulative file reads; excludes decoder RAM pipes."""

    def __init__(self, mib_per_second: float):
        self.started = time.monotonic()
        self.bytes = 0
        self.rate = mib_per_second * 1024**2

    def charge(self, count: int) -> None:
        self.bytes += count
        if self.rate:
            delay = self.bytes / self.rate - (time.monotonic() - self.started)
            if delay > 0:
                time.sleep(delay)


def sha_file(path: Path, budget: ReadBudget | None = None) -> dict:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        while block := handle.read(1024**2):
            if budget:
                budget.charge(len(block))
            digest.update(block)
            count += len(block)
    return {"path": str(path), "bytes": count, "sha256": digest.hexdigest()}


def exact_read(handle, count: int) -> bytes:
    blocks = []
    left = count
    while left:
        block = handle.read(left)
        if not block:
            raise ValueError(f"truncated Y4M plane: {left} bytes missing")
        blocks.append(block)
        left -= len(block)
    return b"".join(blocks)


def header_shape(header: bytes) -> tuple[int, int]:
    tokens = header.decode("ascii").strip().split()
    if not tokens or tokens[0] != "YUV4MPEG2":
        raise ValueError("not a Y4M stream")
    chroma = next((x for x in tokens if x.startswith("C")), "C420jpeg")
    if chroma not in {"C420jpeg", "C420mpeg2", "C420paldv", "C420"}:
        raise ValueError(f"unsupported pixel format {chroma}")
    width = int(next(x[1:] for x in tokens if x.startswith("W")))
    height = int(next(x[1:] for x in tokens if x.startswith("H")))
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("invalid 4:2:0 dimensions")
    return height, width


def y4m_frames(handle, *, budget=None, digest=None):
    header = handle.readline()
    height, width = header_shape(header)
    if digest:
        digest.update(header)
    if budget:
        budget.charge(len(header))
    while marker := handle.readline():
        if not marker.startswith(b"FRAME") or not marker.endswith(b"\n"):
            raise ValueError("invalid Y4M frame marker")
        plane = exact_read(handle, height * width)
        chroma = exact_read(handle, height * width // 2)
        if digest:
            digest.update(marker + plane + chroma)
        if budget:
            budget.charge(len(marker) + len(plane) + len(chroma))
        yield np.frombuffer(plane, dtype=np.uint8).reshape(height, width)


def load_source(path: Path, expected_frames: int, budget: ReadBudget):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        frames = list(y4m_frames(handle, budget=budget, digest=digest))
    if len(frames) != expected_frames:
        raise ValueError(f"{path}: {len(frames)} != {expected_frames} frames")
    pixels = np.stack(frames)
    return pixels, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "luma_sha256": hashlib.sha256(pixels).hexdigest(),
        "shape": list(pixels.shape),
    }


def verify_rgb_window(window: Path, luma: np.ndarray, budget: ReadBudget) -> dict:
    """Verify the retained RGB window maps exactly to the scored source luma.

    This identifies retained frames; it does not recertify their extraction
    from the original MP4, independent mask truth, or development exposure.
    """
    files = sorted(window.glob("frame_*.png"))
    if len(files) != len(luma):
        raise ValueError("registered RGB window has missing or excess frames")
    rgb_digest = hashlib.sha256()
    identities = []
    for index, path in enumerate(files):
        encoded = path.read_bytes()
        budget.charge(len(encoded))
        with Image.open(io.BytesIO(encoded)) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape != (*luma[index].shape, 3):
            raise ValueError("RGB window and stored luma have different geometry")
        rgb_digest.update(rgb.tobytes())
        floating = rgb.astype(np.float64)
        converted = np.clip(
            0.299 * floating[..., 0] + 0.587 * floating[..., 1] + 0.114 * floating[..., 2], 0, 255
        ).astype(np.uint8)
        if not np.array_equal(converted, luma[index]):
            raise ValueError(f"{path}: stored BT.601 luma does not match RGB window")
        identities.append(
            {
                "filename": path.name,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    return {
        "directory": str(window),
        "frames": len(files),
        "files": identities,
        "contiguous_rgb_sha256": rgb_digest.hexdigest(),
        "conversion": "0.299 R + 0.587 G + 0.114 B, float64 then uint8 truncation",
        "all_luma_pixels_identical": True,
    }


def mse_to_db(value: float) -> float | None:
    # JSON null represents +infinity explicitly, never a dropped frame.
    return 10 * math.log10(255**2 / value) if value else None


def score_frame(reference: np.ndarray, decoded: np.ndarray, mask: np.ndarray) -> dict:
    if reference.shape != decoded.shape or reference.shape != mask.shape:
        raise ValueError("shape mismatch")
    if not np.any(mask) or np.all(mask):
        raise ValueError("each frame needs nonempty foreground and background")
    delta = reference.astype(np.float64) - decoded.astype(np.float64)
    squared = delta * delta
    return {
        "whole": float(squared.mean()),
        "foreground": float(squared[mask].mean()),
        "background": float(squared[~mask].mean()),
        "foreground_pixels": int(mask.sum()),
        "frame_pixels": int(mask.size),
    }


def summarize_scores(rows: list[dict]) -> dict:
    answer = {"frames": len(rows), "per_frame": rows}
    for region in ["whole", "foreground", "background"]:
        mses = [r[region] for r in rows]
        dbs = [mse_to_db(x) for x in mses]
        finite_dbs = [x for x in dbs if x is not None]
        counts = [
            r["frame_pixels"]
            if region == "whole"
            else r["foreground_pixels"]
            if region == "foreground"
            else r["frame_pixels"] - r["foreground_pixels"]
            for r in rows
        ]
        # Do not quietly exclude identical or missing frames from the mean.
        answer[region] = {
            "mean_frame_psnr_db": float(np.mean(finite_dbs))
            if len(finite_dbs) == len(dbs)
            else None,
            "identical_frames": sum(x == 0 for x in mses),
            "pooled_pixel_mse": float(np.average(mses, weights=counts)),
            "pooled_pixel_psnr_db": mse_to_db(float(np.average(mses, weights=counts))),
            "selected_pixels": sum(counts),
        }
    return answer


def compare_curves(anchor: list[tuple[float, float]], candidate: list[tuple[float, float]]) -> dict:
    ax, ay = np.array(sorted(anchor)).T
    cx, cy = np.array(sorted(candidate)).T
    low, high = max(min(ax), min(cx)), min(max(ax), max(cx))
    result = {
        "overlap": [float(low), float(high)],
        "has_overlap": bool(high > low),
        "points_each": [len(ax), len(cx)],
        "method": "PCHIP log10(actual stream bytes) over common mean-frame Y-PSNR",
    }
    if high <= low:
        result["bd_rate_percent"] = None
        return result
    if np.any(np.diff(ax) <= 0) or np.any(np.diff(cx) <= 0):
        raise ValueError("nonunique quality points")
    if np.any(np.diff(ay) <= 0) or np.any(np.diff(cy) <= 0):
        result["bd_rate_percent"] = None
        result["ineligible_reason"] = (
            "nonmonotone rate versus quality; do not integrate a reversed segment"
        )
        return result
    delta = (
        PchipInterpolator(cx, np.log10(cy)).integrate(low, high)
        - PchipInterpolator(ax, np.log10(ay)).integrate(low, high)
    ) / (high - low)
    result["bd_rate_percent"] = float(100 * (10**delta - 1))
    return result


def select_reported_streams(
    base: Path, codec: str, rates: list[float]
) -> tuple[list[Path], list[dict]]:
    """Resolve only the immutable reported curve; preserve unrelated old files.

    The exact reported byte counts identify the retained points before any new
    quality is observed. Ambiguous identities fail instead of choosing a file.
    """
    suffix = ".ivf" if codec == "av1" else ".vvc"
    found = list(base.glob(codec + "_qp*" + suffix))
    selected = []
    for rate in rates:
        matches = [p for p in found if p.stat().st_size == rate]
        if len(matches) != 1 or rate <= 0:
            raise ValueError(f"{base}: expected exactly one native stream for saved {rate} bytes")
        selected.append(matches[0])
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError("need three distinct reported points")
    excluded = [
        {
            "path": str(p),
            "bytes": p.stat().st_size,
            "reason": "not in immutable reported three-point curve",
        }
        for p in sorted(found)
        if p not in selected
    ]
    return selected, excluded


def group_sensitivity(report: dict, draws: int = 10000, seed: int = 20260930) -> dict:
    result: dict[str, Any] = {
        "draws": draws,
        "seed": seed,
        "units": "percent saved; fixed saved per-scene BP21 estimates",
        "interpretation": "Source-cluster resampling sensitivity, not a population confidence guarantee or new codec measurement.",
        "codecs": {},
    }
    rng = np.random.default_rng(seed)
    for codec, records in report["fg"].items():
        values = {name: 100 * row["plate_vs_original"]["saving"] for name, row in records.items()}
        groups = sorted({name.split("/")[0] for name in values})
        index = rng.integers(0, len(groups), size=(draws, len(groups)))
        sizes = np.array([sum(name.startswith(group + "/") for name in values) for group in groups])
        sums = np.array(
            [
                sum(v for name, v in values.items() if name.startswith(group + "/"))
                for group in groups
            ]
        )
        means = sums / sizes
        weighted = sums[index].sum(axis=1) / sizes[index].sum(axis=1)
        equal = means[index].mean(axis=1)
        result["codecs"][codec] = {
            "scenes": len(values),
            "source_groups": len(groups),
            "scene_mean_percent": float(np.mean(list(values.values()))),
            "equal_source_mean_percent": float(means.mean()),
            "scene_estimator_cluster_percentile_95": np.percentile(weighted, [2.5, 97.5]).tolist(),
            "equal_source_percentile_95": np.percentile(equal, [2.5, 97.5]).tolist(),
            "leave_one_source_out_scene_mean_range": [
                float(min((sums.sum() - s) / (sizes.sum() - n) for s, n in zip(sums, sizes))),
                float(max((sums.sum() - s) / (sizes.sum() - n) for s, n in zip(sums, sizes))),
            ],
            "source_means_percent": dict(zip(groups, means.tolist())),
        }
    return result


def health_guard(max_load: float) -> None:
    if os.getloadavg()[0] > max_load:
        raise RuntimeError("pause: host CPU load exceeded predeclared limit")
    response = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    if response.stdout.strip():
        raise RuntimeError("pause: another GPU workload appeared on this CPU study's host")


def decode_score(
    ffmpeg: str,
    stream: Path,
    targets: dict[str, np.ndarray],
    masks: np.ndarray,
    threads: int,
    log: Path,
) -> dict:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-hwaccel",
        "none",
        "-threads",
        str(threads),
        "-i",
        str(stream),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
        "-threads",
        str(threads),
        "-pix_fmt",
        "yuv420p",
        "-f",
        "yuv4mpegpipe",
        "pipe:1",
    ]
    scores: dict[str, list[dict[str, Any]]] = {key: [] for key in targets}
    digest = hashlib.sha256()
    started = time.monotonic()
    with log.open("wb") as errors:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        assert child.stdout is not None
        try:
            for index, frame in enumerate(y4m_frames(child.stdout)):
                if index >= len(masks):
                    raise ValueError("decoder returned excess frames")
                digest.update(frame)
                for name, reference in targets.items():
                    scores[name].append(score_frame(reference[index], frame, masks[index]))
            if child.wait(timeout=30):
                raise RuntimeError(f"native decode failed: {log}")
        except BaseException:
            child.kill()
            child.wait()
            raise
        finally:
            child.stdout.close()
    if any(len(rows) != len(masks) for rows in scores.values()):
        raise ValueError("decoder returned missing frames")
    return {
        "command": command,
        "fresh_decode_luma_sha256": digest.hexdigest(),
        "seconds_not_a_speed_benchmark": time.monotonic() - started,
        "scores": {name: summarize_scores(rows) for name, rows in scores.items()},
    }


def run(args) -> dict:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    budget = ReadBudget(args.read_mib_per_second)
    root = args.data_root.resolve() / "outputs/bp21-headroom"
    source_record = sha_file(root / "report.json", budget)
    if source_record["sha256"] != EXPECTED_REPORT:
        raise ValueError("BP21 report identity changed")
    report = json.loads((root / "report.json").read_text())
    result = {
        "schema": "pointstream.cpu_headroom_replay.v1",
        "smoke": args.smoke,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "historical_report": source_record,
        "code_revision": args.code_revision,
        "worker": sha_file(Path(__file__)),
        "environment": {
            "host": platform.node(),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "affinity": sorted(os.sched_getaffinity(0)),
            "cpu_thread_cap": 8,
            "ffmpeg": sha_file(Path(args.ffmpeg)),
            "ffmpeg_version": subprocess.check_output(
                [args.ffmpeg, "-version"], text=True
            ).splitlines(),
            "gpus_observed_not_used": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True
            ),
        },
        "policy": {
            "codecs": ["av1", "vvc"],
            "targets": ["original", "plate"],
            "reencode": False,
            "metric": "8-bit stored BT.601 luma; equal-frame mean Y-PSNR; foreground/background use stored BP21 masks",
            "physical_read_limit_mib_per_second": args.read_mib_per_second,
            "max_host_load": args.max_load,
            "pause_if_gpu_work_appears": True,
            "limitations": [
                "No candidate package decoded; no Gate A recertification",
                "No new subjective/task truth",
                "Changed-target diagnostic, not reconstructive codec",
                "Source-video clustering is not verified match independence",
            ],
        },
        "points": [],
        "clips": [],
        "group_sensitivity": group_sensitivity(report),
        "controls": {
            "identical": score_frame(
                np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8), np.eye(4, dtype=bool)
            ),
            "offset_8": score_frame(
                np.zeros((4, 4), np.uint8), np.full((4, 4), 8, np.uint8), np.eye(4, dtype=bool)
            ),
        },
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA devices must be explicitly hidden")
    clips = report["clips"][:1] if args.smoke else report["clips"]
    for clip in clips:
        health_guard(args.max_load)
        name = clip["video"] + "/" + clip["scene"]
        mask_path = root / "clips" / name / "masks.npz"
        mask_identity = sha_file(mask_path, budget)
        with np.load(mask_path, allow_pickle=False) as archive:
            masks = np.array(archive["masks"], dtype=bool)
        expected_shape = (48, 2160, 3840)
        if masks.shape != expected_shape:
            raise ValueError(f"mask shape {masks.shape}")
        if abs(float(masks.mean()) - clip["player_area"]) > 1e-12:
            raise ValueError("historical player area mismatch")
        clip_record = {
            "name": name,
            "mask": mask_identity,
            "shape": list(masks.shape),
            "player_fraction": float(masks.mean()),
            "sources": {},
            "comparisons": {},
        }
        for codec in ["av1"] if args.smoke else ["av1", "vvc"]:
            base = root / "encode" / name / codec
            original, original_id = load_source(base / "original/source.y4m", 48, budget)
            plate, plate_id = load_source(base / "plate/source.y4m", 48, budget)
            if codec == "av1":
                clip_record["rgb_window"] = verify_rgb_window(
                    root / "clips" / name / "window", original, budget
                )
            if original.shape != masks.shape or plate.shape != masks.shape:
                raise ValueError("source geometry mismatch")
            # Removal must leave every unmasked source luma pixel unchanged.
            outside_changed = sum(
                int(np.count_nonzero((a != b) & (~m))) for a, b, m in zip(original, plate, masks)
            )
            if outside_changed:
                raise ValueError("plate changed unmasked source pixels")
            clip_record["sources"][codec] = {
                "original": original_id,
                "plate": plate_id,
                "outside_mask_changed_pixels": outside_changed,
            }
            saved = report["fg"][codec][name]
            rows = []
            for target in ["original", "plate"]:
                expected = saved["original_curve" if target == "original" else "plate_curve"]
                streams, excluded = select_reported_streams(base / target, codec, expected["rates"])
                clip_record.setdefault("excluded_old_streams", []).extend(excluded)
                for index, stream in enumerate(streams[:1] if args.smoke else streams):
                    health_guard(args.max_load)
                    identity = sha_file(stream, budget)
                    point = {
                        "clip": name,
                        "codec": codec,
                        "target": target,
                        "qp": int(stream.stem.split("qp")[-1]),
                        "stream": identity,
                    }
                    point.update(
                        decode_score(
                            args.ffmpeg,
                            stream,
                            {"original": original, "plate": plate},
                            masks,
                            args.decoder_threads,
                            out
                            / f"{clip['video']}-{clip['scene']}-{codec}-{target}-{point['qp']}.log",
                        )
                    )
                    point["historical_own_target_psnr_db"] = expected["qualities"][index]
                    point["historical_mean_psnr_delta_db"] = (
                        point["scores"][target]["whole"]["mean_frame_psnr_db"]
                        - expected["qualities"][index]
                    )
                    rows.append(point)
                    result["points"].append(point)
                    write_json(out / "partial-report.json", result)
                    progress = os.environ.get("PS_JOB_DIR")
                    if progress:
                        write_json(
                            Path(progress) / "progress.json",
                            {
                                "stage": "native CPU decode/rescore",
                                "completed": len(result["points"]),
                                "updated": time.time(),
                            },
                        )
                    print(
                        f"completed {len(result['points'])}: {name}/{codec}/{target}/QP{point['qp']}",
                        flush=True,
                    )
            if not args.smoke:
                anchor = [
                    (x["scores"]["original"]["whole"]["mean_frame_psnr_db"], x["stream"]["bytes"])
                    for x in rows
                    if x["target"] == "original"
                ]
                own = [
                    (x["scores"]["plate"]["whole"]["mean_frame_psnr_db"], x["stream"]["bytes"])
                    for x in rows
                    if x["target"] == "plate"
                ]
                original_target = [
                    (x["scores"]["original"]["whole"]["mean_frame_psnr_db"], x["stream"]["bytes"])
                    for x in rows
                    if x["target"] == "plate"
                ]
                clip_record["comparisons"][codec] = {
                    "own_target_changed_diagnostic": compare_curves(anchor, own),
                    "original_target_unrestored_plate": compare_curves(anchor, original_target),
                }
            del original, plate
        result["clips"].append(clip_record)
        del masks
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["complete"] = True
    result["physical_file_bytes_read"] = budget.bytes
    write_json(out / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="/opt/local/bin/ffmpeg")
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--decoder-threads", type=int, default=4)
    parser.add_argument("--read-mib-per-second", type=float, default=20)
    parser.add_argument("--max-load", type=float, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.decoder_threads <= 4:
        parser.error("decoder threads must be between 1 and 4")
    if not 0 < args.read_mib_per_second <= 20:
        parser.error("file-read budget must be positive and at most 20 MiB/s")
    if not 0 < args.max_load <= 16:
        parser.error("host-load limit must be positive and at most 16")
    try:
        result = run(args)
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "points": len(result["points"]),
                    "out": str(args.out),
                }
            )
        )
        return 0
    except Exception as exc:
        if args.out.is_dir():
            write_json(args.out / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
