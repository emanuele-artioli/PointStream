"""Bounded CPU measurements of actual side-information packets on saved inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis.k6b_rate_budget import DIAGNOSTIC_SHA
from experiments.tennis.k6b_rate_streams import (
    HEADER,
    decode_keypoints,
    decode_masks,
    decode_reference,
    encode_keypoints,
    encode_masks,
    encode_reference,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha(array: np.ndarray) -> str:
    return sha(np.ascontiguousarray(array).tobytes())


def write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")


def native_equivalence(decoded: list[dict[str, Any]], paths: dict[str, Path]) -> dict[str, Any]:
    from experiments.tennis.k6b import pose_map, run_boxes, runs, to_crop

    checks: dict[str, Any] = {}
    for unit in decoded:
        with np.load(paths[unit["name"]]) as source:
            times = source["t"]
            boxes = source["box"]
            maps = source[
                "pose"
            ]  # Read once: per-frame npz indexing would repeatedly decompress it.
            run_ids = source["run"]
            if not np.array_equal(times, unit["t"]):
                raise ValueError("Transport changes coded frame/timing coverage")
            cursor = 0
            for run_id, ts in enumerate(runs(times.tolist())):
                kps = list(unit["kp"][cursor : cursor + len(ts)])
                computed = run_boxes(kps)
                if not np.array_equal(boxes[cursor : cursor + len(ts)], computed):
                    raise ValueError("Transport changes target crop geometry")
                for kp, box in zip(kps, computed):
                    if run_ids[cursor] != run_id or not np.array_equal(
                        pose_map(to_crop(kp, box)), maps[cursor]
                    ):
                        raise ValueError("Transport changes pose-map pixels or run identity")
                    cursor += 1
            checks[unit["name"]] = {
                "frames": len(times),
                "pose_maps_and_boxes_exact": True,
                "source_npz_sha256": sha(paths[unit["name"]].read_bytes()),
            }
    return checks


def measure_window(
    row: dict[str, Any],
    anchor: Path,
    prepared: dict[str, Path],
    prepared_meta: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    from experiments.tennis import k4
    from experiments.tennis.k6 import frame_regions
    from experiments.tennis.k6b import (
        GEN_H,
        GEN_W,
        CROP_MARGIN,
        SIZE_PERCENTILE,
        CENTRE_SMOOTH,
        unit_name,
    )
    from src.segmentation.masks import ClipMasks, decompress

    if k4.KP_CONF != 0.5:
        raise ValueError("Visibility coding contract differs from native renderer")
    destination.mkdir(parents=True)
    native = json.loads((anchor / "window.json").read_text())
    if native["id"] != row["window"] or native["frames"] != 240:
        raise ValueError("Source window identity/span changed")
    for name, digest in native["files"].items():
        if sha((anchor / name).read_bytes()) != digest:
            raise ValueError("Source anchor member changed")
    kp = json.loads(decompress((anchor / "keypoints.json.zst").read_bytes()))["decoded"][
        "q4-velocity-e4"
    ]
    players = native["players"]
    units = []
    references: dict[str, Any] = {
        name + "_bytes": 0 for name in ["png", "jpeg70", "jpeg85", "jpeg95"]
    }
    reference_files = []
    for player in players:
        name = unit_name(row["window"], player)
        with np.load(prepared[name]) as a:
            times = a["t"]
            units.append(
                {
                    "name": name,
                    "t": times,
                    "kp": np.asarray([kp[f"{player}@{int(t)}"] for t in times], np.float64),
                }
            )
            image = a["reference"]
            metadata = {
                "unit": name,
                "reference_frame": int(prepared_meta[name]["reference_t"]),
                "reference_box": a["reference_box"].tolist(),
                "pixels": "full_prepared_conditioning_crop",
            }
            for quality in (None, 70, 85, 95):
                label = "png" if quality is None else f"jpeg{quality}"
                blob = encode_reference(image, metadata, quality)
                _, restored = decode_reference(blob)
                exact = bool(np.array_equal(image, restored))
                if quality is None and not exact:
                    raise ValueError("Lossless reference pixels changed")
                path = destination / f"{name}.{label}.ref"
                path.write_bytes(blob)
                references[label + "_bytes"] += len(blob)
                mse = float(np.mean((image.astype(float) - restored.astype(float)) ** 2))
                reference_files.append(
                    {
                        "file": path.name,
                        "codec": label,
                        "bytes": len(blob),
                        "sha256": sha(blob),
                        "decoded_sha256": array_sha(restored),
                        "source_sha256": array_sha(image),
                        "exact": exact,
                        "psnr_diagnostic_only": 10 * np.log10(255**2 / mse) if mse else None,
                    }
                )
    metadata = {
        "window": row["window"],
        "native_frames": 240,
        "fps": [int(native["fps"]), 1],
        "start_tick": 0,
        "time_base": [1, int(native["fps"])],
        "clip_local_source_start": native["start"],
        "registered_frame_bitmap": np.packbits(np.isin(np.arange(240), native["scored"]))
        .tobytes()
        .hex(),
        "compositing_order": sorted(players, key=lambda p: native["median_height"][p]),
        "model_profile": "C-q4-guidance2-DDIM30",
        "common_base": "tennis-base@fit-100",
        "generation_size": [GEN_W, GEN_H],
        "crop_rule": [CROP_MARGIN, SIZE_PERCENTILE, CENTRE_SMOOTH],
        "context": [24, 4],
        "seed_rule": "sha256(unit_name)[:7] then seed*1009+run",
    }
    blob = encode_keypoints(metadata, units)
    _, decoded = decode_keypoints(blob)
    for a, b in zip(units, decoded):
        if not np.array_equal(a["kp"][:, :, :2], b["kp"][:, :, :2]) or not np.array_equal(
            a["kp"][:, :, 2] >= 0.5, b["kp"][:, :, 2] >= 0.5
        ):
            raise ValueError("Keypoint coordinates or every-frame visibility changed")
    equivalence = native_equivalence(decoded, prepared)
    path = destination / "keypoints.kps"
    path.write_bytes(blob)
    keypoints = {
        "bytes": len(blob),
        "framing_bytes": HEADER.size,
        "compressed_metadata_and_symbols_bytes": len(blob) - HEADER.size,
        "file": path.name,
        "sha256": sha(blob),
        "equivalent": True,
        "contract": metadata,
        "native_equivalence": equivalence,
    }
    features = json.loads((anchor / "features.json").read_text())
    clip = ClipMasks.load(anchor / "sam-masks.rle")
    frames = []
    # Exact native-grid reduction is an equality check, not a lossy mask assumption.
    for t in range(240):
        regions = frame_regions(clip.frames[t], features["aliases"], players)["regions"]
        pair = []
        for player in players:
            full = regions.get(player, np.zeros((1080, 1920), bool))
            reduced = full[::2, ::2]
            if not np.array_equal(np.repeat(np.repeat(reduced, 2, 0), 2, 1), full):
                raise ValueError("Native 960x540 mask grid is not exactly reconstructible")
            pair.append(reduced.copy())
        frames.append(pair)
    masks = np.asarray(frames, bool)
    del frames
    mattes = {}
    for label, factor in [("exact", 1), ("half_grid", 2), ("quarter_grid", 4)]:
        small = masks[:, :, ::factor, ::factor].copy()
        candidates = []
        for mode in ("rle", "packed", "xor"):
            blob = encode_masks(small, mode, factor)
            _, restored = decode_masks(blob)
            if not np.array_equal(small, restored):
                raise ValueError("Matte transport round trip changed its coded pixels")
            path = destination / f"{label}-{mode}.matte"
            path.write_bytes(blob)
            candidates.append(
                {
                    "file": path.name,
                    "mode": mode,
                    "bytes": len(blob),
                    "sha256": sha(blob),
                    "decoded_sha256": array_sha(restored),
                }
            )
            del restored
        best = min(candidates, key=lambda c: c["bytes"])
        # Pixel overlap is only a transport distortion diagnostic, never generated quality.
        expanded = np.repeat(np.repeat(small, factor, 2), factor, 3)
        union = int(np.count_nonzero(masks | expanded))
        iou = int(np.count_nonzero(masks & expanded)) / union if union else 1.0
        mattes[label] = {
            **best,
            "candidates": candidates,
            "factor_beyond_native_grid": factor,
            "all_240_frames_included": True,
            "native_mask_iou_diagnostic": iou,
            "exact_target_matte": factor == 1,
        }
        del expanded, small
    return {
        "window": row["window"],
        "match": row["match"],
        "references": references,
        "reference_files": reference_files,
        "keypoints": keypoints,
        "mattes": mattes,
        "source_anchor_window_sha256": sha((anchor / "window.json").read_bytes()),
        "source_native_grid_sha256": array_sha(masks),
    }


def validate(stage: Path, expected: int | None = None) -> dict[str, Any]:
    result = json.loads((stage / "transport.json").read_text())
    checks = {
        "cpu_only": result["cuda_visible"] == "",
        "unique_windows": len({r["window"] for r in result["rows"]}) == len(result["rows"]),
        "row_count": len(result["rows"]) == (expected or result["expected_windows"]),
    }
    root = stage / "packets"
    from experiments.tennis.k4 import safe

    for row in result["rows"]:
        directory = root / safe(row["window"])
        k = row["keypoints"]
        blob = (directory / k["file"]).read_bytes()
        _, decoded = decode_keypoints(blob)
        checks[row["window"] + ":keypoints"] = (
            sha(blob) == k["sha256"]
            and len(blob) == k["bytes"]
            and len(decoded) == 2
            and k["equivalent"]
        )
        for f in row["reference_files"]:
            blob = (directory / f["file"]).read_bytes()
            _, image = decode_reference(blob)
            checks[f["file"]] = (
                sha(blob) == f["sha256"]
                and array_sha(image) == f["decoded_sha256"]
                and (f["codec"] != "png" or array_sha(image) == f["source_sha256"])
            )
        for v in row["mattes"].values():
            for c in v["candidates"]:
                blob = (directory / c["file"]).read_bytes()
                _, masks = decode_masks(blob)
                checks[row["window"] + ":" + c["file"]] = (
                    sha(blob) == c["sha256"]
                    and array_sha(masks) == c["decoded_sha256"]
                    and len(blob) == c["bytes"]
                )
                del masks
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "gpu_hours": 0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", type=Path)
    p.add_argument("--anchor", type=Path, action="append", default=[])
    p.add_argument("--prepared", type=Path, action="append", default=[])
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        out = validate(stage)
        write(Path(os.environ["PS_VALIDATION_PATH"]), out)
        print(
            json.dumps(
                {
                    "passed": out["passed"],
                    "checks": len(out["checks"]),
                    "cpu_seconds": out["cpu_seconds"],
                }
            )
        )
        return int(not out["passed"])
    started = time.monotonic()
    import cv2

    cv2.setNumThreads(4)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or a.diagnostic is None:
        raise ValueError("CPU-only visibility and diagnostic required")
    if sha(a.diagnostic.read_bytes()) != DIAGNOSTIC_SHA:
        raise ValueError("Frozen diagnostic changed")
    from experiments.tennis.k4 import safe

    scratch = Path(os.environ["PS_SCRATCH_DIR"]) / "rate-audit"
    scratch.mkdir(parents=True)
    anchors = {}
    units = {}
    prepared_meta = {}
    receipts = []
    for kind, archives in [("anchor", a.anchor), ("prepared", a.prepared)]:
        for i, path in enumerate(archives):
            root = scratch / f"{kind}{i}"
            with tarfile.open(path) as tar:
                tar.extractall(root, filter="data")
            receipts.append({"kind": kind, "path": str(path), "sha256": sha(path.read_bytes())})
            if kind == "anchor":
                for q in (root / "publish").glob("*/window.json"):
                    wid = json.loads(q.read_text())["id"]
                    if wid in anchors:
                        raise ValueError("Duplicate anchor window")
                    anchors[wid] = q.parent
            else:
                manifest = json.loads((root / "publish/prepare.json").read_text())
                for window in manifest["windows"]:
                    for unit in window["units"]:
                        prepared_meta[unit["unit"]] = unit
                for q in (root / "publish/units").glob("*.npz"):
                    if q.stem in units:
                        raise ValueError("Duplicate prepared unit")
                    units[q.stem] = q
    diagnostic = json.loads(a.diagnostic.read_text())
    rows = diagnostic["rows"]
    if a.max_windows:
        smoke = ["alcaraz_hurkacz/031@129", "medvedev_struff/014@194", "alcaraz_hurkacz/016@273"][
            : a.max_windows
        ]
        rows = [r for wid in smoke for r in rows if r["window"] == wid]
    result: dict[str, Any] = {
        "rows": [],
        "expected_windows": len(rows),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "provenance": {
            "diagnostic_sha256": DIAGNOSTIC_SHA,
            "archives": receipts,
            "command": sys.argv,
            "python": sys.version,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "codec_build": [
                s.strip()
                for s in cv2.getBuildInformation().splitlines()
                if any(w in s for w in ("JPEG:", "PNG:", "ZLib:"))
            ],
        },
        "quality_measurements": False,
        "gpu_hours": 0,
    }
    (stage / "packets").mkdir()
    for row in rows:
        item = measure_window(
            row,
            anchors[row["window"]],
            units,
            prepared_meta,
            stage / "packets" / safe(row["window"]),
        )
        result["rows"].append(item)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["cpu_seconds"] = usage.ru_utime + usage.ru_stime
        result["wall_seconds"] = time.monotonic() - started
        write(stage / "transport.json", result)
        print(
            json.dumps(
                {
                    "window": row["window"],
                    "keypoint_bytes": item["keypoints"]["bytes"],
                    "png_bytes": item["references"]["png_bytes"],
                    "matte_bytes": item["mattes"]["exact"]["bytes"],
                    "completed": len(result["rows"]),
                    "cpu_seconds": result["cpu_seconds"],
                }
            ),
            flush=True,
        )
    proof = validate(stage)
    write(stage / "scientific-validation.json", proof)
    if not proof["passed"]:
        return 1
    with tarfile.open(stage / "published.tar", "w") as tar:
        for name in ("packets", "transport.json", "scientific-validation.json"):
            tar.add(stage / name, arcname=name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
