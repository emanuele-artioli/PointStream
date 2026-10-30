"""Frozen pose-prompted SAM tracker on archived unmasked generated RGB."""

from __future__ import annotations

import argparse
import ctypes
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis.k6b_rate_transport import array_sha, sha, write


def restrict_reads(paths: list[Path]) -> int:
    """Linux Landlock read allowlist, inherited by all later worker threads."""
    lib = ctypes.CDLL(None, use_errno=True)
    abi = int(lib.syscall(444, 0, 0, 1))
    if abi < 1:
        raise RuntimeError("Landlock unavailable; no unisolated predictor fallback")
    handled = ctypes.c_uint64(4 | 8)  # READ_FILE | READ_DIR
    fd = int(lib.syscall(444, ctypes.byref(handled), ctypes.sizeof(handled), 0))
    if fd < 0 or lib.prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "Cannot create read sandbox")

    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    try:
        for path in paths:
            if not path.exists():
                continue
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Rule(4 | (8 if path.is_dir() else 0), parent)
                if lib.syscall(445, fd, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), f"Cannot allow {path}")
            finally:
                os.close(parent)
        if lib.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), "Cannot restrict reads")
    finally:
        os.close(fd)
    return abi


def select_mask(detections: list[Any], kp: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    visible = kp[kp[:, 2] >= 0.5, :2].astype(int)
    h, w = shape
    visible = visible[
        (visible[:, 0] >= 0) & (visible[:, 0] < w) & (visible[:, 1] >= 0) & (visible[:, 1] < h)
    ]
    if not detections:
        return np.zeros(shape, bool)

    def rank(d: Any) -> tuple[int, int, float]:
        mask = d[1]
        return int(mask[visible[:, 1], visible[:, 0]].sum()), int(mask.sum()), float(d[2])

    return np.asarray(max(detections, key=rank)[1], bool)


def project(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    import cv2
    from experiments.tennis.k6b import int_box

    x, y, w, h = int_box(box)
    out = np.zeros((1080, 1920), bool)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(1920, x + w), min(1080, y + h)
    if x1 > x0 and y1 > y0:
        large = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        out[y0:y1, x0:x1] = large[y0 - y : y1 - y, x0 - x : x1 - x]
    return out


def transport_root(extracted: Path) -> Path:
    root = extracted / "publish" if (extracted / "publish").is_dir() else extracted
    if not (root / "packets").is_dir() or not (root / "transport.json").is_file():
        raise ValueError("Native transport layout is incomplete")
    return root


def pose_prompt(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Box corner tokens plus confident torso positives, only from decoded KPS."""
    valid = pose[(pose[:, 2] >= 0.5) & np.isfinite(pose).all(1)]
    valid = valid[(valid[:, 0] >= 0) & (valid[:, 0] < 512)
                  & (valid[:, 1] >= 0) & (valid[:, 1] < 784)]
    if len(valid) < 2:
        raise ValueError("Insufficient paid pose for deterministic SAM initialization")
    lo, hi = valid[:, :2].min(0), valid[:, :2].max(0)
    margin = np.maximum((hi - lo) * 0.1, 4)
    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, [511, 783])
    torso = pose[[5, 6, 11, 12]]
    torso = torso[(torso[:, 2] >= 0.5) & (torso[:, 0] >= 0) & (torso[:, 0] < 512)
                  & (torso[:, 1] >= 0) & (torso[:, 1] < 784)]
    positive = torso[:, :2] if len(torso) else valid[:1, :2]
    points = np.vstack([lo, hi, positive]).astype(np.float32) / [512, 784]
    return points.astype(np.float32), np.array([2, 3, *([1] * len(positive))], np.int32)


def worker(a: argparse.Namespace) -> None:
    own = a.out.resolve()
    own.mkdir()
    (own / "tmp").mkdir()
    (own / "config").mkdir()
    allowed = [Path(p) for p in ("/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/opt")]
    # /dev/shm contains fleet caches, including forbidden source/mask packets.
    # Allow concrete GPU/runtime devices, never the /dev directory recursively.
    allowed += [
        Path(p) for p in ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random", "/dev/dri")
    ]
    allowed += list(Path("/dev").glob("nvidia*"))
    allowed += [
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        Path(__file__).resolve().parents[2],
        own,
        a.kps.resolve(),
        a.weights.resolve(),
        a.text.resolve(),
        *[p.resolve() for p in a.generated],
    ]
    abi = restrict_reads(allowed)
    denied = {}
    for path in a.forbidden:
        try:
            with path.open("rb") as f:
                f.read(1)
        except PermissionError:
            denied[str(path)] = True
        else:
            raise RuntimeError(f"Predictor can read forbidden input {path}")
    if not denied:
        raise ValueError("Required denied-read probes missing")
    import torch
    from PIL import Image
    from src.segmentation.sam31_tracker import Sam31MaskTracker, runtime
    from experiments.tennis.k6b import load_generated, to_crop
    from experiments.tennis.k6b_matte_cpu import box_maps
    from experiments.tennis.k6b_rate_streams import decode_keypoints

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, 12000 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    )
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tracker = Sam31MaskTracker(a.weights)
    model = tracker.model
    generated = load_generated([str(p) for p in a.generated], own / "rgb")
    all_units = {}
    for path in sorted(a.kps.glob("*.kps")):
        meta, units = decode_keypoints(path.read_bytes())
        for unit, boxes in zip(units, box_maps(units)):
            name = unit["name"]
            all_units[name] = (unit, boxes)
    rows: list[dict[str, Any]] = []
    kernels: list[str] = []
    began = time.monotonic()
    for (name, profile), g in sorted(generated.items()):
        unit, boxes = all_units[name]
        kp = dict(zip(unit["t"].tolist(), unit["kp"]))
        ticks = g["t"][: a.max_judged or None].tolist()
        masks, inputs = [], []
        loop_started = time.monotonic()
        prompts_record = []
        step = 1 if a.independent_frames else 20
        for start in range(0, len(ticks), step):
            segment = ticks[start:start + step]
            images = torch.stack([
                torch.from_numpy(np.array(Image.fromarray(g["crops"][start+j]).resize(
                    (int(model.image_size), int(model.image_size)))).transpose(2, 0, 1).copy())
                .to(dtype=torch.float16).div_(255).sub_(0.5).div_(0.5)
                for j in range(len(segment))
            ])
            seed_pose = to_crop(kp[segment[0]], boxes[segment[0]])
            points, labels = pose_prompt(seed_pose)
            prompts_record.append({"t": segment[0], "points": points.tolist(), "labels": labels.tolist()})
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) if not rows and start == 0 else None
            with prof if prof is not None else nullcontext():
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = model.init_state(video_height=784, video_width=512,
                                             num_frames=len(segment), offload_video_to_cpu=True)
                    state["images"] = images
                    prompted = model.add_new_points(inference_state=state, frame_idx=0, obj_id=1,
                                         points=torch.from_numpy(points), labels=torch.from_numpy(labels),
                                         clear_old_points=True, rel_coordinates=True)
                    outputs = {}
                    if a.independent_frames:
                        frame, ids, low, video = prompted
                        if int(frame) != 0 or list(ids) != [1] or len(segment) != 1:
                            raise ValueError("Independent prompt output identity/frame changed")
                        outputs[0] = (video[0, 0] > 0).cpu().numpy()
                    else:
                        model.propagate_in_video_preflight(state, run_mem_encoder=True)
                        for frame, ids, low, video, *rest in model.propagate_in_video(
                                state, start_frame_idx=0, max_frame_num_to_track=len(segment)-1,
                                reverse=False, tqdm_disable=True):
                            if list(ids) != [1]:
                                raise ValueError("SAM tracker changed object identity")
                            outputs[int(frame)] = (video[0, 0] > 0).cpu().numpy()
            if prof is not None:
                kernels = sorted({event.key for event in prof.key_averages()
                                  if "flash" in event.key.lower()})
                if not kernels:
                    raise RuntimeError("SAM first-class smoke did not execute flash attention")
            if set(outputs) != set(range(len(segment))):
                raise ValueError("Missing sampled tracker output")
            for j, t in enumerate(segment):
                rgb = g["crops"][start+j]
                pose = to_crop(kp[t], boxes[t])
                masks.append(outputs[j])
                inputs.append({"t": t, "rgb": array_sha(rgb), "pose": array_sha(pose),
                               "box": array_sha(boxes[t])})
            del state, images, outputs
        # Recomputable initialization metadata is part of the installed profile.
        masks_array = np.stack(masks)
        path = own / (name + ".npz")
        np.savez_compressed(path, t=np.array(ticks), masks=masks_array)
        rows.append(
            {
                "name": name,
                "profile": profile,
                "ticks": ticks,
                "input_sha256": sha(json.dumps(inputs, sort_keys=True).encode()),
                "output_sha256": array_sha(masks_array),
                "file": path.name,
                "inference_seconds": time.monotonic() - loop_started,
                "frames": [array_sha(m) for m in masks_array],
                "prompts": prompts_record,
                "area": [int(m.sum()) for m in masks_array],
            }
        )
    if not next(model.parameters()).is_cuda:
        raise ValueError("Predictor model is not on CUDA")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    write(
        own / "prediction.json",
        {
            "rows": rows,
            "landlock_abi": abi,
            "denied_reads": denied,
            "on_cuda": True,
            "sam_load": tracker.load_report,
            "independent_frames": bool(a.independent_frames),
            "flash_events": kernels,
            "sam_runtime": runtime(),
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "seconds": time.monotonic() - began,
            "cpu_seconds": usage.ru_utime + usage.ru_stime,
            "weights_sha256": sha(a.weights.read_bytes()),
            "text_sha256": sha(a.text.read_bytes()),
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "code_sha256": sha(Path(__file__).read_bytes()),
        },
    )


def predict_twice(a: argparse.Namespace, scratch: Path, kps: Path) -> tuple[dict[str, Any], Path]:
    records = []
    began = time.monotonic()
    for n in range(2):
        out = scratch / f"isolated{n}"
        env = {
            **os.environ,
            "HOME": str(out),
            "TMPDIR": str(scratch / f"tmp{n}"),
            "YOLO_CONFIG_DIR": str(out / "config"),
            "PYTHONNOUSERSITE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        }
        # Temporary/config paths belong to the worker's own allowlisted tree.
        env["TMPDIR"] = str(out / "tmp")
        command = [
            sys.executable,
            "-m",
            "experiments.tennis.k6b_receiver_sam",
            "--worker",
            "--out",
            str(out),
            "--kps",
            str(kps),
            "--weights",
            str(a.weights),
            "--text",
            str(a.text),
            "--max-judged",
            str(a.max_judged),
        ]
        if a.independent_frames:
            command += ["--independent-frames"]
        for p in a.generated:
            command += ["--generated", str(p)]
        for p in [*a.source, *a.control, a.transport]:
            command += ["--forbidden", str(p)]
        with (scratch / f"worker{n}.log").open("w") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        log_path = scratch / f"worker{n}.log"
        (scratch / "publish" / log_path.name).write_bytes(log_path.read_bytes())
        if completed.returncode:
            print(log_path.read_text()[-6000:], flush=True)
            raise RuntimeError(f"Isolated predictor exited {completed.returncode}; log preserved")
        records.append(json.loads((out / "prediction.json").read_text()))
    for left, right in zip(records[0]["rows"], records[1]["rows"]):
        if {k: v for k, v in left.items() if k != "inference_seconds"} != {
            k: v for k, v in right.items() if k != "inference_seconds"
        }:
            raise ValueError("Fresh reset receiver prediction/input hashes disagree")
    if len(records[0]["rows"]) != len(records[1]["rows"]) or any(
        records[0][k] != records[1][k] for k in ("weights_sha256", "text_sha256")
    ):
        raise ValueError("Fresh reset model/unit identity differs")
    return {
        "fresh_reset_equal": True,
        "runs": records,
        "seconds": time.monotonic() - began,
    }, scratch / "isolated0"


def assessment(
    diagnostic: dict[str, Any],
    reference: dict[str, Any],
    rows: list[dict[str, Any]],
    installed_bytes: int,
) -> dict[str, Any]:
    from experiments.tennis.k6b_rate_budget import comparison

    originals = {r["window"]: r for r in diagnostic["rows"]}
    refs = {r["window"]: r for r in reference["settings"]["full"]["rows"]}
    cells = []
    for r in rows:
        d, ref = originals[r["id"]], refs[r["id"]]
        rate = ref["kbps"]
        actual = {**d, "method": r["means"]["-1"]}
        cells.append(
            {
                "window": r["id"],
                "quality": r["means"]["-1"],
                "kbps": rate,
                "rate_components": ref["rate_components"],
                "dynamic_matte_bits": 0,
                "prompt_init_bits": 0,
                "primary": {m: comparison(actual, m, rate) for m in ("lpips", "dists")},
                "base_and_segmenter_charged": {
                    m: comparison(
                        actual,
                        m,
                        rate + (d["base_checkpoint_bytes"] + installed_bytes) * 8 / 7200 / 1000,
                    )
                    for m in ("lpips", "dists")
                },
                "installation_bytes": {
                    "base": d["base_checkpoint_bytes"],
                    "segmenter_and_text": installed_bytes,
                },
            }
        )
    wins = sum(c["primary"][m]["won"] is True for c in cells for m in ("lpips", "dists"))
    return {
        "rows": cells,
        "strict_wins": wins,
        "passes_3_of_4": wins >= 3,
        "all240_correction": "unmeasured",
        "background": "target-derived Telea granted",
        "scope": "quality on archived80 ticks only",
    }


def denied_probe_count(report: dict[str, Any]) -> int:
    """Forbidden-read probes the predictor must fail: frozen pair = two sources, the control and the transport; manifest windows = their sources and the transport (no control)."""
    return report["windows"] + 1 if "windows" in report else 4


def validate(archive: Path, stage: str, *, independent_frames: bool = False) -> dict[str, Any]:
    from experiments.tennis.k4 import safe

    with tarfile.open(archive) as t:
        f = t.extractfile("publish/receiver.json")
        if f is None:
            raise ValueError("Missing result")
        r = json.load(f)
        checks = {
            "fresh_reset": r["prediction"]["fresh_reset_equal"],
            "four_units": len(r["prediction"]["runs"][0]["rows"]) == 2 * r.get("windows", 2),
            "on_cuda": all(x["on_cuda"] for x in r["prediction"]["runs"]),
            "denied_reads": all(
                all(x["denied_reads"].values()) and len(x["denied_reads"]) == denied_probe_count(r)
                for x in r["prediction"]["runs"]
            ),
            "two_windows": len(r["rows"]) == r.get("windows", 2),
            "cpu_bound": r["cpu_seconds"] < (600 if stage == "smoke" else 2100),
        }
        expected = 2 if stage == "smoke" else 80
        if independent_frames:
            checks["independent_profile"] = all(
                run.get("independent_frames") is True
                and all(len(row["prompts"]) == expected
                        and [p["t"] for p in row["prompts"]] == row["ticks"]
                        for row in run["rows"])
                for run in r["prediction"]["runs"]
            )
        checks["memory"] = r["peak_gib"] * 1024 <= 12000
        checks["qualified_class"] = r["device"]["class_matches"]
        first, second = r["prediction"]["runs"]
        for left, right in zip(first["rows"], second["rows"]):
            name = left["name"]
            checks[name + ":reset_hashes"] = {
                k: v for k, v in left.items() if k != "inference_seconds"
            } == {k: v for k, v in right.items() if k != "inference_seconds"}
            stream = t.extractfile("publish/" + name + "-masks.npz")
            if stream is None:
                raise ValueError("Actual decoded predictor masks missing")
            with np.load(io.BytesIO(stream.read()), allow_pickle=False) as z:
                masks = z["masks"]
                checks[name + ":mask_arrays"] = (
                    masks.shape == (expected, 784, 512)
                    and masks.dtype == bool
                    and z["t"].tolist() == list(range(0, expected * 3, 3))
                    and array_sha(masks) == left["output_sha256"]
                    and [array_sha(m) for m in masks] == left["frames"]
                )
        checks["frozen_weights"] = all(
            run["weights_sha256"]
            == "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
            and run["text_sha256"]
            == "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
            for run in (first, second)
        )
        checks["sam_native"] = all(
            run["sam_load"]["sdpa_backend_policy"] == "native_flash"
            and not run["sam_load"]["missing_keys"] and not run["sam_load"]["unexpected_keys"]
            and run["sam_runtime"]["capability"][0] >= 8 and bool(run["flash_events"])
            for run in (first, second)
        )
        checks["ticks"] = all(
            len(x["judged"]) == expected and x["judged"] == x["expected_judged"] for x in r["rows"]
        )
        for row in r["rows"]:
            checks[row["id"] + ":control"] = row["control_max_abs_difference"] <= 1e-4
            checks[row["id"] + ":full_ticks"] = (
                stage == "smoke" or row["judged"] == row["original_judged"]
            )
            checks[row["id"] + ":finite"] = all(
                np.isfinite(list(v.values())).all() for v in row["means"].values()
            )
            score_stream = t.extractfile("publish/" + safe(row["id"]) + "/scores.npz")
            if score_stream is None:
                raise ValueError("Actual per-frame neural scores missing")
            with np.load(io.BytesIO(score_stream.read()), allow_pickle=False) as z:
                ticks = row["judged"]
                checks[row["id"] + ":per_frame_scores"] = all(
                    z[m].shape == (2, 240)
                    and np.isfinite(z[m][:, ticks]).all()
                    and np.allclose(
                        z[m][:, ticks].mean(1),
                        [row["means"][str(s)][m] for s in (0, -1)],
                        atol=1e-12,
                        rtol=0,
                    )
                    for m in ("lpips", "dists")
                )
            checks[row["id"] + ":shape_and_source_hashes"] = all(
                set(row[key]) == {str(tick) for tick in row["judged"]}
                for key in (
                    "prediction_shape",
                    "source_rgb_frame_sha256",
                    "source_player_masks_sha256",
                    "scoring_masks",
                )
            )
            checks[row["id"] + ":coverage"] = all(
                len(v) == 2
                and all(
                    c["requested_union"] - c["written_union"] == c["missing_union"] >= 0 for c in v
                )
                for v in row["crop_coverage"].values()
            )
            for file, digest in row["files"].items():
                stream = t.extractfile("publish/" + safe(row["id"]) + "/" + file)
                checks[row["id"] + ":" + file] = stream is not None and sha(stream.read()) == digest
        if stage == "smoke":
            checks["projected_full_under1500"] = r["projected_full_seconds"] < 1500
    return {"passed": all(checks.values()), "checks": checks}


def main(*, independent_default: bool = False) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "text", "out", "kps", "diagnostic", "reference-report", "transport"):
        p.add_argument("--" + name, type=Path)
    for name in ("generated", "source", "control", "anchor", "backbone", "forbidden"):
        p.add_argument("--" + name, type=Path, action="append", default=[])
    p.add_argument("--independent-frames", action="store_true", default=independent_default)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--max-judged", type=int, default=0)
    p.add_argument("--windows-file", type=Path, help="window manifest instead of the frozen Al/Med pair (no --diagnostic/--reference-report/--control)")
    p.add_argument("--window", action="append", help="manifest window name (repeat; default all)")
    a = p.parse_args()
    stage = Path(os.environ.get("PS_STAGE_DIR", "/tmp"))
    if a.validate:
        result = validate(stage / "published.tar", os.environ["PS_STAGE"], independent_frames=a.independent_frames)
        write(Path(os.environ["PS_VALIDATION_PATH"]), result)
        return int(not result["passed"])
    if a.worker:
        worker(a)
        return 0
    from experiments.tennis import k6b_matte_judge as judge
    from experiments.tennis import window_manifest as wm
    from experiments.tennis.k6b_foreground_video import PILOTS, SOURCE_SHA
    from experiments.tennis.k6b_joint_matte import REFERENCE_REPORT_SHA, CONTROL_SHA
    from experiments.tennis.k6b import load_generated
    from experiments.tennis.k6b_matte_cpu import box_maps
    from experiments.tennis.k6b_rate_streams import decode_keypoints
    from experiments.tennis.k6b_rate_budget import DIAGNOSTIC_SHA

    # Frozen pair by default; with --windows-file the manifest windows (no frozen old-run inputs exist for them; their control is in-job).
    if a.windows_file is None:
        pilots, pilot_shas = list(PILOTS), list(SOURCE_SHA)
    else:
        chosen = wm.select(wm.load(a.windows_file), a.window or None)
        pilots, pilot_shas = [w.id for w in chosen], [w.source_sha256 for w in chosen]
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir()
    if a.windows_file is None and (
        sha(a.diagnostic.read_bytes()) != DIAGNOSTIC_SHA
        or sha(a.reference_report.read_bytes()) != REFERENCE_REPORT_SHA
        or sha(a.control[0].read_bytes()) != CONTROL_SHA
    ):
        raise ValueError("Frozen evaluation inputs changed")
    kps = scratch / "kps"
    kps.mkdir()
    # Sanitize only KPS1 packets: predictor never receives the mask-bearing TAR.
    with tarfile.open(a.transport) as t:
        for member in t:
            if member.isfile() and member.name.endswith("keypoints.kps"):
                stream = t.extractfile(member)
                assert stream is not None
                blob = stream.read()
                meta, _ = decode_keypoints(blob)
                if meta["window"] in pilots:
                    (
                        kps / (meta["window"].replace("/", "__").replace("@", "_at") + ".kps")
                    ).write_bytes(blob)
    if len(list(kps.glob("*.kps"))) != len(pilots):
        raise ValueError("Every window's paid keypoint stream required")
    prediction, output = predict_twice(a, scratch, kps)
    write(publish / "prediction.json", prediction)
    # Evaluation only starts after both isolated masks are frozen and match.
    import torch
    import cv2
    from experiments.audit.env_smoke import device_record
    from experiments.visor.h3b import load_dists
    from src.codecs.quality import load_lpips, lpips_record

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, 12000 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    )
    cv2.setNumThreads(4)
    device = device_record("RTX 6000 Ada" if "Ada" in torch.cuda.get_device_name() else "RTX A6000")
    if not device["class_matches"]:
        raise ValueError("Qualified Ada required")
    anchors = judge.extract(a.anchor, scratch / "anchors")
    controls = judge.extract(a.control, scratch / "controls") if a.control else {}
    packets = scratch / "wire"
    packets.mkdir()
    with tarfile.open(a.transport) as t:
        t.extractall(packets, filter="data")
    packets = transport_root(packets)
    generated = load_generated([str(q) for q in a.generated], scratch / "generated")
    nets = {"lpips": load_lpips(a.backbone[0], "cuda"), "dists": load_dists(a.backbone[1], "cuda")}
    result = {
        "prediction": prediction,
        "rows": [],
        "device": device,
        "lpips": lpips_record(a.backbone[0]),
        "dists_sha256": sha(a.backbone[1].read_bytes()),
        "code_sha256": sha(Path(__file__).read_bytes()),
        "judge_code_sha256": sha(Path(judge.__file__).read_bytes()),
        "input_sha256": {
            str(q): sha(q.read_bytes())
            for q in [
                a.diagnostic,
                a.reference_report,
                a.transport,
                a.weights,
                a.text,
                *a.source,
                *a.anchor,
                *a.control,
                *a.generated,
            ]
            if q is not None
        },
    }
    sources = {sha(q.read_bytes()): q for q in a.source}
    if a.windows_file is not None:
        result["windows"] = len(pilots)
    began = time.monotonic()
    for wid, digest in zip(pilots, pilot_shas):
        _, units = decode_keypoints(
            (kps / (wid.replace("/", "__").replace("@", "_at") + ".kps")).read_bytes()
        )
        masks = {}
        for i, (unit, boxes) in enumerate(zip(units, box_maps(units))):
            name = unit["name"]
            with np.load(output / (name + ".npz")) as z:
                masks[i] = {int(t): project(m, boxes[int(t)]) for t, m in zip(z["t"], z["masks"])}
            # Keep actual cropped decoder masks, not just scoring summaries.
            (publish / (name + "-masks.npz")).write_bytes((output / (name + ".npz")).read_bytes())
        result["rows"].append(
            judge.judge_window(
                wid,
                anchors[wid],
                controls.get(wid),
                packets / "packets",
                scratch,
                {},
                generated,
                sources[digest],
                digest,
                nets,
                scratch,
                publish,
                a.max_judged,
                control_variant="full",
                predicted_masks=masks,
            )
        )
        from experiments.jobs.monitor import publish_progress

        publish_progress(str(stage), len(result["rows"]))
    result["judge_seconds"] = time.monotonic() - began
    result["peak_gib"] = max(
        torch.cuda.max_memory_allocated() / 2**30, *[r["peak_gib"] for r in prediction["runs"]]
    )
    result["projected_full_seconds"] = (
        60
        + prediction["seconds"]
        + 1.5
        * sum(x["inference_seconds"] for run in prediction["runs"] for x in run["rows"])
        * (80 / (a.max_judged or 80) - 1)
        + sum(
            r["setup_seconds"]
            + 1.5 * sum(r["scored_frame_seconds"].values()) * 80 / len(r["judged"])
            for r in result["rows"]
        )
    )
    u = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["cpu_seconds"] = u.ru_utime + u.ru_stime + c.ru_utime + c.ru_stime
    if not a.max_judged:
        result["assessment"] = assessment(
            json.loads(a.diagnostic.read_text()),
            json.loads(a.reference_report.read_text()),
            result["rows"],
            a.weights.stat().st_size,
        )
    if not a.max_judged and a.independent_frames:
        diagnostic = json.loads(a.diagnostic.read_text())
        reference = json.loads(a.reference_report.read_text())
        result["frozen_quality_sensitivity"] = {}
        for delta in (-0.001, 0.0, 0.001):
            adjusted = [{**row, "means": {**row["means"], "-1": {
                metric: value + delta for metric, value in row["means"]["-1"].items()
            }}} for row in result["rows"]]
            result["frozen_quality_sensitivity"][str(delta)] = assessment(
                diagnostic, reference, adjusted, a.weights.stat().st_size
            )
    write(publish / "receiver.json", result)
    write(stage / "receiver.json", result)
    for n in range(2):
        (publish / f"worker{n}.log").write_bytes((scratch / f"worker{n}.log").read_bytes())
    with tarfile.open(stage / "published.tar", "w") as t:
        t.add(publish, arcname="publish")
    print(
        json.dumps(
            {
                "windows": len(result["rows"]),
                "cpu_seconds": result["cpu_seconds"],
                "projected_full_seconds": result["projected_full_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
