"""Finish the common-player witness from paid geometry and authenticated RGB crops."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import med_mixed_channel as mixed
from experiments.tennis import k6b, k6b_receiver_sam as sam
from experiments.tennis import w8_receiver_finish as w8
from experiments.tennis.k6b_real_base_compose import compose
from experiments.tennis.player_requalification import array_sha
from experiments.tennis.w8_input_job import file_sha
from src.pointstream.tracked_patch import yuv_to_rgb as yuv_to_rgb

YUV_SHA = "f2b9dd6a2370b3a6267a0379339360307bfe7900f58e7a503b8b2a2b1756bafb"
SMOKE = [0, 11, 23, 43]
FULL = list(range(240))
KPS_SHA = "0a81274aaaaa272dddb61fc131f43e3a551976c01230f93514e7bfa0647546ce"


def geometry(blob: bytes, expected: str) -> tuple[dict, bytes, dict]:
    """Read the two sections needed after the JPEG-capable generator verified the carrier."""
    if len(blob) > 4 * 1024 * 1024 or hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("invalid authenticated player carrier")
    profile = packet._embedded_profile(blob)
    packet._profile(profile)
    sections, roles = packet.wire.unpack(blob, profile)
    order = profile["order"]
    if (
        len(order) != 2
        or profile["mode"] != "players"
        or profile["canvas_size"] != [1920, 1080]
        or profile["coded_size"] != [160, 96]
        or profile["cuts"]
        or sum(roles.values()) != len(blob)
        or profile["resources"]["native_decoder"] != w8.FFMPEG_SHA
    ):
        raise ValueError("unsupported common-player witness profile")
    packet._ivf(sections["base"], 160, 96, (25, 1))
    if (
        packet.wire.sha(sections["base"])
        != "8466fe0f097c79a2383351ea9ccce438ea62cd1e0283d25d5c68b08ee4b96c8d"
        or packet.wire.sha(sections["keypoints"]) != KPS_SHA
    ):
        raise ValueError("paid native base or KPS2 differs from frozen witness")
    meta, units = packet._bounded_keypoints(sections["keypoints"], order)
    if meta != {"native_frames": 240, "names": order, "profile": packet.rates.KPS2_PROFILE}:
        raise ValueError("KPS2 geometry profile")
    geometry_by_unit = {}
    for unit in units:
        name = unit["name"]
        runs = [r for r in profile["runs"] if r["unit"] == name]
        if (
            len(runs) != 1
            or runs[0]["start"] != 0
            or runs[0]["end"] != 240
            or not np.array_equal(unit["t"], FULL)
            or unit["kp"].shape != (240, 17, 3)
        ):
            raise ValueError("witness requires the two continuous full-length player tracks")
        geometry_by_unit[name] = (unit["t"], unit["kp"], k6b.run_boxes(list(unit["kp"])))
    if list(geometry_by_unit) != order:
        raise ValueError("KPS2 and packet compositing order mismatch")
    return profile, sections["base"], geometry_by_unit


def _isolation(proof: dict) -> None:
    if (
        proof.get("landlock_abi", 0) < 3
        or not proof.get("denials")
        or proof.get("interfaces") != ["lo"]
        or any(err not in (1, 13) for err in proof["denials"].values())
    ):
        raise ValueError("missing measured receiver containment")


def _verified(row: dict) -> Path:
    path = Path(row["path"])
    if not path.is_absolute() or file_sha(path) != row["sha256"]:
        raise ValueError("staged file hash mismatch")
    return path


def checked_crops(name: str, generated: Path, geom: tuple, frames: int) -> np.ndarray:
    with np.load(generated, allow_pickle=False) as fields:
        times, _, boxes = geom
        if (
            set(fields.files) != {"t", "box", "crops"}
            or not np.array_equal(fields["t"], times[:frames])
            or not np.array_equal(fields["box"], boxes[:frames])
        ):
            raise ValueError(f"generated ticks or boxes differ from paid geometry for {name}")
        crops = fields["crops"]
    if crops.dtype != np.uint8 or crops.shape != (frames, 784, 512, 3):
        raise ValueError(f"generated RGB coverage for {name}")
    return crops


def generated_input(name: str, config: dict, geom: tuple, ticks: list[int]) -> np.ndarray:
    generated = _verified(config["generated"][name])
    seal = _verified(config["generation_receipts"][name])
    receipt = json.loads(seal.read_text())
    _isolation(receipt["isolation"])
    frames = receipt["frames"]
    if (
        receipt["carrier_sha256"] != config["carrier_sha256"]
        or receipt["conditioning_only"] is not False
        or frames not in (44, 120, 240)
        or frames <= max(ticks)
    ):
        raise ValueError("generation seal does not cover this carrier and stage")
    rows = [row for row in receipt["outputs"] if row["unit"] == name]
    if (
        len(rows) != 1
        or rows[0]["crops_file_sha256"] != config["generated"][name]["sha256"]
        or rows[0]["kernels"]["kernel_launches"] <= 0
        or rows[0]["array_sha256"]["t"] != array_sha(geom[0][:frames])
        or rows[0]["array_sha256"]["box"] != array_sha(geom[2][:frames])
        or rows[0]["seed"] != int(hashlib.sha256(name.encode()).hexdigest()[:7], 16) * 1009
    ):
        raise ValueError("generation, kernel, seed or geometry binding differs")
    return checked_crops(name, generated, geom, frames)


def crop_prompt(kp: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        return sam.pose_prompt(k6b.to_crop(kp, box))
    except ValueError as exc:
        if str(exc) != "Insufficient paid pose for deterministic SAM initialization":
            raise
        return None


def _predict(
    model: Any, crop: np.ndarray, prompt: tuple, profile: bool
) -> tuple[np.ndarray, list[str]]:
    """W8 independent-frame SAM reset and normalized positive/corner prompt."""
    import torch
    from PIL import Image

    size = int(model.image_size)
    image = torch.from_numpy(
        np.array(Image.fromarray(crop).resize((size, size))).transpose(2, 0, 1).copy()
    )
    image = image.to(dtype=torch.float16).div_(255).sub_(0.5).div_(0.5)[None]
    profiler = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        )
        if profile
        else None
    )
    with (
        profiler if profiler is not None else nullcontext(),
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        state = model.init_state(
            video_height=784, video_width=512, num_frames=1, offload_video_to_cpu=True
        )
        state["images"] = image
        frame, ids, _, video = model.add_new_points(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=torch.from_numpy(prompt[0]),
            labels=torch.from_numpy(prompt[1]),
            clear_old_points=True,
            rel_coordinates=True,
        )
        if int(frame) != 0 or list(ids) != [1]:
            raise ValueError("SAM frame or object identity")
        mask = (video[0, 0] > 0).cpu().numpy()
    kernels = (
        []
        if profiler is None
        else sorted({e.key for e in profiler.key_averages() if "flash" in e.key.lower()})
    )
    if mask.shape != (784, 512) or mask.dtype != bool or (profile and not kernels):
        raise RuntimeError("SAM crop mask geometry or flash-attention execution failed")
    return mask, kernels


def compose_tick(
    base: np.ndarray, order: list[str], active: dict, masks: dict
) -> tuple[np.ndarray, dict]:
    """Project masks in packet order; each absence falls back to that pixel of the canvas."""
    crops, boxes, projected, reasons = {}, {}, {}, {}
    for name in order:
        if name not in active or name not in masks:
            projected[name] = np.zeros(base.shape[:2], bool)
            reasons[name] = "missing_pose" if name not in active else "insufficient_prompt"
            continue
        crop, box = active[name]
        mask = masks[name]
        if mask.shape != (784, 512) or mask.dtype != bool:
            raise ValueError("invalid SAM crop mask")
        boxes[name], crops[name] = box, crop
        projected[name] = sam.project(mask, box)
        reasons[name] = (
            "composed" if projected[name].any() else "offscreen" if mask.any() else "empty_mask"
        )
    image, written = compose(base, crops, boxes, projected, order)
    return image, {
        "coverage_pixels": int(written.sum()),
        "status": reasons,
        "projected_pixels": {name: int(mask.sum()) for name, mask in projected.items()},
    }


def run(config: dict, isolation: dict) -> dict:
    """Called only by the isolated fleet launcher, after fresh measured access probes."""
    import cv2
    import torch
    from src.segmentation.sam31_tracker import Sam31MaskTracker, runtime

    _isolation(isolation)
    start = time.monotonic()
    ticks = config["ticks"]
    carrier = Path(config["carrier"])
    expected = config["carrier_sha256"]
    verified = manifest.verify(Path(config["manifest"]), expected)
    if carrier.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("oversized player carrier")
    channel = None
    if "joint" in config:
        channel = mixed.decode(Path(config["joint"]), Path(config["cache"]), carrier, expected)
        profile, base, geom = channel["profile"], channel["base"], channel["geometry"]
        run_start, run_stop = channel["start"], channel["stop"]
        smoke = mixed.smoke_ticks(run_start, run_stop)
    else:
        profile, base, geom = geometry(carrier.read_bytes(), expected)
        run_start, run_stop, smoke = 0, 240, SMOKE
    if ticks not in (smoke, FULL) or any(type(t) is not int for t in ticks):
        raise ValueError("unsupported SAM receiver stage")
    order = profile["order"]
    if set(config["generated"]) != set(order) or set(config["generation_receipts"]) != set(order):
        raise ValueError("expected both generated player streams and their receipts")
    output = Path(config["output"])
    output.mkdir(exist_ok=False)
    ivf, yuv = output / "base.ivf", output / "base.yuv"
    ivf.write_bytes(base)
    binary = Path(config["ffmpeg"])
    if (
        not binary.is_absolute()
        or yuv.exists()
        or file_sha(binary) != w8.FFMPEG_SHA
        or file_sha(ivf) != hashlib.sha256(base).hexdigest()
    ):
        raise ValueError("native decoder or target IVF identity differs")
    args = [
        str(binary),
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "1",
        "-c:v",
        "libdav1d",
        "-i",
        str(ivf),
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rawvideo",
        "-n",
        str(yuv),
    ]
    decode_start = time.monotonic()
    subprocess.run(args, check=True, timeout=120, stdin=subprocess.DEVNULL, capture_output=True)
    if yuv.stat().st_size != 240 * 160 * 144 or (channel is None and file_sha(yuv) != YUV_SHA):
        raise ValueError("incomplete native decode")
    native = np.memmap(yuv, np.uint8, mode="r", shape=(240, 144, 160))
    native_seconds = time.monotonic() - decode_start
    model_start = time.monotonic()
    weight = Path(config["sam"])
    identity = w8._qualification_gpu(torch)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    limit = int(os.environ["PS_GPU_MEMORY_LIMIT_MIB"])
    if limit <= 0:
        raise ValueError("missing assigned SAM memory budget")
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    )
    tracker = Sam31MaskTracker(
        weight, checkpoint_sha256=verified["sam_weight"], source_revision=verified["sam_commit"]
    )
    if (
        tracker.load_report.get("sdpa_backend_policy") != "native_flash"
        or tracker.load_report.get("missing_keys")
        or tracker.load_report.get("unexpected_keys")
        or not next(tracker.model.parameters()).is_cuda
    ):
        raise RuntimeError("SAM source, weights or device changed")
    model_seconds = time.monotonic() - model_start
    local_ticks = [t - run_start for t in ticks if run_start <= t < run_stop]
    crops = {name: generated_input(name, config, geom[name], local_ticks) for name in order}
    dest = np.lib.format.open_memmap(
        output / "frames.npy", mode="w+", dtype=np.uint8, shape=(len(ticks), 1080, 1920, 3)
    )
    rows: list[dict] = []
    kernels: list[str] = []
    mask_arrays: dict[str, list[np.ndarray]] = {name: [] for name in order}
    loop_start = time.monotonic()
    for i, tick in enumerate(ticks):
        tick_start = time.monotonic()
        sam_seconds = 0.0
        canvas = mixed.canvas(native[tick], tick, channel)
        active, masks = {}, {}
        for name in order:
            if not run_start <= tick < run_stop:
                mask_arrays[name].append(np.zeros((784, 512), bool))
                continue
            local = tick - run_start
            _, kp, boxes = geom[name]
            box = boxes[local]
            crop = crops[name][local]
            active[name] = crop, box
            prompt = crop_prompt(kp[local], box)
            if prompt is not None:
                predict_start = time.monotonic()
                mask, events = _predict(tracker.model, crop, prompt, not kernels)
                sam_seconds += time.monotonic() - predict_start
                kernels.extend(events)
                masks[name] = mask
            mask_arrays[name].append(masks[name] if name in masks else np.zeros((784, 512), bool))
        frame, row = compose_tick(canvas, order, active, masks)
        dest[i] = frame
        rows.append(
            {
                "tick": tick,
                "tick_seconds": time.monotonic() - tick_start,
                "sam_seconds": sam_seconds,
                "base_sha256": array_sha(canvas),
                "frame_sha256": array_sha(frame),
                **row,
            }
        )
    dest.flush()
    del dest
    if not kernels or any(
        not any(row["status"][name] == "composed" for row in rows) for name in order
    ):
        raise ValueError("missing real SAM execution or player coverage")
    loop_seconds = time.monotonic() - loop_start
    mask_files = {}
    for name in order:
        path = output / f"{name}-masks.npz"
        np.savez_compressed(path, t=np.array(ticks), masks=np.stack(mask_arrays[name]))
        mask_files[name] = file_sha(path)
    result = {
        "scope": "common-player receiver",
        "ticks": ticks,
        "rows": rows,
        "order": order,
        "carrier_sha256": expected,
        "carrier_bytes": carrier.stat().st_size,
        "manifest_bytes": verified["manifest_bytes"],
        "total_receiver_bytes": carrier.stat().st_size + verified["manifest_bytes"],
        "manifest_sha256": verified["manifest_sha256"],
        "native_sha256": w8.FFMPEG_SHA,
        "yuv_sha256": file_sha(yuv),
        "mask_files": mask_files,
        "frames_file_sha256": file_sha(output / "frames.npy"),
        "sam_weight": verified["sam_weight"],
        "sam_runtime": runtime(),
        "sam_load": tracker.load_report,
        "gpu": identity,
        "kernels": kernels,
        "isolation": isolation,
        "generated": config["generated"],
        "generation_receipts": config["generation_receipts"],
        "native_decode_seconds": native_seconds,
        "sam_model_load_seconds": model_seconds,
        "initialization_seconds": loop_start - start,
        "loop_seconds": loop_seconds,
        "finalization_seconds": time.monotonic() - loop_start - loop_seconds,
        "elapsed_seconds": time.monotonic() - start,
    }
    if channel is not None:
        result["wire"] = channel["wire"]
    (output / "finish.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return result
