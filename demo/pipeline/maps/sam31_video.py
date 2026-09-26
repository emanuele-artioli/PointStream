"""SAM 3.1 Object Multiplex video masks, billed as the shared CRF AV1.

Runs inside the pointstream-sam31 env via subprocess. Weight load is timed
and stored on the sidecar, and is not included in extract_ms_p50.

Each text prompt replaces the previous one (the multiplex tracker resets on
add_prompt), so every role is grounded in its own propagate pass and the
masks are composited. Frames stay at the source resolution. Ada uses PyTorch
flash SDPA in bf16; FlashAttention 3 is Hopper-only and is left off.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CONCEPTS = ("hand", "tool", "workbench")
CONCEPT_CODE = {"workbench": 1, "tool": 2, "hand": 3}
DEFAULT_PYTHON = "/home/itec/emanuele/.conda/envs/pointstream-sam31/bin/python"
# Native 1080p frames. The image encoder itself is 1008, the same constant as
# tennis Sam3Detector; SAM 3.1 does not take a YOLO-style imgsz above that.
# Masks are resized back onto the original frame. The 540p cap was only for the
# 32 GiB Volta card, which has no flash attention.
SAM_ENCODER_IMGSZ = 1008
SAM_FRAME_SCALE: str | None = None
SAM_PROB_THRESH = 0.35
DEFAULT_CHECKPOINT = (
    "/home/itec/emanuele/.cache/huggingface/hub/models--facebook--sam3.1/"
    "snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt"
)


def resolve_checkpoint() -> Path | None:
    env = os.environ.get("SAM31_CHECKPOINT")
    candidates = [Path(env)] if env else []
    candidates.append(Path(DEFAULT_CHECKPOINT))
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_python() -> str:
    return os.environ.get("SAM31_PYTHON", DEFAULT_PYTHON)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def extract_frames(
    clip: Path,
    frames_dir: Path,
    *,
    scale: str | None = None,
    ffmpeg: str = "ffmpeg",
) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    # SAM's loader wants "<index>.jpg" starting at 0. q:v 2 is its recommended JPEG.
    pattern = frames_dir / "%05d.jpg"
    cmd = [ffmpeg, "-y", "-i", str(clip)]
    if scale:
        cmd.extend(["-vf", f"scale={scale}"])
    cmd.extend(["-q:v", "2", "-start_number", "0", str(pattern)])
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = sorted(frames_dir.glob("*.jpg"))
    if res.returncode != 0 or not frames:
        tail = res.stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"frame extract failed for {clip}: {tail}")
    return len(frames)


def launch(clip: Path, out_dir: Path, *, max_frames: int | None = None) -> None:
    from demo.evaluation.profile_map import gpu_name
    from demo.pipeline.maps.av1_crf import AV1_CRF, AV1_PRESET, AV1_SCALE
    from demo.pipeline.maps.contract import MapStream, write_sidecar

    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError(
            "sam3.1_multiplex.pt is not on disk. Set SAM31_CHECKPOINT. Do not auto-download."
        )
    python = resolve_python()
    if not Path(python).is_file():
        raise FileNotFoundError(f"SAM 3.1 python not found: {python}")

    from demo.pipeline.maps.yoloe_masks import load_prompt_map

    out_dir.mkdir(parents=True, exist_ok=True)
    role_prompts = load_prompt_map("sam", clip.stem)
    prompts_path = out_dir / "prompts.json"
    prompts_path.write_text(json.dumps(role_prompts))
    frames_dir = out_dir / "frames"
    n_written = extract_frames(clip, frames_dir, scale=SAM_FRAME_SCALE)
    if max_frames is not None:
        for extra in sorted(frames_dir.glob("*.png"))[max_frames:]:
            extra.unlink()
        n_written = max_frames
    payload = out_dir / "masks.av1.mp4"
    timing_path = out_dir / "sam31_timing.json"
    cmd = [
        python, "-m", "demo.pipeline.maps.sam31_video", "--worker",
        "--frames", str(frames_dir),
        "--out", str(payload),
        "--timing", str(timing_path),
        "--checkpoint", str(checkpoint),
        "--prompts", str(prompts_path),
    ]
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    log_path = out_dir / "sam31_worker.log"
    log_path.write_text(res.stderr.decode("utf-8", errors="replace")[-20000:])
    if res.returncode != 0 or not timing_path.is_file():
        raise RuntimeError(
            f"SAM 3.1 worker failed ({res.returncode}). See {log_path}"
        )
    timing = json.loads(timing_path.read_text())
    step_ms = [float(v) for v in timing.get("propagate_step_ms") or []]
    # propagate_in_video yields most frames in bursts, so the median step is ~0.
    # The per-frame cost is the mean of the whole propagate, load excluded.
    steady = step_ms
    n_frames = int(timing["n_frames"])
    fps = float(timing["fps"])
    stream = MapStream(
        map="sam31_masks",
        backend="sam3.1_multiplex.pt",
        payload_path=str(payload),
        payload_bytes=payload.stat().st_size,
        preview_path=str(payload),
        preview_bytes=payload.stat().st_size,
        duration_s=n_frames / fps,
        n_frames=n_frames,
        fps=fps,
        extract_ms_p50=(sum(steady) / len(steady)) if steady else 0.0,
        extract_ms_p95=_percentile(steady, 95),
        pack_ms_p50=0.0,
        codec_ms_p50=0.0,
        decode_ms_p50=0.0,
        gpu=str(timing.get("gpu") or gpu_name()),
        kind="native",
        extra={
            "payload_format": "av1",
            "payload_schema": "pointstream.maps.mask_av1.v1",
            "av1_scale": AV1_SCALE,
            "av1_preset": AV1_PRESET,
            "av1_crf": AV1_CRF,
            "classes": list(CONCEPTS),
            "model_load_s": timing.get("model_load_s"),
            "prompt_s": timing.get("prompt_s"),
            "latency_excludes_load": True,
            "n_nonempty": timing.get("n_nonempty"),
            "concept_ids": timing.get("concept_ids"),
            "overlay_path": str(payload),
            "overlay_key": "black",
            "overlay": "av1-black-key",
            "frames_extracted": n_written,
            "model_frame_scale": SAM_FRAME_SCALE or "native",
            "encoder_imgsz": SAM_ENCODER_IMGSZ,
            "class_prompts": role_prompts,
            "prob_thresh": SAM_PROB_THRESH,
            "attention": "pytorch_flash_sdp",
            "dtype": "bfloat16",
            "use_fa3": False,
        },
    )
    write_sidecar(stream, out_dir / "sidecar.json")
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)


def _worker(argv: list[str]) -> int:
    import inspect
    import uuid

    import numpy as np
    import torch
    from PIL import Image
    from torch import nn

    from demo.pipeline.maps.av1_crf import CLASS_COLORS_BGR, PAINT_ORDER, pipe_bgr_av1

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    args = parser.parse_args(argv)
    role_prompts = json.loads(args.prompts.read_text())
    prob_thresh = float(os.environ.get("SAM_PROB_THRESH", SAM_PROB_THRESH))

    frame_paths = sorted(args.frames.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"no frames in {args.frames}")
    sample = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
    height, width = int(sample.shape[0]), int(sample.shape[1])
    n_frames = len(frame_paths)
    fps = 30.0

    sys.path.insert(0, "/home/itec/emanuele/.cache/sam3-meta")
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the SAM 3.1 env")
    load_audit: list[dict[str, Any]] = []
    original_load = nn.Module.load_state_dict

    def audited_load(self, state_dict, *a, **k):
        missing, unexpected = original_load(self, state_dict, *a, **k)
        load_audit.append({
            "module": type(self).__name__,
            "missing": len(list(missing)),
            "unexpected": len(list(unexpected)),
        })
        return missing, unexpected

    nn.Module.load_state_dict = audited_load
    build_t0 = time.perf_counter()
    try:
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(args.checkpoint),
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            default_output_prob_thresh=prob_thresh,
        )
    finally:
        nn.Module.load_state_dict = original_load
    model_load_s = time.perf_counter() - build_t0

    def compatible_start_session(self, resource_path, session_id=None, offload_video_to_cpu=False, offload_state_to_cpu=False):
        init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        valid = inspect.signature(self.model.init_state).parameters
        state = self.model.init_state(**{key: value for key, value in init_kwargs.items() if key in valid})
        session = session_id or str(uuid.uuid4())
        now = time.time()
        self._all_inference_states[session] = {
            "state": state,
            "session_id": session,
            "start_time": now,
            "last_use_time": now,
        }
        return {"session_id": session}

    predictor.start_session = compatible_start_session.__get__(predictor, type(predictor))
    started = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(args.frames),
    })
    session_id = started["session_id"]

    def unpack(outputs: dict) -> tuple[list[int], np.ndarray]:
        ids = outputs.get("out_obj_ids", [])
        masks = outputs.get("out_binary_masks", [])
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().numpy()
        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()
        ids = [int(v) for v in list(np.asarray(ids).reshape(-1))]
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.size == 0:
            masks = np.zeros((0, height, width), dtype=bool)
        return ids, masks.astype(bool, copy=False)

    # add_prompt resets tracker state, so each role is a separate full-video pass.
    # Paint workbench, then tool, then hand. Identical texts run once.
    labels = np.zeros((n_frames, height, width), dtype=np.uint8)
    frame_ms = [0.0] * n_frames
    concept_ids: dict[str, list[int]] = {name: [] for name in PAINT_ORDER}
    prompt_s = 0.0
    seen_text: dict[str, str] = {}
    for concept in PAINT_ORDER:
        text = str(role_prompts.get(concept) or concept)
        if text in seen_text:
            continue
        seen_text[text] = concept
        torch.cuda.synchronize()
        prompt_t0 = time.perf_counter()
        predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text,
            "output_prob_thresh": prob_thresh,
        })
        torch.cuda.synchronize()
        prompt_s += time.perf_counter() - prompt_t0
        t0 = time.perf_counter()
        seen_ids: set[int] = set()
        for response in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "output_prob_thresh": prob_thresh,
        }):
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            index = int(response["frame_index"])
            t0 = time.perf_counter()
            if index < 0 or index >= n_frames:
                continue
            frame_ms[index] += dt_ms
            ids, masks = unpack(response["outputs"])
            plane = labels[index]
            code = CONCEPT_CODE[concept]
            for object_id, mask in zip(ids, masks):
                if object_id not in seen_ids:
                    seen_ids.add(object_id)
                    concept_ids[concept].append(object_id)
                if mask.shape != (height, width):
                    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (width, height), Image.Resampling.NEAREST
                    )
                    mask = np.asarray(resized) > 0
                plane[mask.astype(bool)] = code
    step_ms = frame_ms

    proc = pipe_bgr_av1(width, height, fps, args.out)
    assert proc.stdin is not None
    n_nonempty = 0
    try:
        for index in range(n_frames):
            plane = labels[index]
            painted = np.zeros((height, width, 3), dtype=np.uint8)
            for name in PAINT_ORDER:
                code = CONCEPT_CODE[name]
                binary = plane == code
                if binary.any():
                    painted[binary] = CLASS_COLORS_BGR[name]
            if int(painted.max()) > 0:
                n_nonempty += 1
            proc.stdin.write(painted.tobytes())
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        code = proc.wait()
        if code != 0:
            tail = stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"SAM mask AV1 encode failed ({code}): {tail}")

    args.timing.write_text(json.dumps({
        "n_frames": n_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "model_load_s": round(model_load_s, 3),
        "prompt_s": round(prompt_s, 3),
        "propagate_step_ms": [round(v, 3) for v in step_ms],
        "n_nonempty": n_nonempty,
        "concept_ids": concept_ids,
        "class_prompts": role_prompts,
        "prob_thresh": prob_thresh,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_loads": load_audit[-3:],
    }) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in args:
        return _worker(args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    ns = parser.parse_args(args)
    launch(ns.clip, ns.out, max_frames=ns.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
