"""Environment-audit smoke: each phase-1 component on one real sample, per GPU class.

Infrastructure smoke, never evidence (docs/experiments.md). One fleet job runs a
group of components; each component runs in its own process (one failure does
not hide the others) and writes ``components/<name>.json`` under PS_STAGE_DIR.
Every GPU component records the device it ran on, the CUDA kernels it launched
(torch profiler) and the attention kernel family, and asserts:

* the device is the claimed GPU of the expected class, and the model is on it;
* CUDA kernels ran (no silent CPU fallback);
* the attention family is the one this GPU class should use;
* a substantive output check against the dataset's own labels.

    python -m experiments.audit.env_smoke run --group G --gpu-class NAME --frames N --input ...
    python -m experiments.audit.env_smoke validate --group G --gpu-class NAME
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

GROUPS = {
    "sam31": ("visor", "sam31"),
    "yoloe-codecs": ("visor", "yoloe", "svtav1", "dcvc"),
    "hamer": ("visor", "hamer", "hot3d"),
    "wilor": ("visor", "wilor"),
}
HANDS = ("left hand", "right hand")
COMPONENT_SECONDS = 420


# ----------------------------------------------------------------- helpers

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def scratch() -> Path:
    root = Path(os.environ.get("PS_SCRATCH_DIR") or os.environ["PS_STAGE_DIR"]) / "env-smoke"
    root.mkdir(parents=True, exist_ok=True)
    return root


def polygon_mask(annotations: list[dict[str, Any]], names: set[str], shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("1", (shape[1], shape[0]))
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        if annotation["name"] in names:
            for polygon in annotation["segments"]:
                if len(polygon) >= 3:
                    draw.polygon([tuple(point) for point in polygon], fill=1)
    return np.array(image, dtype=bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def device_record(expected_class: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name, "capability": [props.major, props.minor],
        "uuid": str(getattr(props, "uuid", "")), "total_mib": props.total_memory // 2**20,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": torch.cuda.device_count(), "expected_class": expected_class,
        "class_matches": expected_class in props.name,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "arch_list": torch.cuda.get_arch_list(),
    }


def attention_family(kernels: list[str]) -> list[str]:
    families = set()
    for name in kernels:
        lower = name.lower()
        if "flash" in lower:
            families.add("flash")
        elif "fmha" in lower or "efficient_attention" in lower or "mem_eff" in lower or "attentionkernel" in lower:
            families.add("mem_efficient")
        elif "cudnn" in lower and ("sdpa" in lower or "mha" in lower or "attn" in lower):
            families.add("cudnn")
    return sorted(families)


def profile_cuda(fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Run ``fn`` under the torch profiler; return its result and the CUDA kernels."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        result = fn()
        torch.cuda.synchronize()
    totals: dict[str, float] = {}
    counts: Counter[str] = Counter()
    for event in prof.events():
        if str(getattr(event, "device_type", "")).endswith("CUDA") and event.name:
            us = getattr(event, "device_time", None) or getattr(event, "cuda_time", 0) or 0
            totals[event.name] = totals.get(event.name, 0.0) + float(us)
            counts[event.name] += 1
    names = list(totals)
    return result, {
        "kernel_launches": int(sum(counts.values())), "distinct_kernels": len(names),
        "top_kernels_us": [[name[:160], round(us, 1)] for name, us in sorted(totals.items(), key=lambda kv: -kv[1])[:25]],
        "attention_family": attention_family(names),
        "tensor_core_like": sorted({tag for name in names for tag in ("cutlass", "xmma", "gemm", "s1688", "s16816", "wmma", "hmma", "sm80", "sm75", "sm70", "sm89")
                                    if tag in name.lower()}),
    }


def model_on_cuda(module: Any) -> bool:
    params = list(module.parameters())
    return bool(params) and all(p.is_cuda for p in params)


# ----------------------------------------------------------------- VISOR

def epic_frame_to_video_index(epic_frame: int, fps: float) -> int:
    """Decode-order index of EPIC-KITCHENS rgb frame ``epic_frame`` (1-indexed).

    Hypothesis under test, from P32_07: the rgb frames were extracted at 60 fps
    from the 59.94 fps videos, so frame k shows video time (k - 1) / 60 s. Other
    rates are assumed to be extracted at their own rate. B1 decides the rule.
    """
    extraction = 60.0 if abs(fps - 59.94) < 0.01 else fps
    return int(round((epic_frame - 1) * fps / extraction))


def _decode_window(container: Any, stream: Any, first: int, last: int) -> dict[int, np.ndarray]:
    """Frames ``first..last`` by decode-order index (seek, then decode forward)."""
    rate, base, start_pts = stream.average_rate, stream.time_base, stream.start_time or 0
    container.seek(int(start_pts + max(first - 2, 0) / rate / base), stream=stream, backward=True, any_frame=False)
    frames: dict[int, np.ndarray] = {}
    for frame in container.decode(stream):
        at = int(round(float((frame.pts - start_pts) * base * rate)))
        if first <= at <= last:
            frames[at] = frame.to_ndarray(format="rgb24")
        if at >= last:
            break
    return frames


def load_visor(video: Path, sample: Path, frames: int, work: Path) -> dict[str, Any]:
    """The sample's sparse frames (each matched within +-3 decoded frames) and a clip.

    The clip is ``frames`` consecutive decoded frames starting at the first sparse
    frame's time-rule index, so its first frame carries the human masks.
    """
    import av
    from PIL import Image

    with tarfile.open(sample) as tar:
        names = tar.getnames()
        ann_name = next(n for n in names if n.startswith("annotations/"))
        entries = json.load(tar.extractfile(ann_name))  # type: ignore[arg-type]
        mapping = json.load(tar.extractfile("frame_mapping.json"))  # type: ignore[arg-type]
        jpegs = {n.split("/", 1)[1]: tar.extractfile(n).read() for n in names if n.startswith("rgb_frames/")}  # type: ignore[union-attr]
    video_id = Path(ann_name).stem
    mapping = mapping[video_id]
    chosen = sorted((e for e in entries["video_annotations"] if e["image"]["name"] in jpegs), key=lambda e: e["image"]["name"])
    started = time.time()
    sparse, clip = [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate)
        info = {"codec": stream.codec_context.name, "fps": fps, "width": stream.width,
                "height": stream.height, "frames_declared": stream.frames}
        for entry in chosen:
            epic = int(re.search(r"frame_(\d+)", mapping[entry["image"]["name"]]).group(1))  # type: ignore[union-attr]
            mapped, ruled = epic - 1, epic_frame_to_video_index(epic, fps)
            window = _decode_window(container, stream, min(mapped, ruled) - 3, max(mapped, ruled) + 3)
            released = np.array(Image.open(io.BytesIO(jpegs[entry["image"]["name"]])).convert("RGB")).astype(np.int16)
            mae = {i: float(np.abs(f.astype(np.int16) - released).mean()) for i, f in window.items()}
            best = min(mae, key=lambda i: mae[i])
            frame = window[ruled]
            shape = (int(frame.shape[0]), int(frame.shape[1]))
            masks = {name: polygon_mask(entry["annotations"], {name}, shape) for name in HANDS}
            objects = {a["name"] for a in entry["annotations"]} - set(HANDS)
            masks["objects"] = polygon_mask(entry["annotations"], objects, shape)
            sparse.append({"index": ruled, "mapped_index": mapped, "best_index": best, "name": entry["image"]["name"],
                           "mae": {"time_rule": round(mae[ruled], 3), "mapping_minus_one": round(mae[mapped], 3), "best": round(mae[best], 3)},
                           "frame": frame, "masks": masks,
                           "contacts": {a["name"]: a.get("in_contact_object") for a in entry["annotations"] if a["name"] in HANDS}})
        first = sparse[0]["index"]
        window = _decode_window(container, stream, first, first + frames - 1)
        clip = [window[i] for i in range(first, first + frames)]
    return {"video_id": video_id, "info": info, "sparse": sparse, "clip": clip, "first": first,
            "decode_seconds": round(time.time() - started, 3), "work": work}


def c_visor(args: argparse.Namespace, visor: dict[str, Any]) -> dict[str, Any]:
    rows = [{"name": item["name"], "time_rule_index": item["index"], "mapping_minus_one_index": item["mapped_index"],
             "best_index": item["best_index"], "mae_vs_released_jpeg": item["mae"],
             "mask_pixels": {k: int(v.sum()) for k, v in item["masks"].items()}, "contacts": item["contacts"]}
            for item in visor["sparse"]]
    checks = {
        "frames_decoded": len(visor["clip"]) == args.frames and len(rows) >= 1,
        # The video repeats frames in pairs, so the rule may land on the twin of the best frame.
        "time_rule_matches_released_frames": all(r["mae_vs_released_jpeg"]["time_rule"] < r["mae_vs_released_jpeg"]["best"] + 0.5 for r in rows),
        "hand_masks_nonempty": all(r["mask_pixels"][h] > 0 for r in rows for h in HANDS),
        "resolution_1080p": visor["info"]["width"] == 1920 and visor["info"]["height"] == 1080,
    }
    return {"device": "cpu", "video": visor["video_id"], "video_info": visor["info"],
            "decode_seconds": visor["decode_seconds"], "sparse_frames": rows, "checks": checks}


def clip_jpegs(visor: dict[str, Any], directory: Path) -> Path:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(visor["clip"]):
        Image.fromarray(frame).save(directory / f"{index:05d}.jpg", quality=95)
    return directory


# ----------------------------------------------------------------- segmentation

def c_sam31(args: argparse.Namespace, visor: dict[str, Any]) -> dict[str, Any]:
    import torch

    from src.segmentation.sam31 import DEFAULT_CHECKPOINT_SHA256, Sam31SequenceSegmenter

    device = device_record(args.gpu_class)
    t0 = time.time()
    segmenter = Sam31SequenceSegmenter(checkpoint_path=args.sam31_checkpoint, checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256)
    load_seconds = time.time() - t0
    frames_dir = clip_jpegs(visor, visor["work"] / "sam31-frames")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    clip, kernels = profile_cuda(lambda: segmenter.segment_frames(frames_dir, {"hand": "hand"}, policy="offline_causal"))
    run_seconds = time.time() - t0
    first = visor["sparse"][0]
    truth = first["masks"]["left hand"] | first["masks"]["right hand"]
    predicted = np.zeros_like(truth)
    for instance in clip.frames[0]:
        predicted |= instance.mask().astype(bool)
    score = iou(predicted, truth)
    model = segmenter._require_predictor().model
    expected = "flash" if device["capability"][0] >= 8 else "mem_efficient"
    checks = {
        "claimed_gpu_class": device["class_matches"] and device["device_count"] == 1,
        "model_on_cuda": model_on_cuda(model),
        "cuda_kernels_ran": kernels["kernel_launches"] > 0,
        "attention_family_expected": expected in kernels["attention_family"],
        "pinned_revision": segmenter.model_revision == "2345a4ad109ac29c569da749c91d84f10dc08c40",
        "hand_iou_vs_visor_frame0_gt_0.3": score > 0.3,
        "frames_masked": len(clip) == len(visor["clip"]),
    }
    return {"device": device, "kernels": kernels, "expected_attention": expected,
            "sdpa_backend_policy": segmenter.sdpa_backend_policy, "checkpoint_sha256": segmenter.checkpoint_hash,
            "load_seconds": round(load_seconds, 2), "run_seconds": round(run_seconds, 2),
            "frames": len(clip), "hand_iou_frame0": round(score, 4),
            "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1), "checks": checks}


class _HandDomain:
    classes = ("hand",)

    def prompts_for(self, backend: str) -> dict[str, str]:
        return {"hand": "hand"}

    def options_for(self, backend: str) -> dict[str, Any]:
        return {}


def c_yoloe(args: argparse.Namespace, visor: dict[str, Any]) -> dict[str, Any]:
    import torch

    from src.segmentation.yoloe import YoloeSegmenter

    device = device_record(args.gpu_class)
    models = visor["work"] / "models"
    (models / "YOLO").mkdir(parents=True, exist_ok=True)
    link = models / "YOLO" / "mobileclip2_b.ts"
    if not link.exists():
        link.symlink_to(args.yoloe_text_encoder)
    os.environ["PS_MODELS_ROOT"] = str(models)
    segmenter = YoloeSegmenter("x", weights=args.yoloe_weights)
    t0 = time.time()
    model = segmenter.load(_HandDomain())
    load_seconds = time.time() - t0
    first = visor["sparse"][0]
    bgr = np.ascontiguousarray(first["frame"][..., ::-1])
    torch.cuda.reset_peak_memory_stats()
    results, kernels = profile_cuda(lambda: model.predict(source=bgr, imgsz=1088, conf=0.25, retina_masks=True,
                                                          half=True, device=0, verbose=False))
    t0 = time.time()
    for frame in visor["clip"]:
        model.predict(source=np.ascontiguousarray(frame[..., ::-1]), imgsz=1088, conf=0.25, retina_masks=True,
                      half=True, device=0, verbose=False)
    torch.cuda.synchronize()
    per_frame_ms = (time.time() - t0) / len(visor["clip"]) * 1000
    truth = first["masks"]["left hand"] | first["masks"]["right hand"]
    predicted = np.zeros_like(truth)
    result = results[0]
    if result.masks is not None:
        for mask in result.masks.data.cpu().numpy() > 0.5:
            if mask.shape == truth.shape:
                predicted |= mask
    score = iou(predicted, truth)
    torch_model = model.model
    checks = {
        "claimed_gpu_class": device["class_matches"] and device["device_count"] == 1,
        "model_on_cuda": model_on_cuda(torch_model),
        "cuda_kernels_ran": kernels["kernel_launches"] > 0,
        "weights_from_staged_path": str(segmenter.weights_path) == str(args.yoloe_weights),
        "hand_iou_vs_visor_frame0_gt_0.3": score > 0.3,
    }
    import ultralytics

    return {"device": device, "kernels": kernels, "ultralytics": ultralytics.__version__,
            "weights_sha256": sha256_file(Path(args.yoloe_weights)), "load_seconds": round(load_seconds, 2),
            "per_frame_ms_1088": round(per_frame_ms, 1), "detections_frame0": int(len(result.boxes or [])),
            "hand_iou_frame0": round(score, 4), "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "checks": checks}


# ----------------------------------------------------------------- codecs

def _tool(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"{name} is not on PATH")
    flag = {"SvtAv1EncApp": "--version", "dav1d": "--version", "ffmpeg": "-version"}[name]
    out = subprocess.run([path, flag], capture_output=True, text=True, timeout=30)
    first = (out.stdout or out.stderr).strip().splitlines()[0]
    return {"path": path, "real_path": os.path.realpath(path), "sha256": sha256_file(Path(os.path.realpath(path))), "version": first}


def read_y4m(path: Path) -> list[np.ndarray]:
    """Luma planes of a yuv420p Y4M file."""
    data = path.read_bytes()
    header, rest = data.split(b"\n", 1)
    fields = dict((f[:1].decode(), f[1:].decode()) for f in header.split()[1:])
    width, height = int(fields["W"]), int(fields["H"])
    if not fields.get("C", "420").startswith("420"):
        raise ValueError(f"unexpected chroma {fields.get('C')}")
    size = width * height * 3 // 2
    planes, offset = [], 0
    while offset < len(rest):
        line_end = rest.index(b"\n", offset)
        offset = line_end + 1
        frame = np.frombuffer(rest, dtype=np.uint8, count=size, offset=offset)
        planes.append(frame[: width * height].reshape(height, width))
        offset += size
    return planes


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def c_svtav1(args: argparse.Namespace, visor: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    work = visor["work"] / "svtav1"
    work.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(visor["clip"]):
        Image.fromarray(frame).save(work / f"{index:05d}.png")
    tools = {name: _tool(name) for name in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    source = work / "source.y4m"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(visor["info"]["fps"]), "-i", str(work / "%05d.png"),
                    "-pix_fmt", "yuv420p", "-strict", "-1", str(source)], check=True, timeout=120)
    stream, decoded = work / "out.ivf", work / "decoded.y4m"
    t0 = time.time()
    subprocess.run([tools["SvtAv1EncApp"]["path"], "-i", str(source), "-b", str(stream), "--preset", "8", "--crf", "35",
                    "--keyint", "-1"], check=True, capture_output=True, timeout=300)
    encode_seconds = time.time() - t0
    subprocess.run([tools["dav1d"]["path"], "-i", str(stream), "-o", str(decoded), "--quiet"], check=True, timeout=120)
    ref, dec = read_y4m(source), read_y4m(decoded)
    values = [psnr(a, b) for a, b in zip(ref, dec)]
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    decoders = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True).stdout
    bits = stream.stat().st_size * 8
    checks = {
        "decoded_frame_count": len(dec) == len(ref) == args.frames,
        "luma_psnr_above_30": min(values) > 30.0,
        "ffmpeg_has_libsvtav1": "libsvtav1" in encoders,
        "ffmpeg_has_libdav1d": "libdav1d" in decoders,
    }
    return {"device": "cpu", "tools": tools, "preset": 8, "crf": 35, "stream_bytes": stream.stat().st_size,
            "bpp": round(bits / (len(ref) * ref[0].size), 5), "luma_psnr": [round(v, 2) for v in values],
            "encode_seconds": round(encode_seconds, 2), "checks": checks}


def c_dcvc(args: argparse.Namespace, visor: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    from src.codecs.dcvc_uf_worker import dcvc_command

    device = device_record(args.gpu_class)
    work = visor["work"] / "dcvc"
    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(visor["clip"]):
        Image.fromarray(frame).save(frames_dir / f"im{index + 1:05d}.png")
    out_dir = work / "decoded"
    out_dir.mkdir()
    height, width = visor["clip"][0].shape[:2]
    plan = {"structure": "hts", "qp": 32, "frame_count": len(visor["clip"]), "height": height, "width": width,
            "frames_dir": str(frames_dir), "container": str(work / "stream.psdc"), "out_dir": str(out_dir),
            "image_ckpt": args.dcvc_image, "image_sha256": sha256_file(Path(args.dcvc_image)),
            "video_ckpt": args.dcvc_video, "video_sha256": sha256_file(Path(args.dcvc_video)), "profile": True}
    plan_path = work / "plan.json"
    write_json(plan_path, plan)
    reports = {}
    for action in ("encode", "decode"):
        report = work / f"{action}.json"
        command, env, cwd = dcvc_command(Path(sys.prefix), action, plan_path, report)
        done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, capture_output=True, text=True, timeout=COMPONENT_SECONDS)
        if done.returncode:
            raise RuntimeError(f"DCVC {action} failed ({done.returncode}):\n{done.stderr[-4000:]}")
        reports[action] = json.loads(report.read_text())
    decoded = [np.load(out_dir / f"{i:05d}.npy") for i in range(len(visor["clip"]))]
    values = [psnr(a, b) for a, b in zip(visor["clip"], decoded)]
    enc, dec = reports["encode"], reports["decode"]
    kernels = enc.get("kernels", {})
    checks = {
        "claimed_gpu_class": device["class_matches"] and device["device_count"] == 1,
        "worker_on_claimed_gpu": enc["environment"]["cuda_visible_devices"] == device["cuda_visible_devices"],
        "extension_variant_for_class": enc["environment"]["extension"]["variant"] == ("sm89" if device["capability"] == [8, 9] else "sm80"),
        "cuda_kernels_ran": kernels.get("kernel_launches", 0) > 0,
        "decoded_frame_count": len(decoded) == len(visor["clip"]),
        "decoder_matches_encoder_intra": dec["passes"][0]["rgb_sha256"][0] == enc["i_recon_rgb_sha256"],
        "decode_deterministic": bool(dec["deterministic"]),
        "rgb_psnr_above_28": min(values) > 28.0,
    }
    bits = enc["container_bytes"] * 8
    return {"device": device, "kernels": kernels, "encode": {k: v for k, v in enc.items() if k != "kernels"},
            "decode": dec, "rgb_psnr": [round(v, 2) for v in values],
            "bpp": round(bits / (len(decoded) * height * width), 5), "checks": checks}


# ----------------------------------------------------------------- hands

def _crop_batch(dataset_cls: Any, cfg: Any, visor_item: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    import torch

    boxes, rights, meta = [], [], []
    for side in HANDS:
        box = bbox(visor_item["masks"][side])
        if box is not None:
            boxes.append(box)
            rights.append(1.0 if side == "right hand" else 0.0)
            meta.append({"side": side, "box": box})
    bgr = np.ascontiguousarray(visor_item["frame"][..., ::-1])
    dataset = dataset_cls(cfg, bgr, np.array(boxes, dtype=np.float32), np.array(rights, dtype=np.float32), rescale_factor=2.0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(boxes), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in batch.items()}, meta


def _keypoints_check(output: dict[str, Any], batch: dict[str, Any], meta: list[dict[str, Any]], visor_item: dict[str, Any]) -> dict[str, Any]:
    """Fraction of the 21 projected keypoints inside each hand's VISOR mask box (padded 25%)."""
    kp = output["pred_keypoints_2d"].detach().float().cpu().numpy().copy()
    right = batch["right"].cpu().numpy()
    centre = batch["box_center"].cpu().numpy()
    size = batch["box_size"].cpu().numpy()
    rows = []
    for i, info in enumerate(meta):
        points = kp[i].copy()
        points[:, 0] *= 2 * right[i] - 1
        points = centre[i][None] + points * size[i]
        x0, y0, x1, y1 = info["box"]
        pad_x, pad_y = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
        inside = ((points[:, 0] >= x0 - pad_x) & (points[:, 0] <= x1 + pad_x) & (points[:, 1] >= y0 - pad_y) & (points[:, 1] <= y1 + pad_y))
        rows.append({"side": info["side"], "box": info["box"], "inside_fraction": round(float(inside.mean()), 3)})
    return {"hands": rows, "frame": visor_item["name"]}


def _hand_model(kind: str, args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    if kind == "hamer":
        from hamer.configs import get_config
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.models import HAMER as Model

        cfg = get_config(args.hamer_config, update_cachedir=True)
        checkpoint, mean_params = args.hamer_checkpoint, args.hamer_mean_params
    else:
        from wilor.configs import get_config
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.models import WiLoR as Model

        tree = Path(sys.prefix) / "opt" / "WiLoR"
        cfg = get_config(str(tree / "pretrained_models" / "model_config.yaml"), update_cachedir=True)
        checkpoint, mean_params = args.wilor_checkpoint, str(tree / "mano_data" / "mano_mean_params.npz")
    cfg.defrost()
    if "BBOX_SHAPE" not in cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    cfg.MANO.MODEL_PATH = args.mano_dir
    cfg.MANO.MEAN_PARAMS = mean_params
    cfg.MANO.DATA_DIR = str(Path(mean_params).parent)
    cfg.freeze()
    model = Model(cfg, init_renderer=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(("discriminator", "mano."))]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(("discriminator",))]
    model = model.cuda().eval()
    load = {"checkpoint_sha256": sha256_file(Path(checkpoint)), "missing_keys": missing[:20], "missing_count": len(missing),
            "unexpected_keys": unexpected[:20], "unexpected_count": len(unexpected), "ignored_prefixes": ["discriminator", "mano."]}
    return model, cfg, ViTDetDataset, load


def _hand_component(kind: str, args: argparse.Namespace, visor: dict[str, Any]) -> dict[str, Any]:
    import torch

    device = device_record(args.gpu_class)
    t0 = time.time()
    model, cfg, dataset_cls, load = _hand_model(kind, args)
    load_seconds = time.time() - t0
    rows = []
    torch.cuda.reset_peak_memory_stats()
    kernels: dict[str, Any] = {}
    for position, item in enumerate(visor["sparse"]):
        batch, meta = _crop_batch(dataset_cls, cfg, item)
        with torch.no_grad():
            if position == 0:
                output, kernels = profile_cuda(lambda: model(batch))
            else:
                output = model(batch)
        rows.append(_keypoints_check(output, batch, meta, item))
        mano = output["pred_mano_params"]
        rows[-1]["mano_param_shapes"] = {k: list(v.shape) for k, v in mano.items()}
    fractions = [h["inside_fraction"] for r in rows for h in r["hands"]]
    checks = {
        "claimed_gpu_class": device["class_matches"] and device["device_count"] == 1,
        "model_on_cuda": model_on_cuda(model),
        "cuda_kernels_ran": kernels.get("kernel_launches", 0) > 0,
        "checkpoint_keys_complete": load["missing_count"] == 0 and load["unexpected_count"] == 0,
        "mano_pose_output": all(r["mano_param_shapes"].get("hand_pose", [0])[0] >= 1 for r in rows),
        "median_keypoints_inside_visor_hand_box_0.8": float(np.median(fractions)) >= 0.8,
    }
    extra: dict[str, Any] = {}
    if kind == "wilor":
        extra["detector"] = _wilor_detector(args, visor)
        checks["detector_loads_and_finds_hands"] = extra["detector"]["detections"] >= 1
    return {"device": device, "kernels": kernels, "load": load, "load_seconds": round(load_seconds, 2),
            "frames": rows, "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1), "checks": checks, **extra}


def _wilor_detector(args: argparse.Namespace, visor: dict[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    detector = YOLO(args.wilor_detector)
    item = visor["sparse"][0]
    result = detector.predict(np.ascontiguousarray(item["frame"][..., ::-1]), conf=0.3, device=0, verbose=False)[0]
    classes = [result.names[int(c)] for c in result.boxes.cls.cpu().numpy()] if result.boxes is not None else []
    return {"weights_sha256": sha256_file(Path(args.wilor_detector)), "detections": len(classes), "classes": classes,
            "trained_with": "ultralytics 8.1.34 (WiLoR requirements); loaded by the environment's ultralytics"}


def c_hamer(args: argparse.Namespace, visor: dict[str, Any]) -> dict[str, Any]:
    return _hand_component("hamer", args, visor)


def c_wilor(args: argparse.Namespace, visor: dict[str, Any]) -> dict[str, Any]:
    return _hand_component("wilor", args, visor)


def c_hot3d(args: argparse.Namespace, visor: dict[str, Any]) -> dict[str, Any]:
    """Render HOT3D MANO hands through the clip's fisheye camera and compare with the labelled boxes."""
    from PIL import Image

    from hand_tracking_toolkit import camera, visualization
    from hand_tracking_toolkit.dataset import HandShapeCollection, decode_hand_pose
    from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
    from hand_tracking_toolkit.rasterizer import rasterize_mesh
    import torch

    mano = MANOHandModel(args.mano_dir)
    rows = []
    with tarfile.open(args.hot3d_clip) as tar:
        shapes = json.load(tar.extractfile("__hand_shapes.json__"))  # type: ignore[arg-type]
        shape = HandShapeCollection(mano_beta=torch.tensor(shapes["mano"]), umetrack=None)
        for frame_key in ("000000", "000075"):
            cameras = json.load(tar.extractfile(f"{frame_key}.cameras.json"))  # type: ignore[arg-type]
            cam = camera.from_json(cameras["214-1"])
            hands = json.load(tar.extractfile(f"{frame_key}.hands.json"))  # type: ignore[arg-type]
            image = np.array(Image.open(tar.extractfile(f"{frame_key}.image_214-1.jpg")).convert("RGB"))  # type: ignore[arg-type]
            for side, pose in decode_hand_pose(hands).items():
                _, verts, faces = visualization.get_keypoints_and_mesh(hand_pose=pose, hand_shape=shape, mano_model=mano, pose_type="mano")
                _, mask, _ = rasterize_mesh(verts=np.asarray(verts), faces=np.asarray(faces), camera=cam)
                mask = np.asarray(mask) > 0
                height, width = mask.shape
                raw = hands[side.value]["boxes_amodal"].get("214-1")
                # Amodal boxes may extend past the image; the rendered mask cannot.
                labelled = None if raw is None else [max(0.0, raw[0]), max(0.0, raw[1]), min(float(width), raw[2]), min(float(height), raw[3])]
                rendered = bbox(mask)
                box_iou = None
                if labelled is not None and rendered is not None:
                    a = np.zeros(mask.shape, bool)
                    b = np.zeros(mask.shape, bool)
                    a[rendered[1]:rendered[3], rendered[0]:rendered[2]] = True
                    lx0, ly0, lx1, ly1 = (int(round(v)) for v in labelled)
                    b[ly0:ly1, lx0:lx1] = True
                    box_iou = iou(a, b)
                rows.append({"frame": frame_key, "side": side.value, "mask_pixels": int(mask.sum()),
                             "rendered_box": rendered, "labelled_box_amodal": labelled,
                             "box_iou": None if box_iou is None else round(box_iou, 3),
                             "image_shape": list(image.shape), "camera": type(cam).__name__})
    ious = [r["box_iou"] for r in rows if r["box_iou"] is not None]
    checks = {
        "fisheye_camera": all(r["camera"] != "PinholePlaneCameraModel" for r in rows),
        "hands_rendered": len(rows) >= 2 and all(r["mask_pixels"] > 0 for r in rows),
        "rendered_box_matches_labelled_amodal_box_0.7": bool(ious) and min(ious) > 0.7,
    }
    return {"device": "cpu", "clip_sha256": sha256_file(Path(args.hot3d_clip)), "rows": rows, "checks": checks}


COMPONENTS: dict[str, Callable[[argparse.Namespace, dict[str, Any]], dict[str, Any]]] = {
    "visor": c_visor, "sam31": c_sam31, "yoloe": c_yoloe, "svtav1": c_svtav1, "dcvc": c_dcvc,
    "hamer": c_hamer, "wilor": c_wilor, "hot3d": c_hot3d,
}


# ----------------------------------------------------------------- driver

def run_component(args: argparse.Namespace) -> int:
    out = Path(os.environ["PS_STAGE_DIR"]) / "components" / f"{args.component}.json"
    started = time.time()
    record: dict[str, Any] = {"component": args.component, "gpu_class": args.gpu_class, "citable": False}
    try:
        work = scratch() / args.component
        work.mkdir(parents=True, exist_ok=True)
        if args.mano_left and args.mano_right:
            # Staged files keep their cache names; MANO loaders want one directory.
            mano = work / "mano"
            mano.mkdir(exist_ok=True)
            for name, source in (("MANO_LEFT.pkl", args.mano_left), ("MANO_RIGHT.pkl", args.mano_right)):
                if not (mano / name).exists():
                    (mano / name).symlink_to(Path(source).resolve())
            args.mano_dir = str(mano)
        visor = load_visor(Path(args.visor_video), Path(args.visor_sample), args.frames, work)
        record.update(COMPONENTS[args.component](args, visor))
        record["checks"] = {name: bool(value) for name, value in record["checks"].items()}
        record["passed"] = bool(record["checks"]) and all(record["checks"].values())
    except Exception as exc:
        record.update(passed=False, error=repr(exc), traceback=traceback.format_exc()[-6000:])
    record["seconds"] = round(time.time() - started, 2)
    write_json(out, record)
    return 0 if record["passed"] else 1


def run_group(args: argparse.Namespace, argv: list[str]) -> int:
    from experiments.jobs.monitor import publish_progress

    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    environment = {
        "python": sys.executable, "prefix": sys.prefix,
        "opt_provenance": json.loads((Path(sys.prefix) / "opt" / "PROVENANCE.json").read_text())
        if (Path(sys.prefix) / "opt" / "PROVENANCE.json").exists() else None,
        "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True).stdout.splitlines(),
    }
    write_json(stage_dir / "environment.json", environment)
    status: dict[str, int | str] = {}
    # Ultralytics pip-installs missing packages at run time unless told not to;
    # the environment must already hold everything.
    child_env = {**os.environ, "YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "true"}
    for done, component in enumerate(GROUPS[args.group], start=1):
        command = [sys.executable, "-m", "experiments.audit.env_smoke", "component", component, *argv]
        try:
            result = subprocess.run(command, timeout=COMPONENT_SECONDS, capture_output=True, text=True, env=child_env)
            status[component] = result.returncode
            (stage_dir / "components").mkdir(exist_ok=True)
            (stage_dir / "components" / f"{component}.log").write_text(result.stdout[-20000:] + "\n--- stderr\n" + result.stderr[-20000:])
        except subprocess.TimeoutExpired:
            status[component] = "timeout"
        publish_progress(os.environ.get("PS_STAGE", "smoke"), done)
    write_json(stage_dir / "result.json", {"group": args.group, "gpu_class": args.gpu_class, "status": status, "citable": False})
    return 0


def validate(args: argparse.Namespace) -> int:
    from experiments.jobs.monitor import write_json as publish

    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    checks: dict[str, bool] = {}
    for component in GROUPS[args.group]:
        path = stage_dir / "components" / f"{component}.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        for name, value in (record.get("checks") or {"ran": False}).items():
            checks[f"{component}.{name}"] = bool(value)
        checks[f"{component}.passed"] = bool(record.get("passed"))
        if record.get("device") not in (None, "cpu"):
            checks[f"{component}.gpu_class"] = args.gpu_class in record["device"]["name"]
    passed = bool(checks) and all(checks.values())
    publish(Path(os.environ["PS_VALIDATION_PATH"]), {"passed": passed, "checks": checks, "citable": False})
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("run", "component", "validate"))
    parser.add_argument("component", nargs="?")
    parser.add_argument("--group", choices=sorted(GROUPS))
    parser.add_argument("--gpu-class", required=True)
    parser.add_argument("--frames", type=int, default=9)
    for name in ("visor-video", "visor-sample", "sam31-checkpoint", "yoloe-weights", "yoloe-text-encoder",
                 "dcvc-image", "dcvc-video", "hamer-checkpoint", "hamer-config", "hamer-mean-params",
                 "wilor-checkpoint", "wilor-detector", "mano-left", "mano-right", "hot3d-clip"):
        parser.add_argument(f"--{name}")
    args = parser.parse_args(argv)
    args.mano_dir = None
    if args.mode == "validate":
        return validate(args)
    if args.mode == "component":
        return run_component(args)
    rest = [a for a in argv if a != "run"]
    return run_group(args, rest)


if __name__ == "__main__":
    raise SystemExit(main())
