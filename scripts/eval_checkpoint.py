"""Fast, cheap checkpoint-quality signal for the G2 training campaign (report 10, Phase 5.4).

Given a checkpoint + the probe-set manifest written by `scripts/select_probe_set.py`,
runs inference on every probe clip's conditioning frames (skeleton/canny), scores
the generated frames against ground truth with PSNR/SSIM/VMAF/FVD (+ an
uncalibrated VGG "LPIPS-like" distance, see `src/shared/lpips_metric.py`), and
appends one JSONL record to an on-disk log so a human (or `scripts/train_campaign.py`)
can plot a curve across steps without waiting for a full training run to finish.

Architectures supported (dispatch table at the bottom, `ARCH_INFERENCE`):
  - pix2pix       -> scripts/train_pix2pix.py's UNetGenerator (skeleton+ref -> color)
  - spade4tennis  -> scripts/train_spade4tennis.py's SPADEResNet9Generator (same I/O)
  - controlnet    -> scripts/train_controlnet.py's ControlNetModel + base SD1.5 pipeline
                     (heavy: full diffusion sampling per frame; only exercised for
                     real ControlNet checkpoints, not needed for the pix2pix/spade4tennis
                     smoke path)

Metrics are computed **per probe clip** (not concatenated across clips) so FVD's
I3D temporal window never straddles a seam between two unrelated clips; the
per-clip scores are then averaged into one aggregate record. This keeps the
per-clip breakdown available for debugging while still writing one simple
number per metric per step to the log.

Usage
-----
    conda run -n pointstream python scripts/eval_checkpoint.py \\
        --checkpoint assets/weights/pix2pix_generator.pt --arch pix2pix \\
        --manifest assets/probe_set/manifest.json --dataset-root assets/dataset \\
        --step 1000 --variant pix2pix --log outputs/campaign/pix2pix_probe_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.encoder.video_io import encode_video_frames_ffmpeg
from src.experiment_evaluation import _compute_fvd, _compute_psnr, _compute_ssim_ffmpeg, _compute_vmaf_ffmpeg
from src.shared.lpips_metric import compute_lpips_from_frames

_FRAME_ID_RE = re.compile(r"frame_(\d+)\.png$")

DEFAULT_METRICS = ("psnr", "ssim", "vmaf", "fvd")
ARCH_CHOICES = ("pix2pix", "spade4tennis", "controlnet")


# ---------------------------------------------------------------------------
# Manifest / dataset path resolution (pure, fast, unit-testable without models)
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def clip_color_dir(dataset_root: Path, clip: dict) -> Path:
    return dataset_root / clip["video"] / "segmentations" / clip["scene"] / clip["track"]


def clip_condition_dir(dataset_root: Path, clip: dict, condition: str) -> Path:
    suffix = {"skeleton": "_skeleton", "canny": "_canny"}[condition]
    return dataset_root / clip["video"] / "segmentations" / clip["scene"] / f"{clip['track']}{suffix}"


def clip_frame_paths(directory: Path, frame_ids: list[int]) -> list[Path]:
    return [directory / f"frame_{fid:06d}.png" for fid in frame_ids]


def resolve_reference_frame_path(dataset_root: Path, clip: dict) -> Path:
    """Deterministic reference frame: the earliest frame in the *whole track*
    (not just the probe window), preferring one outside the probe window so
    the reference doesn't trivially leak the target frame. Falls back to the
    window's own first frame if the track has nothing else."""
    color_dir = clip_color_dir(dataset_root, clip)
    all_frame_ids = sorted(
        int(m.group(1)) for f in color_dir.glob("frame_*.png") if (m := _FRAME_ID_RE.search(f.name)) is not None
    )
    window_ids = set(clip["frame_ids"])
    outside_window = [fid for fid in all_frame_ids if fid not in window_ids]
    chosen = outside_window[0] if outside_window else clip["frame_ids"][0]
    return color_dir / f"frame_{chosen:06d}.png"


# ---------------------------------------------------------------------------
# Image <-> tensor helpers
# ---------------------------------------------------------------------------


def load_image_rgb01(path: Path, size: int) -> torch.Tensor:
    """Load, pad to square (black fill), resize, return [3, size, size] in [0, 1]."""
    from PIL import Image

    from src.shared.tennis_dataset import pad_to_square

    img = Image.open(path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    else:
        img = img.convert("RGB")
    img = pad_to_square(img, fill=0)
    img = img.resize((size, size), resample=Image.BILINEAR)
    array = np.asarray(img, dtype=np.float32) / 255.0  # Shape: [H, W, 3]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()  # Shape: [3, H, W]


def load_clip_tensor(paths: list[Path], size: int) -> torch.Tensor:
    """paths -> [N, 3, size, size] float32 in [0, 1]."""
    return torch.stack([load_image_rgb01(p, size) for p in paths], dim=0)


def rgb01_to_bgr_uint8_frames(frames_rgb01: torch.Tensor) -> list[np.ndarray]:
    """[N, 3, H, W] in [0,1] -> list of [H, W, 3] BGR uint8 arrays for encode_video_frames_ffmpeg."""
    frames_np = (frames_rgb01.clamp(0, 1) * 255.0).round().byte().numpy()  # Shape: [N, 3, H, W]
    out = []
    for frame in frames_np:
        hwc_rgb = np.transpose(frame, (1, 2, 0))  # Shape: [H, W, 3] RGB
        hwc_bgr = hwc_rgb[:, :, ::-1].copy()
        out.append(hwc_bgr)
    return out


# ---------------------------------------------------------------------------
# Architecture inference dispatch (heavy — loads real checkpoints/models)
# ---------------------------------------------------------------------------


def run_pix2pix_inference(  # pragma: no cover - requires real checkpoint + GPU, exercised by integration/smoke run
    checkpoint_path: Path,
    condition_frames: torch.Tensor,
    reference_frame: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """condition_frames/reference_frame in [0,1]; UNetGenerator expects [-1,1]. Returns [N,3,H,W] in [0,1]."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_pix2pix_module", Path(__file__).with_name("train_pix2pix.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = module.UNetGenerator(in_channels=6, out_channels=3).to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()

    skeleton = (condition_frames.to(device) - 0.5) * 2.0  # Shape: [N, 3, H, W]
    ref = (reference_frame.to(device) - 0.5) * 2.0  # Shape: [3, H, W]
    ref_batch = ref.unsqueeze(0).expand(skeleton.shape[0], -1, -1, -1)  # Shape: [N, 3, H, W]

    with torch.no_grad():
        gen_input = torch.cat((skeleton, ref_batch), dim=1)  # Shape: [N, 6, H, W]
        predicted = generator(gen_input)  # Shape: [N, 3, H, W] in [-1, 1]

    return ((predicted.clamp(-1, 1) + 1.0) / 2.0).cpu()  # Shape: [N, 3, H, W] in [0, 1]


def run_spade4tennis_inference(  # pragma: no cover - requires real checkpoint + GPU, exercised by integration/smoke run
    checkpoint_path: Path,
    condition_frames: torch.Tensor,
    reference_frame: torch.Tensor,
    device: str,
) -> torch.Tensor:
    from src.shared.spade4tennis_arch import SPADEResNet9Generator

    generator = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=64, n_blocks=9).to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    generator.load_state_dict(state_dict, strict=False)
    generator.eval()

    skeleton = (condition_frames.to(device) - 0.5) * 2.0
    ref = (reference_frame.to(device) - 0.5) * 2.0
    ref_batch = ref.unsqueeze(0).expand(skeleton.shape[0], -1, -1, -1)

    with torch.no_grad():
        predicted = generator(skeleton, ref_batch)  # Shape: [N, 3, H, W] in [-1, 1]

    return ((predicted.clamp(-1, 1) + 1.0) / 2.0).cpu()


def run_controlnet_inference(  # pragma: no cover - requires real SD1.5 + ControlNet checkpoint + GPU
    checkpoint_path: Path,
    condition_frames: torch.Tensor,
    reference_frame: torch.Tensor,  # unused (ControlNet is not reference-conditioned); kept for a uniform signature
    device: str,
    base_model_id: str = "assets/weights/stable-diffusion-v1-5",
    prompt: str = "photorealistic tennis player, broadcast sports shot",
    num_inference_steps: int = 20,
    generator_seed: int = 0,
) -> torch.Tensor:
    """Runs the diffusers ControlNet pipeline frame-by-frame. Slow (full diffusion
    sampling per frame) — this is why the campaign smoke test in this workstream
    exercises pix2pix/spade4tennis instead, not ControlNet, for wall-clock reasons."""
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

    controlnet = ControlNetModel.from_pretrained(str(checkpoint_path)).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id, controlnet=controlnet, safety_checker=None
    ).to(device)

    outputs = []
    for i in range(condition_frames.shape[0]):
        cond_img = condition_frames[i].permute(1, 2, 0).numpy()  # Shape: [H, W, 3] in [0,1]
        from PIL import Image

        cond_pil = Image.fromarray((cond_img * 255).astype(np.uint8))
        gen = torch.Generator(device=device).manual_seed(generator_seed)
        result = pipe(prompt, image=cond_pil, num_inference_steps=num_inference_steps, generator=gen).images[0]
        result_array = np.asarray(result, dtype=np.float32) / 255.0  # Shape: [H, W, 3]
        outputs.append(torch.from_numpy(result_array).permute(2, 0, 1))

    return torch.stack(outputs, dim=0)  # Shape: [N, 3, H, W] in [0, 1]


ARCH_INFERENCE: dict[str, Callable[..., torch.Tensor]] = {
    "pix2pix": run_pix2pix_inference,
    "spade4tennis": run_spade4tennis_inference,
    "controlnet": run_controlnet_inference,
}

ARCH_CONDITION: dict[str, str] = {
    "pix2pix": "skeleton",
    "spade4tennis": "skeleton",
    "controlnet": "skeleton",  # overridden by --controlnet-condition-type
}


# ---------------------------------------------------------------------------
# Metric scoring
# ---------------------------------------------------------------------------


def compute_metrics_for_clip(
    ground_truth_rgb01: torch.Tensor,
    predicted_rgb01: torch.Tensor,
    fps: float,
    metrics: tuple[str, ...],
    include_lpips: bool,
    tmp_dir: Path,
) -> dict[str, Any]:
    """Builds two tiny mp4s for one clip and scores them with the requested metrics."""
    height, width = ground_truth_rgb01.shape[-2:]
    ref_path = tmp_dir / "ref.mp4"
    pred_path = tmp_dir / "pred.mp4"
    ref_path.unlink(missing_ok=True)
    pred_path.unlink(missing_ok=True)

    encode_video_frames_ffmpeg(
        ref_path, rgb01_to_bgr_uint8_frames(ground_truth_rgb01), fps=fps, width=width, height=height
    )
    encode_video_frames_ffmpeg(
        pred_path, rgb01_to_bgr_uint8_frames(predicted_rgb01), fps=fps, width=width, height=height
    )

    result: dict[str, Any] = {}
    if "psnr" in metrics:
        result.update(_compute_psnr(ref_path, pred_path))
    if "ssim" in metrics:
        result.update(_compute_ssim_ffmpeg(ref_path, pred_path))
    if "vmaf" in metrics:
        result.update(_compute_vmaf_ffmpeg(ref_path, pred_path))
    if "fvd" in metrics:
        result.update(_compute_fvd(ref_path, pred_path))
    if include_lpips:
        result.update(compute_lpips_from_frames(ground_truth_rgb01, predicted_rgb01))

    return result


_AGGREGATE_NUMERIC_KEYS = (
    "psnr_mean",
    "ssim_mean",
    "vmaf_mean",
    "fvd",
    "lpips_vgg_uncalibrated",
)


def aggregate_clip_metrics(per_clip: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of each numeric metric across clips, skipping None/missing entries."""
    aggregate: dict[str, Any] = {}
    for key in _AGGREGATE_NUMERIC_KEYS:
        values = [clip[key] for clip in per_clip if clip.get(key) is not None]
        aggregate[key] = float(sum(values) / len(values)) if values else None
    aggregate["num_clips_scored"] = sum(1 for clip in per_clip if any(clip.get(k) is not None for k in _AGGREGATE_NUMERIC_KEYS))
    aggregate["num_clips_total"] = len(per_clip)
    return aggregate


def append_jsonl_log(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(record, sort_keys=False) + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_path: Path,
    arch: str,
    manifest: dict,
    dataset_root: Path,
    img_size: int,
    device: str,
    metrics: tuple[str, ...],
    include_lpips: bool,
    fps: float,
    condition_type: str | None = None,
    arch_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Runs inference + scoring for every probe clip in `manifest`; returns per-clip and aggregate metrics."""
    arch_kwargs = arch_kwargs or {}
    condition = condition_type or ARCH_CONDITION[arch]
    infer_fn = ARCH_INFERENCE[arch]

    per_clip_metrics = []
    with tempfile.TemporaryDirectory(prefix="eval_checkpoint_") as tmp:
        tmp_dir = Path(tmp)
        for clip in manifest["probe_clips"]:
            color_paths = clip_frame_paths(clip_color_dir(dataset_root, clip), clip["frame_ids"])
            cond_paths = clip_frame_paths(clip_condition_dir(dataset_root, clip, condition), clip["frame_ids"])
            ref_path = resolve_reference_frame_path(dataset_root, clip)

            ground_truth = load_clip_tensor(color_paths, img_size)  # Shape: [N, 3, size, size]
            condition_tensor = load_clip_tensor(cond_paths, img_size)  # Shape: [N, 3, size, size]
            reference_tensor = load_image_rgb01(ref_path, img_size)  # Shape: [3, size, size]

            predicted = infer_fn(checkpoint_path, condition_tensor, reference_tensor, device, **arch_kwargs)

            clip_metrics = compute_metrics_for_clip(
                ground_truth, predicted, fps=fps, metrics=metrics, include_lpips=include_lpips, tmp_dir=tmp_dir
            )
            clip_metrics["clip_key"] = f"{clip['video']}/{clip['scene']}/{clip['track']}"
            per_clip_metrics.append(clip_metrics)

    return {"per_clip": per_clip_metrics, "aggregate": aggregate_clip_metrics(per_clip_metrics)}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arch", choices=ARCH_CHOICES, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("assets/dataset"))
    parser.add_argument("--step", type=int, required=True, help="Training step/epoch this checkpoint corresponds to")
    parser.add_argument("--variant", type=str, required=True, help="Candidate/variant name for the log (e.g. 'pix2pix')")
    parser.add_argument("--log", type=Path, required=True, help="JSONL log to append the scored record to")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--fps", type=float, default=24.0, help="Nominal fps for the tiny per-clip scoring videos")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--controlnet-condition-type", type=str, default="pose", choices=["pose", "canny"])
    parser.add_argument("--controlnet-base-model", type=str, default="assets/weights/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-num-inference-steps", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics = tuple(m.strip().lower() for m in args.metrics.split(",") if m.strip())

    manifest = load_manifest(args.manifest)

    arch_kwargs: dict[str, Any] = {}
    condition_type = None
    if args.arch == "controlnet":
        condition_type = "canny" if args.controlnet_condition_type == "canny" else "skeleton"
        arch_kwargs = {
            "base_model_id": args.controlnet_base_model,
            "num_inference_steps": args.controlnet_num_inference_steps,
        }

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        arch=args.arch,
        manifest=manifest,
        dataset_root=args.dataset_root,
        img_size=args.img_size,
        device=device,
        metrics=metrics,
        include_lpips=not args.skip_lpips,
        fps=args.fps,
        condition_type=condition_type,
        arch_kwargs=arch_kwargs,
    )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "step": args.step,
        "variant": args.variant,
        "arch": args.arch,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        **result["aggregate"],
    }
    append_jsonl_log(args.log, record)

    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
