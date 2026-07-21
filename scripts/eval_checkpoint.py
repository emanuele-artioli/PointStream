"""Fast, cheap checkpoint-quality signal for the G2 training campaign (report 10, Phase 5.4).

Given a checkpoint + the probe-set manifest written by `scripts/select_probe_set.py`,
runs inference on every probe clip, scores the generated frames against ground
truth with PSNR/SSIM/VMAF/FVD (+ an uncalibrated VGG "LPIPS-like" distance, see
`src/shared/lpips_metric.py`), and appends one JSONL record to an on-disk log so
a human (or `scripts/train_campaign.py`) can plot a curve across steps without
waiting for a full training run to finish.

**Inference runs the decoder's own strategy classes** (`build_eval_strategy` ->
`src.decoder.genai_compositor.build_genai_strategy`), configured through a real
`PointstreamConfig`. This is the Residual-Guarantee symmetry principle applied
to evaluation: one code path, or the number is fiction. It is not optional
hygiene -- this script previously reimplemented inference and diverged, scoring
ControlNet as text-to-image from pure noise (reference frame unused) while the
decoder ran img2img seeded from the reference crop. That divergence, not model
quality, produced the G2 campaign's PSNR 9.76 / VMAF 0.11.

Strategies render their own condition from **raw keypoints**, exactly as the
decoder does from an `ActorPacket`. The pre-rendered `_skeleton`/`_canny` PNG
directories are training-time artefacts the decoder never sees, so eval does not
feed them either.

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
from typing import Any

import numpy as np
import torch

from src.encoder.video_io import encode_video_frames_ffmpeg
from src.experiment_evaluation import _compute_fvd, _compute_psnr, _compute_ssim_ffmpeg, _compute_vmaf_ffmpeg
from src.shared.lpips_metric import compute_lpips_from_frames

_FRAME_ID_RE = re.compile(r"frame_(\d+)\.png$")

DEFAULT_METRICS = ("psnr", "ssim", "vmaf", "fvd")
ARCH_CHOICES = ("pix2pix", "spade4tennis", "controlnet", "multi-controlnet")


# ---------------------------------------------------------------------------
# Manifest / dataset path resolution (pure, fast, unit-testable without models)
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def clip_color_dir(dataset_root: Path, clip: dict) -> Path:
    return dataset_root / clip["video"] / "segmentations" / clip["scene"] / clip["track"]


def clip_condition_dir(dataset_root: Path, clip: dict, condition: str) -> Path:
    suffix = {"skeleton": "_skeleton", "canny": "_canny", "pose": "_skeleton"}[condition]
    return dataset_root / clip["video"] / "segmentations" / clip["scene"] / f"{clip['track']}{suffix}"


def clip_frame_paths(directory: Path, frame_ids: list[int]) -> list[Path]:
    return [directory / f"frame_{fid:06d}.png" for fid in frame_ids]


def clip_condition_frame_paths(dataset_root: Path, clip: dict, condition: str) -> list[Path]:
    cond_dir = clip_condition_dir(dataset_root, clip, condition)
    color_dir = clip_color_dir(dataset_root, clip)
    all_frame_ids = sorted(
        int(m.group(1)) for f in color_dir.glob("frame_*.png") if (m := _FRAME_ID_RE.search(f.name)) is not None
    )
    id_to_index = {fid: idx for idx, fid in enumerate(all_frame_ids)}
    all_cond_paths = sorted(cond_dir.glob("frame_*.png"))
    return [all_cond_paths[id_to_index[fid]] for fid in clip["frame_ids"]]


def clip_track_index(dataset_root: Path, clip: dict) -> dict[int, int]:
    """Map absolute frame id -> positional index within the track.

    The three per-track artefacts use *different* id conventions and must be
    joined positionally, never by frame_id:
      - `frame_*.png` filenames and `track_*_metadata.json` carry the **absolute**
        source frame id (e.g. 493, 498, 499 -- note the gaps, tracks are not
        contiguous);
      - `track_*_keypoints.json` carries a **0-based index** into the track.
    """
    color_dir = clip_color_dir(dataset_root, clip)
    all_frame_ids = sorted(
        int(m.group(1)) for f in color_dir.glob("frame_*.png") if (m := _FRAME_ID_RE.search(f.name)) is not None
    )
    return {fid: idx for idx, fid in enumerate(all_frame_ids)}


def clip_keypoints(dataset_root: Path, clip: dict, frame_ids: list[int]) -> list[np.ndarray]:
    """Per-frame DWPose keypoints in crop-local coordinates. Shape: [18, 3] each.

    This is what the decoder actually receives (via `ActorPacket`) and renders
    into a condition image itself -- the pre-rendered `_skeleton` PNGs are a
    training-time artefact the decoder never sees.
    """
    scene_dir = clip_color_dir(dataset_root, clip).parent
    payload = json.loads((scene_dir / f"{clip['track']}_keypoints.json").read_text())
    index_of = clip_track_index(dataset_root, clip)
    out: list[np.ndarray] = []
    for fid in frame_ids:
        record = payload[index_of[fid]]
        out.append(np.asarray(record["keypoints"], dtype=np.float32))  # Shape: [18, 3]
    return out


def clip_racket_bboxes(dataset_root: Path, clip: dict, frame_ids: list[int]) -> list[list[float] | None]:
    """Per-frame racket bbox in crop-local coordinates, or None where absent."""
    scene_dir = clip_color_dir(dataset_root, clip).parent
    meta_path = scene_dir / f"{clip['track']}_metadata.json"
    if not meta_path.exists():
        return [None] * len(frame_ids)
    payload = json.loads(meta_path.read_text())
    index_of = clip_track_index(dataset_root, clip)
    return [payload[index_of[fid]].get("racket_bbox_crop") for fid in frame_ids]


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

    img: Image.Image = Image.open(path)
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    else:
        img = img.convert("RGB")
    img = pad_to_square(img, fill=0)
    img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    array = np.asarray(img, dtype=np.float32) / 255.0  # Shape: [H, W, 3]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()  # Shape: [3, H, W]


def load_clip_tensor(paths: list[Path], size: int) -> torch.Tensor:
    """paths -> [N, 3, size, size] float32 in [0, 1]."""
    return torch.stack([load_image_rgb01(p, size) for p in paths], dim=0)


def load_seg_rgb01(path: Path, size: int) -> torch.Tensor:
    from PIL import Image
    from src.shared.tennis_dataset import pad_to_square
    
    img = Image.open(path)
    if img.mode == 'RGBA':
        alpha = img.split()[-1]
        seg_mask = Image.merge("RGB", (alpha, alpha, alpha))
    else:
        seg_mask = Image.new("RGB", img.size, (255, 255, 255))
        
    seg_mask = pad_to_square(seg_mask, fill=0)
    seg_mask = seg_mask.resize((size, size), resample=Image.Resampling.BILINEAR)
    array = np.asarray(seg_mask, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()

def load_seg_clip_tensor(paths: list[Path], size: int) -> torch.Tensor:
    return torch.stack([load_seg_rgb01(p, size) for p in paths], dim=0)


def rgb01_to_bgr_uint8_tensor(frame_rgb01: torch.Tensor) -> torch.Tensor:
    """[3, H, W] float in [0,1] RGB -> [3, H, W] uint8 BGR.

    BGR uint8 is the `reference_crop_tensor` convention every decoder strategy
    expects (it originates from `cv2.imdecode` in the encoder).
    """
    rgb_uint8 = (frame_rgb01.clamp(0, 1) * 255.0).round().to(torch.uint8)  # Shape: [3, H, W] RGB
    return rgb_uint8.flip(0).contiguous()  # Shape: [3, H, W] BGR


def bgr_uint8_to_rgb01(frame_bgr_uint8: torch.Tensor, size: int) -> torch.Tensor:
    """[3, h, w] uint8 BGR (strategy output, at the crop's own size) -> [3, size, size] float RGB in [0,1].

    Matches `load_image_rgb01`'s geometry so predictions and ground truth are
    comparable: pad to square with black, then resize.
    """
    from PIL import Image

    from src.shared.tennis_dataset import pad_to_square

    rgb = frame_bgr_uint8.flip(0).permute(1, 2, 0).cpu().numpy()  # Shape: [h, w, 3] RGB
    img = pad_to_square(Image.fromarray(rgb.astype(np.uint8)), fill=0)
    img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
    array = np.asarray(img, dtype=np.float32) / 255.0  # Shape: [size, size, 3]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()  # Shape: [3, size, size]


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


ARCH_TO_BACKEND: dict[str, str] = {
    "pix2pix": "pix2pix",
    "spade4tennis": "spade4tennis",
    "controlnet": "caption-controlnet",
    "multi-controlnet": "multi-controlnet",
}


def build_eval_strategy(arch: str, checkpoint_path: Path, config_overrides: dict[str, Any] | None = None) -> Any:
    """Construct the *decoder's* strategy for `arch`, pointed at `checkpoint_path`.

    Evaluation must run the same code path the decoder runs. The previous
    implementation reimplemented inference here and diverged badly: ControlNet
    was scored with `StableDiffusionControlNetPipeline` (text-to-image from pure
    noise, with the reference frame explicitly unused) while the decoder runs
    `StableDiffusionControlNetImg2ImgPipeline` seeded from the reference crop.
    That divergence produced the G2 campaign's PSNR 9.76 / VMAF 0.11 -- the
    arithmetic expectation when an unconditional sample is compared against a
    specific target, not a model-quality result.
    """
    from src.decoder.genai_compositor import build_genai_strategy
    from src.shared.config import PointstreamConfig

    config = PointstreamConfig()
    config.genai_backend = ARCH_TO_BACKEND[arch]
    config.genai_checkpoint_override = str(checkpoint_path)
    for key, value in (config_overrides or {}).items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown PointstreamConfig field for eval override: {key!r}")
        setattr(config, key, value)
    return build_genai_strategy(config.genai_backend, config)


def run_strategy_inference(  # pragma: no cover - requires real checkpoint + GPU
    strategy: Any,
    keypoints: list[np.ndarray],
    reference_frame_rgb01: torch.Tensor,
    device: str,
    img_size: int,
    seed: int = 0,
) -> torch.Tensor:
    """Run the decoder strategy frame-by-frame. Returns [N, 3, img_size, img_size] RGB in [0, 1].

    `keypoints` are crop-local DWPose coordinates -- exactly what the decoder
    receives in an `ActorPacket` and renders into a condition image itself. The
    pre-rendered `_skeleton` PNGs are a training-time artefact the decoder never
    sees, so eval must not feed them either.
    """
    reference_bgr = rgb01_to_bgr_uint8_tensor(reference_frame_rgb01)  # Shape: [3, H, W] BGR uint8
    torch_device = torch.device(device)

    outputs = []
    for pose in keypoints:
        pose_tensor = torch.from_numpy(pose)  # Shape: [18, 3]
        generated = strategy.generate(
            reference_crop_tensor=reference_bgr,
            dense_dwpose_tensor=pose_tensor,
            seed=seed,
            device=torch_device,
        )  # Shape: [3, h, w] BGR uint8
        outputs.append(bgr_uint8_to_rgb01(generated, img_size))

    return torch.stack(outputs, dim=0)  # Shape: [N, 3, img_size, img_size]


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
    seed: int = 0,
) -> dict[str, Any]:
    """Runs inference + scoring for every probe clip in `manifest`; returns per-clip and aggregate metrics.

    Inference goes through the decoder's own strategy classes (see
    `build_eval_strategy`) so the score describes the system as it actually
    runs. `condition_type` is accepted for CLI compatibility but no longer
    selects a pre-rendered condition directory: the strategy renders its own
    condition from raw keypoints, exactly as the decoder does.
    """
    del condition_type  # strategies render their own condition from keypoints
    strategy = build_eval_strategy(arch, checkpoint_path, arch_kwargs)

    per_clip_metrics = []
    with tempfile.TemporaryDirectory(prefix="eval_checkpoint_") as tmp:
        tmp_dir = Path(tmp)
        for clip in manifest["probe_clips"]:
            frame_ids = sorted(clip["frame_ids"])
            color_paths = clip_frame_paths(clip_color_dir(dataset_root, clip), frame_ids)
            ref_path = resolve_reference_frame_path(dataset_root, clip)

            ground_truth = load_clip_tensor(color_paths, img_size)  # Shape: [N, 3, size, size]
            reference_tensor = load_image_rgb01(ref_path, img_size)  # Shape: [3, size, size]
            keypoints = clip_keypoints(dataset_root, clip, frame_ids)  # list of Shape: [18, 3]

            predicted = run_strategy_inference(
                strategy=strategy,
                keypoints=keypoints,
                reference_frame_rgb01=reference_tensor,
                device=device,
                img_size=img_size,
                seed=seed,
            )  # Shape: [N, 3, size, size]

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
    parser.add_argument("--seed", type=int, default=0, help="Generation seed, pinned so runs are comparable")
    # ControlNet knobs map onto PointstreamConfig fields, so eval and the
    # decoder are configured identically rather than through a parallel set of
    # kwargs that only eval understood.
    parser.add_argument("--controlnet-num-inference-steps", type=int, default=None, dest="controlnet_steps")
    parser.add_argument(
        "--controlnet-strength",
        type=float,
        default=None,
        help="img2img denoising strength. Lower preserves more of the reference appearance; "
        "the decoder default (0.65) is high for a reconstruction task.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics = tuple(m.strip().lower() for m in args.metrics.split(",") if m.strip())

    manifest = load_manifest(args.manifest)

    # Only overrides the user actually set; everything else keeps the decoder's
    # own defaults so eval and the pipeline stay configured identically.
    arch_kwargs: dict[str, Any] = {}
    if args.controlnet_steps is not None:
        arch_kwargs["controlnet_steps"] = args.controlnet_steps
    if args.controlnet_strength is not None:
        arch_kwargs["controlnet_strength"] = args.controlnet_strength
    arch_kwargs["controlnet_width"] = args.img_size
    arch_kwargs["controlnet_height"] = args.img_size

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
        arch_kwargs=arch_kwargs,
        seed=args.seed,
    )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "step": args.step,
        "variant": args.variant,
        "arch": args.arch,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "seed": args.seed,
        "config_overrides": arch_kwargs,
        **result["aggregate"],
    }
    append_jsonl_log(args.log, record)

    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
