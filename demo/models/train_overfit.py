"""Overfit the 4-channel SPADE hand generator on clip 1 (SAM matte, DWB2 poses)."""

from __future__ import annotations

import sqlite3  # noqa: F401
import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from demo.evaluation.pose_backends import BACKENDS
from demo.models.dataset import EgocentricHandDataset, build_curated_samples
from demo.models.hand_objective import hand_step_loss
from demo.models.matte import dwb2_roundtrip, interpolate_hand_alphas, read_hand_alphas
from demo.models.unet_generator import HandPix2PixUNet, HandSPADEUNet
from demo.pipeline.hand_keypoints import extract_video_hand_poses

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CURATED_DIR = Path("/home/itec/emanuele/Datasets/Egocentric-10K/curated")
DEFAULT_OUTPUT_DIR = Path("demo/outputs/models")


def train(
    curated_dir: Path = DEFAULT_CURATED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    frames_per_clip: int = 400,
    epochs: int = 10_000,
    batch_size: int = 8,
    lr: float = 2e-4,
    device_str: str = "cuda:0",
    model_type: str = "spade",
    smoke: bool = False,
    pose_backend: str = "rtm_hand",
    mask_video: Path | None = None,
    dwb2: bool = True,
    clip_indices: list[int] | None = None,
    max_minutes: float = 60.0,
    patience_epochs: int = 5,
    min_delta: float = 1e-3,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = curated_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if clip_indices is None:
        clip_indices = [0]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Using training device: {device}")

    all_samples = []
    all_anchors = {}

    for clip_idx in clip_indices:
        item = manifest[clip_idx]
        clip_path = Path(item["path"])
        logger.info(
            "Extracting poses and building samples for Clip %s: %s (%s frames)...",
            clip_idx + 1,
            clip_path.name,
            frames_per_clip,
        )
        extractor = BACKENDS.get(pose_backend, extract_video_hand_poses)
        logger.info("Pose backend for training: %s", pose_backend)
        poses = extractor(clip_path, frames_per_clip)
        probe = cv2.VideoCapture(str(clip_path))
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        probe.release()
        frame_alphas = None
        if mask_video is not None:
            # SAM only. Gaps of a few frames are interpolated; no YOLOE.
            frame_alphas = interpolate_hand_alphas(
                read_hand_alphas(mask_video, width, height, frames_per_clip)
            )
            covered = sum(1 for mask in frame_alphas if int(np.count_nonzero(mask)) > 200)
            logger.info(
                "Hand matte frames for clip %s: %d (%d with a hand)",
                clip_idx + 1,
                len(frame_alphas),
                covered,
            )
        if dwb2:
            poses, payload_bytes = dwb2_roundtrip(poses, width, height, stabilize=True)
            logger.info("DWB2 roundtrip clip %s: %d bytes", clip_idx + 1, payload_bytes)
        samples, anchors, _anchor_bytes = build_curated_samples(
            clip_path,
            poses,
            image_size=256,
            max_frames=frames_per_clip,
            clip_id=clip_idx,
            frame_alphas=frame_alphas,
        )
        logger.info("  Extracted %d valid hand crop samples.", len(samples))
        all_samples.extend(samples)
        all_anchors[clip_idx] = anchors

    if not all_samples:
        raise ValueError("No hand samples were extracted! Check video format and hand presence.")

    logger.info("Total training samples: %d", len(all_samples))

    if smoke:
        all_samples = all_samples[:16]
        epochs = 1
        logger.info("Running smoke test mode (16 samples, 1 epoch)...")

    dataset = EgocentricHandDataset(all_samples, image_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)

    out_channels = 4 if mask_video else 3
    if model_type == "spade":
        model = HandSPADEUNet(in_channels=6, out_channels=out_channels).to(device)
        logger.info("Initialized HandSPADEUNet (%d output channels).", out_channels)
    else:
        model = HandPix2PixUNet(in_channels=6, out_channels=out_channels).to(device)
        logger.info("Initialized baseline HandPix2PixUNet (%d output channels).", out_channels)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    criterion_l1 = nn.L1Loss()
    criterion_lpips = None
    try:
        import lpips

        criterion_lpips = lpips.LPIPS(net="alex", verbose=False).to(device)
        criterion_lpips.eval()
        for p in criterion_lpips.parameters():
            p.requires_grad = False
        logger.info("Loaded LPIPS perceptual loss for training.")
    except Exception as e:
        logger.warning(f"Could not load LPIPS for training: {e}")

    # Cosine over a long horizon; early stop usually fires first.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 200), eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    logger.info(
        "Starting %s train (max %.0f min, patience %d epochs, min_delta %.4f)...",
        model_type.upper(),
        max_minutes,
        patience_epochs,
        min_delta,
    )
    start_time = time.time()
    deadline = start_time + max_minutes * 60.0
    best_loss = float("inf")
    best_state = None
    stale = 0
    avg_loss = float("inf")
    stop_reason = "max_epochs"
    loss_still_falling = False
    epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_app = 0.0
        epoch_matte = 0.0
        num_batches = 0

        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                if targets.shape[1] == 4:
                    loss, loss_app, loss_matte = hand_step_loss(outputs, targets, criterion_lpips)
                else:
                    loss_l1 = criterion_l1(outputs, targets)
                    if criterion_lpips is not None:
                        loss = loss_l1 + 0.8 * criterion_lpips(outputs, targets).mean()
                    else:
                        loss = loss_l1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if targets.shape[1] == 4:
                epoch_app += float(loss_app.detach())
                epoch_matte += float(loss_matte.detach())
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, num_batches)
        elapsed = time.time() - start_time
        logger.info(
            "Epoch [%02d] | SmoothMax: %.4f | Appearance: %.4f | Matte: %.4f | Elapsed: %.1fs",
            epoch,
            avg_loss,
            epoch_app / max(1, num_batches),
            epoch_matte / max(1, num_batches),
            elapsed,
        )

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if time.time() >= deadline:
            # Was loss still falling steeply?
            loss_still_falling = stale == 0 or (best_loss - avg_loss) > min_delta * 2
            stop_reason = "time_limit"
            logger.info(
                "Hit %.0f minute limit at epoch %d (loss=%.4f, best=%.4f, still_falling=%s)",
                max_minutes,
                epoch,
                avg_loss,
                best_loss,
                loss_still_falling,
            )
            break
        if stale >= patience_epochs:
            stop_reason = "early_stop"
            logger.info(
                "Early stop at epoch %d (no improve for %d epochs; best=%.4f)",
                epoch,
                patience_epochs,
                best_loss,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_total = time.time() - start_time
    meta = {
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "epochs": epoch,
        "final_loss": float(avg_loss),
        "best_loss": float(best_loss if best_loss < float("inf") else avg_loss),
        "image_size": 256,
        "out_channels": out_channels,
        "pose_backend": pose_backend,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed_total,
        "loss_still_falling_at_cutoff": loss_still_falling,
        "batch_size": batch_size,
        "device": str(device),
        "anchors": {
            str(k): {side: anchor.tolist() for side, anchor in v.items()}
            for k, v in all_anchors.items()
        },
    }
    checkpoint_path = output_dir / ("spade_generator.pt" if model_type == "spade" else "overfit_generator.pt")
    torch.save(meta, checkpoint_path)
    latest_path = output_dir / "overfit_generator.pt"
    torch.save(meta, latest_path)
    (output_dir / "train_log.json").write_text(
        json.dumps(
            {
                "epochs": epoch,
                "elapsed_seconds": elapsed_total,
                "final_loss": float(avg_loss),
                "best_loss": float(best_loss if best_loss < float("inf") else avg_loss),
                "stop_reason": stop_reason,
                "loss_still_falling_at_cutoff": loss_still_falling,
                "batch_size": batch_size,
                "device": str(device),
            },
            indent=2,
        )
    )
    logger.info(
        "Model saved to %s (%d bytes) stop=%s elapsed=%.1fs",
        checkpoint_path,
        checkpoint_path.stat().st_size,
        stop_reason,
        elapsed_total,
    )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit Hand generator on clip 1")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=400, help="Frames per clip to process")
    parser.add_argument("--epochs", type=int, default=10_000, help="Upper bound; early stop / time usually fire first")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model-type", choices=["spade", "unet"], default="spade")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test")
    parser.add_argument("--pose-backend", default="rtm_hand", choices=list(BACKENDS))
    parser.add_argument("--mask-video", type=Path, default=None, help="SAM hand-color video for clip 1")
    parser.add_argument("--dwb2", action="store_true", default=True)
    parser.add_argument("--no-dwb2", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--clip-index", type=int, default=0)
    args = parser.parse_args()

    train(
        curated_dir=args.curated_dir,
        output_dir=args.output_dir,
        frames_per_clip=args.frames,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        model_type=args.model_type,
        smoke=args.smoke,
        pose_backend=args.pose_backend,
        mask_video=args.mask_video,
        dwb2=not args.no_dwb2,
        clip_indices=[args.clip_index],
        max_minutes=args.max_minutes,
        patience_epochs=args.patience,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()
