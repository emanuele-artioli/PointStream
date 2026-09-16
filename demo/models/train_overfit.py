"""Rapidly overfit the HandPix2PixUNet on the 3 curated worker clips."""

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from demo.models.dataset import EgocentricHandDataset, build_curated_samples
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
    epochs: int = 35,
    batch_size: int = 8,
    lr: float = 2e-4,
    device_str: str = "cuda:1",
    model_type: str = "spade",
    smoke: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = curated_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Using training device: {device}")

    all_samples = []
    all_anchors = {}

    # Extract poses and build dataset for each curated clip
    for clip_idx, item in enumerate(manifest[:3]):
        clip_path = Path(item["path"])
        logger.info(f"Extracting poses and building samples for Clip {clip_idx + 1}: {clip_path.name} ({frames_per_clip} frames)...")
        poses = extract_video_hand_poses(clip_path, max_frames=frames_per_clip)
        samples, anchors = build_curated_samples(
            clip_path,
            poses,
            image_size=256,
            max_frames=frames_per_clip,
            clip_id=clip_idx,
        )
        logger.info(f"  Extracted {len(samples)} valid hand crop samples.")
        all_samples.extend(samples)
        all_anchors[clip_idx] = anchors

    if not all_samples:
        raise ValueError("No hand samples were extracted! Check video format and hand presence.")

    logger.info(f"Total training samples across all 3 clips: {len(all_samples)}")

    if smoke:
        all_samples = all_samples[:16]
        epochs = 1
        logger.info("Running smoke test mode (16 samples, 1 epoch)...")

    dataset = EgocentricHandDataset(all_samples, image_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)

    if model_type == "spade":
        model = HandSPADEUNet(in_channels=6, out_channels=3).to(device)
        logger.info("Initialized HandSPADEUNet with SPADE modulation blocks.")
    else:
        model = HandPix2PixUNet(in_channels=6, out_channels=3).to(device)
        logger.info("Initialized baseline HandPix2PixUNet.")

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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    logger.info(f"Starting model overfitting ({model_type.upper()}) for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss_l1 = criterion_l1(outputs, targets)
                if criterion_lpips is not None:
                    loss_lpips = criterion_lpips(outputs, targets).mean()
                    loss = loss_l1 + 0.8 * loss_lpips
                else:
                    loss = loss_l1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, num_batches)
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] | Combined Loss: {avg_loss:.4f} | Elapsed: {elapsed:.1f}s")

    checkpoint_path = output_dir / ("spade_generator.pt" if model_type == "spade" else "overfit_generator.pt")
    torch.save(
        {
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "final_loss": avg_loss,
            "image_size": 256,
            "anchors": {
                str(k): {side: anchor.tolist() for side, anchor in v.items()}
                for k, v in all_anchors.items()
            },
        },
        checkpoint_path,
    )
    # Also save as overfit_generator.pt so standard pipeline loads the latest best model
    if model_type == "spade":
        latest_path = output_dir / "overfit_generator.pt"
        torch.save(
            {
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "epochs": epochs,
                "final_loss": avg_loss,
                "image_size": 256,
                "anchors": {
                    str(k): {side: anchor.tolist() for side, anchor in v.items()}
                    for k, v in all_anchors.items()
                },
            },
            latest_path,
        )
    logger.info(f"Model successfully saved to {checkpoint_path} ({checkpoint_path.stat().st_size} bytes)")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit Hand generator on Egocentric clips")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=400, help="Frames per clip to process")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model-type", choices=["spade", "unet"], default="spade")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test")
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
    )


if __name__ == "__main__":
    main()
