#!/usr/bin/env python3
"""
PointStream Client – Skeleton Reconstruction + AnimateAnyone Inference.

Reconstructs player videos from the compact data produced by the server:
  1. Load DWPose keypoints CSV + reference images from an experiment directory.
  2. Draw skeleton images in the exact format AnimateAnyone expects.
  3. Run AnimateAnyone Pose2Video inference per player.

Requirements:
    Must run in the **animate-anyone** conda environment::

        conda activate animate-anyone
        cd /home/itec/emanuele/Moore-AnimateAnyone
        python /home/itec/emanuele/pointstream/pointstream_client.py \\
            --experiment_dir /path/to/experiment

Alternatively the ``--animate_anyone_dir`` flag can override the repo location.
"""

import argparse
import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure AnimateAnyone repo is importable
# ---------------------------------------------------------------------------

_DEFAULT_AA_DIR = Path("/home/itec/emanuele/Moore-AnimateAnyone")

def _setup_animate_anyone_path(aa_dir):
    aa_dir = Path(aa_dir).resolve()
    if str(aa_dir) not in sys.path:
        sys.path.insert(0, str(aa_dir))
    os.chdir(aa_dir)  # AnimateAnyone loads weights with relative paths
    return aa_dir


# ---------------------------------------------------------------------------
# Skeleton reconstruction from keypoints CSV
# ---------------------------------------------------------------------------

# Import the drawing function from the pointstream dwpose module (pure numpy/cv2)
_POINTSTREAM_DIR = Path(__file__).resolve().parent
if str(_POINTSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(_POINTSTREAM_DIR))

from dwpose import draw_pose, keypoints_to_pose_dict


def load_keypoints(experiment_dir):
    """
    Load the merged metadata CSV (preferred) or fallback to the original
    `dwpose_keypoints.csv` produced by the server.

    Returns:
        (df, fps)
        - df: DataFrame containing `frame_index`, `player_id`, `keypoints`,
              `scores`, `detect_width`, `detect_height` and (if available)
              tracking columns merged in.
        - fps: source video frames-per-second (float) or None if unknown.
    """
    exp_dir = Path(experiment_dir)

    # 1) Prefer merged metadata if present
    merged_files = sorted(exp_dir.glob("merged_metadata*.csv"))
    if merged_files:
        merged_path = merged_files[0]
        df = pd.read_csv(merged_path)

        # Parse detect size / fps from filename (we store constant values in filename)
        fname = merged_path.stem
        fps = None
        detect_w = None
        detect_h = None
        # patterns: ..._w{W}_h{H}_fps_{FPS}
        import re
        m = re.search(r"w(\d+)_h(\d+)", fname)
        if m:
            detect_w = int(m.group(1))
            detect_h = int(m.group(2))
        m2 = re.search(r"fps_([0-9]+\.?[0-9]*)", fname)
        if m2:
            try:
                fps = float(m2.group(1))
            except Exception:
                fps = None

        # If merged file doesn't include detect_width/height columns (we removed them),
        # add them to the returned DataFrame for client compatibility
        if detect_w and "detect_width" not in df.columns:
            df["detect_width"] = detect_w
        if detect_h and "detect_height" not in df.columns:
            df["detect_height"] = detect_h

        return df, fps

    # 2) Fallback: use dwpose_keypoints.csv + tracking_metadata.csv (if present)
    dw_csv = exp_dir / "dwpose_keypoints.csv"
    if not dw_csv.exists():
        raise FileNotFoundError(f"Keypoints CSV not found: {dw_csv}")
    pose_df = pd.read_csv(dw_csv)

    tracking_csv = exp_dir / "tracking_metadata.csv"
    fps = None
    if tracking_csv.exists():
        track_df = pd.read_csv(tracking_csv)
        if "video_fps" in track_df.columns:
            fps_vals = track_df["video_fps"].dropna().unique()
            fps = float(fps_vals[0]) if len(fps_vals) > 0 else None
        # merge tracking columns into pose dataframe if helpful (left join)
        merged = pd.merge(
            pose_df,
            track_df,
            left_on=["frame_index", "player_id"],
            right_on=["frame_index", "id"],
            how="left",
            suffixes=("_pose", "_track"),
        )
        return merged, fps

    return pose_df, fps


def build_skeleton_image(kpts_json, scores_json, detect_w, detect_h,
                         output_h=512, output_w=512):
    """
    Reconstruct a single DWPose skeleton image from serialised keypoints.

    The output matches the format returned by AnimateAnyone's DWposeDetector,
    so it can be fed directly to the Pose2Video pipeline.

    Returns:
        PIL.Image (RGB, output_h x output_w).
    """
    kpts = np.array(json.loads(kpts_json), dtype=np.float64)   # (134, 2)
    scores = np.array(json.loads(scores_json), dtype=np.float64)  # (134,)

    pose = keypoints_to_pose_dict(kpts, scores, detect_w, detect_h)

    canvas = draw_pose(pose, output_h, output_w)          # (H, W, 3) BGR uint8
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(canvas_rgb)


def build_all_skeletons(df, player_id, output_h=512, output_w=512):
    """
    Build an ordered list of skeleton PIL images for one player.

    Returns:
        list[PIL.Image], sorted by frame_index.
    """
    player_df = df[df["player_id"] == player_id].sort_values("frame_index")
    skeletons = []
    for _, row in tqdm(player_df.iterrows(), total=len(player_df),
                       desc=f"Drawing skeletons for player {player_id}"):
        skel = build_skeleton_image(
            row["keypoints"], row["scores"],
            int(row["detect_width"]), int(row["detect_height"]),
            output_h, output_w,
        )
        skeletons.append(skel)
    return skeletons


# ---------------------------------------------------------------------------
# AnimateAnyone inference
# ---------------------------------------------------------------------------

def run_inference(ref_image_path, skeleton_pils, config_path,
                  width=512, height=784, length=None, steps=30, cfg=3.5,
                  seed=42, fps=None, save_dir=None, player_id=0):
    """
    Run AnimateAnyone Pose2Video inference with a reference image and
    skeleton sequence (both as PIL).

    The heavy imports are deferred so the module can be loaded without the
    animate-anyone environment for skeleton-only usage.
    """
    import torch
    import torchvision
    from diffusers import AutoencoderKL, DDIMScheduler
    from einops import repeat
    from omegaconf import OmegaConf
    from torchvision import transforms
    from transformers import CLIPVisionModelWithProjection

    from src.models.pose_guider import PoseGuider
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel
    from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
    from src.utils.util import save_videos_grid

    config = OmegaConf.load(config_path)
    weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32

    # --- Load models --------------------------------------------------------
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path, subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda",
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    # --- Load weights -------------------------------------------------------
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
        strict=False,
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
        strict=False,
    )

    pipe = Pose2VideoPipeline(
        vae=vae, image_encoder=image_enc,
        reference_unet=reference_unet, denoising_unet=denoising_unet,
        pose_guider=pose_guider, scheduler=scheduler,
    ).to("cuda", dtype=weight_dtype)

    generator = torch.manual_seed(seed)

    # --- Prepare inputs -----------------------------------------------------
    ref_image_pil = Image.open(ref_image_path).convert("RGB")

    pose_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])

    # Use full skeleton sequence if `length` is None, otherwise cap to available frames
    L = len(skeleton_pils) if length is None else min(length, len(skeleton_pils))
    print(f"    Using sequence length L={L} (requested: {length}, available: {len(skeleton_pils)})")
    pose_list = skeleton_pils[:L]
    pose_tensor_list = [pose_transform(p) for p in pose_list]

    ref_image_tensor = pose_transform(ref_image_pil).unsqueeze(1).unsqueeze(0)
    ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=L)

    pose_tensor = torch.stack(pose_tensor_list, dim=0).transpose(0, 1).unsqueeze(0)

    # --- Inference ----------------------------------------------------------
    video = pipe(
        ref_image_pil, pose_list,
        width, height, L, steps, cfg,
        generator=generator,
    ).videos

    # --- Save ---------------------------------------------------------------
    if save_dir is None:
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        save_dir = Path(f"output/{date_str}/{time_str}--pointstream")
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    video_with_refs = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
    out_path = save_dir / f"player_{player_id}_{height}x{width}_{int(cfg)}.mp4"
    out_fps = fps or 12
    save_videos_grid(video_with_refs, str(out_path), n_rows=3, fps=out_fps)
    print(f"  Saved: {out_path} (fps={out_fps})")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PointStream Client – reconstruct skeletons & animate players",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Experiment directory produced by pointstream_server.py")
    parser.add_argument("--animate_anyone_dir", type=str,
                        default=str(_DEFAULT_AA_DIR),
                        help="Path to Moore-AnimateAnyone repo")
    parser.add_argument("--config", type=str, default=None,
                        help="AnimateAnyone YAML config "
                             "(default: configs/prompts/run_finetuned.yaml)")
    parser.add_argument("--player_ids", type=int, nargs="*", default=None,
                        help="Which player IDs to animate (default: all)")
    parser.add_argument("-W", type=int, default=512, help="Output width (per-frame)")
    parser.add_argument("-H", type=int, default=512, help="Output height (per-frame) — default changed to 512 for square frames)")
    parser.add_argument("-L", type=int, default=None, help="Sequence length (frames) — default: all available frames from the DWpose CSV")
    parser.add_argument("--steps", type=int, default=30, help="Denoising steps")
    parser.add_argument("--cfg", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--skeletons_only", action="store_true",
                        help="Only reconstruct skeletons (skip inference)")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    aa_dir = _setup_animate_anyone_path(args.animate_anyone_dir)

    if args.config is None:
        args.config = str(aa_dir / "configs" / "prompts" / "run_finetuned.yaml")

    print(f"\n{'#'*80}")
    print(f"#  PointStream Client")
    print(f"#  Experiment: {exp_dir}")
    print(f"{'#'*80}\n")

    # Load keypoints (merged metadata preferred) — returns (df, fps)
    kp_df, source_fps = load_keypoints(exp_dir)
    all_player_ids = sorted(kp_df["player_id"].unique())
    if source_fps:
        print(f"Detected source video FPS: {source_fps}")

    if args.player_ids is not None:
        player_ids = [pid for pid in args.player_ids if pid in all_player_ids]
    else:
        player_ids = all_player_ids

    print(f"Players to animate: {player_ids}")

    for pid in player_ids:
        print(f"\n{'='*60}")
        print(f"Player {pid}")
        print(f"{'='*60}")

        # Reference image
        ref_path = exp_dir / "reference" / f"id{pid}.png"
        if not ref_path.exists():
            print(f"  Reference image not found: {ref_path} – skipping")
            continue

        # Build skeletons
        skeletons = build_all_skeletons(kp_df, pid, args.H, args.W)
        print(f"  Built {len(skeletons)} skeleton frames")

        if not skeletons:
            print("  No skeletons built – skipping")
            continue

        # Optionally save skeleton images for debugging
        skel_debug_dir = exp_dir / "debug_skeletons" / f"id{pid}"
        skel_debug_dir.mkdir(parents=True, exist_ok=True)
        for i, skel in enumerate(skeletons[:5]):
            skel.save(skel_debug_dir / f"{i:05d}.png")
        print(f"  Debug skeletons (first 5) saved to {skel_debug_dir}")

        if args.skeletons_only:
            # Save all skeletons when in skeleton-only mode
            for i, skel in enumerate(skeletons):
                skel.save(skel_debug_dir / f"{i:05d}.png")
            print(f"  All {len(skeletons)} skeletons saved (--skeletons_only mode)")
            continue

        # Run inference
        out_path = run_inference(
            ref_image_path=str(ref_path),
            skeleton_pils=skeletons,
            config_path=args.config,
            width=args.W,
            height=args.H,
            length=args.L,
            steps=args.steps,
            cfg=args.cfg,
            seed=args.seed,
            fps=source_fps,
            save_dir=args.output_dir,
            player_id=pid,
        )

        # Copy generated video(s) back into the experiment folder for easy access
        try:
            exp_out_dir = exp_dir
            exp_out_dir.mkdir(parents=True, exist_ok=True)
            dst_name = exp_out_dir / f"output_player_{pid}.mp4"
            shutil.copy2(str(out_path), str(dst_name))
            print(f"  Copied generated video to: {dst_name}")
        except Exception as e:
            print(f"  Warning: failed to copy generated video into experiment dir: {e}")

    print(f"\n{'='*80}")
    print(f"Client finished.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
