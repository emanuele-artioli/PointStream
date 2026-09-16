"""Generates Rate-Distortion and Rate-Computation Pareto plots from benchmark JSON."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_JSON = Path("demo/outputs/results/comparison_results.json")
DEFAULT_OUT_DIR = Path("demo/outputs/results")


def generate_plots(json_path: Path = DEFAULT_JSON, out_dir: Path = DEFAULT_OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "r") as f:
        data = json.load(f)

    clips = data.get("clips", [])
    if not clips:
        logger.warning("No clips found in comparison JSON.")
        return []

    plot_paths = []

    # Iterate over each clip and plot RD curves
    for clip in clips:
        clip_name = clip["clip_name"]
        ps = clip["pointstream"]
        av1_arms = clip["av1_arms"]

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"PointStream vs AV1: Rate-Distortion-Utility ({clip_name})", fontsize=15, fontweight="bold")

        # Separate AV1 arms by tier
        av1_540p = sorted([a for a in av1_arms if a.get("scale") == "960:540"], key=lambda x: x["actual_kbps"])
        av1_720p = sorted([a for a in av1_arms if a.get("scale") == "1280:720"], key=lambda x: x["actual_kbps"])
        av1_1080p = sorted([a for a in av1_arms if a.get("scale") is None and not a.get("deblocked", False)], key=lambda x: x["actual_kbps"])
        av1_deb = sorted([a for a in av1_arms if a.get("deblocked", False)], key=lambda x: x["actual_kbps"])

        # 1. Bitrate vs Hand-ROI PSNR
        ax = axs[0, 0]
        # Plot PointStream variants
        ps_variants = clip.get("pointstream_variants", [ps])
        ps_markers = {
            "PointStream (Standard, 250k bg)": ("blue", "*", 220),
            "PointStream (Ultra-Low, 90k bg)": ("darkgoldenrod", "P", 180),
            "PointStream (Plate 2s)": ("darkorange", "D", 140),
            "PointStream": ("blue", "*", 220),
        }

        # 1. Bitrate vs Hand-ROI PSNR
        ax = axs[0, 0]
        if av1_540p:
            ax.plot([a["actual_kbps"] for a in av1_540p], [a["metrics"]["hand_roi_psnr_dB"] for a in av1_540p], "g-^", label="AV1 (540p, p7)", linewidth=1.8, markersize=7)
        if av1_720p:
            ax.plot([a["actual_kbps"] for a in av1_720p], [a["metrics"]["hand_roi_psnr_dB"] for a in av1_720p], "m-v", label="AV1 (720p, p7)", linewidth=1.8, markersize=7)
        if av1_1080p:
            ax.plot([a["actual_kbps"] for a in av1_1080p], [a["metrics"]["hand_roi_psnr_dB"] for a in av1_1080p], "r-o", label="AV1 (1080p, p6)", linewidth=1.8, markersize=7)
        if av1_deb:
            ax.plot([a["actual_kbps"] for a in av1_deb], [a["metrics"]["hand_roi_psnr_dB"] for a in av1_deb], "c--s", label="AV1 (1080p Deblocked)", linewidth=1.2)

        for p_arm in ps_variants:
            p_name = p_arm.get("name", "PointStream")
            color, marker, s = ps_markers.get(p_name, ("blue", "*", 180))
            ax.scatter([p_arm["bitrate_kbps"]], [p_arm["metrics"]["hand_roi_psnr_dB"]], color=color, s=s, zorder=6, label=f"{p_name} ({p_arm['bitrate_kbps']}k)", marker=marker)

        ax.set_xlabel("Total Bitrate (kbps)", fontsize=11)
        ax.set_ylabel("Hand-ROI PSNR (dB) [Higher is better]", fontsize=11)
        ax.set_title("Hand Interaction ROI Quality", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

        # 2. Bitrate vs LPIPS
        ax = axs[0, 1]
        if av1_540p:
            ax.plot([a["actual_kbps"] for a in av1_540p], [a["metrics"]["lpips"] for a in av1_540p], "g-^", label="AV1 (540p, p7)", linewidth=1.8, markersize=7)
        if av1_720p:
            ax.plot([a["actual_kbps"] for a in av1_720p], [a["metrics"]["lpips"] for a in av1_720p], "m-v", label="AV1 (720p, p7)", linewidth=1.8, markersize=7)
        if av1_1080p:
            ax.plot([a["actual_kbps"] for a in av1_1080p], [a["metrics"]["lpips"] for a in av1_1080p], "r-o", label="AV1 (1080p, p6)", linewidth=1.8, markersize=7)
        if av1_deb:
            ax.plot([a["actual_kbps"] for a in av1_deb], [a["metrics"]["lpips"] for a in av1_deb], "c--s", label="AV1 (1080p Deblocked)", linewidth=1.2)

        for p_arm in ps_variants:
            p_name = p_arm.get("name", "PointStream")
            color, marker, s = ps_markers.get(p_name, ("blue", "*", 180))
            ax.scatter([p_arm["bitrate_kbps"]], [p_arm["metrics"]["lpips"]], color=color, s=s, zorder=6, label=f"{p_name}", marker=marker)

        ax.set_xlabel("Total Bitrate (kbps)", fontsize=11)
        ax.set_ylabel("LPIPS Perceptual Distance [Lower is better]", fontsize=11)
        ax.set_title("Perceptual Distortion (LPIPS)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

        # 3. Bitrate vs Hand Detection Rate
        ax = axs[1, 0]
        if av1_540p:
            ax.plot([a["actual_kbps"] for a in av1_540p], [a["teleop_utility"]["detection_rate"] * 100 for a in av1_540p], "g-^", label="AV1 (540p, p7)", linewidth=1.8, markersize=7)
        if av1_720p:
            ax.plot([a["actual_kbps"] for a in av1_720p], [a["teleop_utility"]["detection_rate"] * 100 for a in av1_720p], "m-v", label="AV1 (720p, p7)", linewidth=1.8, markersize=7)
        if av1_1080p:
            ax.plot([a["actual_kbps"] for a in av1_1080p], [a["teleop_utility"]["detection_rate"] * 100 for a in av1_1080p], "r-o", label="AV1 (1080p, p6)", linewidth=1.8, markersize=7)
        if av1_deb:
            ax.plot([a["actual_kbps"] for a in av1_deb], [a["teleop_utility"]["detection_rate"] * 100 for a in av1_deb], "c--s", label="AV1 (1080p Deblocked)", linewidth=1.2)

        for p_arm in ps_variants:
            p_name = p_arm.get("name", "PointStream")
            color, marker, s = ps_markers.get(p_name, ("blue", "*", 180))
            ax.scatter([p_arm["bitrate_kbps"]], [p_arm["teleop_utility"]["detection_rate"] * 100], color=color, s=s, zorder=6, label=f"{p_name}", marker=marker)

        ax.set_xlabel("Total Bitrate (kbps)", fontsize=11)
        ax.set_ylabel("Hand Detection Success Rate (%)", fontsize=11)
        ax.set_title("Teleop Utility: Hand Tracking Detection Rate", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

        # 4. Bitrate vs MPJPE
        ax = axs[1, 1]
        if av1_540p:
            ax.plot([a["actual_kbps"] for a in av1_540p], [a["teleop_utility"]["mpjpe_pixels"] for a in av1_540p], "g-^", label="AV1 (540p, p7)", linewidth=1.8, markersize=7)
        if av1_720p:
            ax.plot([a["actual_kbps"] for a in av1_720p], [a["teleop_utility"]["mpjpe_pixels"] for a in av1_720p], "m-v", label="AV1 (720p, p7)", linewidth=1.8, markersize=7)
        if av1_1080p:
            ax.plot([a["actual_kbps"] for a in av1_1080p], [a["teleop_utility"]["mpjpe_pixels"] for a in av1_1080p], "r-o", label="AV1 (1080p, p6)", linewidth=1.8, markersize=7)
        if av1_deb:
            ax.plot([a["actual_kbps"] for a in av1_deb], [a["teleop_utility"]["mpjpe_pixels"] for a in av1_deb], "c--s", label="AV1 (1080p Deblocked)", linewidth=1.2)

        for p_arm in ps_variants:
            p_name = p_arm.get("name", "PointStream")
            color, marker, s = ps_markers.get(p_name, ("blue", "*", 180))
            ax.scatter([p_arm["bitrate_kbps"]], [p_arm["teleop_utility"]["mpjpe_pixels"]], color=color, s=s, zorder=6, label=f"{p_name}", marker=marker)

        ax.set_xlabel("Total Bitrate (kbps)", fontsize=11)
        ax.set_ylabel("Mean Joint Position Error (pixels) [Lower is better]", fontsize=11)
        ax.set_title("Teleop Utility: Hand Joint Error (MPJPE)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

        plt.tight_layout()
        out_png = out_dir / f"rd_curves_{clip_name}.png"
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        logger.info(f"Saved RD plot to {out_png}")
        plot_paths.append(out_png)

    # 5. Latency Profile Bar Chart
    lat = data.get("latency_profile", {})
    if lat:
        fig, ax = plt.subplots(figsize=(11, 6))
        stages = [
            "MediaPipe Pose (GPU)",
            "Keypoint Pack (CPU)",
            "SVT-AV1 Bg (CPU)",
            "Keypoint Unpack",
            "SPADE Gen. (GPU)",
            "Compositing",
            "Parallel E2E",
            "Serial E2E",
        ]
        values = [
            lat.get("encode_pose_extraction_ms", 15.1),
            lat.get("encode_keypoint_pack_ms", 0.05),
            lat.get("encode_background_svtav1_ms", 13.6),
            lat.get("decode_keypoint_unpack_ms", 0.02),
            lat.get("decode_generator_inference_ms", lat.get("decode_unet_inference_ms", 7.6)),
            lat.get("decode_compositing_ms", 0.06),
            lat.get("parallel_end_to_end_latency_ms", lat.get("end_to_end_latency_ms", 22.8)),
            lat.get("serial_end_to_end_latency_ms", 36.4),
        ]
        colors = ["#2b5c8f", "#3e82c5", "#5dade2", "#e67e22", "#d35400", "#e74c3c", "#27ae60", "#229954"]
        bars = ax.bar(stages, values, color=colors, width=0.55)
        ax.axhline(50.0, color="red", linestyle="--", linewidth=1.5, label="50ms Teleoperation Budget")
        ax.set_ylabel("Latency (milliseconds)", fontsize=12)
        ax.set_title("PointStream End-to-End Latency Breakdown (RTX 6000 Ada)", fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(fontsize=11)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f} ms",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")

        plt.xticks(rotation=20, ha="right", fontsize=10)
        plt.tight_layout()
        lat_png = out_dir / "latency_profile.png"
        plt.savefig(lat_png, dpi=150)
        plt.close(fig)
        logger.info(f"Saved latency profile plot to {lat_png}")
        plot_paths.append(lat_png)

    return plot_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RD curves from benchmark results")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    generate_plots(json_path=args.json, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

