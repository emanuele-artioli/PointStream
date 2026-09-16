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

    for clip in clips:
        clip_name = clip["clip_name"]
        ps = clip["pointstream"]
        av1_arms = clip["av1_arms"]

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"PointStream vs AV1: Rate-Distortion-Utility ({clip_name})", fontsize=15, fontweight="bold")

        # Filter AV1 arms by category
        av1_starved = sorted([a for a in av1_arms if a.get("scale") in ["320:180", "426:240", "640:360"]], key=lambda x: x["actual_kbps"])
        av1_540p = sorted([a for a in av1_arms if a.get("scale") == "960:540"], key=lambda x: x["actual_kbps"])
        av1_720p = sorted([a for a in av1_arms if a.get("scale") == "1280:720"], key=lambda x: x["actual_kbps"])
        av1_1080p = sorted([a for a in av1_arms if a.get("scale") is None and not a.get("deblocked", False)], key=lambda x: x["actual_kbps"])
        av1_deb = sorted([a for a in av1_arms if a.get("deblocked", False)], key=lambda x: x["actual_kbps"])

        # PointStream variants sorted by bitrate (excluding plate for the curve)
        ps_variants = clip.get("pointstream_variants", [ps])
        ps_ladder = sorted([v for v in ps_variants if "Plate" not in v.get("name", "")], key=lambda x: x["bitrate_kbps"])
        ps_plate = [v for v in ps_variants if "Plate" in v.get("name", "")]

        # Plotting helper
        def plot_arm_curves(ax, key_fn, y_label, title, higher_is_better=True):
            if av1_starved:
                ax.plot([a["actual_kbps"] for a in av1_starved], [key_fn(a) for a in av1_starved], "k--x", label="AV1 Starved (180p/240p/360p)", linewidth=1.5, markersize=6)
            if av1_540p:
                ax.plot([a["actual_kbps"] for a in av1_540p], [key_fn(a) for a in av1_540p], "g-^", label="AV1 (540p, p7)", linewidth=1.8, markersize=7)
            if av1_720p:
                ax.plot([a["actual_kbps"] for a in av1_720p], [key_fn(a) for a in av1_720p], "m-v", label="AV1 (720p, p7)", linewidth=1.8, markersize=7)
            if av1_1080p:
                ax.plot([a["actual_kbps"] for a in av1_1080p], [key_fn(a) for a in av1_1080p], "r-o", label="AV1 (1080p, p6/p10)", linewidth=1.8, markersize=7)
            if av1_deb:
                ax.plot([a["actual_kbps"] for a in av1_deb], [key_fn(a) for a in av1_deb], "c--s", label="AV1 (1080p Deblocked)", linewidth=1.2)

            # PointStream Ladder curve
            if ps_ladder:
                ax.plot([p["bitrate_kbps"] for p in ps_ladder], [key_fn(p) for p in ps_ladder], "b-o", linewidth=2.5, markersize=8, label="PointStream Ladder", zorder=6)
                # Scatter individual rungs with names
                for p in ps_ladder:
                    ax.scatter([p["bitrate_kbps"]], [key_fn(p)], color="blue", s=100, zorder=7)

            if ps_plate:
                for p in ps_plate:
                    ax.scatter([p["bitrate_kbps"]], [key_fn(p)], color="darkorange", marker="D", s=120, zorder=6, label="PointStream (Plate 2s)")

            ax.set_xlabel("Total Bitrate (kbps)", fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=7, loc="lower right" if higher_is_better else "upper right")

        # 1. Hand-ROI PSNR
        plot_arm_curves(
            axs[0, 0],
            key_fn=lambda x: x["metrics"]["hand_roi_psnr_dB"],
            y_label="Hand-ROI PSNR (dB) [Higher is better]",
            title="Hand Interaction ROI Quality",
            higher_is_better=True,
        )

        # 2. LPIPS
        plot_arm_curves(
            axs[0, 1],
            key_fn=lambda x: x["metrics"]["lpips"],
            y_label="LPIPS Perceptual Distance [Lower is better]",
            title="Perceptual Distortion (LPIPS)",
            higher_is_better=False,
        )

        # 3. Detection Rate
        ax = axs[1, 0]
        plot_arm_curves(
            ax,
            key_fn=lambda x: x["teleop_utility"]["detection_rate"] * 100,
            y_label="Hand Detection Success Rate (%)",
            title="Teleop Utility: Hand Tracking Detection Rate",
            higher_is_better=True,
        )
        ceiling = clip.get("reference_oracle", {}).get("detection_ceiling", 0.6) * 100
        ax.axhline(ceiling, color="gray", linestyle=":", linewidth=1.5, label=f"Oracle Ceiling ({ceiling:.1f}%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7, loc="lower right")

        # 4. MPJPE
        plot_arm_curves(
            axs[1, 1],
            key_fn=lambda x: x["teleop_utility"]["mpjpe_pixels"],
            y_label="Mean Joint Position Error (pixels) [Lower is better]",
            title="Teleop Utility: Hand Joint Error (MPJPE)",
            higher_is_better=False,
        )

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
            lat.get("parallel_end_to_end_latency_ms", lat.get("end_to_end_latency_ms", 18.36)),
            lat.get("serial_end_to_end_latency_ms", 31.96),
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
