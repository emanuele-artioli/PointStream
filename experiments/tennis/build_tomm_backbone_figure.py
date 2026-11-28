"""Render the TOMM multi-backbone quality-vs-speed figure from verified docs/experiments JSON receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def build_figure(receipt_dir: Path, out_pdf: Path, out_png: Path | None = None) -> None:
    mb_path = receipt_dir / "tomm-multi-backbone-20261128-scores.json"
    if not mb_path.exists():
        mb_path = receipt_dir / "tomm-multi-backbone-scores.json"
    mb = _load(mb_path)
    mm = _load(receipt_dir / "tomm-backbone-mimicmotion-20261128-scores.json")
    wan = _load(receipt_dir / "tomm-backbone-wan-animate-20261128-scores.json")
    wv2 = _load(receipt_dir / "tomm-warp-v2-pilot-scores.json")
    w240 = _load(receipt_dir / "tomm-warp-native240-match102-scores.json")

    frames = 48.0
    aa = mb["backbones"]["aa_dpm10_ov8_smooth"]
    cn = mb["backbones"]["cn_openpose_sd15_dpm10"]
    k6 = mb["backbones"]["k6_ref_warp"]
    mm_b = mm["backbones"]["mimicmotion_svdx"]
    wan_b = wan["backbones"]["wan_animate_14b"]
    mls_sam = wv2["variants"]["v2_mls_sam"]
    mls_e2e = w240["arms"]["mls_step8_bgsub_alpha"]

    ddim30_fps = frames / float(mb["ddim30_48f_seconds_ref"])
    ddim30_dists = 0.3032

    points_a = [
        ("DDIM-30 (Moore-AA)", ddim30_fps, ddim30_dists, "gpu_lora", (0, 15), "left"),
        ("Wan2.2-Animate\n(14B DiT, 0-shot)", frames / wan_b["render_s_total"], wan_b["masked_dists"], "gpu_zs", (-10, -28), "left"),
        ("MimicMotion 1.1\n(SVD-XT, 0-shot)", frames / mm_b["render_s_total"], mm_b["masked_dists"], "gpu_zs", (12, -26), "left"),
        ("Moore-AA\n(DPM-10, LoRA)", frames / aa["render_s_total"], aa["masked_dists"], "gpu_lora", (6, 12), "left"),
        ("ControlNet-SD1.5\n(2D, 0-shot)", frames / cn["render_s_total"], cn["masked_dists"], "gpu_2d", (8, -2), "left"),
        ("MLS v2\n(SAM, 1-CPU)", frames / mls_sam["timing"]["total_receiver_seconds"], mls_sam["dists_mean"], "cpu", (-8, 15), "right"),
        ("MLS v2 0-SAM\n(e2e 30.8 fps)", mls_e2e["timing"]["fps_end_to_end"], mls_e2e["prefix48_dists_mean"], "cpu_e2e", (-10, -24), "right"),
        ("Gaussian warp\n(k6, 1-CPU)", frames / k6["render_s_total"], k6["masked_dists"], "cpu", (4, 18), "right"),
    ]

    points_b = [
        ("Moore-AA (DPM-10)", aa["tlpips_mean"], aa["yolo26x_pose_oks_mean"], "gpu_lora", (-8, 3), "right"),
        ("Wan2.2-Animate\n(14B DiT)", wan_b["tlpips_mean"], wan_b["yolo26x_pose_oks_mean"], "gpu_zs", (-8, -2), "right"),
        ("MimicMotion 1.1\n(SVD-XT)", mm_b["tlpips_mean"], mm_b["yolo26x_pose_oks_mean"], "gpu_zs", (-8, 2), "right"),
        ("ControlNet-SD1.5\n(2D)", cn["tlpips_mean"], cn["yolo26x_pose_oks_mean"], "gpu_2d", (8, 0), "left"),
        ("MLS v2 (1-CPU)", mls_sam["tlpips_mean"], mls_sam["yolo_pose_mean_oks"], "cpu", (-8, 0), "right"),
        ("Gaussian warp\n(k6, 1-CPU)", k6["tlpips_mean"], k6["yolo26x_pose_oks_mean"], "cpu", (-8, -6), "right"),
    ]

    styles = {
        "gpu_lora": dict(color="#1b5f8a", marker="o", s=64, edgecolor="white", linewidth=0.8, zorder=5),
        "gpu_zs": dict(color="#6c4f99", marker="D", s=56, edgecolor="white", linewidth=0.8, zorder=5),
        "gpu_2d": dict(color="#c45147", marker="X", s=64, edgecolor="white", linewidth=0.8, zorder=5),
        "cpu": dict(color="#2a8a5b", marker="s", s=56, edgecolor="white", linewidth=0.8, zorder=5),
        "cpu_e2e": dict(color="#e07a1f", marker="*", s=125, edgecolor="#5c2d06", linewidth=0.6, zorder=6),
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.4, 3.65), gridspec_kw={"width_ratios": [1.35, 1.0]}, layout="constrained"
    )

    ax1.axvspan(29.97, 130, color="#2a8a5b", alpha=0.07, zorder=0)
    ax1.axvline(29.97, color="#2a8a5b", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
    ax1.text(33.0, 0.415, "29.97 fps\nreal-time", fontsize=7.2, color="#1d6643", va="top", ha="left")

    ax1.axhline(0.524, color="#666666", linestyle=":", linewidth=1.1, zorder=1)
    ax1.text(2.4, 0.519, "SVT-AV1 (17.7 kbps, 0.524)", fontsize=7.2, color="#555555", va="bottom", ha="left")
    ax1.axhline(0.438, color="#8a5a44", linestyle="-.", linewidth=1.1, zorder=1)
    ax1.text(0.055, 0.443, "DCVC-UF (17.4 kbps, 0.438)", fontsize=7.2, color="#734834", va="top", ha="left")

    pareto_pts = sorted(
        [(fps, dists) for _, fps, dists, _, _, _ in points_a],
        key=lambda t: t[0],
        reverse=True,
    )
    frontier = []
    best_d = 999.0
    for fps, dists in pareto_pts:
        if dists < best_d:
            frontier.append((fps, dists))
            best_d = dists
    frontier.reverse()
    ax1.plot(
        [p[0] for p in frontier],
        [p[1] for p in frontier],
        color="#1b5f8a",
        linestyle="-",
        linewidth=1.1,
        alpha=0.35,
        zorder=2,
    )

    for label, fps, dists, grp, (dx, dy), ha in points_a:
        ax1.scatter([fps], [dists], **styles[grp])
        ax1.annotate(
            label,
            xy=(fps, dists),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            ha=ha,
            va="center",
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5, shrinkA=0, shrinkB=4)
            if abs(dx) > 7 or abs(dy) > 10
            else None,
        )

    ax1.set_xscale("log")
    ax1.set_xlim(0.048, 125)
    ax1.set_ylim(0.555, 0.268)
    ax1.set_xlabel("Effective 2-player throughput (video FPS, log scale) $\\rightarrow$ faster")
    ax1.set_ylabel("Masked player DISTS (inverted: higher is better)")
    ax1.set_title("(a) Perceptual quality vs. speed (15.5 kbps warm)", fontsize=9.0, fontweight="bold")
    ax1.grid(alpha=0.22, which="major")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1b5f8a", markersize=6.5, label="GPU 3D diffusion (LoRA)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#6c4f99", markersize=6.0, label="GPU 3D diffusion (zero-shot)"),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#c45147", markersize=6.5, label="GPU 2D diffusion (zero-shot)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2a8a5b", markersize=6.0, label="1-CPU warp (SAM masks)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e07a1f", markeredgecolor="#5c2d06", markersize=9.0, label="1-CPU warp (0-GPU, 0-SAM e2e)"),
    ]
    ax1.legend(
        handles=legend_handles,
        fontsize=6.6,
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="#dddddd",
    )

    for label, tlpips, oks, grp, (dx, dy), ha in points_b:
        ax2.scatter([tlpips], [oks], **styles[grp])
        ax2.annotate(
            label,
            xy=(tlpips, oks),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            ha=ha,
            va="center",
            color="#222222",
        )

    ax2.set_xlim(0.122, 0.036)
    ax2.set_ylim(0.71, 0.935)
    ax2.set_xlabel("Temporal LPIPS (inverted: right is smoother)")
    ax2.set_ylabel("Post-decode YOLO26x-pose OKS $\\uparrow$")
    ax2.set_title("(b) Temporal stability vs. pose fidelity (48f)", fontsize=9.0, fontweight="bold")
    ax2.grid(alpha=0.22, which="major")
    ax2.annotate(
        "Better",
        xy=(0.088, 0.922),
        xytext=(0.112, 0.902),
        fontsize=7.4,
        color="#1d6643",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1d6643", lw=1.1),
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--receipt-dir", type=Path, default=Path("docs/experiments"))
    p.add_argument("--out-pdf", type=Path, required=True)
    p.add_argument("--out-png", type=Path, default=None)
    args = p.parse_args()
    build_figure(args.receipt_dir, args.out_pdf, args.out_png)
    print(f"Saved {args.out_pdf}")


if __name__ == "__main__":
    main()
