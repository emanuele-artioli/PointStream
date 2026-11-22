"""Plot held-out downstream response, preserving each reference image and complete initialization price."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
CARDS = HERE.parents[2] / "experiments"
TRANSFER = CARDS / "reference-transfer-al028-20261122"


def load(path):
    return json.loads(path.read_text())


def main():
    metrics = load(TRANSFER / "all-arm-results.json")
    q90 = load(TRANSFER / "image-oracle.json")
    images = load(TRANSFER / "jpeg-webp-oracle.json")["rows"]
    stream = load(CARDS / "reviewed-foreground-20261122/matched-three.json")
    running = stream["actual_kbps"]["joint"]
    assert stream["joint_startup_bytes"] == q90["source_cache_bytes"]
    sizes = {
        m: (images[m]["image_bytes"], images[m]["envelope_bytes"]) for m in ("jpeg40", "jpeg70")
    }
    sizes["q90"] = q90["avif_image_bytes"], q90["avif_cache_bytes"]
    sizes["lossless"] = images["webp"]["image_bytes"], images["webp"]["envelope_bytes"]
    rows = {}
    for m, (image_bytes, envelope) in sizes.items():
        rows[m] = dict(
            image_bytes=image_bytes,
            initialization_bytes=envelope,
            source_model_dists=metrics["source_model"][m]["dists"],
            body_rgb_psnr=metrics["source_model"][m]["rgb_psnr"],
            temporal_error=metrics["temporal"][m],
            reviewed_tick_dists=metrics["human_accepted"][m]["dists"],
            beats_lossless_dists_ticks=sum(
                v["scores"][m]["dists"] < v["scores"]["lossless"]["dists"]
                for v in metrics["body_scores"]
            ),
            active_seconds_at_33_kbps=8 * envelope / (1000 * (33 - running)),
            equivalent_kbps_by_valid_active_seconds={
                str(t): running + 8 * envelope / (t * 1000) for t in (9.6, 30, 120, 300)
            },
        )
    output = dict(
        shot="Alcaraz–Hurkacz/028",
        measured_stream_kbps_with_cache=running,
        actual_original_png_init_bytes=q90["source_cache_bytes"],
        actual_original_png_image_bytes=q90["source_image_bytes"],
        actual_original_png_cold_9_6s_kbps=running + 8 * q90["source_cache_bytes"] / 9600,
        mode_rows=rows,
        note="Physical initializations, identical non-appearance streams. The lossy generator assessment is the first 24 frames only. Combined longer-play prices are scenarios conditional on retaining a valid appearance.",
    )
    (TRANSFER / "wire-price-and-quality.json").write_text(json.dumps(output, indent=2) + "\n")
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    for name, key, axis_title in [
        ("rd", "source_model_dists", "Player-body DISTS  ↓"),
        ("motion", "temporal_error", "Body motion residual error  ↓"),
    ]:
        fig, ax = plt.subplots(figsize=(4.25, 2.6))
        fig.subplots_adjust(left=0.175, right=0.97, top=0.9, bottom=0.23)
        for mode, tag, colour, marker in [
            ("jpeg40", "JPEG Q40", "#b34d00", "^"),
            ("jpeg70", "JPEG Q70", "#b34d00", "^"),
            ("q90", "AVIF Q90", "#00787e", "o"),
            ("lossless", "Lossless WebP", "#753c77", "D"),
        ]:
            point = rows[mode]
            x, y = point["image_bytes"] / 1000, point[key]
            ax.scatter([x], [y], s=25, marker=marker, color=colour, zorder=3)
            offset = (21, 2) if mode == "jpeg70" else (0, 7)
            ax.annotate(
                tag, (x, y), xytext=offset, textcoords="offset points", ha="center", fontsize=7
            )
        ax.plot(
            [rows[m]["image_bytes"] / 1000 for m in ("jpeg40", "jpeg70")],
            [rows[m][key] for m in ("jpeg40", "jpeg70")],
            color="#b34d00",
            linewidth=1.1,
            zorder=2,
        )
        ymin, ymax = min(v[key] for v in rows.values()), max(v[key] for v in rows.values())
        span = max(ymax - ymin, 1e-4)
        ax.set(
            xlabel="Two transmitted reference images (kB)",
            ylabel=axis_title,
            xlim=(18, 110),
            ylim=(ymin - 0.20 * span, ymax + 0.35 * span),
        )
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", color="#d7dce0", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"al028-reference-{name}.{ext}", dpi=210)
        plt.close(fig)


if __name__ == "__main__":
    main()
