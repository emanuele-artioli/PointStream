"""Reproduce player mask and pose diagnostics from archived experiment receipts."""
import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "docs/experiments"
OUT = Path(__file__).resolve().parent
COLORS = ("#00796d", "#b15c22")


def read(path):
    return json.loads((EXP / path).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e9eb", linewidth=.7)
    ax.set_axisbelow(True)


def calculate_masks():
    audit_path = EXP / "receiver-replay-20261118/independent-idle-20261118/finish-full-audit.json"
    coverage_path = EXP / "clean-base-comparison-20261118/mask-coverage.json"
    review_path = EXP / "foreground-references-20261118/finished-review-1621utc.json"
    audit, coverage, review = [json.loads(p.read_text()) for p in (audit_path, coverage_path, review_path)]
    assert audit["all_frames_exact"] and audit["ticks"] == 240 and [r["tick"] for r in audit["rows"]] == list(range(240))
    assert review["ticks"] == list(range(10, 240, 20))
    unprompted = set(review["unprompted_ticks"])
    assert unprompted == {10, 90, 170}
    outputs = []
    audit_rows = {r["tick"]: r for r in audit["rows"]}
    for row in coverage["rows"]:
        tick = row["tick"]
        for index in (0, 1):
            ident = f"person:{index:012d}"
            unit = f"nadal_kyrgios__000_at120__person_{index:012d}"
            body = row["players"][ident]["body_pixels"]
            intersection = row["players"][ident]["covered_pixels"]
            area = audit_rows[tick]["projected_pixels"][unit]
            assert intersection <= body and intersection <= area
            assert intersection + row["players"][ident]["omitted_pixels"] == body
            union = body + area - intersection
            assert union > 0
            outputs.append(dict(tick=tick, player=index + 1, proposed=tick not in unprompted,
                                body_pixels=body, projected_pixels=area,
                                intersection_pixels=intersection, union_pixels=union, iou=intersection / union))
    assert len(outputs) == 24 and sorted({r["tick"] for r in outputs}) == review["ticks"]
    groups = {}
    for player in (1, 2):
        groups[str(player)] = {}
        for label, selection in (("all", lambda r: True), ("assisted", lambda r: r["proposed"]),
                                 ("unprompted", lambda r: not r["proposed"])):
            selected = [r for r in outputs if r["player"] == player and selection(r)]
            groups[str(player)][label] = {"count": len(selected),
                                          "mean_iou": float(np.mean([r["iou"] for r in selected]))}
    return dict(scope="one opened match, 12 reviewed frames; source-mask and receiver projection IoU",
                inputs={str(p.relative_to(ROOT)): sha(p) for p in (audit_path, coverage_path, review_path)},
                rows=outputs, means=groups)


def mask_figure(result):
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    for player, color in zip((1, 2), COLORS, strict=True):
        rows = [r for r in result["rows"] if r["player"] == player]
        ax.plot([r["tick"] for r in rows], [r["iou"] for r in rows],
                color=color, linewidth=1.3, label=f"Player {player}")
        for proposed, marker in ((True, "o"), (False, "s")):
            subset = [r for r in rows if r["proposed"] == proposed]
            ax.scatter([r["tick"] for r in subset], [r["iou"] for r in subset], color=color,
                       marker=marker, s=26, zorder=3)
    ax.set_ylim(.4, 1.01)
    ax.set_xlim(0, 240)
    ax.set_yticks([.4, .6, .8, 1.0])
    ax.set_xticks([10, 50, 90, 130, 170, 210, 230])
    ax.set_xlabel("Source frame in 240-frame rally")
    ax.set_ylabel("Intersection over union")
    ax.set_title("Receiver player masks compared with source labels", loc="left", fontweight="bold", fontsize=12)
    style(ax)
    player1 = result["means"]["1"]["all"]["mean_iou"]
    player2 = result["means"]["2"]["all"]["mean_iou"]
    ax.legend(loc="lower left", ncol=2, frameon=False, title=f"Frame means   P1 {player1:.3f}   P2 {player2:.3f}")
    fig.text(.14, .005, "□ Source drawn without proposal (3 frames)    ● Proposal reviewed by author (9 frames)", fontsize=8)
    fig.subplots_adjust(left=.13, right=.98, top=.84, bottom=.23)
    save(fig, "receiver-masks")


def pose_figure():
    source = EXP / "k4b.md"
    section = source.read_text().split("### 2026-10-21 — K4b.3: keypoint rate against keypoint error", 1)[1]
    section = section.split("18 of the 24 codes pass", 1)[0]
    records = []
    for line in section.splitlines():
        if not line.startswith("  |") or " | q" not in line.replace("**", ""):
            continue
        cells = [v.strip().replace("**", "") for v in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not cells[0].startswith("q "):
            continue
        rate = float(cells[1])
        far = float(re.match(r"\d+\.\d+", cells[2]).group())
        near = float(re.match(r"\d+\.\d+", cells[3]).group())
        median, p95 = [float(s.strip()) for s in cells[4].split("/")]
        records.append(dict(code=cells[0], rate=rate, far_oks=far, near_oks=near, median_px=median, p95_px=p95))
    assert len(records) == 6 and [(r["rate"], r["p95_px"]) for r in records] == [
        (15., .6), (11.7, 1.2), (9.5, 2.4), (6.6, 4.39), (3.3, 8.07), (2.6, 8.56)]
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for key, label, color, marker in (("far_oks", "Far player", COLORS[0], "o"),
                                      ("near_oks", "Near player", COLORS[1], "s")):
        ax[0].scatter([r["rate"] for r in records], [r[key] for r in records],
                      color=color, marker=marker, label=label, s=35, zorder=3)
    ax[0].axhline(.95, color="#829099", linewidth=1, linestyle="--")
    ax[0].set_ylim(.86, 1.01)
    ax[0].set_ylabel("Median OKS vs uncoded YOLO")
    ax[0].legend(frameon=False, loc="lower right")
    ax[1].scatter([r["rate"] for r in records], [r["p95_px"] for r in records], color=COLORS[0], s=35)
    ax[1].set_ylabel("95th percentile displacement (px)")
    ax[1].set_ylim(0, 10)
    for a in ax:
        a.set_xlim(0, 16.5)
        a.set_xlabel("Estimated rate for two players (kbps)")
        style(a)
    ax[0].set_title("Keypoint agreement", loc="left", fontweight="bold")
    ax[1].set_title("Large keypoint errors", loc="left", fontweight="bold")
    fig.suptitle("Keypoint coding against the original detector output", x=.10, ha="left", fontsize=12, fontweight="bold")
    fig.text(.10, .895, "48 development windows · rate is pooled entropy estimate · each point is a different code", fontsize=8)
    fig.text(.10, .025, "Selected code: 4 px quantization, every fourth frame · 3.3 kbps, 8.07 px p95", fontsize=8)
    fig.subplots_adjust(left=.10, right=.98, top=.75, bottom=.23, wspace=.35)
    save(fig, "pose-coding")
    return dict(scope="K4b.3 six reported examples; no human keypoint ground truth", source=str(source.relative_to(ROOT)),
                source_sha256=sha(source), records=records)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    masks = calculate_masks()
    mask_figure(masks)
    pose = pose_figure()
    (OUT / "results.json").write_text(json.dumps(dict(mask=masks, pose=pose), indent=2) + "\n")
