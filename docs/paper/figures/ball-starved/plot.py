"""Plot real three-match low-rate receipts; no interpolation or padded bytes."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RECEIPTS = HERE.parents[2] / "experiments" / "ball-confirmation-starved-20261122"
matches = ("209", "269", "235")
out = []
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True, constrained_layout=True)
for ax, match in zip(axes, matches):
    target = "target" if match == "235" else "final"
    chosen = json.loads((RECEIPTS / f"match{match}-{target}-result.json").read_text())
    tracked = next(r for r in chosen["results"] if r["mode"] == "tracked")
    rate = tracked["kbps"]
    candidates = []
    for f in sorted(RECEIPTS.glob(f"match{match}-*result.json")):
        data = json.loads(f.read_text())
        for row in data["results"]:
            if row["mode"] in ("plain", "roi") and row["kbps"] <= rate:
                candidates.append((sum(c["dists"] for c in row["crop"]) / len(row["crop"]), row, f.name))
    score, anchor, filename = min(candidates, key=lambda x: x[0])
    test_score = sum(c["dists"] for c in tracked["crop"]) / len(tracked["crop"])
    ax.plot([anchor["kbps"], rate], [score, test_score], color="0.6", lw=1)
    ax.scatter(anchor["kbps"], score, color="#244a73", marker="s", s=65, label="Best affordable AV1")
    ax.scatter(rate, test_score, color="#b84322", marker="o", s=65, label="Tracked ball + AV1")
    ax.annotate(f'{anchor["kbps"]:.2f} kbps\n{score:.3f}', (anchor["kbps"], score), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.annotate(f'{rate:.2f} kbps\n{test_score:.3f}', (rate, test_score), xytext=(0, -24), textcoords="offset points", ha="center", fontsize=8)
    ax.set_title(f'Match {match}  ·  {chosen["visible_labels"]} visible labels')
    ax.set_xlim(45, 65)
    ax.set_ylim(.19, .39)
    ax.grid(alpha=.2)
    ax.set_xlabel("Actual total rate (kbps)")
    out.append({"match": match, "visible_labels": chosen["visible_labels"], "anchor": {"mode": anchor["mode"], "crf": anchor["crf"], "kbps": anchor["kbps"], "dists": score, "receipt": filename}, "tracked": {"kbps": rate, "dists": test_score, "receipt": f"match{match}-{target}-result.json"}})
axes[0].set_ylabel("Ball crop DISTS (lower is better)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
fig.savefig(HERE / "ball-starved.pdf")
fig.savefig(HERE / "ball-starved.png", dpi=180)
(HERE / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
