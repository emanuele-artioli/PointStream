"""Draw measured continuous two-player coverage for all 12 official TEST matches."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
COV_DIR = EXP / "official-test-player-coverage-20261122"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def contiguous(ticks: set[int]) -> list[tuple[int, int]]:
    starts = [t for t in sorted(ticks) if t - 1 not in ticks]
    stops = [t + 1 for t in sorted(ticks) if t + 1 not in ticks]
    return list(zip(starts, stops, strict=True))


def coverage(cases: list[dict]) -> list[dict]:
    rows = []
    for item in cases:
        match = item["match"]
        record: dict = dict(match=match, event=item["event"], court_ticks=item["court_ticks"])
        if item["court_ticks"]:
            candidates = [
                EXP / "test-crop-joint-20261122" / f"{match}-result.json",
                EXP / "test-joint-extension-20261122" / f"{match}-crop-result.json",
                COV_DIR / f"{match}-crop-result.json",
                HERE / f"{match}-crop-result.json",
            ]
            sources = [p for p in candidates if p.exists()]
            if len(sources) != 1:
                raise ValueError(f"Expected 1 source crop receipt for {match}, got {sources}")
            result = load(sources[0])
            assert result["event"] == match and result["frames"] == 240
            tracks = list(result["tracks"].values())
            assert len(tracks) == 2
            missed = [set(t["missing"]) for t in tracks]
            assert all(len(m) == 240 - t["fitted"] for t, m in zip(tracks, missed, strict=True))
            paired = set(range(240)).difference(*missed)
            record.update(
                status="measured",
                receipt=str(sources[0].relative_to(EXP)),
                source_sha256=result["source_sha256"],
                runs=contiguous(paired),
                paired_ticks=len(paired),
                full=len(paired) == 240,
            )
        else:
            record.update(status="ordinary", runs=[], paired_ticks=0, full=False)
        rows.append(record)
    return rows


def plot(rows: list[dict], path: Path) -> None:
    colors = dict(both="#12695d", missing="#dbdfe2", fallback="#e0bb86")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for y, row in enumerate(rows):
        color = colors["missing"] if row["court_ticks"] else colors["fallback"]
        ax.broken_barh([(0, 240)], (y - 0.32, 0.64), facecolors=color)
        ax.broken_barh(
            [(a, b - a) for a, b in row["runs"]], (y - 0.32, 0.64), facecolors=colors["both"]
        )
        ax.text(
            244,
            y,
            f"{row['paired_ticks']}/240" if row["status"] == "measured" else "ordinary",
            va="center",
            fontsize=7.1,
        )
    ax.set_yticks(range(len(rows)), [r["event"] for r in rows], fontsize=7.2)
    ax.set(
        xlim=(0, 310),
        ylim=(len(rows) - 0.5, -0.5),
        xlabel="Native frame in the selected shot",
        xticks=[0, 60, 120, 180, 240],
    )
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 240)
    ax.legend(
        handles=[
            Patch(color=colors[k], label=s)
            for k, s in (
                ("both", "Both players tracked"),
                ("missing", "Incomplete tracks"),
                ("fallback", "Court selector chose ordinary video"),
            )
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncol=2,
        frameon=False,
        fontsize=7.1,
        handlelength=1.2,
        columnspacing=1,
    )
    fig.subplots_adjust(left=0.23, right=0.95, top=0.85, bottom=0.14)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def main() -> None:
    test = load(EXP / "cohort-expansion-20261122" / "cohort-results.json")["rows"]
    assert len(test) == 12 and len({r["match"] for r in test}) == 12
    rows = coverage(test)
    summary = dict(
        rows=rows,
        enrolled_matches=len(rows),
        measured_pose_matches=sum(r["status"] == "measured" for r in rows),
        full_joint_pose_matches=sum(r["full"] is True for r in rows),
        partial_pose_matches=sum(r["status"] == "measured" and not r["full"] for r in rows),
        ordinary_route_matches=sum(r["status"] == "ordinary" for r in rows),
        unmeasured_pose_matches=0,
        paired_ticks_measured=sum(r["paired_ticks"] for r in rows if r["status"] == "measured"),
        ticks_in_measured_pose_tests=240 * sum(r["status"] == "measured" for r in rows),
        ticks_enrolled=240 * len(rows),
    )
    (HERE / "coverage-summary-12.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(rows, HERE / "native-player-coverage.pdf")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
