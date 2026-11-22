"""Draw measured continuous two-player coverage for all twelve prespecified matches."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CARD = Path(__file__).resolve().parent
EXP = CARD.parent


def load(path):
    return json.loads(path.read_text())


def contiguous(ticks):
    starts = [t for t in sorted(ticks) if t - 1 not in ticks]
    stops = [t + 1 for t in sorted(ticks) if t + 1 not in ticks]
    return list(zip(starts, stops, strict=True))


def coverage(cases):
    terminal = load(CARD / "terminal-source-jobs.json")
    rows = []
    for item in cases:
        match = item["match"]
        record = dict(match=match, event=item["event"], court_ticks=item["court_ticks"])
        if item["court_ticks"]:
            expected = [
                EXP / "test-crop-joint-20261122" / f"{match}-result.json",
                EXP / "test-joint-extension-20261122" / f"{match}-crop-result.json",
                CARD / f"{match}-crop-result.json",
            ]
            sources = [p for p in expected if p.exists()]
            if not sources:
                jobs = {k: v for k, v in terminal.items() if k.startswith(match + "-")}
                assert jobs and any(v["state"] == "expired" for v in jobs.values())
                record.update(
                    status="unmeasured",
                    runs=[],
                    paired_ticks=None,
                    full=None,
                    expired_jobs=[v["job"] for v in jobs.values() if v["state"] == "expired"],
                )
                rows.append(record)
                continue
            if len(sources) != 1:
                raise ValueError(f"Expected one authenticated native source result for {match}")
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


def plot(rows, path):
    colors = dict(both="#12695d", missing="#dbdfe2", fallback="#e0bb86")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for y, row in enumerate(rows):
        color = colors["missing"] if row["court_ticks"] else colors["fallback"]
        if row["status"] == "unmeasured":
            ax.broken_barh(
                [(0, 240)],
                (y - 0.32, 0.64),
                facecolors="white",
                edgecolors="#9399a0",
                hatch="///",
                linewidth=0.5,
            )
        else:
            ax.broken_barh([(0, 240)], (y - 0.32, 0.64), facecolors=color)
        ax.broken_barh(
            [(a, b - a) for a, b in row["runs"]], (y - 0.32, 0.64), facecolors=colors["both"]
        )
        ax.text(
            244,
            y,
            f"{row['paired_ticks']}/240"
            if row["status"] == "measured"
            else ("ordinary" if row["status"] == "ordinary" else "unmeasured"),
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
        ]
        + [
            Patch(facecolor="white", edgecolor="#9399a0", hatch="///", label="Pose test unmeasured")
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


if __name__ == "__main__":
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
        unmeasured_pose_matches=sum(r["status"] == "unmeasured" for r in rows),
        paired_ticks_measured=sum(r["paired_ticks"] for r in rows if r["status"] == "measured"),
        ticks_in_measured_pose_tests=240 * sum(r["status"] == "measured" for r in rows),
        ticks_enrolled=240 * len(rows),
    )
    (CARD / "coverage-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(rows, CARD / "native-player-coverage.pdf")
