"""Calculate equal-source, paired outcomes for every frozen official TEST clip."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261122
REPLICATES = 10000
GROUPS = ("target", "other-matched", "sensitivity", "unmatched")


def aggregate(rows, key):
    use = [r for r in rows if r.get("candidate_dists") is not None and r.get(key) is not None]
    if not use:
        return None
    values = np.array([[r["candidate_dists"], r[key]] for r in use])
    rng = np.random.default_rng(SEED)
    resampled = values[rng.integers(len(values), size=(REPLICATES, len(values)))].mean(axis=1)
    means = values.mean(axis=0)
    return dict(
        n=len(use),
        candidate=float(means[0]),
        ordinary=float(means[1]),
        difference=float(means[0] - means[1]),
        gain_percent=float(100 * (1 - means[0] / means[1])),
        mean_within_clip_gain_percent=float(np.mean(100 * (1 - values[:, 0] / values[:, 1]))),
        difference_95=np.quantile(resampled[:, 0] - resampled[:, 1], [0.025, 0.975]).tolist()
        if len(values) > 1
        else None,
        gain_percent_95=np.quantile(
            100 * (1 - resampled[:, 0] / resampled[:, 1]), [0.025, 0.975]
        ).tolist()
        if len(values) > 1
        else None,
        wins=int((values[:, 0] < values[:, 1]).sum()),
    )


def counts(rows):
    vis = [r for r in rows if r["visibility"] == 1]
    absent = [r for r in rows if r["visibility"] == 0]
    return dict(
        visible=len(vis),
        hit5=sum(
            r["error_native_pixels"] is not None and r["error_native_pixels"] <= 5 for r in vis
        ),
        hit10=sum(
            r["error_native_pixels"] is not None and r["error_native_pixels"] <= 10 for r in vis
        ),
        marked_absent=len(absent),
        emitted_absent=sum(r["present"] for r in absent),
        visible_position=sum(r["present"] for r in vis),
        unknown=sum(r["visibility"] not in (0, 1) for r in rows),
    )


def summarize(data):
    stage = json.loads((HERE / "staged.json").read_text())["selection"]
    rows = []
    all_marks = []
    rgb_rows = {arm: [] for arm in ("source", "ordinary", "candidate")}
    srctotal = []
    for m in stage:
        match = m["match"]
        dest = data / f"{match}-score.json"
        row = dict(match=match, rank=len(rows) + 1, fps=m["video"]["avg_frame_rate"])
        rows.append(row)
        if not dest.exists():
            row["missing_primary"] = True
            continue
        s = json.loads(dest.read_text())
        assert (
            s["source_sha256"] == m["source"]["sha256"]
            and s["labels_sha256"] == m["labels"]["sha256"]
        )
        row.update(
            selection=s["selection"],
            cadence=s["fps"],
            court_ticks=s["court_ticks"],
            ordinary_kbps=s["ordinary_kbps"],
            candidate_kbps=s["candidate_kbps"],
            gap_fraction=s["gap_fraction"],
            ordinary_dists=s["ordinary"]["ball_dists"],
            candidate_dists=s["candidate"]["ball_dists"],
            strongest_dists=None
            if not s["best_affordable"]
            else s["best_affordable"]["ball_dists"],
            strongest_kbps=None if not s["best_affordable"] else s["best_affordable"]["kbps"],
            ordinary_psnr=s["ordinary"]["whole_rgb_psnr"],
            candidate_psnr=s["candidate"]["whole_rgb_psnr"],
            unlisted=s["unlisted"],
            direct=counts(s["direct"]),
        )
        assert row["selection"] in GROUPS
        all_marks += s["direct"]
        neural = data / f"{match}-neuroscore.json"
        if neural.exists():
            ns = json.loads(neural.read_text())
            assert (
                ns["source_rgb_sha256"] == s["source_rgb_sha256"] and ns["visible"] == s["visible"]
            )
            row.update(
                neural_kbps=ns["kbps"],
                neural_dists=ns["ball_dists"],
                neural_psnr=ns["whole_rgb_psnr"],
                neural_affordable=ns["bytes"] <= max(s["candidate_bytes"], s["ordinary_bytes"]),
            )
        judge = data / f"{match}-judge-ada-fixed.json"
        if not judge.exists():
            judge = data / f"{match}-judge-ada.json"
        if not judge.exists():
            judge = data / f"{match}-judge.json"
        if judge.exists():
            j = json.loads(judge.read_text())
            assert j["rgb_sha256"]["source"] == s["source_rgb_sha256"]
            row["source_qualified"] = j["source_qualified"]
            row["rgb"] = {k: counts(v["rows"]) for k, v in j["detections"].items()}
            srctotal += j["detections"]["source"]["rows"]
            if j["source_qualified"]:
                for key, vs in j["detections"].items():
                    rgb_rows[key] += vs["rows"]
    groups = {}
    for name, gr in [(name, [r for r in rows if r.get("selection") == name]) for name in GROUPS] + [
        ("matched-actual", [r for r in rows if r.get("selection") in GROUPS[:2]])
    ]:
        paired = [r for r in gr if r.get("candidate_dists") is not None]
        groups[name] = dict(
            clips=len(gr),
            same_geometry=aggregate(gr, "ordinary_dists"),
            strongest_affordable=aggregate(gr, "strongest_dists"),
            matched_psnr_delta_db=np.mean(
                [r["candidate_psnr"] - r["ordinary_psnr"] for r in gr]
            ).item()
            if gr
            else None,
            with_visible=len(paired),
            affordable_neural=aggregate(
                [r for r in gr if r.get("neural_affordable")], "neural_dists"
            ),
        )
    visible = [r for r in rows if r.get("direct", {}).get("visible")]
    return dict(
        seed=SEED,
        resamples=REPLICATES,
        count=len(rows),
        scored=sum("direct" in r for r in rows),
        scored_regardless_of_scene_eligibility=True,
        rows=rows,
        groups=groups,
        combined_direct=counts(all_marks),
        macro_hit10=float(np.mean([r["direct"]["hit10"] / r["direct"]["visible"] for r in visible]))
        if visible
        else None,
        source_qualified=sum(bool(r.get("source_qualified")) for r in rows),
        judge_count=sum("source_qualified" in r for r in rows),
        rgb_source_all_scored=counts(srctotal),
        rgb_source_qualified={k: counts(v) for k, v in rgb_rows.items()},
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE / "test-physical")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.data)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        result["scored"],
        "/",
        result["count"],
        result["combined_direct"],
        result["groups"]["target"],
    )
