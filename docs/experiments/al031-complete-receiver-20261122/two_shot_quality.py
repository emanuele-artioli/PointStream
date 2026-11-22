"""Compare genuine two-shot byte budgets on identical fixed source times and model region counts."""

import argparse
import json
from itertools import product
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
FIRST = ROOT.parent / "al028-neural-display-20261122"
TICKS = set(range(10, 240, 20))


def read(p):
    return json.loads(p.read_text())


def fixed(rows):
    chosen = [r for r in rows if r["tick"] in TICKS]
    assert len(chosen) == 24 and all(sum(r["tick"] == t for r in chosen) == 2 for t in TICKS)
    return mean(r["dists"] for r in chosen)


def return_ordinary(folders):
    output = {}
    for folder in folders:
        for f in sorted(folder.glob("*.json")):
            if f.name in ["arms.json", "summary.json"]:
                continue
            v = read(f)
            if "body_rows" not in v:
                continue
            a = v["arm"]
            assert (
                a["name"].startswith("ordinary")
                and v["frames"] == 240
                and len(v["body_rows"]) == 480
            )
            row = dict(
                bytes=a["bytes"],
                rate=a["kbps"],
                stream=a["stream_sha256"],
                dists=fixed(v["body_rows"]),
                supports=v["supports_sha256"],
                source=v["source_sha256"],
                fixed_rows=[r for r in v["body_rows"] if r["tick"] in TICKS],
            )
            if row["stream"] in output:
                assert output[row["stream"]] == row
            output[row["stream"]] = row
    return list(output.values())


def run(extended=False, include_frontier=False, retained=False, continuous=False):
    prices = read(ROOT / "two-shot-price.json")
    first = read(FIRST / "player-joint.json")
    assert (
        first["source_space"] == "review"
        and first["native_rates"]["joint"] == prices["first_scene_bytes"] * 8 / 9600
    )
    initial = fixed([r for r in first["rows"] if r["arm"] == "joint"])
    one = [
        dict(bytes=r["bytes"], stream=r["stream_sha256"], dists=fixed(r["rows"]))
        for r in read(FIRST / "ordinary-three-size-curves.json")["points"]
    ]
    two = read(ROOT / "return-fixed12-ordinary-arms.json")
    assert len(one) == 52 and len(two) >= 16 and len({r["stream"] for r in two}) == len(two)
    if retained:
        extra_return = sorted(ROOT.glob("ordinary-retained-*-scores.json"))
        assert len(extra_return) == 4
        digests = {v["stream"] for v in two}
        for file in extra_return:
            score = read(file)
            a = score["arm"]
            assert a["stream_sha256"] not in digests
            assert score["frames"] == 240 and len(score["body_rows"]) == 480
            two.append(
                dict(bytes=a["bytes"], stream=a["stream_sha256"], dists=fixed(score["body_rows"]))
            )
            digests.add(a["stream_sha256"])
    result = dict(
        source_ticks=sorted(TICKS),
        people_per_shot=2,
        first_joint_dists=initial,
        first_ordinary_count=len(one),
        second_ordinary_count=len(two),
        cases={},
    )
    if extended:
        extra = read(ROOT / "first-shot-extra-controls.json")
        for r in extra["points"]:
            item = dict(bytes=r["bytes"], stream=r["stream_sha256"], dists=fixed(r["rows"]))
            if item["stream"] not in {v["stream"] for v in one}:
                one.append(item)
        result["first_ordinary_count"] = len(one)
    paired = [
        dict(
            first_bytes=a["bytes"],
            second_bytes=b["bytes"],
            sum_bytes=a["bytes"] + b["bytes"],
            first_dists=a["dists"],
            second_dists=b["dists"],
            mean_dists=(a["dists"] + b["dists"]) / 2,
            first_sha=a["stream"],
            second_sha=b["stream"],
        )
        for a, b in product(one, two)
    ]
    if include_frontier:
        frontier = []
        for row in sorted(paired, key=lambda r: (r["sum_bytes"], r["mean_dists"])):
            if frontier and row["mean_dists"] >= frontier[-1]["mean_dists"]:
                continue
            frontier.append(dict(**row, kbps=row["sum_bytes"] * 8 / 19200))
        result["ordinary_measured_frontier"] = frontier
    for name, mode in [("send_new", "fresh"), ("reuse", "prior")]:
        allowance = prices[name]["total_bytes"]
        choices = [r for r in paired if r["sum_bytes"] <= allowance]
        assert choices
        best = min(choices, key=lambda r: (r["mean_dists"], r["sum_bytes"]))
        close = min(choices, key=lambda r: (allowance - r["sum_bytes"], r["mean_dists"]))
        for x in [best, close]:
            x["two_shot_kbps"] = x["sum_bytes"] * 8 / 19200
        case = dict(
            bytes=allowance,
            rate=prices[name]["two_shot_kbps"],
            affordable_pairs=len(choices),
            best_ordinary=best,
            nearest_ordinary=close,
        )
        scored = ROOT / f"{mode}-joint-q50-scores.json"
        if scored.exists():
            d = read(scored)
            curr = fixed(d["body_rows"])
            value = (initial + curr) / 2
            case.update(
                return_dists=curr,
                mean_pointstream_dists=value,
                percent_lower_than_best_ordinary=100 * (1 - value / best["mean_dists"]),
            )
        if continuous:
            stream = read(ROOT / "two-window-continuous-result.json")["points"][name]
            assert stream["bytes"] <= allowance and len(stream["rows"]) == 48
            assert abs(stream["dists"] - mean(stream["per_shot"])) < 1e-8
            case["one_stream_ordinary"] = {
                "bytes": stream["bytes"],
                "kbps": stream["kbps"],
                "dists": stream["dists"],
                "per_shot": stream["per_shot"],
                "frame_at_return": stream["at_return"],
                "stream_sha256": stream["stream_sha256"],
            }
            winner = min(
                [
                    dict(
                        coding="independent", kbps=best["two_shot_kbps"], dists=best["mean_dists"]
                    ),
                    dict(coding="one_stream", kbps=stream["kbps"], dists=stream["dists"]),
                ],
                key=lambda item: (item["dists"], item["kbps"]),
            )
            case["best_ordinary_any"] = winner
            case["break_even_return_dists"] = 2 * winner["dists"] - initial
            if "mean_pointstream_dists" in case:
                case["percent_lower_than_best_including_one_stream"] = 100 * (
                    1 - case["mean_pointstream_dists"] / winner["dists"]
                )
        result["cases"][name] = case
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ordinary-folders", type=Path, nargs="+")
    p.add_argument("--extended", action="store_true")
    p.add_argument("--frontier", action="store_true")
    p.add_argument("--retained", action="store_true")
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    data = (
        return_ordinary(a.ordinary_folders)
        if a.ordinary_folders
        else run(a.extended, a.frontier, a.retained, a.continuous)
    )
    a.out.write_text(json.dumps(data, indent=2) + "\n")
