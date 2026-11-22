"""Summarize physical tennis comparisons with matches as the observation units."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from statistics import mean

BASELINES = ("av1", "dcvc_uf")
METRIC_DIRECTIONS = {
    "dists": "lower",
    "lpips": "lower",
    "ssim": "higher",
    "psnr": "higher",
    "ms_ssim": "higher",
    "temporal_luma_error": "lower",
}


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    return ordered[low] + (position - low) * (
        ordered[min(low + 1, len(ordered) - 1)] - ordered[low]
    )


def summarize(by_match, draws):
    means = list(by_match.values())
    rng = random.Random(20261123)
    bootstrap = [mean(rng.choices(means, k=len(means))) for _ in range(draws)]
    return {
        "match_means": by_match,
        "estimate": mean(means),
        "match_bootstrap95": [quantile(bootstrap, p) for p in (0.025, 0.975)],
        "favorable_matches": sum(v > 0 for v in means),
    }


def validate_row(row, resolution):
    if row["route"] not in ("joint", "ball", "conventional"):
        raise ValueError("unknown delivery route")
    if set(row["streams"]) != {"pointstream", *BASELINES}:
        raise ValueError("expected one complete registered physical stream per codec family")
    if row["ticks"] != list(range(*resolution)):
        raise ValueError("delivery must cover every registered native frame")
    for arm in row["streams"].values():
        if not 0 < arm["bytes"] <= row["byte_ceiling"]:
            raise ValueError("a physical stream exceeds the registered byte ceiling")
        if len(arm["sha256"]) != 64:
            raise ValueError("stream has no registered SHA256")
        for scores in arm["scores"].values():
            if scores is not None and any(
                name not in METRIC_DIRECTIONS or not math.isfinite(value)
                for name, value in scores.items()
            ):
                raise ValueError("unregistered metric or invalid score")
    ps = row["streams"]["pointstream"]
    if row["route"] == "conventional" and not ps.get("fallback_codec"):
        raise ValueError("conventional fallback needs a named delivered codec")
    for area in row["supports"]:
        if len(row["supports"][area]["sha256"]) != 64:
            raise ValueError("paired support lacks an identity")
        if any(area not in arm["scores"] for arm in row["streams"].values()):
            raise ValueError("all codec scores need the same support declarations")


def aggregate(manifest, records, draws=10000):
    events = manifest["events"]
    if len(events) != 6 or any(len(event["rallies"]) != 3 for event in events):
        raise ValueError("expected six registered events with three rallies each")
    if sum(event["held_out"] for event in events) < 2:
        raise ValueError("the study needs two registered held-out matches")
    if len({event["event_id"] for event in events}) != 6:
        raise ValueError("match IDs must be distinct")
    enrolled = {
        rally["rally_id"]: (event["event_id"], event["held_out"], rally["native_range"])
        for event in events
        for rally in event["rallies"]
    }
    if len(enrolled) != 18:
        raise ValueError("rally IDs must be distinct")
    seen = {}
    routes = Counter()
    for row in records:
        name = row["rally_id"]
        if name not in enrolled or name in seen:
            raise ValueError(f"unexpected or duplicate rally {name}")
        match, _, resolution = enrolled[name]
        if resolution is None or row["event_id"] != match:
            raise ValueError(f"record does not identify a frozen source rally {name}")
        validate_row(row, resolution)
        seen[name] = row
        routes[row["route"]] += 1
    out = {
        "enrolled": 18,
        "complete_deliveries": len(seen),
        "unresolved_rallies": sorted(set(enrolled) - seen.keys()),
        "routes": dict(routes),
        "results": {},
    }
    if len(seen) != 18:
        return out

    targets = set.intersection(*(set(r["supports"]) for r in seen.values()))
    for target in sorted(targets):
        for metric, direction in METRIC_DIRECTIONS.items():
            for baseline in BASELINES:
                by_match = {}
                for name, row in seen.items():
                    streams = row["streams"]
                    scores = [streams[k]["scores"][target] for k in ("pointstream", baseline)]
                    if any(score is None or metric not in score for score in scores):
                        break
                    difference = scores[0][metric] - scores[1][metric]
                    if direction == "lower":
                        difference *= -1
                    by_match.setdefault(enrolled[name][0], []).append(difference)
                else:
                    match_means = {k: mean(xs) for k, xs in by_match.items()}
                    pooled = summarize(match_means, draws)
                    held_out = {
                        event["event_id"]: match_means[event["event_id"]]
                        for event in events
                        if event["held_out"]
                    }
                    pooled["held_out_match_means"] = held_out
                    out["results"][f"{target}/{metric}/{baseline}"] = pooled
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = [p.read_bytes() for p in (args.manifest, args.records)]
    result = aggregate(*(json.loads(blob) for blob in source))
    result["input_sha256"] = dict(
        zip(("manifest", "records"), (sha256(blob).hexdigest() for blob in source), strict=True)
    )
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
