"""Summarize existing London complete-stream prices and reviewed body controls."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics

HERE = Path(__file__).resolve().parent
INPUTS = (
    HERE / "tenniset-ball-summary.json",
    HERE / "london-reviewed-scores.json",
    HERE / "route-summary.json",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    deliveries = json.loads(INPUTS[0].read_text())["rows"]
    scores = json.loads(INPUTS[1].read_text())["rows"]
    routes = {row["rally_id"]: row for row in json.loads(INPUTS[2].read_text())["rows"]}
    samples = defaultdict(list)
    for row in scores:
        samples[(row["rally"], str(row["ceiling_bps"]))].append(row)
    results = []
    for delivery in deliveries:
        name = delivery["rally_id"]
        row = {"rally": name, "frames": delivery["frames"], "fallback_ticks": delivery["fallback_ticks"], "ceilings": {}}
        assert routes[name]["evaluated_range"] == [0, delivery["frames"]]
        for rate, stream in delivery["selected"].items():
            chosen = samples[(name, rate)]
            assert len(chosen) == 8
            body_rows = [record["scores"]["ball_stream"]["players"] for record in chosen]
            populated = [v for v in body_rows if v]
            metrics = {
                metric: statistics.mean(statistics.mean(person[metric] for person in frame) for frame in populated)
                if populated else None
                for metric in ("dists", "lpips", "luma_ssim", "rgb_psnr")
            }
            row["ceilings"][rate] = {
                "bytes": stream["bytes"],
                "ceiling_bytes": stream["ceiling"],
                "residual_bytes": stream["ceiling"] - stream["bytes"],
                "body_reviewed_frames": len(populated),
                "body_instances": sum(map(len, populated)),
                "metrics_frame_then_player_mean": metrics,
            }
            assert stream["ceiling"] >= stream["bytes"]
        results.append(row)
    print(json.dumps({
        "scope": "Six saved ball-plus-base physical streams, regional scores on accepted automatic source proposals; no player-generation result",
        "input_sha256": {p.name: digest(p) for p in INPUTS},
        "script_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "results": results,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
