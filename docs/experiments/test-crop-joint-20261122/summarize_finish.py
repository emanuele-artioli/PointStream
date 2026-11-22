"""Summarize complete native receiver coverage and actual staged component times."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def main(a):
    f = json.loads(a.finish.read_text())
    timing = json.loads(a.timing.read_text())
    sender = json.loads(a.sender.read_text())
    paid = next(r for r in sender["rates"] if r["crf"] == sender["selected_crf"])
    assert (
        timing["archive_sha256"] == a.archive_sha and f["carrier_sha256"] == paid["carrier_sha256"]
    )
    assert f["yuv_sha256"] == paid["decoded_sha256"] and f["frames_file_sha256"]
    assert (
        f["wire"]["joint_sha256"] == paid["joint_sha256"]
        and f["wire"]["cache_sha256"] == paid["init_sha256"]
    )
    assert f["wire"]["active_ticks"] == [0, 240] and f["wire"]["ball_positions"] == a.ball_positions
    assert f["ticks"] == list(range(240)) and [r["tick"] for r in f["rows"]] == f["ticks"]
    assert (
        f["isolation"]["interfaces"] == ["lo"]
        and f["isolation"]["mutation_denied"]
        and f["kernels"]
    )
    assert set(f["order"]) == {r["unit"] for r in sender["foreground"]} and len(f["order"]) == 2
    result = dict(
        job=a.job,
        archive_sha256=a.archive_sha,
        archive_bytes=timing["archive_bytes"],
        receipt_sha256=hashlib.sha256(a.finish.read_bytes()).hexdigest(),
        source=sender["source_sha256"],
        frames_sha256=f["frames_file_sha256"],
        native_yuv_sha256=f["yuv_sha256"],
        frame_sha256=[r["frame_sha256"] for r in f["rows"]],
        base_frame_sha256=[r["base_sha256"] for r in f["rows"]],
        wire=f["wire"],
        source_denied=True,
        actual_gpu=f["gpu"],
        mask_file_sha256=f["mask_files"],
        composited={},
        stage_seconds={k: v for k, v in f.items() if k.endswith("seconds")},
        worker=timing,
    )
    result["stage_seconds"]["sum_sam_predictions"] = sum(r["sam_seconds"] for r in f["rows"])
    result["stage_seconds"]["sum_composition_ticks"] = sum(r["tick_seconds"] for r in f["rows"])
    for name in f["order"]:
        counts = Counter(r["status"][name] for r in f["rows"])
        assert sum(counts.values()) == 240 and counts["composed"] > 0
        areas = [r["projected_pixels"][name] for r in f["rows"]]
        result["composited"][name] = dict(
            status_counts=dict(counts),
            min_pixels=min(areas),
            median_pixels=statistics.median(areas),
            max_pixels=max(areas),
        )
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(a.job, timing["worker_wall_seconds"], result["composited"], result["stage_seconds"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("finish", "timing", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--sender", type=Path, default=HERE / "match102-sender.json")
    p.add_argument("--ball-positions", type=int, default=217)
    p.add_argument("--archive-sha", required=True)
    p.add_argument("--job", required=True)
    main(p.parse_args())
