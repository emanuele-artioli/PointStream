"""Separate measured native rendering, mask, base decode, and archive costs."""

import argparse
from collections import Counter
import json
from pathlib import Path


def main(a):
    generation = json.loads(a.generation.read_text())
    finish = json.loads(a.finish.read_text())
    r = finish["receipt"]
    assert r["ticks"] == list(range(240)) and len(generation) == 2
    start, stop = r["wire"]["active_ticks"]
    active = [v for v in r["rows"] if start <= v["tick"] < stop]
    other = [v for v in r["rows"] if not start <= v["tick"] < stop]
    assert len(active) + len(other) == 240
    decode = r["native_decode_seconds"]
    sam = sum(x["sam_seconds"] for x in r["rows"])
    assert 0 < sam <= r["loop_seconds"] and decode > 0
    for row in other:
        assert row["coverage_pixels"] == 0 and row["frame_sha256"] == row["base_sha256"]
    tracks = []
    for name, info in generation.items():
        status = dict(Counter(row["status"][name] for row in active))
        assert info["frames"] == stop - start and info["render_seconds"] > 0
        tracks.append(
            dict(
                name=name,
                active_ticks=info["frames"],
                render_and_save_sec=info["render_seconds"],
                frames_per_sec=info["frames"] / info["render_seconds"],
                device=info["gpu"],
                job=info["job_id"],
                archive_sha256=info["archive_sha256"],
                active_status=status,
                nonempty=sum(row["projected_pixels"].get(name, 0) > 0 for row in active),
            )
        )
    result = dict(
        tracks=tracks,
        finish_job=finish["job_id"],
        finish_gpu=r["gpu"],
        native_decode_sec=decode,
        sam_model_load_sec=r["sam_model_load_seconds"],
        sam_per_frame_sum_sec=sam,
        composite_loop_sec=r["loop_seconds"],
        other_loop_work_sec=r["loop_seconds"] - sam,
        isolated_finish_sec=r["elapsed_seconds"],
        archive_sec=finish["timing"]["archive_seconds"],
        phase_cpu_verification_sec=finish["timing"]["verify_seconds"],
        frames=240,
        active_ticks=len(active),
        fallback_ticks=len(other),
        measured_scope="Cumulative render-and-save per individual with actual device; full-image native decode, model load and per-tick SAM/loop times at receiver. Offline verification and archive assembly reported separately. Benchmarked stages may overlap when running different players on independent devices.",
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ["generation", "finish", "out"]:
        p.add_argument("--" + k, required=True, type=Path)
    main(p.parse_args())
