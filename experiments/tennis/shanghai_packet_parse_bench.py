"""Warm offline 240-symbol packet parsing microbenchmark, without image decode."""

import hashlib
import json
import os
import platform
import statistics
import tarfile
import time
from pathlib import Path

from src.pointstream import ball, ball_stream

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
ARCHIVE = ROOT / "jobs/fleet/inbox/20261122T051939Z-58ffb82f/full/published.tar"
EXPECTED_ARCHIVE = "a31bccb26f9c3ce5141d4d09b73b9c6b2002ac40f2ae11a134eab894bd420d96"
OUT = Path("/local/users/emanuele/pointstream/audits/shanghai-packet-parse-bench-20261122")


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse(data):
    p = ball_stream.unpack(data)
    positions, _ = ball.decode_frames(p.ball_payload, 240, 960, 544)
    assert p.frames == 240 and len(positions) == 240
    return positions


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) == 1
    OUT.mkdir(parents=True, exist_ok=False)
    assert sha(ARCHIVE) == EXPECTED_ARCHIVE
    with tarfile.open(ARCHIVE) as t:
        report = json.loads(t.extractfile("publish/report.json").read())
        row = next(g["selected"] for g in report["groups"] if g["width"] == 640)
        data = t.extractfile("publish/" + row["selected_file"]).read()
    assert len(data) == 24388 and hashlib.sha256(data).hexdigest() == row["sha256"]
    reference = parse(data)
    encoded = json.dumps(reference, separators=(",", ":")).encode()
    identity = hashlib.sha256(encoded).hexdigest()
    for _ in range(5):
        assert parse(data) == reference
    measured = []
    for _ in range(30):
        t = time.perf_counter_ns()
        value = parse(data)
        elapsed = time.perf_counter_ns() - t
        assert (
            value == reference
            and hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
            == identity
        )
        measured.append(elapsed / 1e6)
    result = {
        "archive_sha256": EXPECTED_ARCHIVE,
        "packet_sha256": row["sha256"],
        "packet_bytes": len(data),
        "frames": 240,
        "present_symbols": sum(x is not None for x in reference),
        "positions_json_sha256": identity,
        "warmup_repetitions": 5,
        "timed_repetitions": 30,
        "parse_ms_median": statistics.median(measured),
        "parse_ms_min": min(measured),
        "parse_ms_max": max(measured),
        "all_parse_ms": measured,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "platform": platform.platform(),
        "scope": "warm offline whole-sequence PSBL parse plus canonical validation and location decode; no video decode, model load, network, sender or live buffering",
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
