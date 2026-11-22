"""K8b court registration on the fixed original match284 native frames."""

import argparse
import json
import os
from pathlib import Path
import time
from experiments.tennis import k8b
from experiments.visor.b1 import file_sha256

EXPECTED = "0afb0fb191881def1235285b58fe54814556c83fed7815cd99a00b92d26bc86c"


def run(a):
    t = time.monotonic()
    if file_sha256(a.source) != EXPECTED:
        raise ValueError("source identity")
    meta = k8b.probe(a.source, "ffprobe")
    if (meta["width"], meta["height"], meta["r_frame_rate"], meta["frames"]) != (
        1920,
        1080,
        "25/1",
        250,
    ):
        raise ValueError("native source geometry/cadence")
    rows = k8b.register_chunk(str(a.source), 0, a.frames, "ffmpeg")
    if len(rows) != a.frames or [r["frame"] for r in rows] != list(range(a.frames)):
        raise ValueError("native frame coverage")
    result = {
        "source_sha256": EXPECTED,
        "native": meta,
        "frames": a.frames,
        "fit_count": sum(bool(r["fits"]) for r in rows),
        "rows": rows,
        "elapsed_seconds": time.monotonic() - t,
        "stage": os.environ["PS_STAGE"],
    }
    p = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    p.mkdir()
    (p / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(json.dumps(result) + "\n")


def validate(a):
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "result.json").read_text())
    n = 12 if r["stage"] == "smoke" else 240
    checks = {
        "frame_count": r["frames"] == n,
        "coverage": len(r["rows"]) == n,
        "cost": r["stage"] != "smoke" or r["elapsed_seconds"] * 20 * 1.5 + 120 < 2000,
    }
    if n == 240:
        checks["court_view_screen"] = r["fit_count"] / n >= 0.8
    result = {"passed": all(checks.values()), "checks": checks}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result) + "\n")
    if not result["passed"]:
        raise SystemExit(str(checks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate(a) if a.validate else run(a)


if __name__ == "__main__":
    main()
