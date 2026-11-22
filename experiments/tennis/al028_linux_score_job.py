"""Run the frozen paired-base Al028 scorer in the finisher's Linux environment."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import numpy as np

FINISH = "ca9c335e804ec51003ca13e17c9895bb2c231ef79d30e16342d81050f0e679d0"
DCVC = "bac9b77792126c748a7640eee2ce80649f54c2509d5973623d06e1ed35b931bb"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(a):
    if a.frames != (24 if os.environ["PS_STAGE"] == "smoke" else 240):
        raise ValueError("wrong stage frame count")
    os.environ["AL028_ASSETS"] = str(a.assets)
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/experiments/joint-replication-feasibility-20261122/al028_joint_recompose.py"
    )
    spec = importlib.util.spec_from_file_location("al028_score", path)
    if spec is None or spec.loader is None:
        raise ValueError("scorer source missing")
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    out = Path(os.environ["PS_SCRATCH_DIR"]) / "al028-score"
    args = SimpleNamespace(
        finish=a.finish / "publish",
        finish_sha=FINISH,
        gen4=a.player1 / "publish",
        gen6=a.player2 / "publish",
        dcvc=a.assets / "ps-nov22-al028-q24-full/full.yuv",
        dcvc_sha=DCVC,
        out=out,
        frames=a.frames,
        save_rates=[20],
    )
    score.main(args)
    stage = Path(os.environ["PS_STAGE_DIR"])
    shutil.copyfile(out / "result.json", stage / "result.json")
    if a.frames == 240:
        shutil.copyfile(out / "frames-20.npy", stage / "frames-20.npy")
        with tarfile.open(stage / "private-review.tar", "x") as tar:
            tar.add(out / "private-review", arcname="private-review")
    (stage / "output-hashes.json").write_text(
        json.dumps(
            {
                p.name: sha(p)
                for p in stage.iterdir()
                if p.name in ("result.json", "frames-20.npy", "private-review.tar")
            },
            indent=2,
        )
        + "\n"
    )


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "result.json").read_text())
    hashes = json.loads((stage / "output-hashes.json").read_text())
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    checks = {
        "frames": r["frames"] == n,
        "source_witness": len(r["rgb_stream_sha256"]) == 3,
        "receipts": r["finish_receipt_sha256"] == FINISH,
        "result_hash": hashes["result.json"] == sha(stage / "result.json"),
    }
    forecast = 1.5 * r["elapsed_seconds"] * 10 + 300
    if n == 24:
        checks["forecast"] = forecast < 1800
    else:
        frame = stage / "frames-20.npy"
        images = np.load(frame, mmap_mode="r", allow_pickle=False)
        h = hashlib.sha256()
        for im in images:
            h.update(np.ascontiguousarray(im))
        checks["frames20"] = (
            images.shape == (240, 1080, 1920, 3)
            and images.dtype == np.uint8
            and h.hexdigest() == r["rgb_stream_sha256"]["20"]
            and hashes["frames-20.npy"] == sha(frame)
        )
        checks["review"] = len(r["private_review_png_sha256"]) == 12 * 10 and hashes[
            "private-review.tar"
        ] == sha(stage / "private-review.tar")
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": all(checks.values()), "checks": checks, "forecast_seconds": forecast})
        + "\n"
    )
    if not all(checks.values()):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    for field in ("assets", "finish", "player1", "player2"):
        p.add_argument("--" + field, type=Path)
    p.add_argument("--frames", type=int, choices=(24, 240))
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)


if __name__ == "__main__":
    main()
