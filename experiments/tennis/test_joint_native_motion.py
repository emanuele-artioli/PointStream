"""Apply original source optical flow on campus to genuine native warm-budget decodes."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import zlib
import numpy as np

from experiments.tennis.player_requalification import sha256

CARD = Path(__file__).resolve().parents[2] / "docs/experiments/test-crop-joint-20261122"
sys.path.insert(0, str(CARD))
import temporal_native as temporal  # noqa: E402


def inputs(args):
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    anchor = json.loads(
        (CARD.parent / "test102-joint-anchors-20261122/manifest.json").resolve().read_text()
    )
    arms = []
    for mode, w, f in (("ordinary", 320, args.ordinary), ("roi", 640, args.roi)):
        row = next(r for r in anchor["rows"] if r["mode"] == mode and r["width"] == w)
        sel = row["selected"]["warm"]
        assert sha256(f) == row["archive_sha256"]
        with tarfile.open(f) as z:
            ivf = z.extractfile(sel["file"]).read()
            raw = zlib.decompress(z.extractfile(sel["decoded_file"]).read())
        assert hashlib.sha256(ivf).hexdigest() == sel["sha256"] and len(ivf) == sel["bytes"]
        assert hashlib.sha256(raw).hexdigest() == sel["decoded_sha256"]
        dest = scratch / (mode + ".yuv")
        dest.write_bytes(raw)
        del raw
        arms.append(
            dict(
                name=f"{'av1' if mode == 'ordinary' else 'roi'}-{w}-warm",
                width=w,
                height=row["height"],
                kind="yuv",
                bytes=sel["bytes"],
                path=str(dest),
                sha256=sel["decoded_sha256"],
                decoded_sha256=sel["decoded_sha256"],
            )
        )
    with tarfile.open(args.neural) as z:
        r = json.load(z.extractfile("publish/result.json"))
    n = r["native"]
    assert n["accepted"] and n["bytes"] == 17432 and len(r["yuv_frames"]) == 240
    arms.append(
        dict(
            name="dcvc-480-q34",
            width=480,
            height=272,
            kind="yuv_tar",
            archive=str(args.neural),
            member="publish/decoded.yuv",
            bytes=n["bytes"],
            stream_sha256=n["container_sha256"],
            frame_sha256=r["yuv_frames"],
            decoded_sha256=r["raw_sha256"],
        )
    )
    receipt = json.loads((args.witness / "publish/finish.json").read_text())
    assert (
        receipt["wire"]["joint_bytes"] == 17794
        and receipt["wire"]["active_ticks"] == [0, 240]
        and receipt["kernels"]
    )
    arms.append(
        dict(
            name="joint-crf50",
            width=1920,
            height=1080,
            kind="rgb_npy",
            path=str(args.witness / "publish/frames.npy"),
            bytes=17794,
            sha256=receipt["frames_file_sha256"],
        )
    )
    dest = scratch / "arms.json"
    dest.write_text(json.dumps(arms) + "\n")
    return dest


def main(args):
    stage = Path(os.environ["PS_STAGE_DIR"])
    expected = {"smoke": 5, "full": 240}[os.environ["PS_STAGE"]]
    assert args.frames == expected
    if args.validate:
        r = json.loads((stage / "motion.json").read_text())
        timing = json.loads((stage / "timing.json").read_text())
        assert len(r["summary"]) == 4 and r["recorded_ticks"] == expected
        source_values = None
        for name, rr in r["rows"].items():
            assert len(rr) == 2 * (expected - 1) and np.isfinite([i["error"] for i in rr]).all()
            assert r["summary"][name]["n"] == len(rr)
            a = [(i["tick"], i["role"], i["pixels"], i["source_delta"]) for i in rr]
            if source_values is None:
                source_values = a
            else:
                assert source_values == a
        assert timing["elapsed_s"] < 595
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(
            json.dumps(
                dict(
                    passed=True,
                    checks=dict(
                        arms=4,
                        source_correspondence=True,
                        transitions=2 * (expected - 1),
                        receipt_sha256=sha256(stage / "motion.json"),
                    ),
                )
            )
            + "\n"
        )
        return
    start = time.monotonic()
    f = inputs(args)
    grid = json.loads((CARD / "match102-source-grid.json").read_text())
    can = json.loads((CARD / "match102-canonical-source.json").read_text())
    assert (
        sha256(args.source) == grid["source_sha256"] == can["source_sha256"]
        and sha256(args.pixels) == can["lossless_sha256"]
    )
    p = argparse.Namespace(
        source=args.source,
        source_pixels=args.pixels,
        source_hashes=grid["native_rgb_typed_sha256"],
        features=args.features,
        arms=f,
        scratch=Path(os.environ["PS_SCRATCH_DIR"]),
        out=stage / "motion.json",
        frames=args.frames,
    )
    temporal.main(p)
    (stage / "timing.json").write_text(json.dumps(dict(elapsed_s=time.monotonic() - start)) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "pixels", "features", "ordinary", "roi", "neural", "witness"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--frames", type=int, choices=(5, 240), required=True)
    p.add_argument("--validate", action="store_true")
    main(p.parse_args())
