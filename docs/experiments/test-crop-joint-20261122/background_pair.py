"""Pair native untouched and cleaned base encodes on the same source-defined background."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from build_joint import HERE, FPS, TOTAL, sha, spatial_masks
from src.codecs import svtav1
from src.pointstream import ball


def main(a):
    cv2.setNumThreads(1)
    t0 = time.process_time()
    a.out.mkdir(exist_ok=False)
    summary = json.loads((HERE / "match102-sender.json").read_text())
    receipt = json.loads((HERE / "match102-result.json").read_text())
    assert (
        sha(a.source) == summary["source_sha256"]
        and sha(a.features) == receipt["prior_feature_archive_sha256"]
    )
    bs = a.ball.read_bytes()
    pos, appear = ball.decode_frames(bs, 240, 960, 544)
    assert ball.encode_frames(pos, appear, 960, 544) == bs
    masks = spatial_masks(a, receipt, pos)
    assert sha(a.sender / "clean.yuv") == summary["donor"]["clean_yuv_sha256"]
    raw = np.memmap(a.sender / "source.yuv", np.uint8, "r", shape=(240, 816, 960))
    clean = np.memmap(a.sender / "clean.yuv", np.uint8, "r", shape=raw.shape)
    for n in range(240):
        bg = masks[n] == 0
        assert np.array_equal(raw[n, :544][bg], clean[n, :544][bg])
    ff = svtav1.tool("ffmpeg")
    enc = svtav1.tool("SvtAv1EncApp")
    dec = svtav1.tool("dav1d")
    for label, td in (("ffmpeg", ff), ("svt", enc), ("dav1d", dec)):
        assert td == summary["tools"][label]
    ref = a.out / "original-low.yuv"
    subprocess.run(
        [
            ff["path"],
            "-v",
            "error",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            "960x544",
            "-r",
            str(FPS),
            "-i",
            str(a.sender / "source.yuv"),
            "-vf",
            "scale=160:96:flags=area",
            "-frames:v",
            "240",
            "-f",
            "rawvideo",
            "-n",
            str(ref),
        ],
        check=True,
        timeout=100,
    )
    rows = []
    for old in summary["rates"]:
        crf = old["crf"]
        d = a.out / f"raw-crf{crf}"
        d.mkdir()
        candidate = svtav1.code(
            ref,
            d,
            width=160,
            height=96,
            fps=Fraction(30000, 1001),
            frames=240,
            crf=crf,
            preset=4,
            threads=1,
            full_range=False,
            encoder=enc["path"],
            decoder=dec["path"],
            timeout=120,
        )
        stats = []
        for name, data, nb, expected in (
            ("untouched", Path(candidate["decoded"]), (d / "stream.ivf").stat().st_size, None),
            ("donor", Path(old["native"]["decoded"]), old["base_bytes"], old["decoded_sha256"]),
        ):
            output_sha = sha(data)
            if expected:
                assert expected == output_sha
            v = np.memmap(data, np.uint8, "r", shape=(240, 144, 160))
            outside_error = 0.0
            full_error = 0.0
            count = 0
            for n in range(TOTAL):
                expanded = cv2.resize(v[n, :96], (960, 544), interpolation=cv2.INTER_CUBIC)
                difference = np.subtract(expanded, raw[n, :544], dtype=np.float32)
                outside = difference[masks[n] == 0]
                outside_error += float(np.sum(outside**2, dtype=np.float64))
                full_error += float(np.sum(difference**2, dtype=np.float64))
                count += outside.size
            stats.append(
                dict(
                    name=name,
                    bytes=nb,
                    decoded_sha256=output_sha,
                    outside_pixels=count,
                    outside_luma_psnr=10 * math.log10(255**2 * count / outside_error),
                    whole_luma_psnr=10 * math.log10(255**2 * TOTAL * 544 * 960 / full_error),
                )
            )
        x, y = stats
        rows.append(
            dict(
                crf=crf,
                arms=stats,
                saving_bytes=x["bytes"] - y["bytes"],
                saving_percent=100 * (1 - y["bytes"] / x["bytes"]),
                outside_psnr_delta=y["outside_luma_psnr"] - x["outside_luma_psnr"],
                unmodified_encoder=candidate,
            )
        )
    result = dict(
        source_sha256=sha(a.source),
        raw_yuv_sha256=sha(a.sender / "source.yuv"),
        clean_sha256=sha(a.sender / "clean.yuv"),
        source_roi_sha256=summary["donor"]["mask_sha256"],
        fps=str(FPS),
        ticks=240,
        compared_area_fraction=float((masks == 0).mean()),
        original_outside_identical=True,
        tools=summary["tools"],
        rows=rows,
        cpu_seconds=time.process_time() - t0,
    )
    report = a.out / "result.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    for r in rows:
        print(r["crf"], r["saving_bytes"], r["saving_percent"], r["outside_psnr_delta"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "ball", "features", "out", "sender"):
        p.add_argument("--" + k, type=Path, required=True)
    main(p.parse_args())
