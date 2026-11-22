"""Recover two actual native decoder outputs at the ideal-cached player test rate."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

from src.codecs import svtav1
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clean-court-regional-screen-20261118"))
import crops  # noqa: E402


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(prior, root):
    root.mkdir(exist_ok=False)
    info = json.loads((prior / "result.json").read_text())
    packet = json.loads((HERE.parent / "starved-base-local-20261122/base-receipt.json").read_text())
    clean = next(r for r in packet["outputs"] if r["arm"] == "clean" and r["crf"] == 60)
    target = clean["total_bytes"] - clean["reference_bytes"]
    raw = prior / "480.yuv"
    roi = prior / "roi.txt"
    tools = info["tools"]
    assert sha(raw) == info["scaled_source_sha256"] and sha(roi) == info["roi_map_sha256"]
    assert all(sha(Path(t["path"])) == t["sha256"] for t in tools.values())
    rows = [r for r in info["all_encodes"] if r["n"] == 240]
    selected = {}
    for arm in ["plain", "roi"]:
        for _ in range(8):
            arm_rows = [r for r in rows if r["mode"] == arm]
            best = min(arm_rows, key=lambda r: abs(r["file_bytes"] - target))
            if abs(best["file_bytes"] / target - 1) <= 0.005:
                break
            ordered = sorted(arm_rows, key=lambda r: r["q"])
            bounds = [
                (lo, hi)
                for lo, hi in zip(ordered, ordered[1:])
                if (lo["file_bytes"] - target) * (hi["file_bytes"] - target) < 0
                and hi["q"] - lo["q"] >= 0.5
            ]
            if not bounds:
                break
            lo, hi = bounds[0]
            q = round((lo["q"] + hi["q"]) * 2) / 4
            dest = root / f"{arm}-{q:g}"
            dest.mkdir()
            new = svtav1.code(
                raw,
                dest,
                width=480,
                height=272,
                fps=Fraction(50),
                frames=240,
                crf=q,
                preset=4,
                threads=2,
                full_range=False,
                encoder=tools["SvtAv1EncApp"]["path"],
                decoder=tools["dav1d"]["path"],
                timeout=90,
                extra=["--roi-map-file", str(roi)] if arm == "roi" else None,
            )
            assert new["decoded_frames"] == new["temporal_units"] == 240
            new.update(
                q=q,
                n=240,
                mode=arm,
                decoded_sha256=sha(Path(new["decoded"])),
                rate=new["file_bytes"] / 600,
            )
            rows.append(new)
        selected[arm] = min(
            (r for r in rows if r["mode"] == arm), key=lambda r: abs(r["file_bytes"] - target)
        )
    assert sha(prior / "crops.zip") == info["crop_sha256"]
    labels = crops.ROOT / "foreground-human-reviewed-20261118" / f"{crops.LABEL}.json"
    assert sha(labels) == crops.LABEL
    with (
        zipfile.ZipFile(prior / "crops.zip") as original,
        zipfile.ZipFile(root / "crops.zip", "x") as out,
    ):
        for key in original.namelist():
            if Path(key).stem in ("source", "mask"):
                out.writestr(key, original.read(key))
        for arm, row in selected.items():
            assert (
                sha(Path(row["decoded"])) == row["decoded_sha256"]
                and sha(Path(row["stream"])) == row["stream_sha256"]
            )
            native = np.memmap(row["decoded"], "uint8", "r", shape=(240, 408, 480))
            for f in json.loads(labels.read_text())["frames"]:
                t = f["tick"]
                rgb = cv2.resize(
                    yuv_to_rgb(native[t], 480, 272, False),
                    (1920, 1080),
                    interpolation=cv2.INTER_CUBIC,
                )
                for region, y, x, _ in crops.supports(f):
                    ok, png = cv2.imencode(".png", rgb[y, x, ::-1])
                    assert ok
                    out.writestr(f"{t:03d}/{region}/{arm}.png", png.tobytes())
    result = dict(
        target_bytes=target,
        cold_bytes=clean["total_bytes"],
        prior_sha256=sha(prior / "result.json"),
        script_sha256=sha(Path(__file__)),
        new_rows=rows[len([r for r in info["all_encodes"] if r["n"] == 240]) :],
        selected=selected,
        crops_sha256=sha(root / "crops.zip"),
    )
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print({k: r["rate"] for k, r in selected.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prior", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    main(a.prior, a.output)
