"""Extract lossless reference PNGs from the frozen Sydney CRF 70 cache so AVIF Q90 can be encoded locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from experiments.tennis import common_player_packet as packet
from experiments.tennis import w8_payload as wire

CACHE_SHA256 = "1f191ca3d078c258c6252003d746b6077c268bb98cff7260c4c960cab5a8d569"


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def run_prepare(args: argparse.Namespace) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    stage_name = os.environ["PS_STAGE"]
    expected = {"smoke": 1, "full": 2}[stage_name]

    cache_bytes = args.cache.read_bytes()
    if sha256_bytes(cache_bytes) != CACHE_SHA256:
        raise ValueError("Sydney cache sha256 mismatch")

    cache_profile = packet._embedded_profile(cache_bytes)
    refs, _ = wire.unpack(cache_bytes, cache_profile)

    pub = scratch / "publish"
    pub.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for idx, (name, raw_ref) in enumerate(refs.items()):
        attrs, png_blob = packet._bounded_transport(raw_ref, b"REF1", packet.MAX_REF)
        if attrs["codec"] != ".png":
            raise ValueError(f"Expected PNG ref for {name}")
        if idx >= expected:
            continue
        out_png = stage / f"{attrs['unit']}.png"
        out_png.write_bytes(png_blob)
        arr = np.asarray(Image.open(out_png).convert("RGB"))
        src_arr = packet._bounded_reference(raw_ref)[1]
        if not np.array_equal(arr, src_arr):
            raise ValueError(f"Decoded pixel mismatch for {name}")
        rows.append(
            {
                "name": name,
                "unit": attrs["unit"],
                "png_bytes": len(png_blob),
                "png_sha256": sha256_bytes(png_blob),
                "shape": list(arr.shape),
            }
        )

    summary = {"match": "match127", "stage": stage_name, "count": expected, "rows": rows}
    (stage / "q90_prepare.json").write_text(json.dumps(summary, indent=2) + "\n")
    archive = stage / "published.tar"
    tmp = stage / ".q90.tar"
    with tarfile.open(tmp, "x") as tf:
        tf.add(stage / "q90_prepare.json", arcname="publish/q90_prepare.json")
    tmp.rename(archive)


def validate(_args: argparse.Namespace) -> None:
    stage = Path(os.environ["PS_STAGE_DIR"])
    expected = {"smoke": 1, "full": 2}[os.environ["PS_STAGE"]]
    data = json.loads((stage / "q90_prepare.json").read_text())
    assert data["count"] == expected and len(data["rows"]) == expected
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": True, "checks": {"count": expected}}, indent=2) + "\n"
    )


def main() -> None:
    import sys

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--count", type=int, choices=(1, 2), required=True)
    p.add_argument("--validate", action="store_true")
    if sys.argv[1:] == ["--validate"]:
        stage = Path(os.environ["PS_STAGE_DIR"])
        argv = json.loads((stage / "dispatch.json").read_text())["command"]
        validate(p.parse_args(["--validate", *argv[3:]]))
    else:
        args = p.parse_args()
        if args.validate:
            validate(args)
        else:
            run_prepare(args)


if __name__ == "__main__":
    main()
