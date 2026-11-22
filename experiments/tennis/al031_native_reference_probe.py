"""Verify that the original appearance cache produces identical RGB inside the exact player runtime."""

import argparse
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys
import time

from experiments.tennis import common_player_packet as packet
from experiments.tennis.player_requalification import array_sha, sha256

CACHE = "91bd3a3fd9b18db301348c98e5af1536b859f2e51dde71e7de56c3beee1c1af2"
EXPECTED = (
    "4efc900c9e414bd768d06545bd70d5c5d2a0299eab1d2f7e044df85839ad9e60",
    "34f19a1b0c83306a069ecf1c6384f06f1115c62dac2490d9053f0050b57e4cdc",
)
LIB = "b0e2bad8c117a3ad7d344d7924272d12b8e363956efb91b5074e43d4e201a08d"


def run(a):
    start = time.monotonic()
    native = Path(sys.prefix) / "lib/libzstd.so.1"
    assert (
        not os.environ.get("CUDA_VISIBLE_DEVICES")
        and find_spec("zstandard") is None
        and sha256(native) == LIB
    )
    cache = a.cache.read_bytes()
    assert len(cache) == 157517 and packet.wire.sha(cache) == CACHE
    assets, _ = packet.wire.unpack(cache, packet._embedded_profile(cache))
    carried, _, units = packet.unpack(a.prior.read_bytes())
    assert set(assets) == {"ref0", "ref1"} and all(carried[k] == v for k, v in assets.items())
    refs = []
    for i, key in enumerate(sorted(assets)[: a.count]):
        blob = assets[key]
        f, shape_bytes, _, _ = packet.rates.HEADER.unpack_from(blob)
        coded = blob[packet.rates.HEADER.size :]
        assert (
            f == b"REF1" and shape_bytes < 784 * (512 * 3 + 1) + 2 * packet.MAX_TRANSPORT_META + 4
        )
        for malformed, n in (
            (coded, shape_bytes + 1),
            (coded[:-1], shape_bytes),
            (coded + coded, shape_bytes),
        ):
            try:
                packet._installed_zstd(malformed, n, packet.MAX_REF)
            except ValueError:
                pass
            else:
                raise ValueError("decoder accepted malformed length or boundary")
        _, image = packet._bounded_reference(blob)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        assert (
            image.shape == (784, 512, 3) and str(image.dtype) == "uint8" and digest == EXPECTED[i]
        )
        assert sum(array_sha(image) == array_sha(v[0]["reference"]) for v in units.values()) == 1
        refs.append(
            dict(
                name=key,
                raw_rgb_sha256=digest,
                typed_rgb_sha256=array_sha(image),
                ref_sha256=packet.wire.sha(blob),
                bytes=len(blob),
                expanded_bytes=shape_bytes,
                malformed_lengths_rejected=3,
            )
        )
    report = dict(
        backend="installed libzstd 1.5.7",
        library_sha256=LIB,
        cache=CACHE,
        prior=sha256(a.prior),
        refs=refs,
        worker_seconds=time.monotonic() - start,
    )
    out = Path(os.environ["PS_STAGE_DIR"]) / "result.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    assert report["worker_seconds"] < 30


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "result.json").read_text())
    checks = dict(
        cache=r["cache"] == CACHE,
        native=r["library_sha256"] == LIB,
        hashes=[v["raw_rgb_sha256"] for v in r["refs"]] == list(EXPECTED[: len(r["refs"])]),
        cases=len(r["refs"]) == (1 if os.environ["PS_STAGE"] == "smoke" else 2),
        malformed=all(v["malformed_lengths_rejected"] == 3 for v in r["refs"]),
        time=r["worker_seconds"] < 30,
    )
    val = dict(passed=all(checks.values()), checks=checks)
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(val) + "\n")
    assert val["passed"], val


if __name__ == "__main__":
    if sys.argv[1:] == ["--validate"]:
        validate()
    else:
        p = argparse.ArgumentParser()
        p.add_argument("--cache", type=Path, required=True)
        p.add_argument("--prior", type=Path, required=True)
        p.add_argument("--count", type=int, choices=[1, 2], required=True)
        run(p.parse_args())
