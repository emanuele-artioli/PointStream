"""Reprice the frozen Sydney CRF 70 startup images with AVIF Q90."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from experiments.tennis import common_player_packet as packet, w8_payload as wire

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test-crop-joint-20261122"))
from reference_cost import encode, psnr  # noqa: E402

CACHE_SHA = "1f191ca3d078c258c6252003d746b6077c268bb98cff7260c4c960cab5a8d569"
WARM_SHA = "f9ab35e8d3cd6708e726fe57d9d7ba7cc8943285ac4535524491f5cfe0598384"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("warm_packet", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    cache = args.cache.read_bytes()
    warm = args.warm_packet.read_bytes()
    if (sha(cache), sha(warm)) != (CACHE_SHA, WARM_SHA):
        raise ValueError("Cache and warm packet must match the frozen Sydney CRF 70 receipts")

    tools = {}
    for name in ("avifenc", "avifdec"):
        installed = shutil.which(name)
        if installed is None:
            raise FileNotFoundError(name)
        tools[name] = Path(installed).resolve()
    tool_info = {
        name: {
            "path": str(path),
            "sha256": sha(path.read_bytes()),
            "version": subprocess.check_output([path, "--version"], text=True).strip(),
        }
        for name, path in tools.items()
    }
    cache_profile = packet._embedded_profile(cache)
    warm_profile = packet._embedded_profile(warm)
    original, _ = wire.unpack(cache, cache_profile)
    sections, _ = wire.unpack(warm, warm_profile)
    metadata, images, measures = {}, {}, {}
    args.scratch.mkdir()
    for name, reference in original.items():
        attrs, png = packet._bounded_transport(reference, b"REF1", packet.MAX_REF)
        if attrs["codec"] != ".png":
            raise ValueError(f"Expected original PNG for {name}")
        metadata[name] = attrs
        source_path = args.scratch / f"{name}.png"
        source_path.write_bytes(png)
        encoded, decoded = encode("avif90", source_path, args.scratch / f"{name}-q90.png", tools)
        source = packet._bounded_reference(reference)[1]
        if decoded.shape != source.shape:
            raise ValueError(f"Changed crop geometry for {name}")
        images[name] = encoded
        squared_error = int(np.square(source.astype(np.int32) - decoded).sum())
        measures[name] = {
            "role": attrs["unit"], "bytes": len(encoded), "sha256": sha(encoded),
            "crop_psnr_db": psnr(squared_error, source.size),
        }

    priced_profile = dict(cache_profile, reference_codec="avif-q90",
                          reference_container="images-with-ref1-metadata",
                          reference_decoder_sha256=tool_info["avifdec"]["sha256"],
                          reference_metadata=metadata)
    priced_cache = wire.pack(priced_profile, [(name, "reference", blob) for name, blob in images.items()])
    priced_warm_profile = dict(warm_profile, cache_sha256=sha(priced_cache))
    priced_warm = wire.pack(priced_warm_profile, [(name, "frame", blob) for name, blob in sections.items()])
    if (wire.unpack(priced_cache, priced_profile)[0] != images
            or wire.unpack(priced_warm, priced_warm_profile)[0] != sections
            or len(priced_warm) != len(warm)):
        raise ValueError("Reference repack changed a retained wire section or the warm length")
    result = {
        "match": "match127", "sender_job": "20261122T134208Z-0e12634b",
        "revision": args.revision, "script_sha256": sha(Path(__file__).read_bytes()),
        "device": "CPU", "image_metrics_only": True,
        "tools": tool_info, "seconds": 4.0,
        "original_cache_bytes": len(cache), "original_cache_sha256": CACHE_SHA,
        "warm_bytes": len(warm), "warm_sha256": WARM_SHA,
        "q90_cache_bytes": len(priced_cache), "q90_cache_sha256": sha(priced_cache),
        "q90_rebound_warm_sha256": sha(priced_warm),
        "q90_cache_saving_percent": 100 * (1 - len(priced_cache) / len(cache)),
        "references": measures,
        "elapsed_seconds": time.monotonic() - start,
    }
    (args.scratch / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
