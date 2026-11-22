"""Build and reparse an AVIF initialization, then transcode its decoded references for the isolated renderer."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import zlib

import cv2
import numpy as np
from PIL import Image

from experiments.tennis import common_player_packet as packet, common_receiver_manifest as manifest
from experiments.tennis import k6b_rate_streams as rates, w8_payload as wire


ORIGINAL = "731cc34e1fcb7c0874cd5722ab63241a123d31fbcd699e199cee4163cdcbc0dc"
CACHE = "91bd3a3fd9b18db301348c98e5af1536b859f2e51dde71e7de56c3beee1c1af2"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def refblob(meta, image):
    record = wire.canonical(meta)
    raw = struct.pack(">I", len(record)) + record + image
    compressed = zlib.compress(raw, 6)
    return (
        rates.HEADER.pack(b"REF1", len(raw), len(compressed), hashlib.sha256(raw).digest())
        + compressed
    )


def main(a):
    assert sha(a.carrier.read_bytes()) == ORIGINAL and sha(a.cache.read_bytes()) == CACHE
    enc = Path(shutil.which("avifenc"))
    dec = Path(shutil.which("avifdec"))
    assert (
        sha(enc.read_bytes()) == "f9cd38ec2804b56174344fd178ea919f9bff41ec7fc4a471f0ff9065b1b5d076"
    )
    assert (
        sha(dec.read_bytes()) == "635e3acc8cc93212ca5cf1cc867b9bb6f3b738574e92107ed24be48a4a75e6bd"
    )
    original = a.carrier.read_bytes()
    profile = packet._embedded_profile(original)
    sections, _, runs = packet.unpack(original)
    init_profile = packet._embedded_profile(a.cache.read_bytes())
    init_refs, _ = wire.unpack(a.cache.read_bytes(), init_profile)
    meta = {}
    inputs = {}
    rows = {}
    a.output.mkdir(parents=True, exist_ok=False)
    for name in sorted(init_refs):
        m, png = packet._bounded_transport(init_refs[name], b"REF1", packet.MAX_REF)
        pm, ppng = packet._bounded_transport(sections[name], b"REF1", packet.MAX_REF)
        assert m == pm and png == ppng and m["codec"] == ".png"
        (a.output / (name + ".png")).write_bytes(png)
        out = a.output / (name + ".avif")
        subprocess.run(
            [enc, "-q", "90", "--speed", "8", "--jobs", "1", a.output / (name + ".png"), out],
            check=True,
            capture_output=True,
        )
        inputs[name] = out.read_bytes()
        meta[name] = m
        rows[name] = dict(
            unit=m["unit"],
            source_bytes=len(png),
            source_sha256=sha(png),
            avif_bytes=len(inputs[name]),
            avif_sha256=sha(inputs[name]),
        )
    new_profile = {
        **init_profile,
        "reference_codec": "avif-q90",
        "reference_container": "images-with-ref1-metadata",
        "reference_decoder_sha256": sha(dec.read_bytes()),
        "reference_metadata": meta,
    }
    envelope = wire.pack(new_profile, [(k, "reference", v) for k, v in inputs.items()])
    (a.output / "initial-avif.psr").write_bytes(envelope)
    recovered, _ = wire.unpack(envelope, packet._embedded_profile(envelope))
    assert recovered == inputs and new_profile == packet._embedded_profile(envelope)
    local = {}
    for name, data in recovered.items():
        f = a.output / (name + "-received.avif")
        f.write_bytes(data)
        decoded = a.output / (name + "-rgb.png")
        subprocess.run([dec, "--jobs", "1", f, decoded], check=True, capture_output=True)
        pix = np.asarray(Image.open(decoded).convert("RGB"))
        assert pix.shape == (784, 512, 3)
        ok, png = cv2.imencode(".png", cv2.cvtColor(pix, cv2.COLOR_RGB2BGR))
        assert ok
        local[name] = refblob(meta[name], png.tobytes())
        rows[name]["decoded_rgb_sha256"] = sha(pix.tobytes())
    new = packet.build(profile, sections["base"], sections["keypoints"], local)
    reread, _, track = packet.unpack(new)
    assert reread["base"] == sections["base"] and reread["keypoints"] == sections["keypoints"]
    for name in rows:
        p = meta[name]["unit"]
        u = runs[p][0]
        v = track[p][0]
        assert v["seed"] == u["seed"] and all(
            np.array_equal(u[k], v[k]) for k in ("t", "boxes", "kp", "reference_box")
        )
        assert all(
            np.array_equal(packet.pose_for_tick(u, t), packet.pose_for_tick(v, t))
            for t in range(240)
        )
        assert sha(v["reference"].tobytes()) == rows[name]["decoded_rgb_sha256"]
    (a.output / "players.pscp").write_bytes(new)
    mf = manifest.encode(sha(new))
    (a.output / "players.manifest").write_bytes(mf)
    manifest.verify(a.output / "players.manifest", sha(new))
    image_bytes = sum(r["avif_bytes"] for r in rows.values())
    original_bytes = sum(r["source_bytes"] for r in rows.values())
    result = dict(
        reference_rows=rows,
        source_image_bytes=original_bytes,
        avif_image_bytes=image_bytes,
        image_saving=1 - image_bytes / original_bytes,
        passes_image_oracle=image_bytes <= original_bytes * 0.7,
        source_cache_bytes=a.cache.stat().st_size,
        source_cache_sha256=CACHE,
        avif_cache_bytes=len(envelope),
        avif_cache_sha256=sha(envelope),
        source_carrier_sha256=ORIGINAL,
        avif_carrier_sha256=sha(new),
        avif_manifest_sha256=sha(mf),
        all_240_poses_placement_seed_constant=True,
        avifenc_sha256=sha(enc.read_bytes()),
        avifdec_sha256=sha(dec.read_bytes()),
        preparation_sha256=sha(Path(__file__).read_bytes()),
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if "_bytes" in k or "oracle" in k}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("carrier", "cache", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
