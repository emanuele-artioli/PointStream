"""Source-free decode of the measured PSB1 cache, cuts, KPS and carriers."""

import hashlib
import json
import struct
import zlib
from pathlib import Path
from experiments.tennis import (
    common_player_packet as p,
    common_player_ball as b,
    w8_payload as w,
    k6b,
    k6b_rate_streams as rates,
    common_receiver_manifest as manifest,
)
from src.codecs import svtav1
from src.segmentation.masks import decompress
import reference_webp_channel as cache

root = Path(
    "/home/itec/emanuele/Datasets/pointstream-data/tennis/cumulative-joint-20261122/three-segment-transport-v2"
)
q = (root / "delivery.psb").read_bytes()
assert (
    hashlib.sha256(q).hexdigest()
    == "7ad5e13d4007d5510e70c2d9892c082817d2f4f59e07cdc38641db172b471f64"
)
hs = struct.unpack_from(">I", q, 4)[0]
header = json.loads(q[8 : 8 + hs])
cursor = 8 + hs
sections = {}
for row in header["entries"]:
    raw = q[cursor : cursor + row["bytes"]]
    cursor += len(raw)
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    sections[row["name"]] = raw
assert cursor == len(q) and sum(x["frames"] for x in header["segments"]) == 11305
im, slots = cache.unpack(sections["cache_initial"])
rm, empty = cache.unpack(sections["cache_return"])
assert not empty and rm["initial_sha256"] == hashlib.sha256(sections["cache_initial"]).hexdigest()
decoded = [cache.ref_decode(s) for s in slots]


def portable(blob):
    magic, size, _, digest = rates.HEADER.unpack_from(blob)
    raw = decompress(blob[rates.HEADER.size :])
    assert len(raw) == size and hashlib.sha256(raw).digest() == digest
    comp = zlib.compress(raw, 6)
    return rates.HEADER.pack(magic, size, len(comp), digest) + comp


recovered = {}
dec = svtav1.tool("dav1d")
for shot in ("initial", "return"):
    wire = sections[shot + "_wire"]
    outer = p._embedded_profile(wire)
    parts, _ = w.unpack(wire, outer)
    profile = outer["player_profile"]
    refs = {}
    _, tracks = rates.decode_keypoints(parts["keypoints"])
    byname = {r["name"]: r for r in tracks}
    for run in profile["runs"]:
        name = run["unit"]
        slot = outer["cache_slots"][name]
        meta, img = decoded[slot]
        meta = {k: v for k, v in meta.items() if k not in ("codec", "quality", "color")}
        if shot == "return":
            meta = {
                "unit": name,
                "reference_frame": 0,
                "reference_box": k6b.run_boxes(list(byname[name]["kp"]))[0].tolist(),
                "pixels": "full_prepared_conditioning_crop",
            }
            refs[run["reference"]] = portable(rates.encode_reference(img, meta))
        else:
            refs[run["reference"]] = rates.encode_reference(img, meta, lossless_auto=True)
    kps = portable(parts["keypoints"]) if shot == "return" else parts["keypoints"]
    carrier = p.build(profile, parts["base"], kps, refs)
    side = sections[shot + "_ball"]
    positions, tile, _ = b._open(carrier, side, dec["sha256"])
    expected_manifest = manifest.encode(hashlib.sha256(carrier).hexdigest())
    assert expected_manifest == sections[shot + "_manifest"]
    assert len(positions) == 240
    recovered[shot] = {
        "carrier_sha256": hashlib.sha256(carrier).hexdigest(),
        "ball_sha256": hashlib.sha256(side).hexdigest(),
        "frames": len(positions),
        "present": sum(x is not None for x in positions),
        "source_free": True,
    }
assert (
    recovered["initial"]["carrier_sha256"]
    == "45a6554b7fc05d07c308c22f148d70c035c6dde838abc08fe38f8aedf0002d1b"
    and recovered["return"]["carrier_sha256"]
    == "a0ec63326642130adb5d765a3e7c30aaf564ad5bbf7cd17d56594ffd82dc5b06"
)
out = Path("/local/users/emanuele/pointstream/audits/source-free-packet-replay-20261122")
out.mkdir(parents=True, exist_ok=False)
r = {
    "packet_sha256": hashlib.sha256(q).hexdigest(),
    "packet_bytes": len(q),
    "segments": header["segments"],
    "recovered": recovered,
    "fallback_sha256": hashlib.sha256(sections["fallback_ivf"]).hexdigest(),
    "source_files_opened": False,
    "scope": "source-free recovery of carriers and ball payloads; player video generation/composition remains separate",
}
(out / "receipt.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r))
