"""Rebuild the exact low-rate initial receiver inputs from paid cache slots."""

import hashlib
import json
import struct
from pathlib import Path
import cv2
import numpy as np
from experiments.tennis import (
    common_player_packet as p,
    common_player_ball as b,
    w8_payload as w,
    k6b_rate_streams as rates,
    common_receiver_manifest as manifest,
)
from src.codecs import svtav1
import reference_webp_channel as cache

root = Path(
    "/home/itec/emanuele/Datasets/pointstream-data/tennis/cumulative-joint-20261122/initial-frozen-52-293"
)
out = Path("/local/users/emanuele/pointstream/audits/initial-cached-player-v3-20261122")
out.mkdir(parents=True, exist_ok=False)


def sha(x):
    return hashlib.sha256(x).hexdigest()


original = (root / "players.pscp").read_bytes()
oldside = (root / "ball.sidecar").read_bytes()
oldmanifest = (root / "receiver.manifest").read_bytes()
assert (sha(original), sha(oldside), sha(oldmanifest)) == (
    "c5d62404913130a759cbe0ced630dc8fa9743140e3dacbc74078d7afd0d5e14f",
    "e05a85788b767f748886761bb0d9307d91effa12c270171939b4ff07e8e8a811",
    "35043669177a9417d0d517de683ec8aac2d3095b6b03c97b7d8e731110120346",
)
profile = p._embedded_profile(original)
sections, _ = w.unpack(original, profile)
initial = (cache.OUTPUT / "initial.psr").read_bytes()
im, slots = cache.unpack(initial)
images = [cache.ref_decode(z)[1] for z in slots]
names = im["tracks"]
assert names == cache.SOURCES
profile["resources"]["native_decoder"] = (
    "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2"
)
profile["resources"]["reference_decoder"] = hashlib.sha256(
    cv2.getBuildInformation().encode()
).hexdigest()
old = p._decode(profile, sections)
refs = {}
slotmap = {}
for run in profile["runs"]:
    name = run["unit"]
    refname = run["reference"]
    slot = names.index(name)
    slotmap[name] = slot
    meta, _ = rates.decode_reference(sections[refname])
    meta = {k: v for k, v in meta.items() if k not in ("codec", "quality", "color")}
    refs[refname] = rates.encode_reference(images[slot], meta, lossless_auto=True)
    assert np.array_equal(old[name][0]["reference"], images[slot])
wire_profile = {
    "kind": "cached-player-initial-v1",
    "player_profile": profile,
    "cache_initial_sha256": sha(initial),
    "cache_slots": slotmap,
}
wire = w.pack(
    wire_profile,
    [("base", "frame", sections["base"]), ("keypoints", "frame", sections["keypoints"])],
)
parts, _ = w.unpack(wire, wire_profile)
rebuilt = p.build(profile, parts["base"], parts["keypoints"], refs)
newparts, _, new = p.unpack(rebuilt)
assert newparts["base"] == sections["base"] and newparts["keypoints"] == sections["keypoints"]
for name in names:
    a, c = old[name][0], new[name][0]
    for k in ("reference", "boxes", "kp", "t"):
        assert np.array_equal(a[k], c[k])
    assert a["seed"] == c["seed"]
old_profile = p._embedded_profile(oldside)
assert old_profile["carrier_sha256"] == sha(original)
side_parts, _ = w.unpack(oldside, old_profile)
tile = side_parts["tile"]
positions = [
    None if pair == (65535, 65535) else pair
    for pair in struct.iter_unpack(">HH", side_parts["positions"])
]
dec = svtav1.tool("dav1d")
side = b.pack(rebuilt, positions, tile, decoder_sha256=dec["sha256"])
assert b._open(rebuilt, side, dec["sha256"])[:2] == (positions, tile)
m = manifest.encode(sha(rebuilt))
(out / "initial.pscw").write_bytes(wire)
(out / "receiver.pscp").write_bytes(rebuilt)
(out / "ball.sidecar").write_bytes(side)
(out / "receiver.manifest").write_bytes(m)
r = {
    "cache_initial_bytes": len(initial),
    "initial_wire_bytes": len(wire),
    "initial_ball_sidecar_bytes": len(side),
    "receiver_manifest_bytes": len(m),
    "initial_transmitted_bytes": len(initial) + len(wire) + len(side) + len(m),
    "rebuilt_carrier_bytes_not_transmitted": len(rebuilt),
    "wire_sha256": sha(wire),
    "carrier_sha256": sha(rebuilt),
    "ball_sha256": sha(side),
    "original_carrier_sha256": sha(original),
    "all_240_generator_inputs_exact": True,
    "base_and_ball_payload_exact": True,
    "scope": "source-free cache plus player input transport; identical paid base and ball, final composed pixels to validate",
}
(out / "receipt.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r))
