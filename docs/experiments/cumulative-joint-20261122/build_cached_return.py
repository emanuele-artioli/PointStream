"""Build a target player carrier at receiver from paid cache, KPS, and base."""

import hashlib
import io
import json
import struct
import tarfile
import zlib
from pathlib import Path
import numpy as np
from experiments.tennis import (
    common_player_packet as p,
    common_player_ball as b,
    w8_payload as w,
    k6b,
    k6b_rate_streams as rates,
    common_receiver_manifest as manifest,
)
from src.codecs import svtav1
import reference_webp_channel as cache

root = Path("/home/itec/emanuele/Datasets/pointstream-data")
out = Path("/local/users/emanuele/pointstream/audits/return-cached-player-v5-20261122")
out.mkdir(parents=True, exist_ok=False)


def sha(x):
    return hashlib.sha256(x).hexdigest()


def portable(blob):
    from src.segmentation.masks import decompress

    magic, size, _, digest = rates.HEADER.unpack_from(blob)
    raw = decompress(blob[rates.HEADER.size :])
    assert len(raw) == size and hashlib.sha256(raw).digest() == digest
    coded = zlib.compress(raw, 6)
    return rates.HEADER.pack(magic, size, len(coded), digest) + coded


def member(path, name):
    with tarfile.open(path) as f:
        return f.extractfile(name).read()


base = member(
    root / "jobs/fleet/inbox/20261121T232531Z-d3660a15/full/published.tar",
    "publish/clean-crf60.ivf",
)
tile = member(
    root / "jobs/fleet/inbox/20261121T232604Z-fbd63629/full/published.tar", "publish/tile.ivf"
)
origins = member(
    root / "jobs/fleet/inbox/20261121T232604Z-fbd63629/full/published.tar", "publish/positions.bin"
)
kps = member(
    root / "jobs/fleet/inbox/20261031T171359Z-aba3845f/full/published.tar",
    "publish/packets/nadal_kyrgios__002@120/keypoints.kps",
)
assert (
    sha(base) == "2fa1f8d752b527707a5e0c1e1be6e070ef34efb91a37aa8ae20ca66215ec2ab9"
    and sha(tile) == "f30860d5924a4d2b9980168a587a343eab8edb17479d4db4aa612997ee6f7b25"
)
_, tracks = rates.decode_keypoints(kps)
names = [x["name"] for x in tracks]
assert set(names) == set(cache.TARGETS)
kps2 = rates.encode_keypoints_kps2({"native_frames": 240}, tracks)
_, again = rates.decode_keypoints(kps2)
assert all(
    np.array_equal(a["kp"], c["kp"]) and np.array_equal(a["t"], c["t"])
    for a, c in zip(tracks, again)
)
initial = (cache.OUTPUT / "initial.psr").read_bytes()
reuse = (cache.OUTPUT / "return-reuse.psr").read_bytes()
im, slots = cache.unpack(initial)
rm, empty = cache.unpack(reuse)
assert not empty and rm["initial_sha256"] == sha(initial)
imgs = [cache.ref_decode(z)[1] for z in slots]
slotmap = dict(zip(rm["tracks"], rm["source_slots"]))
assert set(slotmap) == set(names)
template = (root / "tennis/receiver-replay-20261118-3dd4aa753dfa/qpng.pscp").read_bytes()
assert sha(template) == "3dd4aa753dfa3d15a6fdbd1b368d6b48929b4eb8c933e2217a33bed87f735bfd"
profile = p._embedded_profile(template)
profile["coded_size"] = [240, 136]
profile["order"] = names
profile["runs"] = []
refs = {}
for i, t in enumerate(tracks):
    name = t["name"]
    t0 = int(t["t"][0])
    assert len(t["t"]) == 240 and t0 == 0
    boxes = k6b.run_boxes(list(t["kp"]))
    refname = "ref" + str(i)
    profile["runs"].append({"unit": name, "start": 0, "end": 240, "reference": refname})
    md = {
        "unit": name,
        "reference_frame": 0,
        "reference_box": np.asarray(boxes[0]).tolist(),
        "pixels": "full_prepared_conditioning_crop",
    }
    refs[refname] = portable(rates.encode_reference(imgs[slotmap[name]], md))
# Only base, KPS2, cache-slot mapping and profile are delivered in this record.
wire_profile = {
    "kind": "cached-player-return-v1",
    "player_profile": profile,
    "cache_initial_sha256": sha(initial),
    "cache_return_sha256": sha(reuse),
    "cache_slots": slotmap,
}
wire = w.pack(wire_profile, [("base", "frame", base), ("keypoints", "frame", kps2)])
wire_parts, _ = w.unpack(wire, wire_profile)
rebuilt = p.build(profile, wire_parts["base"], portable(wire_parts["keypoints"]), refs)
sections, _, decoded = p.unpack(rebuilt)
assert sections["base"] == base and sections["keypoints"] == portable(kps2)
# Independent check that decoded paid inputs match the exact consumed model arrays.
golden = root / "tennis/cross-shot-reference-20261122/reuse/conditioning.tar"
for name in names:
    with np.load(io.BytesIO(member(golden, "publish/" + name + ".npz")), allow_pickle=False) as g:
        r = decoded[name][0]
        assert (
            np.array_equal(r["reference"], g["reference"])
            and np.array_equal(r["t"], g["t"])
            and np.array_equal(r["boxes"], g["box"])
        )
        golden_poses = g["pose"]
        assert all(np.array_equal(p.pose_for_tick(r, t), golden_poses[t]) for t in range(240))
positions = [
    None if pair == (65535, 65535) else pair for pair in struct.iter_unpack(">HH", origins)
]
dec = svtav1.tool("dav1d")
side = b.pack(rebuilt, positions, tile, decoder_sha256=dec["sha256"])
assert b._open(rebuilt, side, dec["sha256"])[0] == positions
m = manifest.encode(sha(rebuilt))
(out / "return.pscw").write_bytes(wire)
(out / "receiver.pscp").write_bytes(rebuilt)
(out / "ball.sidecar").write_bytes(side)
(out / "receiver.manifest").write_bytes(m)
result = {
    "cache_initial_bytes": len(initial),
    "cache_return_bytes": len(reuse),
    "return_wire_bytes": len(wire),
    "return_ball_sidecar_bytes": len(side),
    "receiver_manifest_bytes": len(m),
    "return_transmitted_bytes": len(reuse) + len(wire) + len(side) + len(m),
    "receiver_rebuilt_carrier_bytes_not_transmitted": len(rebuilt),
    "kps2_bytes": len(kps2),
    "wire_sha256": sha(wire),
    "carrier_sha256": sha(rebuilt),
    "ball_sha256": sha(side),
    "references_and_240_pose_box_exact": True,
    "scope": "transmitted source-free target player inputs, native base and ball; generated mask/reconstruction still required",
}
(out / "receipt.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
