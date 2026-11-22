"""Serialize the measured three-segment delivery with a real integrity index."""

import hashlib
import json
import struct
import tarfile
from pathlib import Path

root = Path("/home/itec/emanuele/Datasets/pointstream-data")
cache = root / "tennis/cross-shot-reference-20261122/reference-channel-lossless-webp-v1"
local = Path("/local/users/emanuele/pointstream/audits")
init = local / "initial-cached-player-v3-20261122"
ret = local / "return-cached-player-v5-20261122"
out = root / "tennis/cumulative-joint-20261122/three-segment-transport-v2"
out.mkdir(parents=True, exist_ok=False)


def sha(x):
    return hashlib.sha256(x).hexdigest()


def member(path, name):
    with tarfile.open(path) as tar:
        return tar.extractfile(name).read()


gap = member(
    root / "jobs/fleet/inbox/20261121T230405Z-9ee24c23/full/published.tar", "publish/fallback.ivf"
)
assert sha(gap) == "0b4960e92ce07db9dabee33daa98a705c215191ba042c6291b1c546a6a87c1d3"
entries = [
    ("cache_initial", cache / "initial.psr"),
    ("initial_wire", init / "initial.pscw"),
    ("initial_ball", init / "ball.sidecar"),
    ("initial_manifest", init / "receiver.manifest"),
    ("fallback_ivf", gap),
    ("cache_return", cache / "return-reuse.psr"),
    ("return_wire", ret / "return.pscw"),
    ("return_ball", ret / "ball.sidecar"),
    ("return_manifest", ret / "receiver.manifest"),
]
parts = [
    (name, value if isinstance(value, bytes) else value.read_bytes()) for name, value in entries
]
header = {
    "version": 1,
    "fps": [50, 1],
    "source_ticks": [76626, 87931],
    "segments": [
        {"mode": "cached-player-ball", "start": 0, "frames": 240},
        {"mode": "native-fallback", "start": 240, "frames": 10825},
        {"mode": "cached-player-ball", "start": 11065, "frames": 240},
    ],
    "entries": [{"name": n, "bytes": len(v), "sha256": sha(v)} for n, v in parts],
    "preinstalled": "common tennis generator, SAM mask generator, codecs and their pinned runtimes",
}
raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
packet = b"PSB1" + struct.pack(">I", len(raw)) + raw + b"".join(v for _, v in parts)
p = out / "delivery.psb"
p.write_bytes(packet)
# Parse the written stream from scratch; the index alone identifies all payload boundaries.
q = p.read_bytes()
assert q[:4] == b"PSB1"
n = struct.unpack_from(">I", q, 4)[0]
h = json.loads(q[8 : 8 + n])
pos = 8 + n
for r in h["entries"]:
    data = q[pos : pos + r["bytes"]]
    assert len(data) == r["bytes"] and sha(data) == r["sha256"]
    pos += r["bytes"]
assert pos == len(q) and sum(s["frames"] for s in h["segments"]) == 11305
r = {
    "total_bytes": len(packet),
    "component_bytes": sum(len(v) for _, v in parts),
    "index_bytes": 8 + len(raw),
    "packet_sha256": sha(packet),
    "duration_seconds": 226.1,
    "kbps": len(packet) * 8 / 226100,
    "all_sections_reparsed": True,
    "frames": 11305,
    "scope": "actual serialized inputs and cut index; end-to-end player generation, masks and all-frame playback not yet validated",
    "entries": h["entries"],
}
(out / "receipt.json").write_text(json.dumps(r, indent=2))
print(json.dumps(r))
