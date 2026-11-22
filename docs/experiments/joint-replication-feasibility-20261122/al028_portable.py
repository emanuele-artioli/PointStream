"""Source-free local transport normalization for the pinned generator runtime."""

import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
import zstandard as zstd

from experiments.tennis import common_player_packet as players
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import k6b_rate_streams as rates
from experiments.tennis import w8_payload as wire

ROOT = Path("/private/tmp/ps-al028-joint25-packets-20261122")
OUT = Path("/private/tmp/ps-al028-portable-20261122")
JOINT = "fb69663af77820efc3d84c65ec314a15948805ab1f3bc70571fcab0ee2d1e1bd"
CACHE = "91bd3a3fd9b18db301348c98e5af1536b859f2e51dde71e7de56c3beee1c1af2"
CARRIER = "3aa0bbd4bfc7cd2c479dc2ce8128144a24c96770c01ed9f19f55c77f0f0f525f"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def portable(blob):
    magic, size, _, digest = rates.HEADER.unpack_from(blob)
    raw = zstd.ZstdDecompressor().decompress(blob[rates.HEADER.size :])
    if len(raw) != size or hashlib.sha256(raw).digest() != digest:
        raise ValueError("input transport integrity")
    comp = zlib.compress(raw, 6)
    return rates.HEADER.pack(magic, size, len(comp), digest) + comp


def main():
    joint = (ROOT / "rally.psj").read_bytes()
    cache = (ROOT / "initial.psr").read_bytes()
    if sha(joint) != JOINT or sha(cache) != CACHE:
        raise ValueError("paid packet identity")
    jp = players._embedded_profile(joint)
    cp = players._embedded_profile(cache)
    pieces, _ = wire.unpack(joint, jp)
    refs, _ = wire.unpack(cache, cp)
    profile = jp["player"] | {"resources": cp["resources"]}
    original = players.build(profile, pieces["base"], pieces["poses"], refs)
    if sha(original) != CARRIER:
        raise ValueError("paid carrier reconstruction")
    portable_refs = {k: portable(v) for k, v in refs.items()}
    normalized = players.build(profile, pieces["base"], portable(pieces["poses"]), portable_refs)
    first = players.unpack(original)
    second = players.unpack(normalized)
    proofs = {}
    for name in profile["order"]:
        a = first[2][name][0]
        b = second[2][name][0]
        checks = {k: bool(np.array_equal(a[k], b[k])) for k in ("t", "kp", "reference", "boxes")}
        checks["seed"] = a["seed"] == b["seed"]
        checks["all_240_pose_rasters"] = all(
            np.array_equal(players.pose_for_tick(a, i), players.pose_for_tick(b, i))
            for i in range(240)
        )
        if not all(checks.values()):
            raise ValueError("portable conditioning differs")
        proofs[name] = checks
    OUT.mkdir(exist_ok=False)
    (OUT / "players.pscp").write_bytes(normalized)
    (OUT / "players.manifest").write_bytes(manifest.encode(sha(normalized)))
    result = {
        "scope": "Source-free zstd-to-zlib local normalization; paid joint/cache bytes unchanged, internal carrier not transmitted. Requires a preinstalled zstd-capable front end.",
        "joint_sha256": JOINT,
        "cache_sha256": CACHE,
        "original_carrier_sha256": CARRIER,
        "internal_carrier_sha256": sha(normalized),
        "internal_manifest_sha256": sha((OUT / "players.manifest").read_bytes()),
        "internal_bytes": len(normalized),
        "kps_sha256": sha(portable(pieces["poses"])),
        "checks": proofs,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
