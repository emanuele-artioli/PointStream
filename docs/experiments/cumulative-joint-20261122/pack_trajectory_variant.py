"""Replace only return ball tiles with the authenticated fitted trajectory."""

import hashlib
import json
import struct
from pathlib import Path

from src.pointstream import ball

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
INPUT = ROOT / "tennis/cumulative-joint-20261122/three-segment-transport-v2/delivery.psb"
TRAJECTORY = ROOT / "jobs/fleet/inbox/20261122T020713Z-4e37b55b/full/trajectory.bin"
OUTPUT = ROOT / "tennis/cumulative-joint-20261122/three-segment-trajectory-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    packet = INPUT.read_bytes()
    assert sha(packet) == "7ad5e13d4007d5510e70c2d9892c082817d2f4f59e07cdc38641db172b471f64"
    n = struct.unpack_from(">I", packet, 4)[0]
    header = json.loads(packet[8 : 8 + n])
    cursor = 8 + n
    parts = []
    for row in header["entries"]:
        data = packet[cursor : cursor + row["bytes"]]
        cursor += len(data)
        assert len(data) == row["bytes"] and sha(data) == row["sha256"]
        if row["name"] == "return_ball":
            data = TRAJECTORY.read_bytes()
            assert sha(data) == "1106b696731cbc98b7a61985f4c2122d7f6109b57f9bd238e2f0a73a277b1ece"
            row["name"] = "return_trajectory"
        elif row["name"] == "return_manifest":
            manifest = json.loads(data)
            manifest["composition"] = "decoded-base,trajectory,far-to-near-players"
            data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("ascii")
        row["bytes"], row["sha256"] = len(data), sha(data)
        parts.append(data)
    assert cursor == len(packet)
    header["segments"][2]["mode"] = "cached-player-trajectory-before-players"
    header["trajectory"] = {
        "codec": "pointstream-ball-frames-v1",
        "canvas": [960, 544],
        "frames": 240,
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    wire = b"PSB1" + struct.pack(">I", len(encoded)) + encoded + b"".join(parts)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    path = OUTPUT / "delivery.psb"
    path.write_bytes(wire)
    again = path.read_bytes()
    n = struct.unpack_from(">I", again, 4)[0]
    hdr = json.loads(again[8 : 8 + n])
    cursor = 8 + n
    for row in hdr["entries"]:
        data = again[cursor : cursor + row["bytes"]]
        cursor += len(data)
        assert len(data) == row["bytes"] and sha(data) == row["sha256"]
        if row["name"] == "return_trajectory":
            positions, appearances = ball.decode_frames(data, 240, 960, 544)
            assert len(positions) == len(appearances) == 240
    assert cursor == len(again)
    result = {
        "total_bytes": len(wire),
        "index_bytes": 8 + len(encoded),
        "trajectory_bytes": TRAJECTORY.stat().st_size,
        "return_manifest_bytes": next(
            r["bytes"] for r in hdr["entries"] if r["name"] == "return_manifest"
        ),
        "packet_sha256": sha(wire),
        "kbps": len(wire) * 8 / 226100,
        "duration_seconds": 226.1,
        "scope": "actual indexed two-visit transport, streak before players; receiver rendering measured separately",
    }
    (OUTPUT / "receipt.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
