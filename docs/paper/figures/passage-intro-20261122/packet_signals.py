"""Extract the first rally's displayed overlays from its transmitted packet."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

from experiments.tennis import common_player_packet as player
from experiments.tennis import w8_payload as transport

HERE = Path(__file__).resolve().parent
TICK = 90


def signals(path):
    receipt = json.loads((HERE / "source-frame-receipt.json").read_text())
    wire = path.read_bytes()
    assert hashlib.sha256(wire).hexdigest() == receipt["packet_sha256"] and wire[:4] == b"PSB1"
    offset = 8 + int.from_bytes(wire[4:8], "big")
    header = json.loads(wire[8:offset])
    sections = {}
    for row in header["entries"]:
        data = wire[offset : offset + row["bytes"]]
        offset += len(data)
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        if row["name"] in {"initial_wire", "initial_ball"}:
            profile = player._embedded_profile(data)
            parts, _ = transport.unpack(data, profile)
            sections[row["name"]] = (profile, parts)
    assert offset == len(wire)
    profile, parts = sections["initial_wire"]
    layout = profile["player_profile"]
    assert layout["canvas_size"] == [1920, 1080] and layout["recipe"]["fps"] == [50, 1]
    _, tracks = player._bounded_keypoints(parts["keypoints"], layout["order"])
    assert len(tracks) == 2 and all(t["t"][TICK] == TICK for t in tracks)
    positions = list(struct.iter_unpack(">HH", sections["initial_ball"][1]["positions"]))
    assert len(positions) == 240
    # Centers of the transmitted 32-by-32 patches are in native display coordinates.
    tail = [
        None if x == y == 65535 else [x + 16, y + 16] for x, y in positions[TICK - 12 : TICK + 1]
    ]
    assert receipt["rows"][0]["passage_seconds"] == TICK / 50
    return {
        "packet_sha256": receipt["packet_sha256"],
        "tick": TICK,
        "keypoint_section_bytes": len(parts["keypoints"]),
        "first_tail_tick": TICK - 12,
        "tile_centers": tail,
        "tracks": {t["name"]: t["kp"][TICK].tolist() for t in tracks},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x") as out:
        json.dump(signals(args.packet), out, indent=2)
        out.write("\n")
