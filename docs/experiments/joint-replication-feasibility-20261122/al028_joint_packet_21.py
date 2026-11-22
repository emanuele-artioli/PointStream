"""Build, charge and reconstruct the paid 25fps Al028 trajectory/player channel."""

import hashlib
import json
from pathlib import Path

from experiments.tennis import common_player_packet as players
from experiments.tennis import common_receiver_manifest as receiver
from experiments.tennis import w8_payload as wire
from src.pointstream import ball

ROOT = Path("/private/tmp/ps-al028-donor25-crf52-20261122")
PAYLOAD = Path("/private/tmp/ps-al028-ball-sender-full.payload")
OUT = Path("/private/tmp/ps-al028-joint25-crf52-20261122")
EXPECTED = "b430f7304390cd8986444e2c9c4bfc1c2269b40601990a8f5b94419e17601dc9"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    carrier = (ROOT / "players.pscp").read_bytes()
    if sha(carrier) != EXPECTED:
        raise ValueError("carrier identity")
    profile = players._embedded_profile(carrier)
    parts, roles = wire.unpack(carrier, profile)
    recipe = json.loads((ROOT / "players.manifest").read_text())
    if (
        recipe.pop("carrier_sha256") != EXPECTED
        or recipe.pop("composition") != "decoded-base,farthest-to-nearest-players,optional-ball"
    ):
        raise ValueError("manifest identity")
    symbols = PAYLOAD.read_bytes()
    positions, appearance = ball.decode_frames(symbols, 240, 960, 544)
    if ball.encode_frames(positions, appearance, 960, 544) != symbols:
        raise ValueError("ball round-trip")
    cache_profile = {
        "type": "joint-initialization-v1",
        "runtime": recipe,
        "resources": profile["resources"],
    }
    cache = wire.pack(
        cache_profile, [(k, "reference", v) for k, v in parts.items() if k.startswith("ref")]
    )
    joint_profile = {
        "kind": "joint-trajectory-v1",
        "player": {k: v for k, v in profile.items() if k != "resources"},
        "cache_sha256": sha(cache),
        "ball_canvas": [960, 544],
        "display": [1920, 1080],
        "order": ["clean_base", "ball", "far_player", "near_player"],
    }
    joint = wire.pack(
        joint_profile,
        [
            ("base", "frame", parts["base"]),
            ("poses", "frame", parts["keypoints"]),
            ("trajectory", "frame", symbols),
        ],
    )
    cached, _ = wire.unpack(cache, cache_profile)
    delivered, _ = wire.unpack(joint, joint_profile)
    rebuilt_profile = joint_profile["player"] | {"resources": cache_profile["resources"]}
    rebuilt = players.build(rebuilt_profile, delivered["base"], delivered["poses"], cached)
    if (
        rebuilt != carrier
        or receiver.encode(sha(rebuilt)) != (ROOT / "players.manifest").read_bytes()
        or delivered["trajectory"] != symbols
    ):
        raise ValueError("joint channel failed paid receiver reconstruction")
    OUT.mkdir(exist_ok=False)
    (OUT / "initial.psr").write_bytes(cache)
    (OUT / "rally.psj").write_bytes(joint)
    result = {
        "scope": "Exact packet accounting and source-independent input reconstruction, before learned replay. No generated masks are transmitted; masks are receiver-computed. The common model is preinstalled.",
        "ticks": 240,
        "fps": 25,
        "initial_bytes": len(cache),
        "joint_bytes": len(joint),
        "cold_bytes": len(cache) + len(joint),
        "ideal_prewarmed_bytes": len(joint),
        "cold_kbps": (len(cache) + len(joint)) / 1200,
        "ideal_prewarmed_kbps": len(joint) / 1200,
        "cache_sha256": sha(cache),
        "joint_sha256": sha(joint),
        "receiver_carrier_sha256": sha(rebuilt),
        "paid_parts_bytes": {
            "base": len(parts["base"]),
            "poses": len(parts["keypoints"]),
            "ball": len(symbols),
            "reference_blobs": roles["reference"],
        },
        "framing_bytes": len(cache)
        + len(joint)
        - len(parts["base"])
        - len(parts["keypoints"])
        - len(symbols)
        - roles["reference"],
        "order": joint_profile["order"],
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
