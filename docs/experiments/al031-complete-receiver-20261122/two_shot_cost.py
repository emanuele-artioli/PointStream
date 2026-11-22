"""Measure two delivered shots from their signed packets and the referenced caches."""

import argparse
import json
from pathlib import Path

from experiments.tennis import common_player_packet as packet


def run(a):
    previous = a.first_joint.read_bytes()
    prior = a.old_cache.read_bytes()
    first_profile = packet._embedded_profile(previous)
    old_sha = packet.wire.sha(prior)
    assert first_profile["cache_sha256"] == old_sha
    first_fields, _ = packet.wire.unpack(previous, first_profile)
    assert len(first_fields["poses"]) > 0 and first_profile["display"] == [1920, 1080]
    assert (
        packet.wire.sha(previous)
        == "ab12294fec2bba5bb5b21bf373a3a31c620b80cf9463cd92ab0383647d1a8aa9"
    )
    identity = json.loads(Path(__file__).with_name("reference-identity.json").read_text())
    assert identity["old_cache_sha256"] == old_sha
    first = len(previous) + len(prior)
    result = dict(
        first_cache_bytes=len(prior),
        first_scene_bytes=len(previous),
        first_joint_sha256=packet.wire.sha(previous),
        first_cache_sha256=old_sha,
        elapsed_reference_seconds=identity["age_seconds"],
    )
    for mode, root in [("send_new", a.fresh), ("reuse", a.reuse)]:
        price = json.loads((root / "result.json").read_text())
        assert price["native_fps"] == "25"
        k = price["selected_crf"]
        folder = root / f"crf{k}"
        joint = (folder / "rally.psj").read_bytes()
        init = (folder / "initial.psr").read_bytes()
        profile = packet._embedded_profile(joint)
        assert profile["cache_sha256"] == packet.wire.sha(init)
        assert (init == prior) == (mode == "reuse")
        fields, _ = packet.wire.unpack(joint, profile)
        assert set(fields) == set(first_fields)
        extra = 0 if mode == "reuse" else len(init)
        total = first + len(joint) + extra
        result[mode] = dict(
            new_cache_bytes=extra,
            scene_bytes=len(joint),
            joint_sha256=packet.wire.sha(joint),
            total_bytes=total,
            total_video_seconds=19.2,
            two_shot_kbps=total * 8 / 19200,
            return_kbps=(len(joint) + extra) * 8 / 9600,
        )
    result["saved_bytes"] = result["send_new"]["total_bytes"] - result["reuse"]["total_bytes"]
    result["two_shot_saved_fraction"] = result["saved_bytes"] / result["send_new"]["total_bytes"]
    result["coverage"] = (
        "Two 240-frame video intervals at 25 fps; original clock interval between them excluded identically"
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for x in ["first-joint", "old-cache", "fresh", "reuse", "out"]:
        p.add_argument("--" + x, type=Path, required=True)
    run(p.parse_args())
