"""Price a proposed ordinary-video + unchanged pose-section control from receipts.

This does not construct, decode, or score a physical control packet.
"""

from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
INPUTS = {}


def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    INPUTS[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def main():
    wire_path = ROOT / "experiments/tennis/w8_payload.py"
    spec = importlib.util.spec_from_file_location("sidecar_wire", wire_path)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    INPUTS[str(wire_path.relative_to(ROOT))] = hashlib.sha256(wire_path.read_bytes()).hexdigest()
    sender = read("docs/experiments/test-crop-joint-20261122/match102-sender.json")
    rung = next(r for r in sender["rates"] if r["crf"] == 50)
    assert sender["frames"] == 240 and rung["pose_bytes"] == 2301
    fps = Fraction(sender["native_fps"])
    duration = Fraction(sender["frames"], 1) / fps
    results = []
    for width in (320, 480):
        folder = "docs/experiments/shanghai-joint-low-anchors-20261122"
        source = read(f"{folder}/{width}-receipt.json")
        index = read(f"{folder}/{width}-index.json")
        scored = read(f"{folder}/score-ordinary-{width}-warm.json")
        video = source["selected"]["warm"]
        assert source["source_sha256"] == scored["source_sha256"] == sender["source_sha256"]
        assert Fraction(source["fps"]) == fps and source["frames"] == sender["frames"]
        assert video["sha256"] == scored["bitstream_sha256"] == index["selected"]["warm"]["sha256"]
        assert video["bytes"] == scored["bytes"]
        # Uncompressed PSW8 header length depends on section lengths and metadata,
        # not their contents: all SHA-256 hex digests occupy exactly 64 bytes.
        # Zero bodies below are size probes only and are never evidence payloads.
        profile = {
            "kind": "ordinary-video-paid-pose-accounting-v1",
            "frames": sender["frames"], "fps": [fps.numerator, fps.denominator],
            "canvas_size": [1920, 1080], "coded_size": [source["width"], source["height"]],
            "order": [unit["unit"] for unit in sender["foreground"]],
            "pose_payload": "unchanged-joint-poses-section",
            "video_payload": "svt-av1-ivf", "coordinates": "native-canvas",
        }
        sections = [("video", "frame", bytes(video["bytes"])), ("poses", "frame", bytes(rung["pose_bytes"]))]
        probe = wire.pack(profile, sections, compact_header=False)
        decoded, roles = wire.unpack(probe, profile)
        assert all(decoded[name] == body for name, _, body in sections)
        assert sum(roles.values()) == len(probe)
        framing = len(probe) - video["bytes"] - rung["pose_bytes"]
        assert framing == roles["window"]
        results.append({
            "name": f"Shanghai ordinary-{width} + paid poses (accounting only)",
            "source_sha256": sender["source_sha256"], "video_producer_job": index["job_id"],
            "video_archive_sha256": index["archive_sha256"],
            "video_sha256": video["sha256"], "video_bytes": video["bytes"],
            "pose_section_bytes": rung["pose_bytes"],
            "pose_parent_joint_sha256": rung["joint_sha256"],
            "profile": profile, "framing_bytes": framing,
            "proposed_total_bytes": len(probe), "proposed_kbps": float(len(probe) * 8 / duration / 1000),
            "full_obj_warm_bytes": rung["joint_bytes"],
            "spare_bytes_under_full_obj_ceiling": rung["joint_bytes"] - len(probe),
            "fits_full_obj_warm_ceiling": len(probe) <= rung["joint_bytes"],
        })
    halle = read("docs/experiments/halle-foreground-20261123/sender-result.json")
    halle_rung = next(r for r in halle["rates"] if r["crf"] == 50)
    halle_video = read("docs/experiments/halle-foreground-20261123/av1-320-score.json")
    assert halle_video["source_sha256"] == halle["source_sha256"]
    output = {
        "scope": "accounting-only; no physical combined packet, decoded pixels, pose accuracy or latency measured",
        "duration_seconds": float(duration), "shanghai": results,
        "halle": {
            "video_bytes": halle_video["bytes"], "pose_section_bytes": halle_rung["pose_bytes"],
            "lower_bound_before_framing_bytes": halle_video["bytes"] + halle_rung["pose_bytes"],
            "full_obj_warm_bytes": halle_rung["joint_bytes"],
            "fits_even_before_framing": halle_video["bytes"] + halle_rung["pose_bytes"] <= halle_rung["joint_bytes"],
        },
        "input_receipt_sha256": INPUTS,
    }
    Path(sys.argv[1]).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"shanghai_bytes": [r["proposed_total_bytes"] for r in results], "halle_fits": output["halle"]["fits_even_before_framing"]}))


if __name__ == "__main__":
    main()
