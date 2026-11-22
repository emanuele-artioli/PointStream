"""Strictly validate an independently annotated source-only body-joint export."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


def validate(manifest_path, export_path):
    raw = manifest_path.read_bytes()
    spec = json.loads(raw)
    result = json.loads(export_path.read_text())
    if (
        result.get("schema") != spec["schema"]
        or result.get("manifest_sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("schema or source manifest identity mismatch")
    if (
        result.get("source_sha256") != spec["source_sha256"]
        or result.get("width") != spec["width"]
        or result.get("height") != spec["height"]
        or result.get("fps") != spec["fps"]
    ):
        raise ValueError("source identity, geometry or fps mismatch")
    frames = result.get("frames")
    if not isinstance(frames, list) or len(frames) != len(spec["frames"]):
        raise ValueError("frame count mismatch")
    counts = {k: 0 for k in ("visible", "occluded", "unsure", "not_visible")}
    for row, expected in zip(frames, spec["frames"], strict=True):
        if (
            expected.get("file") != f"frames/{expected['tick']:03d}.png"
            or expected["timestamp_seconds"] != expected["tick"] / spec["fps"]
        ):
            raise ValueError("manifest source frame path or timestamp mismatch")
        frame_path = manifest_path.parent / expected["file"]
        if hashlib.sha256(frame_path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("source frame bytes changed")
        with Image.open(frame_path) as image:
            if image.size != (spec["width"], spec["height"]):
                raise ValueError("source image dimensions changed")
        if (
            row.get("tick") != expected["tick"]
            or row.get("timestamp_seconds") != expected["timestamp_seconds"]
            or row.get("source_png_sha256") != expected["sha256"]
        ):
            raise ValueError("frame order, time or image identity mismatch")
        actors = row.get("actors")
        if not isinstance(actors, list) or [a.get("id") for a in actors] != spec["actors"]:
            raise ValueError("actor identity or ordering mismatch")
        for actor in actors:
            joints = actor.get("joints")
            if not isinstance(joints, list) or [j.get("name") for j in joints] != spec["joints"]:
                raise ValueError("joint names, ordering or coverage mismatch")
            for joint in joints:
                status = joint.get("status")
                if status not in counts:
                    raise ValueError("unmarked or unknown joint status")
                x, y = joint.get("x"), joint.get("y")
                if status == "visible":
                    if any(
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not math.isfinite(v)
                        for v in (x, y)
                    ) or not (0 <= x < spec["width"] and 0 <= y < spec["height"]):
                        raise ValueError(
                            "visible joint requires finite in-bounds source coordinates"
                        )
                elif x is not None or y is not None:
                    raise ValueError("non-visible joint must not invent a coordinate")
                counts[status] += 1
    return {
        "schema": spec["schema"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "export_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
        "source_sha256": spec["source_sha256"],
        "frames": len(frames),
        "actors_per_frame": len(spec["actors"]),
        "joint_slots": sum(counts.values()),
        "status_counts": counts,
        "synthetic_fixture": spec["synthetic_fixture"],
        "validated_export": result,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = validate(a.manifest, a.export)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print("validated", result["frames"], "frames", result["status_counts"])


if __name__ == "__main__":
    main()
