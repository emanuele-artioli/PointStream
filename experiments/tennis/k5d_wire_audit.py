"""Audit new wire framing against immutable K5c streams; no codec or model run."""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import subprocess
import tarfile
import time
from typing import Any

from src.pointstream import ball_stream
from src.pointstream.bitstream import Container, Tag

ARCHIVE_SHA256 = "d40a9f7e0a7d2878d01010d4e128467bbdc174f9a899e007a555b9febe256d32"


def audit(path: Path, expected_sha256: str) -> dict[str, Any]:
    started = time.process_time()
    if expected_sha256 != ARCHIVE_SHA256 or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("requires the exact bounded K5c evidence archive")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("archive identity differs")
    results = []
    with tarfile.open(path, "r:") as archive:
        members = archive.getmembers()
        if (len(members) > 100 or sum(m.size for m in members) > 16 * 1024 * 1024
                or len({m.name for m in members}) != len(members)):
            raise ValueError("archive is not the bounded saved pilot")

        def read(name: str) -> bytes:
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError("non-file archive member")
            file = archive.extractfile(member)
            if file is None:
                raise ValueError("missing archive member")
            return file.read()

        report = json.loads(read("publish/pilot.json"))
        width, height = report["size"]
        fps, frames = Fraction(report["fps"]), report["frames"]
        expected_points = {("dcvc", q) for q in (0, 21, 42)} | {("svt", q) for q in (48, 55, 62)}
        if len(report["points"]) != 6 or {(p["codec"], p["point"]) for p in report["points"]} != expected_points:
            raise ValueError("saved six-setting coverage differs")
        for point in report["points"]:
            kind, setting = point["codec"], point["point"]
            directory = f"publish/{kind}-{setting}"
            native = read(f"{directory}/stream." + ("psdc" if kind == "dcvc" else "ivf"))
            packet = read(f"{directory}/ball.psb")
            recorded = point["streak"]["ledger"]
            if (hashlib.sha256(native).hexdigest() != recorded["native_sha256"]
                    or hashlib.sha256(packet).hexdigest() != recorded["ball_sha256"]
                    or len(native) != recorded["native_bytes"] or len(packet) != recorded["ball_container_bytes"]):
                raise ValueError("saved component identity differs")
            original = Container.unpack(packet)
            if ((original.width, original.height, original.fps, original.frames) != (width, height, fps, frames)
                    or len(original.segments) != 1 or original.segments[0][0] != Tag.BALL):
                raise ValueError("saved BALL geometry or segments differ")
            codec = "dcvc-uf" if kind == "dcvc" else "svt-av1-ivf"
            resource = "dcvc-uf-htl@main-32e68872" if kind == "dcvc" else "svt-av1@4.2.0+av1-rgb-v1"
            stream = ball_stream.Stream(width, height, fps, frames, codec, resource, native, original.segments[0][1])
            wire = ball_stream.pack(stream)
            restored = ball_stream.unpack(wire)
            if restored != stream or restored.native != native or restored.ball_payload != original.segments[0][1]:
                raise ValueError("new wire did not retain exact components")
            metadata_bytes = len(wire) - len(native) - len(stream.ball_payload)
            results.append({"codec": codec, "setting": setting, "native_bytes": len(native),
                            "ball_payload_bytes": len(stream.ball_payload), "metadata_and_integrity_bytes": metadata_bytes,
                            "total_wire_bytes": len(wire), "kbps": len(wire) * 8 / float(stream.duration) / 1000,
                            "wire_sha256": hashlib.sha256(wire).hexdigest(), "native_sha256": recorded["native_sha256"],
                            "geometry": [width, height], "frames": frames, "fps": str(fps), "resource_id": resource})
    return {"passed": True, "scope": "framing roundtrip on six saved real native/BALL records; no new pixels, decoding, quality or isolation result",
            "original_job": "20261103T141758Z-f5d52fac", "original_archive_sha256": digest,
            "new_wire_version": 1, "checks": ["exact_native_bytes", "canonical_ball_symbols", "frame_geometry", "full_wire_accounting"],
            "points": results, "cpu_seconds": time.process_time() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive, args.sha256)
    result["code_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({"passed": result["passed"], "settings": len(result["points"]),
                      "cpu_seconds": result["cpu_seconds"], "report": str(args.output)}))


if __name__ == "__main__":
    main()
