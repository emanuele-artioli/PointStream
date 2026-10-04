"""Audit every charged native IVF container's geometry, timebase and frame ledger."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct
import resource
import subprocess
import time


def identity(path):
    digest = hashlib.sha256()
    count = 0
    started = time.monotonic()
    with path.open("rb") as handle:
        while block := handle.read(1024**2):
            if os.getloadavg()[0] > 40:
                raise RuntimeError("native audit host load exceeded40")
            digest.update(block)
            count += len(block)
            delay = count / (20 * 1024**2) - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    return {"bytes": count, "sha256": digest.hexdigest()}


def ivf_header(path):
    with Path(path).open("rb") as handle:
        fields = struct.unpack("<4sHH4sHHIIII", handle.read(32))
        signature, version, length, codec, width, height, rate, scale, count, reserved = fields
        if (signature, version, length, codec) != (b"DKIF", 0, 32, b"AV01"):
            raise ValueError("native IVF format mismatch")
        packets = 0
        while block := handle.read(12):
            if len(block) != 12:
                raise ValueError("truncated IVF packet header")
            size, timestamp = struct.unpack("<IQ", block)
            if handle.tell() + size > Path(path).stat().st_size:
                raise ValueError("truncated IVF payload")
            handle.seek(size, 1)
            packets += 1
        if packets != count:
            raise ValueError("native IVF declared/physical packet denominator mismatch")
    return {
        "geometry": [width, height],
        "fps": str(Fraction(rate, scale)),
        "header_frames": count,
        "physical_packets": packets,
    }


def audit(report_path, ffprobe="/opt/local/bin/ffprobe"):
    report = json.loads(report_path.read_text())
    receipts = {}
    manifests = []
    if report.get("status") != "complete":
        raise ValueError("completed native study required")
    for source in report["sources"]:
        for arm in source["arms"]:
            path = Path(arm["manifest"]["path"])
            actual = identity(path)
            if (
                actual["bytes"] != arm["manifest"]["bytes"]
                or actual["sha256"] != arm["manifest"]["sha256"]
            ):
                raise ValueError("manifest identity mismatch")
            manifest = json.loads(path.read_text())
            total = actual["bytes"]
            cursor = 0
            for packet in manifest["packets"]:
                stream = Path(packet["stream"]["path"])
                expected = 1 if "hold_until" in packet else packet["end"] - packet["frame"]
                end = packet.get("hold_until", packet.get("end"))
                if packet["frame"] != cursor or end > manifest["frames"]:
                    raise ValueError("placement denominator mismatch")
                cursor = end
                if str(stream) not in receipts:
                    receipts[str(stream)] = {
                        "identity": identity(stream),
                        "header": ivf_header(stream),
                    }
                    cmd = [
                        ffprobe,
                        "-v",
                        "error",
                        "-threads",
                        "1",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,width,height,pix_fmt",
                        "-of",
                        "json",
                        str(stream),
                    ]
                    native_info = json.loads(
                        subprocess.run(
                            cmd, check=True, capture_output=True, text=True, timeout=30
                        ).stdout
                    )["streams"]
                    if (
                        len(native_info) != 1
                        or native_info[0]["codec_name"] != "av1"
                        or [native_info[0]["width"], native_info[0]["height"]]
                        != manifest["geometry"]
                        or native_info[0]["pix_fmt"] != "yuv420p"
                    ):
                        raise ValueError(
                            "native probe geometry/codec/pixel-format contract mismatch"
                        )
                    receipts[str(stream)]["native_probe"] = {"command": cmd, "streams": native_info}
                receipt = receipts[str(stream)]
                native = receipt["identity"]
                header = receipt["header"]
                if (
                    native["bytes"] != packet["stream"]["bytes"]
                    or native["sha256"] != packet["stream"]["sha256"]
                ):
                    raise ValueError("native IVF identity mismatch")
                if (
                    header["geometry"] != manifest["geometry"]
                    or header["fps"] != manifest["fps"]
                    or header["header_frames"] != expected
                ):
                    raise ValueError("native IVF geometry/timebase/frame contract mismatch")
                total += native["bytes"]
            if cursor != source["frame_count"] or total != arm["bytes"]:
                raise ValueError("full native package denominator/byte ledger mismatch")
            manifests.append(
                {
                    "source": source["source_id"],
                    "arm": arm["name"],
                    "crf": arm["crf"],
                    "manifest_sha256": actual["sha256"],
                    "charged_bytes": total,
                }
            )
    return {
        "report_sha256": identity(report_path)["sha256"],
        "audit_worker_sha256": identity(Path(__file__))["sha256"],
        "status": "pass",
        "full_manifests": manifests,
        "unique_native_streams": receipts,
        "ffprobe": identity(Path(ffprobe)),
        "ffprobe_version": subprocess.run(
            [ffprobe, "-version"], check=True, capture_output=True, text=True, timeout=30
        ).stdout,
        "scope": "Native IVF container headers and physical packet/byte ledger, combined with existing fresh native decoder parity; not an arbitrary malicious-bitstream security audit.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--code-revision", required=True)
    a = p.parse_args()
    os.nice(19)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True)
    result = audit(a.report)
    result["audit_code_revision"] = a.code_revision
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(identity(a.out)))


if __name__ == "__main__":
    main()
