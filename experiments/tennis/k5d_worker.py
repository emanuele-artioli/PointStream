"""K5d receiver preparation. Study encoding and quality orchestration are not implemented.

The decode entrypoint consumes a PSBL stream and a separately installed resource
map, and writes native YUV and rendered RGB. It establishes packet-path replay,
not filesystem isolation, which requires a separate denial witness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from experiments.tennis.k5d_source_oracle import sha
from src.pointstream import ball_stream


def resource(stream: ball_stream.Stream, mapping: dict[str, Any]) -> dict[str, Any]:
    """Bind the exact on-wire identifier to a preinstalled, hash-checked resource."""
    entries = mapping.get("resources")
    if not isinstance(entries, dict) or stream.resource_id not in entries:
        raise ValueError("installed decoder resource is missing")
    value = entries[stream.resource_id]
    if not isinstance(value, dict) or value.get("codec") != stream.codec:
        raise ValueError("installed resource codec mismatch")
    fields = ("decoder",) if stream.codec == "svt-av1-ivf" else ("image_ckpt", "video_ckpt")
    for name in fields:
        path = value.get(name)
        digest = value.get(name + "_sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or sha(Path(path)) != digest:
            raise ValueError(f"invalid installed resource {name}")
    return value


def validate_native_header(stream: ball_stream.Stream) -> None:
    if stream.codec == "svt-av1-ivf":
        from src.codecs.svtav1 import ivf_frames
        data = stream.native
        if len(data) < 32:
            raise ValueError("truncated IVF")
        magic, version, size, fourcc, w, h, num, den, count, _ = struct.unpack("<4sHH4sHHIIII", data[:32])
        if (magic != b"DKIF" or version != 0 or size != 32 or fourcc != b"AV01"
                or (w, h) != (stream.width, stream.height) or not den
                or Fraction(num, den) != stream.fps or count != stream.frames
                or len(ivf_frames(data)) != stream.frames):
            raise ValueError("IVF header or frame count differs from PSBL")
    else:
        from src.codecs.dcvc_uf_worker import unpack_container
        header, payload = unpack_container(stream.native)
        if header["structure"] != "htl" or header["frame_count"] != stream.frames or not payload:
            raise ValueError("DCVC header differs from PSBL")


def _native_result(process: Any, path: Path) -> None:
    """Persist bounded native diagnostics before inspecting a pre-exec receipt."""
    _, stderr = process.communicate(timeout=3600)
    tail = stderr[-4096:]
    result = {"returncode": process.returncode, "stderr_bytes": len(stderr),
              "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
              "stderr_tail": tail.decode("utf-8", errors="replace")}
    with path.open("x") as output:
        json.dump(result, output, sort_keys=True, allow_nan=False)
        output.write("\n")
    if process.returncode:
        raise RuntimeError(f"native decoder failed: {process.returncode}: {tail!r}")


def _execute_native(command: list[str], work: Path, isolation: dict[str, Any] | None, **kwargs: Any) -> None:
    if isolation is None:
        subprocess.run(command, check=True, capture_output=True, timeout=3600, **kwargs)
        return
    resolved_executable = str(Path(command[0]).resolve(strict=True))
    if resolved_executable not in isolation["actual_policy"]["execute_files"]:
        raise ValueError("native decoder executable is not in actual policy")
    child_env = kwargs.pop("env", None)
    native_env = dict(os.environ if child_env is None else child_env)
    payload_path = work.parent / "native-exec.json"
    payload_path.write_text(json.dumps({"argv": command, "env": native_env, "nonce": isolation["nonce"],
                                        "campaign_pgid": isolation["campaign_host"]["campaign_pgid"]}, sort_keys=True))
    wrapper_env = dict(native_env)
    wrapper_env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    wrapper = [sys.executable, "-S", "-m", "experiments.tennis.k5d_isolated_receiver",
               "native-wrapper", str(payload_path)]
    with subprocess.Popen(wrapper, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          close_fds=True, env=wrapper_env, **kwargs) as process:
        try:
            birth = time.monotonic_ns()
            observer_path = work.parent / "native-observer.json"
            deadline = time.monotonic() + 30
            while not observer_path.is_file():
                if process.poll() is not None:
                    _native_result(process, work.parent / "native-process-result.json")
                    raise RuntimeError("native decoder exited before observation")
                if time.monotonic() >= deadline:
                    raise RuntimeError("outer native decoder observation missing")
                time.sleep(0.01)
            observation = json.loads(observer_path.read_text())
            if (observation.get("nonce") != isolation["nonce"]
                    or observation.get("namespace_pids", [])[-1:] != [process.pid]
                    or observation.get("namespaces") != isolation["namespaces"]["namespaces"]
                    or observation.get("host_pgid") != isolation["campaign_host"]["campaign_pgid"]):
                raise ValueError("native decoder observer identity differs")
            _native_result(process, work.parent / "native-process-result.json")
            preexec = json.loads((work.parent / "native-preexec.json").read_text())
            if (preexec.get("pid") != process.pid or preexec.get("birth") != observation.get("birth")
                    or preexec.get("observer") != observation or preexec.get("closed_extra_fds") is not True
                    or preexec.get("argv_sha256") != hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()):
                raise ValueError("native pre-exec identity differs")
        except BaseException:
            process.kill()
            process.wait()
            raise
        isolation["decoder"] = {"pid": process.pid, "started_monotonic_ns": birth, "argv": command,
                                "executable_resolved": resolved_executable, "outer_observation": observation, "preexec": preexec,
                                "argv_sha256": hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest(),
                                "returncode": process.returncode}



def decode(wire_path: Path, map_path: Path, work: Path, *, isolation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decode persisted native bytes and render persisted symbols with no source inputs."""
    import numpy as np
    from experiments.tennis import windows as tw

    if wire_path.stat().st_size > ball_stream.MAX_NATIVE + ball_stream.MAX_BALL + 4096:
        raise ValueError("PSBL exceeds receiver size limit")
    wire = wire_path.read_bytes()
    stream = ball_stream.unpack(wire)
    validate_native_header(stream)
    mapping = json.loads(map_path.read_text())
    installed = resource(stream, mapping)
    if stream.width % 2 or stream.height % 2:
        raise ValueError("YUV420 requires even geometry")
    work.mkdir(parents=True, exist_ok=False)
    native = work / ("native.ivf" if stream.codec == "svt-av1-ivf" else "native.psdc")
    native.write_bytes(stream.native)
    decoded = work / "native.yuv"

    if stream.codec == "svt-av1-ivf":
        from src.codecs.svtav1 import decode_command
        command = decode_command(installed["decoder"], native, decoded, threads=1)
        _execute_native(command, work, isolation)
    else:
        from src.codecs.dcvc_uf_worker import dcvc_command
        plan = {"structure": "htl", "src_type": "yuv420", "container": str(native), "out_file": str(decoded),
                "image_ckpt": installed["image_ckpt"], "image_sha256": installed["image_ckpt_sha256"],
                "video_ckpt": installed["video_ckpt"], "video_sha256": installed["video_ckpt_sha256"]}
        plan_path = work / "receiver-plan.json"
        plan_path.write_text(json.dumps(plan))
        command, env, cwd = dcvc_command(Path(sys.prefix), "decode", plan_path, work / "native-decoder-report.json")
        _execute_native(command, work, isolation, env={**os.environ, **env}, cwd=cwd)
        decoder_report = json.loads((work / "native-decoder-report.json").read_text())
        if not decoder_report.get("deterministic") or decoder_report.get("frame_count") != stream.frames:
            raise ValueError("DCVC decoder determinism or frame count failed")
    expected = stream.width * stream.height * 3 // 2 * stream.frames
    if decoded.stat().st_size != expected:
        raise ValueError("decoded native frame count or geometry differs")
    yuv = np.memmap(decoded, dtype=np.uint8, mode="r", shape=(stream.frames, stream.height * 3 // 2, stream.width))
    rgb = np.stack([tw.yuv_to_rgb(frame, stream.width, stream.height) for frame in yuv])
    rendered = ball_stream.render_from_native_decoded_frames(stream, rgb)
    output = work / "rendered.rgb"
    with output.open("xb") as handle:
        handle.write(rendered.tobytes())
    receipt = {"scope": "fresh packet-path replay; filesystem isolation NOT established",
               "wire_sha256": sha(wire_path), "wire_bytes": len(wire), "native_sha256": sha(native),
               "native_bytes": len(stream.native), "decoded_yuv_sha256": sha(decoded),
               "rendered_rgb_sha256": sha(output), "frames": stream.frames, "size": [stream.width, stream.height],
               "fps": str(stream.fps), "codec": stream.codec, "resource_id": stream.resource_id,
               "resource_map_sha256": sha(map_path), "colour_conversion": "limited-range YUV420 to RGB, BT.709"}
    if isolation is not None:
        from experiments.tennis.k5d_isolated_receiver import SCOPE, verify_receipt
        receipt["scope"] = SCOPE
        receipt["isolation"] = isolation
        verify_receipt(receipt, expected_context=isolation["context"], expected_config_sha256=isolation["config_sha256"])
    (work / "receiver.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    receiver = sub.add_parser("decode", help="decode PSBL with installed resource map")
    receiver.add_argument("--wire", type=Path, required=True)
    receiver.add_argument("--resources", type=Path, required=True)
    receiver.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    scratch = os.environ.get("PS_SCRATCH_DIR")
    if not scratch or not args.work_dir.resolve().is_relative_to(Path(scratch).resolve()):
        parser.error("work-dir must be under PS_SCRATCH_DIR")
    print(json.dumps(decode(args.wire, args.resources, args.work_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
