"""Encode or decode DCVC-UF streams in the installed DCVC environment.

Run as a script by the DCVC interpreter, with the DCVC checkout as cwd and
PYTHONPATH. It imports only DCVC's own modules and mirrors
``test_video.py::run_one_point_with_stream`` at the recorded revision, with
three differences:

* The device stays the dispatcher-claimed GPU. ``test_video.py`` workers
  rewrite CUDA_VISIBLE_DEVICES to a bare index, which can leave the claim.
* Checkpoints are loaded from bytes whose SHA-256 was just verified.
* Decoding runs in its own process from the container alone. Its image
  proxy is created explicitly, because the installed model creates it only
  inside ``compress``.

Container: b"PSDC", version, flags, structure, frame count (big-endian
uint16), then the native DCVC stream. Rate counts the whole file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

MAGIC = b"PSDC"
VERSION = 1
FLAG_FORCE_INTRA = 1
STRUCTURE_CODES = {"ld": 0, "hts": 1, "htl": 2}
HEADER = struct.Struct(">4sBBBH")


def pack_container(native: bytes, *, structure: str, frame_count: int, force_intra: bool) -> bytes:
    if structure not in STRUCTURE_CODES or not 0 < frame_count < 65536:
        raise ValueError("invalid container structure or frame count")
    flags = FLAG_FORCE_INTRA if force_intra else 0
    return HEADER.pack(MAGIC, VERSION, flags, STRUCTURE_CODES[structure], frame_count) + native


def unpack_container(data: bytes) -> tuple[dict, bytes]:
    if len(data) < HEADER.size:
        raise ValueError("truncated DCVC container")
    magic, version, flags, code, frame_count = HEADER.unpack(data[:HEADER.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError("not a version-1 PointStream DCVC container")
    structures = {value: key for key, value in STRUCTURE_CODES.items()}
    if code not in structures or frame_count <= 0:
        raise ValueError("invalid DCVC container header")
    return {
        "structure": structures[code], "frame_count": frame_count,
        "force_intra": bool(flags & FLAG_FORCE_INTRA), "header_bytes": HEADER.size,
    }, data[HEADER.size:]


# factory_bg_rd scoring and the 20261003 preview both passed --reset_interval 0.
RESET_INTERVAL = 0


def reset_flag(frame_index: int, frame_delay: int, reset_interval: int) -> int:
    """test_video.py's feature-memory reset rule for a prediction NAL."""
    return int(reset_interval > 0 and (frame_index + frame_delay) % reset_interval == 1)


def display_schedule(frame_count: int, frame_delay: int, *, force_intra: bool) -> list[list[int]]:
    """Display indices carried by each NAL, matching test_video.py.

    One I frame starts the segment; the rest is coded in groups of
    ``frame_delay``. The last group's encoder padding is not displayed.
    """
    if frame_count <= 0 or frame_delay not in (1, 8):
        raise ValueError("invalid frame count or frame delay")
    if force_intra:
        return [[index] for index in range(frame_count)]
    schedule = [[0]]
    index = 1
    while index < frame_count:
        width = min(frame_delay, frame_count - index)
        schedule.append(list(range(index, index + width)))
        index += width
    return schedule


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_check(expected: dict) -> dict:
    actual = {}
    for rel in sorted(expected):
        path = Path.cwd() / rel
        actual[rel] = _file_sha256(path) if path.is_file() else None
    mismatched = sorted(rel for rel in expected if actual[rel] != expected[rel])
    return {"actual": actual, "mismatched": mismatched, "matches_reference": not mismatched}


def environment_report(with_cuda: bool) -> dict:
    import numpy
    import torch

    report = {
        "python": sys.executable, "python_version": sys.version.split()[0],
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "numpy": numpy.__version__,
        "cwd": str(Path.cwd()), "pythonpath": os.environ.get("PYTHONPATH"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import inference_extensions_cuda as extension

        location = Path(extension.__file__)
        report["extension"] = {"module": "inference_extensions_cuda", "path": str(location), "sha256": _file_sha256(location)}
    except Exception as exc:  # recorded as a blocker by the runner
        report["extension"] = {"error": repr(exc)}
    if with_cuda:
        report["device_name"] = torch.cuda.get_device_name(0)
        report["device_count"] = torch.cuda.device_count()
    return report


def _require_claimed_device() -> None:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise RuntimeError(f"refusing to run without one claimed GPU UUID; CUDA_VISIBLE_DEVICES={visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")


def _load_state(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    """Strict-load input: verified bytes, DCVC's unwrapping, documented prefix only."""
    import torch
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    data = Path(path).read_bytes()
    actual = _sha256(data)
    if actual != expected_sha256:
        raise ValueError(f"checkpoint bytes {actual} differ from verified {expected_sha256}: {path}")
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if "net" in checkpoint:
        checkpoint = checkpoint["net"]
    consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
    dtypes = sorted({str(value.dtype) for value in checkpoint.values()})
    return checkpoint, {"path": str(path), "sha256": actual, "bytes": len(data), "keys": len(checkpoint), "dtypes": dtypes}


def _finalize(net, device):
    import torch

    return net.half().to(device).to(memory_format=torch.channels_last)


def _build(args, *, need_video: bool):
    import torch
    from src.models.image_model import DMCI
    from src.utils.common import ModelStructure, set_torch_env

    set_torch_env()
    device = "cuda:0"
    torch.cuda.set_stream(torch.cuda.Stream(device, 0))
    loaded = {}
    i_state, loaded["image"] = _load_state(args.image_ckpt, args.image_sha256)
    i_net = DMCI().eval()
    i_net.load_state_dict(i_state, strict=True)
    i_net.update(0)
    i_net = _finalize(i_net, device)
    p_net, frame_delay = None, 1
    if need_video:
        structure = ModelStructure(args.structure)
        if structure == ModelStructure.LD:
            from src.models.video_model_ld import DMC, g_frame_delay
            p_net = DMC()
        else:
            from src.models.video_model_ht import DMC, g_frame_delay
            p_net = DMC(model_structure=structure)
        p_state, loaded["video"] = _load_state(args.video_ckpt, args.video_sha256)
        p_net = p_net.eval()
        p_net.load_state_dict(p_state, strict=True)
        p_net.update(0)
        p_net = _finalize(p_net, device)
        frame_delay = g_frame_delay
    loaded["strict_load"] = True
    loaded["precision"] = "float16 (test_video.finalize_model)"
    return i_net, p_net, frame_delay, device, loaded


def _ensure_image_proxy(i_net) -> None:
    """Same proxy construction as DMCI.compress at this revision."""
    if i_net.proxy is None:
        from inference_extensions_cuda import DMCIProxy

        state = i_net.add_cdf_to_state_dict(i_net.state_dict())
        i_net.proxy = DMCIProxy()
        i_net.proxy.set_param(state, i_net.gaussian_encoder.skip_thres)


def _to_rgb_uint8(x_hat, height: int, width: int):
    """test_video.py PNG path: ycbcr2rgb(x_hat + 0.5), clamp*255, round."""
    import torch
    from src.utils.transforms import ycbcr2rgb

    if x_hat.dim() != 4 or x_hat.shape[0] != 1 or x_hat.shape[1] != 3:
        raise ValueError(f"unexpected reconstruction shape {tuple(x_hat.shape)}")
    x_hat = x_hat[:, :, :height, :width]
    rgb = torch.clamp(ycbcr2rgb(x_hat + 0.5) * 255, 0, 255).round().byte()
    return rgb.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def _save_png(pixels, path: Path) -> str:
    from PIL import Image

    Image.fromarray(pixels, mode="RGB").save(path)
    return _file_sha256(path)


def _peak_memory(device) -> dict:
    import torch

    report = {
        "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
    }
    try:
        visible = os.environ["CUDA_VISIBLE_DEVICES"]
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", visible],
            capture_output=True, text=True, timeout=10, check=True,
        )
        report["device_memory_used_mib_at_end"] = float(result.stdout.strip().splitlines()[0])
    except Exception as exc:
        report["device_memory_used_mib_at_end"] = None
        report["nvidia_smi_error"] = repr(exc)
    return report


def _read_source(reader, device, maximum_read: int, frame_delay: int, is_intra: bool):
    """test_video.get_src_frame for src_type png."""
    import torch
    from src.utils.transforms import rgb2ycbcr

    frames = []
    for _ in range(maximum_read):
        rgb = reader.read_one_frame()
        if rgb is None:
            raise ValueError("source ended before the declared frame count")
        x = torch.from_numpy(rgb).unsqueeze(0).to(device=device, non_blocking=True).float() / 255.0
        frames.append(rgb2ycbcr(x))
    padding = 0
    while not is_intra and len(frames) < frame_delay:
        frames.append(frames[-1])
        padding += 1
    x = torch.cat(frames, dim=1).half() - 0.5
    return x.to(memory_format=torch.channels_last), padding


def encode(args) -> dict:
    import torch
    from src.models.image_model import DMCI
    from src.utils.stream_helper import SPSHelper, write_ip, write_sps
    from src.utils.video_reader import PNGReader

    plan = json.loads(Path(args.plan).read_text())
    qp = int(plan["qp"])
    force_intra = bool(plan.get("force_intra"))
    reset_interval = int(plan["reset_interval"])
    _require_claimed_device()
    started = time.time()
    i_net, p_net, frame_delay, device, loaded = _build(args, need_video=not force_intra)
    load_seconds = time.time() - started
    results = []
    with torch.inference_mode():
        for stream in plan["streams"]:
            frame_count = int(stream["frame_count"])
            height, width = int(stream["height"]), int(stream["width"])
            reader = PNGReader(stream["frames_dir"], width, height)
            padding_r, padding_b = DMCI.get_padding_size(height, width, 16)
            output = io.BytesIO()
            sps_helper = SPSHelper()
            nals, padded = [], 0
            stream_started = time.time()
            frame_index = 0
            while frame_index < frame_count:
                is_intra = frame_index == 0 or force_intra
                maximum_read = 1 if is_intra else min(frame_delay, frame_count - frame_index)
                x, padding = _read_source(reader, device, maximum_read, frame_delay, is_intra)
                padded += padding
                torch.cuda.synchronize(device=device)
                begin = time.time()
                if is_intra:
                    encoded = i_net.compress(x, qp, padding_b, padding_r)
                    if not force_intra:
                        p_net.clear_dpb()
                        p_net.add_ref_feature_from_frame(encoded["x_hat"])
                    if frame_index == 0 and stream.get("i_recon_png"):
                        stream["i_recon_sha256"] = _save_png(_to_rgb_uint8(encoded["x_hat"], height, width), Path(stream["i_recon_png"]))
                    reset = 0
                else:
                    reset = reset_flag(frame_index, frame_delay, reset_interval)
                    encoded = p_net.compress(x, qp, reset, padding_b, padding_r)
                sps = {"sps_id": -1, "height": height, "width": width}
                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                sps_bytes = write_sps(output, sps) if sps_new else 0
                nal_bytes = write_ip(output, is_intra, sps_id, qp, encoded["ec_parallel"], reset, encoded["bit_stream"])
                torch.cuda.synchronize(device=device)
                nals.append({
                    "type": "I" if is_intra else "P", "first_display_index": frame_index, "display_frames": maximum_read,
                    "encoder_padding_frames": padding, "reset_feature_memory": reset,
                    "nal_bytes": nal_bytes, "sps_bytes": sps_bytes, "payload_bytes": len(encoded["bit_stream"]),
                    "seconds": round(time.time() - begin, 4),
                })
                frame_index += maximum_read
            reader.close()
            native = output.getvalue()
            container = pack_container(native, structure=args.structure, frame_count=frame_count, force_intra=force_intra)
            target = Path(stream["container"])
            with target.open("xb") as handle:
                handle.write(container)
            results.append({
                "name": stream["name"], "container": str(target), "container_bytes": len(container),
                "container_sha256": _sha256(container), "native_stream_bytes": len(native),
                "container_header_bytes": HEADER.size, "frame_count": frame_count, "qp_i": qp, "qp_p": qp,
                "reset_interval": reset_interval, "intra_policy": "force_intra" if force_intra else "one I frame then prediction",
                "frame_delay": frame_delay, "encoder_padding_frames": padded, "nals": nals,
                "i_recon_sha256": stream.get("i_recon_sha256"), "encode_seconds": round(time.time() - stream_started, 3),
            })
    return {
        "operation": "encode", "structure": args.structure, "checkpoints": loaded, "load_seconds": round(load_seconds, 3),
        "streams": results, "peak_memory": _peak_memory(device), "environment": environment_report(True),
    }


def decode(args) -> dict:
    """Decode containers using only their bytes and the declared checkpoints."""
    import torch
    from src.utils.stream_helper import NalType, SPSHelper, read_header, read_ip_remaining, read_sps_remaining

    plan = json.loads(Path(args.plan).read_text())
    _require_claimed_device()
    containers = {}
    for stream in plan["streams"]:
        data = Path(stream["container"]).read_bytes()
        header, native = unpack_container(data)
        if header["structure"] != args.structure:
            raise ValueError("container structure differs from the declared decoder")
        containers[stream["name"]] = (stream, header, native, _sha256(data))
    force_intra = {h["force_intra"] for _s, h, _n, _d in containers.values()}
    if len(force_intra) != 1:
        raise ValueError("one decode process handles one intra policy")
    force_intra = force_intra.pop()
    started = time.time()
    i_net, p_net, frame_delay, device, loaded = _build(args, need_video=not force_intra)
    _ensure_image_proxy(i_net)
    load_seconds = time.time() - started
    order = [s["name"] for s in plan["streams"]] + list(plan.get("repeat", []))
    results = []
    with torch.inference_mode():
        for position, name in enumerate(order):
            stream, header, native, container_sha = containers[name]
            out_dir = Path(stream["out_dir"] if position < len(plan["streams"]) else stream["out_dir"] + "_repeat")
            out_dir.mkdir(parents=True, exist_ok=False)
            buffer = io.BytesIO(native)
            sps_helper = SPSHelper()
            expected = display_schedule(header["frame_count"], frame_delay, force_intra=force_intra)
            decoded_hashes, nal_types = [], []
            decode_started = time.time()
            for nal_frames in expected:
                header_nal = read_header(buffer)
                while header_nal["nal_type"] == NalType.NAL_SPS:
                    sps_helper.add_sps_by_id(read_sps_remaining(buffer, header_nal["sps_id"]))
                    header_nal = read_header(buffer)
                sps = sps_helper.get_sps_by_id(header_nal["sps_id"])
                qp, ec_part, reset, bit_stream = read_ip_remaining(buffer)
                if header_nal["nal_type"] == NalType.NAL_I:
                    reconstruction = i_net.decompress(bit_stream, sps, qp, ec_part)["x_hat"]
                    if not force_intra:
                        p_net.clear_dpb()
                        p_net.add_ref_feature_from_frame(reconstruction, apply_feature_adaptor=False)
                    nal_types.append("I")
                else:
                    reconstruction = p_net.decompress(bit_stream, sps, qp, ec_part, reset)["x_hat"]
                    nal_types.append("P")
                for offset, display_index in enumerate(nal_frames):
                    frame = reconstruction[offset] if isinstance(reconstruction, list) else reconstruction
                    pixels = _to_rgb_uint8(frame, sps["height"], sps["width"])
                    decoded_hashes.append(_save_png(pixels, out_dir / f"im{display_index + 1:05d}.png"))
            if buffer.read(1):
                raise ValueError(f"{name}: trailing bytes after the declared frame count")
            torch.cuda.synchronize(device=device)
            results.append({
                "name": name, "repeat": position >= len(plan["streams"]), "container_sha256": container_sha,
                "out_dir": str(out_dir), "frame_count": header["frame_count"], "nal_types": nal_types,
                "decoded_png_sha256": decoded_hashes, "decode_seconds": round(time.time() - decode_started, 3),
            })
    return {
        "operation": "decode", "structure": args.structure, "checkpoints": loaded, "load_seconds": round(load_seconds, 3),
        "inputs": "container bytes and checkpoints only; no source frame path is passed",
        "streams": results, "peak_memory": _peak_memory(device), "environment": environment_report(True),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("check")
    check.add_argument("--expected", type=Path, required=True)
    check.add_argument("--report", type=Path, required=True)
    for name in ("encode", "decode"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--report", type=Path, required=True)
        command.add_argument("--expected", type=Path, required=True)
        command.add_argument("--structure", choices=sorted(STRUCTURE_CODES), required=True)
        command.add_argument("--image-ckpt", type=Path, required=True)
        command.add_argument("--image-sha256", required=True)
        command.add_argument("--video-ckpt", type=Path)
        command.add_argument("--video-sha256")
    args = parser.parse_args(argv)
    expected = json.loads(args.expected.read_text())
    reference = reference_check(expected)
    if args.action == "check":
        payload = {"reference": reference, "environment": environment_report(False)}
    else:
        if not reference["matches_reference"]:
            raise SystemExit(f"installed DCVC differs from the mirrored revision: {reference['mismatched']}")
        payload = encode(args) if args.action == "encode" else decode(args)
        payload["reference"] = reference
    with args.report.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
