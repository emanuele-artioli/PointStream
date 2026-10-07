"""Encode or decode one DCVC-UF stream (CVPR 2026, microsoft/DCVC ``cbdae87``).

Run as a script, never imported by PointStream: DCVC's top-level package is
also called ``src``. `dcvc_command` builds the command line; the process runs
with ``cwd`` and ``PYTHONPATH`` set to the vendored DCVC tree
(``<prefix>/opt/DCVC``, env/build.sh). Only DCVC's own modules are imported, and
the stream path mirrors ``test_video.py::run_one_point_with_stream`` with these
differences (ported from the pre-reset adapter, ``archive/pre-reset-2026-10-05``):

* The device stays the fleet-claimed GPU. ``test_video.py`` workers rewrite
  CUDA_VISIBLE_DEVICES to a bare index, which can leave the claim.
* Checkpoints are loaded strictly from bytes whose SHA-256 was just verified.
* Decoding is its own process, from the container bytes alone. The image proxy
  is created explicitly, because DCVC creates it only inside ``compress``.
* The CUDA inference extension is chosen per GPU: ``sm89`` on Ada, ``sm80``
  (SASS for sm_70 to sm_86) elsewhere. DCVC then picks its kernel path at run
  time: plain PyTorch below sm_75, CUTLASS Sm75 kernels on Turing, Sm80 above.
* ``src_type`` selects the input path of ``test_video.py``: ``png`` (RGB frames
  ``im00001.png``..., output RGB ``.npy`` per frame) or ``yuv420`` (one raw 8-bit
  planar file, chroma upsampled by nearest neighbour as upstream; output the
  same raw format in ``out_file``). Upstream truncates the reconstructed chroma
  when it writes YUV; here both planes are rounded.

Container: b"PSDC", version, flags, structure, frame count (big-endian uint16),
then the native DCVC stream. Rate counts the whole file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

MAGIC = b"PSDC"
VERSION = 1
STRUCTURE_CODES = {"ld": 0, "hts": 1, "htl": 2}
HEADER = struct.Struct(">4sBBBH")
# test_video.py's default; one I frame starts the stream.
RESET_INTERVAL = 0


def dcvc_command(prefix: Path, action: str, plan: Path, report: Path) -> tuple[list[str], dict[str, str], Path]:
    """Command, environment overrides and cwd that run this worker on ``prefix``."""
    tree = prefix / "opt" / "DCVC"
    env = {"PYTHONPATH": str(tree), "PYTHONNOUSERSITE": "1"}
    command = [str(prefix / "bin" / "python"), str(Path(__file__).resolve()), action,
               "--plan", str(plan), "--report", str(report)]
    return command, env, tree


def pack_container(native: bytes, *, structure: str, frame_count: int) -> bytes:
    if structure not in STRUCTURE_CODES or not 0 < frame_count < 65536:
        raise ValueError("invalid container structure or frame count")
    return HEADER.pack(MAGIC, VERSION, 0, STRUCTURE_CODES[structure], frame_count) + native


def unpack_container(data: bytes) -> tuple[dict[str, Any], bytes]:
    if len(data) < HEADER.size:
        raise ValueError("truncated DCVC container")
    magic, version, flags, code, frame_count = HEADER.unpack(data[:HEADER.size])
    if magic != MAGIC or version != VERSION or flags != 0:
        raise ValueError("not a version-1 PointStream DCVC container")
    structures = {value: key for key, value in STRUCTURE_CODES.items()}
    if code not in structures or frame_count <= 0:
        raise ValueError("invalid DCVC container header")
    return {"structure": structures[code], "frame_count": frame_count}, data[HEADER.size:]


def display_schedule(frame_count: int, frame_delay: int) -> list[list[int]]:
    """Display indices carried by each NAL: one I frame, then groups of ``frame_delay``."""
    if frame_count <= 0 or frame_delay not in (1, 8):
        raise ValueError("invalid frame count or frame delay")
    schedule, index = [[0]], 1
    while index < frame_count:
        width = min(frame_delay, frame_count - index)
        schedule.append(list(range(index, index + width)))
        index += width
    return schedule


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extension_variant() -> str:
    import torch

    return "sm89" if torch.cuda.get_device_capability(0) == (8, 9) else "sm80"


def _import_extension() -> dict[str, Any]:
    variant = extension_variant()
    directory = Path(sys.prefix) / "opt" / "dcvc-extensions" / variant
    sys.path.insert(0, str(directory))
    import inference_extensions_cuda as extension

    location = Path(extension.__file__)
    if location.parent != directory:
        raise RuntimeError(f"inference_extensions_cuda loaded from {location}, not {directory}")
    return {"variant": variant, "path": str(location), "sha256": _sha256(location.read_bytes())}


def _require_claimed_device() -> None:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-") or "," in visible:
        raise RuntimeError(f"refusing to run without one claimed GPU UUID; CUDA_VISIBLE_DEVICES={visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")


def _load_state(path: str, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return checkpoint, {"path": path, "sha256": actual, "bytes": len(data), "keys": len(checkpoint)}


def _build(plan: dict[str, Any]) -> tuple[Any, Any, int, dict[str, Any]]:
    import torch
    from src.models.image_model import DMCI
    from src.utils.common import ModelStructure, set_torch_env

    set_torch_env()
    torch.cuda.set_stream(torch.cuda.Stream("cuda:0", 0))

    def finalize(net: Any) -> Any:  # test_video.finalize_model
        return net.half().to("cuda:0").to(memory_format=torch.channels_last)

    loaded: dict[str, Any] = {"strict_load": True, "precision": "float16"}
    i_state, loaded["image"] = _load_state(plan["image_ckpt"], plan["image_sha256"])
    i_net = DMCI().eval()
    i_net.load_state_dict(i_state, strict=True)
    i_net.update(0)
    i_net = finalize(i_net)
    structure = ModelStructure(plan["structure"])
    if structure == ModelStructure.LD:
        from src.models.video_model_ld import DMC, g_frame_delay

        p_net = DMC()
    else:
        from src.models.video_model_ht import DMC, g_frame_delay

        p_net = DMC(model_structure=structure)
    p_state, loaded["video"] = _load_state(plan["video_ckpt"], plan["video_sha256"])
    p_net = p_net.eval()
    p_net.load_state_dict(p_state, strict=True)
    p_net.update(0)
    return i_net, finalize(p_net), int(g_frame_delay), loaded


def _ensure_image_proxy(i_net: Any) -> None:
    if i_net.proxy is None:
        from inference_extensions_cuda import DMCIProxy

        state = i_net.add_cdf_to_state_dict(i_net.state_dict())
        i_net.proxy = DMCIProxy()
        i_net.proxy.set_param(state, i_net.gaussian_encoder.skip_thres)


def _to_yuv420_bytes(x_hat: Any, height: int, width: int) -> bytes:
    """test_video.py YUV path: 4:4:4 to 4:2:0 by 2x2 average, clamp*255, round."""
    import torch
    from src.utils.transforms import yuv_444_to_420

    y, uv = yuv_444_to_420(x_hat[:, :, :height, :width] + 0.5)
    planes = [torch.clamp(p * 255, 0, 255).round().byte().squeeze(0).cpu().numpy() for p in (y, uv)]
    return b"".join(plane.tobytes() for plane in planes)


def _to_rgb_uint8(x_hat: Any, height: int, width: int) -> Any:
    """test_video.py PNG path: ycbcr2rgb(x_hat + 0.5), clamp*255, round."""
    import torch
    from src.utils.transforms import ycbcr2rgb

    x_hat = x_hat[:, :, :height, :width]
    rgb = torch.clamp(ycbcr2rgb(x_hat + 0.5) * 255, 0, 255).round().byte()
    return rgb.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def _read_source(reader: Any, maximum_read: int, frame_delay: int, is_intra: bool, src_type: str = "png") -> tuple[Any, int]:
    """test_video.get_src_frame for src_type png or yuv420."""
    import torch
    from src.utils.transforms import rgb2ycbcr, ycbcr420_to_444_np

    frames = []
    for _ in range(maximum_read):
        if src_type == "yuv420":
            y, uv = reader.read_one_frame()
            if y is None:
                raise ValueError("source ended before the declared frame count")
            yuv = torch.from_numpy(ycbcr420_to_444_np(y, uv)).unsqueeze(0).to("cuda:0")
            frames.append(yuv.float() / 255.0)
            continue
        rgb = reader.read_one_frame()
        if rgb is None:
            raise ValueError("source ended before the declared frame count")
        x = torch.from_numpy(rgb).unsqueeze(0).to("cuda:0").float() / 255.0
        frames.append(rgb2ycbcr(x))
    padding = 0
    while not is_intra and len(frames) < frame_delay:
        frames.append(frames[-1])
        padding += 1
    x = torch.cat(frames, dim=1).half() - 0.5
    return x.to(memory_format=torch.channels_last), padding


def _profile(fn: Any) -> dict[str, Any]:
    """CUDA kernels launched by ``fn`` (torch profiler), summed by name."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    totals: dict[str, float] = {}
    launches = 0
    for event in prof.events():
        if str(getattr(event, "device_type", "")).endswith("CUDA") and event.name:
            totals[event.name] = totals.get(event.name, 0.0) + float(getattr(event, "device_time", 0) or 0)
            launches += 1
    top = sorted(totals.items(), key=lambda item: -item[1])[:25]
    return {"kernel_launches": launches, "distinct_kernels": len(totals),
            "top_kernels_us": [[name[:160], round(us, 1)] for name, us in top],
            "cutlass_kernels": sum("cutlass" in name.lower() for name in totals)}


def _environment(extension: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        "python": sys.executable, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "extension": extension,
        "cwd": str(Path.cwd()),
    }


def encode(plan: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from src.models.image_model import DMCI
    from src.utils.stream_helper import SPSHelper, write_ip, write_sps
    from src.utils.video_reader import PNGReader, YUV420Reader

    _require_claimed_device()
    extension = _import_extension()
    started = time.time()
    i_net, p_net, frame_delay, loaded = _build(plan)
    load_seconds = time.time() - started
    qp, count = int(plan["qp"]), int(plan["frame_count"])
    height, width = int(plan["height"]), int(plan["width"])
    src_type = plan.get("src_type", "png")
    if src_type == "yuv420":
        reader = YUV420Reader(plan["frames_file"], width, height)
    elif src_type == "png":
        reader = PNGReader(plan["frames_dir"], width, height)
    else:
        raise ValueError(f"unknown src_type {src_type!r}")
    padding_r, padding_b = DMCI.get_padding_size(height, width, 16)
    output, sps_helper, nals = io.BytesIO(), SPSHelper(), []
    state: dict[str, Any] = {"i_recon_sha256": None}
    torch.cuda.reset_peak_memory_stats()
    encode_started = time.time()

    def run() -> None:
        index = 0
        while index < count:
            is_intra = index == 0
            maximum_read = 1 if is_intra else min(frame_delay, count - index)
            x, padding = _read_source(reader, maximum_read, frame_delay, is_intra, src_type)
            if is_intra:
                encoded = i_net.compress(x, qp, padding_b, padding_r)
                p_net.clear_dpb()
                p_net.add_ref_feature_from_frame(encoded["x_hat"])
                if src_type == "yuv420":
                    state["i_recon_sha256"] = _sha256(_to_yuv420_bytes(encoded["x_hat"], height, width))
                else:
                    pixels = np.ascontiguousarray(_to_rgb_uint8(encoded["x_hat"], height, width))
                    state["i_recon_sha256"] = _sha256(pixels.tobytes())
                reset = 0
            else:
                reset = int(RESET_INTERVAL > 0 and (index + frame_delay) % RESET_INTERVAL == 1)
                encoded = p_net.compress(x, qp, reset, padding_b, padding_r)
            sps = {"sps_id": -1, "height": height, "width": width}
            sps_id, sps_new = sps_helper.get_sps_id(sps)
            sps["sps_id"] = sps_id
            sps_bytes = write_sps(output, sps) if sps_new else 0
            nal_bytes = write_ip(output, is_intra, sps_id, qp, encoded["ec_parallel"], reset, encoded["bit_stream"])
            nals.append({"type": "I" if is_intra else "P", "display_frames": maximum_read,
                         "encoder_padding_frames": padding, "nal_bytes": nal_bytes, "sps_bytes": sps_bytes})
            index += maximum_read
        torch.cuda.synchronize()

    kernels = None
    with torch.inference_mode():
        if plan.get("profile"):
            kernels = _profile(run)
        else:
            run()
    container = pack_container(output.getvalue(), structure=plan["structure"], frame_count=count)
    with Path(plan["container"]).open("xb") as handle:
        handle.write(container)
    recon_key = "i_recon_rgb_sha256" if src_type == "png" else "i_recon_yuv420_sha256"
    return {
        "operation": "encode", "structure": plan["structure"], "src_type": src_type, "qp": qp, "frame_count": count,
        "frame_delay": frame_delay, "checkpoints": loaded, "load_seconds": round(load_seconds, 3),
        "encode_seconds": round(time.time() - encode_started, 3), "nals": nals,
        "container_bytes": len(container), "container_sha256": _sha256(container),
        "i_recon_sha256": state["i_recon_sha256"], recon_key: state["i_recon_sha256"], "kernels": kernels,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "environment": _environment(extension),
    }


def decode(plan: dict[str, Any]) -> dict[str, Any]:
    """Decode the container twice, from its bytes and the checkpoints only."""
    import numpy as np
    import torch
    from src.utils.stream_helper import NalType, SPSHelper, read_header, read_ip_remaining, read_sps_remaining

    _require_claimed_device()
    extension = _import_extension()
    data = Path(plan["container"]).read_bytes()
    header, native = unpack_container(data)
    if header["structure"] != plan["structure"]:
        raise ValueError("container structure differs from the declared decoder")
    started = time.time()
    i_net, p_net, frame_delay, loaded = _build(plan)
    _ensure_image_proxy(i_net)
    load_seconds = time.time() - started
    src_type = plan.get("src_type", "png")
    if src_type not in ("png", "yuv420"):
        raise ValueError(f"unknown src_type {src_type!r}")
    out_dir = Path(plan["out_dir"]) if src_type == "png" else None
    out_file = Path(plan["out_file"]).open("xb") if src_type == "yuv420" else None
    passes = []
    with torch.inference_mode():
        for repeat in (0, 1):
            buffer, sps_helper, hashes = io.BytesIO(native), SPSHelper(), []
            decode_started = time.time()
            for nal_frames in display_schedule(header["frame_count"], frame_delay):
                nal = read_header(buffer)
                while nal["nal_type"] == NalType.NAL_SPS:
                    sps_helper.add_sps_by_id(read_sps_remaining(buffer, nal["sps_id"]))
                    nal = read_header(buffer)
                sps = sps_helper.get_sps_by_id(nal["sps_id"])
                qp, ec_part, reset, bit_stream = read_ip_remaining(buffer)
                if nal["nal_type"] == NalType.NAL_I:
                    recon = i_net.decompress(bit_stream, sps, qp, ec_part)["x_hat"]
                    p_net.clear_dpb()
                    p_net.add_ref_feature_from_frame(recon, apply_feature_adaptor=False)
                else:
                    recon = p_net.decompress(bit_stream, sps, qp, ec_part, reset)["x_hat"]
                for offset, display_index in enumerate(nal_frames):
                    frame = recon[offset] if isinstance(recon, list) else recon
                    if out_file is not None:
                        data = _to_yuv420_bytes(frame, sps["height"], sps["width"])
                        hashes.append(_sha256(data))
                        if repeat == 0:
                            out_file.write(data)
                        continue
                    pixels = np.ascontiguousarray(_to_rgb_uint8(frame, sps["height"], sps["width"]))
                    hashes.append(_sha256(pixels.tobytes()))
                    if repeat == 0 and out_dir is not None:
                        np.save(out_dir / f"{display_index:05d}.npy", pixels)
            if buffer.read(1):
                raise ValueError("trailing bytes after the declared frame count")
            torch.cuda.synchronize()
            key = "rgb_sha256" if src_type == "png" else "yuv420_sha256"
            passes.append({key: hashes, "frame_sha256": hashes, "decode_seconds": round(time.time() - decode_started, 3)})
    if out_file is not None:
        out_file.close()
    return {
        "operation": "decode", "structure": plan["structure"], "src_type": src_type, "frame_count": header["frame_count"],
        "inputs": "container bytes and checkpoints only", "container_sha256": _sha256(data),
        "checkpoints": loaded, "load_seconds": round(load_seconds, 3), "passes": passes,
        "deterministic": passes[0]["frame_sha256"] == passes[1]["frame_sha256"],
        "environment": _environment(extension),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("encode", "decode"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    payload = encode(plan) if args.action == "encode" else decode(plan)
    with args.report.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
