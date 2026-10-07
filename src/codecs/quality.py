"""Frame quality of a decoded 8-bit 4:2:0 video against its source, whole-frame and inside masks.

Both videos are raw planar files (Y, U, V per frame). Every metric sees the
same RGB view of both: chroma upsampled bilinearly, BT.709, full or limited
range as the source declares, rounded to 8 bits (`yuv420_to_rgb`). Per frame:

* Squared error per RGB pixel, summed exactly in integers, over the whole frame
  and over each mask region, so PSNR is identical on every device.
  ``psnr = 10 log10(255^2 / mse)``, capped at ``PSNR_CAP``.
* Y, U and V PSNR on the planes themselves, and their 6:1:1 mean (the DCVC and
  JVET convention).
* MS-SSIM on RGB (torchmetrics, data range 255).
* LPIPS (AlexNet, torchmetrics' v0.1 linear heads) as a spatial map at full
  resolution: its mean is the frame's LPIPS, its mean inside a region the
  region's LPIPS.
* VMAF (`vmaf`, libvmaf's built-in ``vmaf_v0.6.1``) on the planes, whole frame
  only.

A region score is ``None`` on a frame where the region is empty.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PSNR_CAP = 100.0
#: BT.709 luma coefficients.
KR, KB = 0.2126, 0.0722


def psnr(mse: float) -> float:
    return PSNR_CAP if mse <= 0 else min(PSNR_CAP, 10.0 * math.log10(255.0**2 / mse))


def frame_bytes(width: int, height: int) -> int:
    return width * height * 3 // 2


def read_frames(path: Path, width: int, height: int, first: int, count: int) -> np.ndarray:
    """``count`` frames from ``first`` as an array of shape (count, height * 3 / 2, width)."""
    size = frame_bytes(width, height)
    data = np.fromfile(path, dtype=np.uint8, count=size * count, offset=size * first)
    if data.size != size * count:
        raise ValueError(f"{path}: {data.size // size} frames from {first}, wanted {count}")
    return data.reshape(count, height * 3 // 2, width)


def split_planes(frames: Any, width: int, height: int) -> tuple[Any, Any, Any]:
    """Y, U, V of a (n, h*3/2, w) tensor of 4:2:0 frames."""
    luma = frames[:, :height]
    chroma = frames[:, height:].reshape(frames.shape[0], 2, height // 2, width // 2)
    return luma, chroma[:, 0], chroma[:, 1]


def yuv420_to_rgb(frames: Any, width: int, height: int, *, full_range: bool) -> Any:
    """(n, h*3/2, w) uint8 4:2:0 tensor -> (n, 3, h, w) float tensor of 8-bit RGB values."""
    import torch
    import torch.nn.functional as F

    y, u, v = split_planes(frames, width, height)
    y = y.float()
    uv = torch.stack([u, v], dim=1).float()
    uv = F.interpolate(uv, size=(height, width), mode="bilinear", align_corners=False)
    if full_range:
        y_n, cb, cr = y / 255.0, (uv[:, 0] - 128.0) / 255.0, (uv[:, 1] - 128.0) / 255.0
    else:
        y_n, cb, cr = (y - 16.0) / 219.0, (uv[:, 0] - 128.0) / 224.0, (uv[:, 1] - 128.0) / 224.0
    kg = 1.0 - KR - KB
    r = y_n + 2.0 * (1.0 - KR) * cr
    b = y_n + 2.0 * (1.0 - KB) * cb
    g = (y_n - KR * r - KB * b) / kg
    rgb = torch.stack([r, g, b], dim=1) * 255.0
    return torch.clamp(torch.round(rgb), 0.0, 255.0)


def load_lpips(backbone: Path, device: str) -> Any:
    """torchmetrics' LPIPS (AlexNet) with the backbone read from ``backbone``, never downloaded."""
    import torch
    from torchmetrics.functional.image.lpips import _LPIPS

    net = _LPIPS(pretrained=True, net="alex", spatial=True, pnet_rand=True, pnet_tune=False)
    state = torch.load(backbone, map_location="cpu", weights_only=True)
    features = {k[len("features."):]: v for k, v in state.items() if k.startswith("features.")}
    mapped = {}
    for name, _ in net.net.named_parameters():
        _slice, index, kind = name.split(".")
        mapped[name] = features[f"{index}.{kind}"]
    net.net.load_state_dict(mapped, strict=True)
    return net.eval().to(device)


def lpips_record(backbone: Path) -> dict[str, Any]:
    import hashlib

    import torchmetrics
    from torchmetrics.functional.image import lpips as module

    heads = Path(module.__file__).parent / "lpips_models" / "alex.pth"
    return {
        "metric": "LPIPS", "net": "alex", "torchmetrics": torchmetrics.__version__,
        "heads": str(heads), "heads_sha256": hashlib.sha256(heads.read_bytes()).hexdigest(),
        "backbone": str(backbone), "backbone_sha256": hashlib.sha256(Path(backbone).read_bytes()).hexdigest(),
        "pooling": "spatial map at full resolution (bilinear), mean over the frame or region",
    }


def _region_sum(values: Any, mask: Any) -> tuple[Any, Any]:
    """Per-frame sum and count of ``values`` (n, h, w) inside ``mask`` (n, h, w)."""
    return (values * mask).flatten(1).sum(1), mask.flatten(1).sum(1)


def score_batch(
    reference: np.ndarray, decoded: np.ndarray, masks: dict[str, np.ndarray], *, width: int, height: int,
    full_range: bool, device: str, lpips_net: Any,
) -> list[dict[str, Any]]:
    """Per-frame metrics of one batch. ``masks`` maps a region name to (n, h, w) booleans."""
    import torch
    from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as ms_ssim

    ref = torch.from_numpy(reference).to(device)
    dec = torch.from_numpy(decoded).to(device)
    out: list[dict[str, Any]] = [{} for _ in range(ref.shape[0])]
    planes = {}
    for name, a, b in zip("YUV", split_planes(ref, width, height), split_planes(dec, width, height)):
        diff = a.to(torch.int32) - b.to(torch.int32)
        planes[name] = (diff * diff).flatten(1).sum(1).to(torch.int64).cpu().numpy(), a[0].numel()
    ref_rgb = yuv420_to_rgb(ref, width, height, full_range=full_range)
    dec_rgb = yuv420_to_rgb(dec, width, height, full_range=full_range)
    diff = ref_rgb.to(torch.int32) - dec_rgb.to(torch.int32)
    sq = (diff * diff).sum(1).to(torch.int64)
    with torch.inference_mode():
        msssim = ms_ssim(dec_rgb, ref_rgb, data_range=255.0, reduction="none").reshape(-1).cpu().numpy()
        lp = lpips_net(dec_rgb / 127.5 - 1.0, ref_rgb / 127.5 - 1.0)[:, 0]
    region_masks = {"frame": torch.ones_like(sq, dtype=torch.bool)}
    region_masks.update({k: torch.from_numpy(v).to(device) for k, v in masks.items()})
    for region, mask in region_masks.items():
        sse, count = _region_sum(sq, mask.to(torch.int64))
        lp_sum, _ = _region_sum(lp, mask.to(lp.dtype))
        sse_np, count_np, lp_np = sse.cpu().numpy(), count.cpu().numpy(), lp_sum.double().cpu().numpy()
        for i in range(len(out)):
            n = int(count_np[i])
            out[i][region] = {
                "sse": int(sse_np[i]), "pixels": n,
                "psnr": psnr(sse_np[i] / (3 * n)) if n else None,
                "lpips": float(lp_np[i] / n) if n else None,
            }
    for i in range(len(out)):
        values = {k: psnr(planes[k][0][i] / planes[k][1]) for k in "YUV"}
        out[i]["planes"] = {**{f"psnr_{k.lower()}": v for k, v in values.items()},
                            "psnr_yuv": (6 * values["Y"] + values["U"] + values["V"]) / 8}
        out[i]["ms_ssim"] = float(msssim[i])
    return out


def score(
    reference: Path, decoded: Path, masks: Iterable[dict[str, np.ndarray]], *, width: int, height: int,
    frames: int, full_range: bool, device: str, lpips_net: Any, batch: int = 4,
) -> list[dict[str, Any]]:
    """Per-frame metrics of a whole window; ``masks`` yields each frame's {region: (h, w) bool}."""
    out: list[dict[str, Any]] = []
    regions = iter(masks)
    for first in range(0, frames, batch):
        count = min(batch, frames - first)
        frame_masks = [next(regions) for _ in range(count)]
        stacked = {name: np.stack([m[name] for m in frame_masks]) for name in frame_masks[0]}
        out.extend(score_batch(
            read_frames(reference, width, height, first, count), read_frames(decoded, width, height, first, count),
            stacked, width=width, height=height, full_range=full_range, device=device, lpips_net=lpips_net,
        ))
    return out


def host_tool_env(tool_path: str) -> dict[str, str]:
    """Environment for a tool installed outside the packed environment (the hosts' ``/opt/local``).

    Its shared libraries (libvmaf among them) are found through the host's own
    library directories, which the hosts' login shells put on
    ``LD_LIBRARY_PATH``; fleet workloads do not inherit that.
    """
    prefix = Path(os.path.realpath(tool_path)).parent.parent
    paths = ["/usr/lib/x86_64-linux-gnu", str(prefix / "lib" / "x86_64-linux-gnu"), str(prefix / "lib")]
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    return env


def vmaf_command(ffmpeg: str, decoded: Path, reference: Path, log: Path, *, width: int, height: int,
                 fps: Fraction, threads: int) -> list[str]:
    raw = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-r", f"{fps.numerator}/{fps.denominator}"]
    graph = f"[0:v][1:v]libvmaf=model=version=vmaf_v0.6.1:n_threads={threads}:log_fmt=json:log_path={log}"
    return [ffmpeg, "-hide_banner", "-nostats", "-v", "error", *raw, "-i", str(decoded), *raw, "-i", str(reference),
            "-lavfi", graph, "-f", "null", "-"]


def vmaf(ffmpeg: str, decoded: Path, reference: Path, log: Path, *, width: int, height: int, fps: Fraction,
         threads: int, timeout: float = 1800) -> dict[str, Any]:
    """Per-frame VMAF of ``decoded`` against ``reference`` (libvmaf through ``ffmpeg``)."""
    command = vmaf_command(ffmpeg, decoded, reference, log, width=width, height=height, fps=fps, threads=threads)
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=host_tool_env(ffmpeg))
    if done.returncode:
        raise RuntimeError(f"VMAF failed ({done.returncode}):\n{done.stderr[-4000:]}")
    doc = json.loads(log.read_text())
    values = [float(frame["metrics"]["vmaf"]) for frame in doc["frames"]]
    return {"command": command, "per_frame": values, "version": doc.get("version"),
            "pooled_mean": float(doc["pooled_metrics"]["vmaf"]["mean"])}


def vmaf_record(ffmpeg: str) -> dict[str, Any]:
    """The ffmpeg binary and the libvmaf it links, by path, sha256 and version."""
    import hashlib

    from src.codecs.svtav1 import tool

    record = tool(ffmpeg)
    record["ld_library_path"] = host_tool_env(ffmpeg)["LD_LIBRARY_PATH"]
    linked = subprocess.run(["ldd", record["real_path"]], capture_output=True, text=True, timeout=30,
                            env=host_tool_env(ffmpeg)).stdout
    for line in linked.splitlines():
        if "libvmaf" in line and "=>" in line and "not found" not in line:
            path = os.path.realpath(line.split("=>")[1].split("(")[0].strip())
            record["libvmaf"] = {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    record["model"] = "vmaf_v0.6.1 (built in)"
    return record
