"""DCVC-UF HT-L coding a window with its reference replaced at every NAL: PLAN step K3's oracle (docs/experiments.md).

Run as a script, never imported by PointStream: like `experiments.visor.h3c_bits` it runs with ``cwd`` and
``PYTHONPATH`` set to the vendored DCVC tree, whose top-level package is also called ``src``. `experiments.visor.k3`
writes the plan.

HT-L's reference after an I frame or a memory reset is a single frame: ``add_ref_feature_from_frame`` pixel-unshuffles
it and restarts the memory through ``feature_adaptor_i`` (``dmc_htl_proxy.cpp``; a reset uses the previous NAL's last
decoded frame, ``recon_head(..., for_reset=True)``). So any frame can stand in for the previous decoded frame without
new weights. This script codes the window once per arm through DCVC's own PyTorch path (``forward_one_frame``, the
published weights in fp32, symbols rounded as `h3c_bits` does), which is a complete encoder and decoder: the CUDA
proxy cannot run an encoder and an in-loop decoder in one process, and the next reference needs the decoded frame.

Arms (`ARMS`): ``A`` unmodified (recurrent memory); every other arm calls ``clear_dpb`` and
``add_ref_feature_from_frame(ref)`` before each P NAL coding frames t0..t0+7, with ``ref`` built from the previous
decoded frame x̂(t0-1) (8-bit 4:2:0, as a decoder holds it) and the source:
``B`` x̂(t0-1); ``C`` x̂(t0-1) with the source's frame t0 inside R(t0-1) | R(t0); ``D`` the same at the group's middle
frame; ``E`` the source's frame t0. Per arm: the decoded window (8-bit 4:2:0, the worker's conversion) and per NAL
the model's estimated bits and their share on R (`h3c_bits`' cells).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from h3c_bits import FRAME_DELAY, PAD, load_state, region_bits, schedule, spread_z  # noqa: E402

ARMS = ("A", "B", "C", "D", "E")


def target_frame(arm: str, frames: list[int]) -> int:
    """The source frame whose hand (C, D) or whole picture (E) the reference shows."""
    return frames[len(frames) // 2] if arm == "D" else frames[0]


def reference_mask(region: np.ndarray, previous: int, target: int) -> np.ndarray:
    """Where the reference shows the source: the hand's pixels at the previous frame and at the target."""
    return np.asarray(region[previous]) | np.asarray(region[target])


def code_arm(arm: str, plan: dict[str, Any], nets: tuple[Any, Any], planes: np.ndarray, region: np.ndarray,
             out: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from src.models.image_model import DMCI
    from src.utils.transforms import yuv_444_to_420

    device = "cuda"
    i_net, p_net = nets
    width, height, count = int(plan["width"]), int(plan["height"]), int(plan["frame_count"])
    pad_r, pad_b = DMCI.get_padding_size(height, width, PAD)
    qp = torch.full((1,), int(plan["qp"]), dtype=torch.int32, device=device)

    def to444(p: np.ndarray) -> Any:
        """(3, H, W) YCbCr in [-0.5, 0.5] on the GPU, chroma upsampled by nearest neighbour (``ycbcr420_to_444_np``)."""
        luma = p[:height].astype(np.float32)
        chroma = p[height:].reshape(2, height // 2, width // 2).astype(np.float32).repeat(2, 1).repeat(2, 2)
        return torch.from_numpy(np.concatenate([luma[None], chroma]) / 255.0 - 0.5).to(device)

    def padded(x: Any) -> Any:
        return F.pad(x[None], (0, pad_r, 0, pad_b), mode="replicate")

    def to420(x_hat: Any) -> np.ndarray:
        """`src.codecs.dcvc_uf_worker._to_yuv420_bytes`: 4:4:4 to 4:2:0 by 2x2 average, clamp*255, round."""
        y, uv = yuv_444_to_420(x_hat[:, :, :height, :width] + 0.5)
        return np.concatenate([torch.clamp(c * 255, 0, 255).round().byte().squeeze(0).cpu().numpy().reshape(-1, width)
                               for c in (y, uv)])

    captured: dict[str, Any] = {}

    def cells() -> np.ndarray:
        y = captured.pop("y").float().sum(1)[0].cpu().numpy()
        z = captured.pop("z").float().sum(1)[0].cpu().numpy()
        return spread_z(y, z)

    for net in (i_net, p_net):
        get_y, get_z = type(net).get_y_bits, type(net).get_z_bits

        def bits_y(y: Any, sigma: Any, net: Any = net, get_y: Any = get_y) -> Any:
            captured["y"] = get_y(net, y, sigma)
            return captured["y"]

        def bits_z(z: Any, index: Any, net: Any = net, get_z: Any = get_z) -> Any:
            captured["z"] = get_z(net, z, index)
            return captured["z"]

        net.get_y_bits, net.get_z_bits = bits_y, bits_z
        net.add_noise = torch.round  # the quantized symbols, not the training noise

    nals, pasted = [], []
    decoded = np.zeros((count, height * 3 // 2, width), np.uint8)
    with torch.no_grad():
        for frames in schedule(count):
            if frames == [0]:
                result = i_net.forward_one_frame(padded(to444(planes[0])), qp)
                x_hats = [result["x_hat"]]
                p_net.clear_dpb()
                p_net.add_ref_feature_from_frame(result["x_hat"])
            else:
                previous, t0 = frames[0] - 1, frames[0]
                if arm != "A":
                    if arm == "E":
                        ref = to444(planes[t0])
                    else:
                        ref = to444(decoded[previous])
                        if arm in ("C", "D"):
                            target = target_frame(arm, frames)
                            mask = reference_mask(region, previous, target)
                            source = to444(planes[target])
                            m = torch.from_numpy(mask).to(device)
                            ref = torch.where(m[None], source, ref)
                            pasted.append(float(mask.mean()))
                    p_net.clear_dpb()
                    p_net.add_ref_feature_from_frame(padded(ref))
                xs = [padded(to444(planes[t])) for t in frames]
                xs += [xs[-1]] * (FRAME_DELAY - len(xs))
                result = p_net.forward_one_frame(torch.cat(xs, dim=1), qp)
                x_hats = result["x_hat"]
            for offset, t in enumerate(frames):
                decoded[t] = to420(x_hats[offset])
            c = cells()
            span = slice(frames[0], frames[-1] + 1)
            nals.append({"frames": frames, "estimated_bits": float(c.sum()),
                         "estimated_region_bits": region_bits(c, np.asarray(region[span]))})
    decoded.tofile(out / f"{arm}.yuv")
    return {"arm": arm, "nals": nals, "decoded": str(out / f"{arm}.yuv"),
            "pasted_share": float(np.mean(pasted)) if pasted else None}


def run(plan: dict[str, Any]) -> dict[str, Any]:
    import torch
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC, g_frame_delay
    from src.utils.common import ModelStructure

    if g_frame_delay != FRAME_DELAY:
        raise RuntimeError(f"HT-L frame delay {g_frame_delay}, expected {FRAME_DELAY}")
    width, height, count = int(plan["width"]), int(plan["height"]), int(plan["frame_count"])
    raw = np.fromfile(plan["frames_file"], np.uint8)
    if raw.size != count * width * height * 3 // 2:
        raise ValueError(f"{plan['frames_file']}: {raw.size} bytes for {count} frames of {width}×{height}")
    planes = raw.reshape(count, height * 3 // 2, width)
    region = np.load(plan["region_file"], mmap_mode="r")
    if region.shape != (count, height, width):
        raise ValueError(f"region {region.shape}, expected {(count, height, width)}")
    if unknown := set(plan["arms"]) - set(ARMS):
        raise ValueError(f"unknown arms {sorted(unknown)}")
    out = Path(plan["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    i_net = DMCI()
    i_net.load_state_dict(load_state(plan["image_ckpt"], plan["image_sha256"]), strict=True)
    p_net = DMC(model_structure=ModelStructure.HTL)
    p_net.load_state_dict(load_state(plan["video_ckpt"], plan["video_sha256"]), strict=True)
    # The training path: the eval path runs DCVC's CUDA proxy, which reports no per-cell bits and keeps no x̂ in Python.
    nets = (i_net.to("cuda").train(), p_net.to("cuda").train())
    began = time.time()
    arms = [code_arm(arm, plan, nets, planes, region, out) for arm in plan["arms"]]
    return {"qp": int(plan["qp"]), "width": width, "height": height, "frame_count": count, "arms": arms,
            "seconds": round(time.time() - began, 2), "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__,
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if not str(Path.cwd()).endswith("DCVC") or "src" not in os.listdir("."):
        raise SystemExit("run from the vendored DCVC tree (cwd and PYTHONPATH)")
    args.report.write_text(json.dumps(run(json.loads(args.plan.read_text())), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
