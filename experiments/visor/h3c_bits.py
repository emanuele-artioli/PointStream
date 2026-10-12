"""DCVC-UF's own estimate of where its bits go, per 16×16 latent cell: PLAN step H3c's rate on the hand region.

Run as a script, never imported by PointStream: like `src.codecs.dcvc_uf_worker` and
`experiments.background.g5b_train` it runs with ``cwd`` and ``PYTHONPATH`` set to the vendored DCVC tree, whose
top-level package is also called ``src``. `experiments.visor.h3c` writes the plan.

The stream's schedule is the worker's (`display_schedule`: one I frame, then groups of 8, the last group padded
by repeating its last frame). Each NAL is run through the same checkpoints' training path (``forward_one_frame``,
as `g5b_train.run_sequence` and G5b's monitor set do), without gradients and with rounding in place of the
training noise, so its bits are those of the quantized symbols under the model's own probabilities. The bits of
y (per latent cell, summed over channels) and of z (spread evenly over the y cells beneath each z cell) are
spread evenly over each cell's pixels in the NAL's real frames, and summed over the region's pixels. The driver
scales each NAL's region share to the NAL's actual bytes; the estimate is only the split.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

FRAME_DELAY = 8  # HT-L's g_frame_delay
CELL = 16  # pixels per latent cell of y
PAD = 64  # the training path's networks halve y twice more (z): frames are padded to a multiple of 64


def schedule(frame_count: int) -> list[list[int]]:
    """`src.codecs.dcvc_uf_worker.display_schedule` for HT-L."""
    out, index = [[0]], 1
    while index < frame_count:
        width = min(FRAME_DELAY, frame_count - index)
        out.append(list(range(index, index + width)))
        index += width
    return out


def spread_z(y_cells: np.ndarray, z_cells: np.ndarray) -> np.ndarray:
    """``y_cells`` (Hy, Wy) plus each z cell's bits shared evenly by the y cells beneath it."""
    hy, wy = y_cells.shape
    hz, wz = z_cells.shape
    rows = np.arange(hy) * hz // hy
    cols = np.arange(wy) * wz // wy
    per_row = np.bincount(rows, minlength=hz).astype(np.float64)
    per_col = np.bincount(cols, minlength=wz).astype(np.float64)
    shared = z_cells[np.ix_(rows, cols)] / (per_row[rows][:, None] * per_col[cols][None, :])
    return y_cells.astype(np.float64) + shared


def region_bits(cells: np.ndarray, region: np.ndarray, size: int = CELL) -> float:
    """Bits of ``cells`` (Hy, Wy) on ``region`` (frames, h, w) bool: each ``size``-pixel cell's bits spread evenly over
    its pixels in every frame (the padded rows and columns beyond h and w count as outside)."""
    hy, wy = cells.shape
    frames, h, w = region.shape
    if not (0 <= hy * size - h < PAD and 0 <= wy * size - w < PAD):
        raise ValueError(f"latent {hy}×{wy} of {size}-pixel cells does not tile a {h}×{w} frame padded to {PAD}")
    padded = np.zeros((frames, hy * size, wy * size), bool)
    padded[:, :h, :w] = region
    counts = padded.reshape(frames, hy, size, wy, size).sum((0, 2, 4))
    return float((cells * counts).sum() / (frames * size * size))


def load_state(path: str, expected: str) -> dict[str, Any]:
    import torch
    from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"checkpoint bytes {actual} differ from verified {expected}: {path}")
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = state.get("net", state)
    consume_prefix_in_state_dict_if_present(state, prefix="module.")
    return state


def estimate(plan: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC, g_frame_delay
    from src.utils.common import ModelStructure

    if g_frame_delay != FRAME_DELAY:
        raise RuntimeError(f"HT-L frame delay {g_frame_delay}, expected {FRAME_DELAY}")
    device = "cuda"
    width, height, count = int(plan["width"]), int(plan["height"]), int(plan["frame_count"])
    raw = np.fromfile(plan["frames_file"], np.uint8)
    if raw.size != count * width * height * 3 // 2:
        raise ValueError(f"{plan['frames_file']}: {raw.size} bytes for {count} frames of {width}×{height}")
    planes = raw.reshape(count, height * 3 // 2, width)
    region = np.load(plan["region_file"], mmap_mode="r")
    # Further regions billed from the same pass (H3e: the hand region outside a pasted mask), by name.
    extra = {name: np.load(path, mmap_mode="r") for name, path in plan.get("extra_region_files", {}).items()}
    for r in [region, *extra.values()]:
        if r.shape != (count, height, width):
            raise ValueError(f"region {r.shape}, expected {(count, height, width)}")
    # The encoder pads to 16 inside its CUDA proxy; the training path needs 64. The extra rows and columns are
    # edge copies outside R, so they only move a few bits between "outside" cells.
    pad_r, pad_b = DMCI.get_padding_size(height, width, PAD)

    def frame(t: int) -> Any:
        """(1, 3, H, W) YCbCr in [-0.5, 0.5], chroma upsampled by nearest neighbour (``ycbcr420_to_444_np``)."""
        p = planes[t]
        luma = p[:height].astype(np.float32)
        chroma = p[height:].reshape(2, height // 2, width // 2).astype(np.float32).repeat(2, 1).repeat(2, 2)
        x = torch.from_numpy(np.concatenate([luma[None], chroma]) / 255.0 - 0.5)[None].to(device)
        return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")

    i_net = DMCI()
    i_net.load_state_dict(load_state(plan["image_ckpt"], plan["image_sha256"]), strict=True)
    p_net = DMC(model_structure=ModelStructure.HTL)
    p_net.load_state_dict(load_state(plan["video_ckpt"], plan["video_sha256"]), strict=True)
    # The training path: the eval path runs DCVC's CUDA proxy, which reports no per-cell bits.
    i_net, p_net = i_net.to(device).train(), p_net.to(device).train()
    captured: dict[str, Any] = {}
    for net in (i_net, p_net):
        get_y, get_z = net.get_y_bits, net.get_z_bits

        def bits_y(y: Any, sigma: Any, get_y: Any = get_y) -> Any:
            captured["y"] = get_y(y, sigma)
            return captured["y"]

        def bits_z(z: Any, index: Any, get_z: Any = get_z) -> Any:
            captured["z"] = get_z(z, index)
            return captured["z"]

        net.get_y_bits, net.get_z_bits = bits_y, bits_z
        net.add_noise = torch.round  # the quantized symbols, not the training noise

    def cells() -> np.ndarray:
        y = captured.pop("y").float().sum(1)[0].cpu().numpy()
        z = captured.pop("z").float().sum(1)[0].cpu().numpy()
        return spread_z(y, z)

    qp = torch.full((1,), int(plan["qp"]), dtype=torch.int32, device=device)
    nals = []
    began = time.time()
    with torch.no_grad():
        for frames in schedule(count):
            if frames == [0]:
                out = i_net.forward_one_frame(frame(0), qp)
                p_net.clear_dpb()
                p_net.add_ref_feature_from_frame(out["x_hat"], apply_feature_adaptor=False)
            else:
                xs = [frame(t) for t in frames]
                xs += [xs[-1]] * (FRAME_DELAY - len(xs))
                p_net.forward_one_frame(torch.cat(xs, dim=1), qp)
            c = cells()
            span = slice(frames[0], frames[-1] + 1)
            nal = {"frames": frames, "estimated_bits": float(c.sum()), "estimated_region_bits": region_bits(c, np.asarray(region[span]))}
            if extra:
                nal["extra_region_bits"] = {name: region_bits(c, np.asarray(r[span])) for name, r in extra.items()}
            nals.append(nal)
    return {"qp": int(plan["qp"]), "width": width, "height": height, "frame_count": count, "nals": nals,
            "seconds": round(time.time() - began, 2), "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if not str(Path.cwd()).endswith("DCVC") or "src" not in os.listdir("."):
        raise SystemExit("run from the vendored DCVC tree (cwd and PYTHONPATH)")
    args.report.write_text(json.dumps(estimate(json.loads(args.plan.read_text())), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
