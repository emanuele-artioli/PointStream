"""ProPainter video inpainting of a clip's foreground holes, run inside ProPainter's tree (G5c's fill check).

    python propainter_fill.py --plan PLAN.JSON --report REPORT.JSON

Run with ProPainter's repository (sczhou/ProPainter, revision in the plan) as the
working directory and on ``sys.path``, as `g5c.propainter_command` does. The plan
names ``frames`` ((n, h, w, 3) uint8 RGB, ``.npy``), ``holes`` ((n, h, w) bool,
``.npy``), ``out`` and the three checkpoints with their sha256; nothing is
downloaded. The steps and defaults are the repository's `inference_propainter.py`
(mask dilation 4, reference stride 10, neighbor length 10, sub-video length 80,
RAFT 20 iterations, fp16 after RAFT); the one change is that frames are padded
(edge) to a multiple of 8 and cropped back instead of resized, so known pixels
stay exact. Outside the dilated holes the output is the input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULTS = {"mask_dilation": 4, "ref_stride": 10, "neighbor_length": 10, "subvideo_length": 80, "raft_iter": 20,
            "fp16": True}


def checked(path: str, sha256: str) -> str:
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != sha256:
        raise ValueError(f"{path}: sha256 {actual}, expected {sha256}")
    return path


def get_ref_index(mid: int, neighbors: list[int], length: int, stride: int, ref_num: int) -> list[int]:
    """The repository's `get_ref_index`."""
    out: list[int] = []
    if ref_num == -1:
        return [i for i in range(0, length, stride) if i not in neighbors]
    start, end = max(0, mid - stride * (ref_num // 2)), min(length, mid + stride * (ref_num // 2))
    for i in range(start, end, stride):
        if i not in neighbors:
            if len(out) > ref_num:
                break
            out.append(i)
    return out


def inpaint(frames_rgb: np.ndarray, holes: np.ndarray, plan: dict[str, Any], device: str = "cuda") -> np.ndarray:
    import scipy.ndimage
    import torch
    from model.modules.flow_comp_raft import RAFT_bi
    from model.propainter import InpaintGenerator
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet

    o = {**DEFAULTS, **plan.get("options", {})}
    n, h0, w0, _ = frames_rgb.shape
    h, w = h0 + (-h0) % 8, w0 + (-w0) % 8
    pad = ((0, 0), (0, h - h0), (0, w - w0))
    frames_np = np.pad(frames_rgb, pad + ((0, 0),), mode="edge")
    holes_np = np.pad(holes, pad, mode="edge")
    dilated = np.stack([scipy.ndimage.binary_dilation(m, iterations=o["mask_dilation"]) for m in holes_np])
    frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().div(255).unsqueeze(0).to(device) * 2 - 1
    flow_masks = torch.from_numpy(dilated).float()[None, :, None].to(device)
    masks_dilated = flow_masks.clone()  # the script dilates both masks by the same mask_dilation

    raft = RAFT_bi(checked(plan["raft"], plan["raft_sha256"]), device)
    complete = RecurrentFlowCompleteNet(checked(plan["flow_completion"], plan["flow_completion_sha256"]))
    for p in complete.parameters():
        p.requires_grad = False
    complete.to(device).eval()
    model = InpaintGenerator(model_path=checked(plan["propainter"], plan["propainter_sha256"])).to(device).eval()

    with torch.no_grad():
        short = 12 if w <= 640 else 8 if w <= 720 else 4 if w <= 1280 else 2
        if n > short:
            ff, fb = [], []
            for f in range(0, n, short):
                end = min(n, f + short)
                a, b = raft(frames[:, f:end] if f == 0 else frames[:, f - 1:end], iters=o["raft_iter"])
                ff.append(a)
                fb.append(b)
                torch.cuda.empty_cache()
            gt = (torch.cat(ff, 1), torch.cat(fb, 1))
        else:
            gt = raft(frames, iters=o["raft_iter"])
        if o["fp16"]:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt = (gt[0].half(), gt[1].half())
            complete, model = complete.half(), model.half()

        length = gt[0].size(1)
        sub = o["subvideo_length"]
        if length > sub:
            pf, pb = [], []
            pad_len = 5
            for f in range(0, length, sub):
                s, e = max(0, f - pad_len), min(length, f + sub + pad_len)
                ps, pe = max(0, f) - s, e - min(length, f + sub)
                part = (gt[0][:, s:e], gt[1][:, s:e])
                pred, _ = complete.forward_bidirect_flow(part, flow_masks[:, s:e + 1])
                pred = complete.combine_flow(part, pred, flow_masks[:, s:e + 1])
                pf.append(pred[0][:, ps:e - s - pe])
                pb.append(pred[1][:, ps:e - s - pe])
                torch.cuda.empty_cache()
            flows = (torch.cat(pf, 1), torch.cat(pb, 1))
        else:
            pred, _ = complete.forward_bidirect_flow(gt, flow_masks)
            flows = complete.combine_flow(gt, pred, flow_masks)

        masked = frames * (1 - masks_dilated)
        prop_len = min(100, sub)
        if n > prop_len:
            uf, um = [], []
            pad_len = 10
            for f in range(0, n, prop_len):
                s, e = max(0, f - pad_len), min(n, f + prop_len + pad_len)
                ps, pe = max(0, f) - s, e - min(n, f + prop_len)
                b, t = masks_dilated[:, s:e].shape[:2]
                imgs, local = model.img_propagation(masked[:, s:e], (flows[0][:, s:e - 1], flows[1][:, s:e - 1]),
                                                    masks_dilated[:, s:e], "nearest")
                upd = frames[:, s:e] * (1 - masks_dilated[:, s:e]) + imgs.view(b, t, 3, h, w) * masks_dilated[:, s:e]
                uf.append(upd[:, ps:e - s - pe])
                um.append(local.view(b, t, 1, h, w)[:, ps:e - s - pe])
                torch.cuda.empty_cache()
            updated, updated_masks = torch.cat(uf, 1), torch.cat(um, 1)
        else:
            b, t = masks_dilated.shape[:2]
            imgs, local = model.img_propagation(masked, flows, masks_dilated, "nearest")
            updated = frames * (1 - masks_dilated) + imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = local.view(b, t, 1, h, w)

    comp: list[np.ndarray | None] = [None] * n
    stride = o["neighbor_length"] // 2
    ref_num = sub // o["ref_stride"] if n > sub else -1
    for f in range(0, n, stride):
        neighbors = list(range(max(0, f - stride), min(n, f + stride + 1)))
        refs = get_ref_index(f, neighbors, n, o["ref_stride"], ref_num)
        ids = neighbors + refs
        with torch.no_grad():
            pred = model(updated[:, ids], (flows[0][:, neighbors[:-1]], flows[1][:, neighbors[:-1]]),
                         masks_dilated[:, ids], updated_masks[:, ids], len(neighbors))
            pred = ((pred.view(-1, 3, h, w) + 1) / 2).float().cpu().permute(0, 2, 3, 1).numpy() * 255
        binary = dilated[neighbors][..., None].astype(np.uint8)
        for i, idx in enumerate(neighbors):
            img = pred[i].astype(np.uint8) * binary[i] + frames_np[idx] * (1 - binary[i])
            prev = comp[idx]
            comp[idx] = img if prev is None else (prev.astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5)
            comp[idx] = comp[idx].astype(np.uint8)  # type: ignore[union-attr]
        torch.cuda.empty_cache()
    out = np.stack(comp)[:, :h0, :w0]  # type: ignore[arg-type]
    keep = ~dilated[:, :h0, :w0]
    out[keep] = frames_rgb[keep]
    return out


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())
    frames, holes = np.load(plan["frames"]), np.load(plan["holes"])
    began = time.time()
    out = inpaint(frames, holes, plan)
    torch.cuda.synchronize()
    np.save(plan["out"], out)
    Path(args.report).write_text(json.dumps({
        "seconds": round(time.time() - began, 1), "frames": len(out), "options": {**DEFAULTS, **plan.get("options", {})},
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
