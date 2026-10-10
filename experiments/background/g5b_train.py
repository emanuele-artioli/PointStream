"""Fine-tune DCVC-UF's video model (HT-L) on one scene: PLAN step G5b.

Run as a script, never imported by PointStream: like `src.codecs.dcvc_uf_worker`
it runs with ``cwd`` and ``PYTHONPATH`` set to the vendored DCVC tree
(``<prefix>/opt/DCVC`` at ``cbdae87``), whose top-level package is also called
``src``. `experiments.background.g5b` writes the plan and the frames.

The loop follows ``train_video.py`` (stage 2 of HT-L: cascaded, 512×512 patches,
an I frame from the frozen image model, then groups of 8 frames, one optimizer
step per sequence, AdamW, gradient norm clipped at 0.2, a random QP per sample
with λ from `get_training_lambdas`), with these differences:

* The data are one scene's frames in memory (4:2:0 as the codec is fed at test
  time, chroma upsampled by nearest neighbour as ``test_video.py`` does), not
  a folder of PNGs; sequences never cross a gap in the fitted frames.
* The distortion counts only the visible background V (the loss mask), with
  DCVC's own YUV/RGB weighting (`mse_weighted_average`) and per-frame weights
  (`mse_8frames_sum`). The rate is the whole patch's, as in DCVC.
* Optionally LPIPS on V (AlexNet, foreground pasted back from the source as in
  G2's scoring) is added to the distortion with weight ``lpips_weight``.
* A fixed monitor set (crops and QPs drawn once from the fitted frames) is
  evaluated every ``monitor_every`` steps without gradients; its curve decides
  convergence. The held-out frames are never seen unless the plan fits on them
  (the upper bound).
* The state is saved every ``save_every`` steps for resumption.

Only the video model is trained; the image model codes the first frame of a
stream and stays as published.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def emit(record: dict[str, Any]) -> None:
    print(json.dumps(record), flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def load_lpips(backbone: str, device: str) -> Any:
    """torchmetrics' LPIPS (AlexNet), backbone read from ``backbone`` (as `src.codecs.quality.load_lpips`)."""
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
    for p in net.parameters():
        p.requires_grad_(False)
    return net.eval().to(device)


def sequence_starts(fit: list[int], length: int) -> list[int]:
    """Starts s with s, s+1, ..., s+length-1 all fitted (no sequence crosses a gap)."""
    have = set(fit)
    return [s for s in sorted(have) if all(s + k in have for k in range(length))]


class Scene:
    """One scene's 4:2:0 frames and loss masks, cropped into training sequences."""

    def __init__(self, frames: np.ndarray, keep: np.ndarray):
        self.frames, self.keep = frames, keep  # (n, h*3/2, w) uint8; (n, h, w) bool
        self.height, self.width = keep.shape[1:]

    def crop(self, start: int, length: int, y: int, x: int, ph: int, pw: int, flip: bool) -> tuple[Any, Any]:
        """(length, 3, ph, pw) YCbCr in [-0.5, 0.5] and (length, 1, ph, pw) V, for even ``y`` and ``x``."""
        import torch

        h, w = self.height, self.width
        seq = self.frames[start:start + length]
        luma = seq[:, y:y + ph, x:x + pw].astype(np.float32)
        chroma = seq[:, h:].reshape(length, 2, h // 2, w // 2)[:, :, y // 2:(y + ph) // 2, x // 2:(x + pw) // 2]
        chroma = chroma.astype(np.float32).repeat(2, axis=2).repeat(2, axis=3)  # nearest, as ycbcr420_to_444_np
        yuv = np.concatenate([luma[:, None], chroma], axis=1) / 255.0 - 0.5
        mask = self.keep[start:start + length, y:y + ph, x:x + pw][:, None].astype(np.float32)
        if flip:
            yuv, mask = yuv[..., ::-1], mask[..., ::-1]
        return torch.from_numpy(np.ascontiguousarray(yuv)), torch.from_numpy(np.ascontiguousarray(mask))

    def batch(self, rng: np.random.Generator, starts: list[int], length: int, size: int, ph: int, pw: int,
              qp_num: int) -> tuple[Any, Any, Any]:
        import torch

        xs, ms = [], []
        for _ in range(size):
            s = int(rng.choice(starts))
            y = 2 * int(rng.integers(0, (self.height - ph) // 2 + 1))
            x = 2 * int(rng.integers(0, (self.width - pw) // 2 + 1))
            a, b = self.crop(s, length, y, x, ph, pw, bool(rng.integers(0, 2)))
            xs.append(a)
            ms.append(b)
        qp = torch.from_numpy(rng.integers(0, qp_num, size).astype(np.int32))
        return torch.stack(xs, 1), torch.stack(ms, 1), qp  # (length, B, 3, ph, pw), (length, B, 1, ph, pw), (B,)


def masked_distortion(x: Any, x_hat: Any, mask: Any) -> Any:
    """DCVC's `mse_weighted_average` over the pixels of ``mask`` only, per sample (B,)."""
    import torch
    from src.utils.transforms import ycbcr2rgb

    count = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mse_yuv = ((x - x_hat) ** 2 * mask).sum(dim=(2, 3)) / count[:, None]
    rgb_diff = ycbcr2rgb(x + 0.5, clamp=False) - ycbcr2rgb(x_hat + 0.5, clamp=False)
    mse_rgb = (rgb_diff ** 2 * mask).sum(dim=(1, 2, 3)) / count
    y, u, v = mse_yuv[:, 0], mse_yuv[:, 1], mse_yuv[:, 2]
    yuv = torch.exp(0.0833 * (10 * torch.log(y.clamp_min(1e-6)) + torch.log(u.clamp_min(1e-6))
                              + torch.log(v.clamp_min(1e-6)))) * 3
    return yuv * 0.8 + mse_rgb * 0.2


def masked_lpips(net: Any, x: Any, x_hat: Any, mask: Any) -> Any:
    """LPIPS on V with the foreground pasted back from the source, per sample (B,)."""
    import torch
    from src.utils.transforms import ycbcr2rgb

    source = ycbcr2rgb(x + 0.5)
    pasted = mask * ycbcr2rgb(x_hat + 0.5) + (1 - mask) * source
    spatial = net(pasted * 2 - 1, source * 2 - 1)
    if spatial.shape[-2:] != mask.shape[-2:]:
        spatial = torch.nn.functional.interpolate(spatial, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    return (spatial * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)


def run_sequence(p_net: Any, i_net: Any, x: Any, mask: Any, qp: Any, lambdas: Any, lpips: Any,
                 lpips_weight: float, delay: int, measure_lpips: bool = False) -> tuple[Any, dict[str, float]]:
    """One cascaded sequence: I frame from the frozen image model, then groups of ``delay`` frames. LPIPS on V
    enters the distortion with ``lpips_weight``; with ``measure_lpips`` it is only reported."""
    import torch
    from src.layers.layers import mse_8frames_sum

    with torch.inference_mode():
        ref = i_net.forward_one_frame(x[0], qp, recon_only=True)
    p_net.clear_dpb()
    p_net.add_ref_feature_from_frame(ref.clone(), apply_feature_adaptor=False)
    weights = [1.5, 0.16, 0.4]  # DCVC-UF's per-frame weights within a group (`DMC.get_rd_info`)
    losses, dists, lps, bpps = [], [], [], []
    for g in range((x.shape[0] - 1) // delay):
        frames = range(1 + g * delay, 1 + (g + 1) * delay)
        group = torch.cat([x[t] for t in frames], dim=1)
        out = p_net.forward_one_frame(group, qp)
        per_frame, mse_frame, lp_frame = [], [], []
        for t, x_hat in zip(frames, out["x_hat"]):
            d = masked_distortion(x[t], x_hat, mask[t])
            mse_frame.append(d.detach())
            if lpips is not None and (lpips_weight > 0 or measure_lpips):
                lp = masked_lpips(lpips, x[t], x_hat, mask[t])
                lp_frame.append(lp.detach())
                if lpips_weight > 0:
                    d = d + lpips_weight * lp
            per_frame.append(d)
        losses.append(lambdas * mse_8frames_sum(per_frame, weights) + out["bpp"])
        dists.append(mse_8frames_sum(mse_frame, weights))
        bpps.append(out["bpp"].detach())
        if lp_frame:
            lps.append(torch.stack(lp_frame).mean(0).detach())
    loss = torch.stack(losses).mean()
    info = {"loss": float(loss.detach()), "dist": float(torch.stack(dists).mean()),
            "bpp": float(torch.stack(bpps).mean())}
    if lps:
        info["lpips_v"] = float(torch.stack(lps).mean())
    return loss, info


def train(plan: dict[str, Any]) -> dict[str, Any]:
    import torch
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC, g_frame_delay
    from src.utils.common import ModelStructure, get_training_lambdas

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    out = Path(plan["out"])
    out.mkdir(parents=True, exist_ok=True)
    frames = np.load(plan["frames"], mmap_mode="r")
    keep = np.load(plan["keep"], mmap_mode="r")
    scene = Scene(np.ascontiguousarray(frames), np.ascontiguousarray(keep))
    length = 1 + plan["groups"] * g_frame_delay
    starts = sequence_starts(plan["fit"], length)
    if not starts:
        raise SystemExit(f"no {length}-frame run among the fitted frames")
    ph, pw = plan["patch"]
    qp_num = DMCI.qp_num()
    lambda_table = torch.tensor(get_training_lambdas(plan["lambdas"], qp_num), dtype=torch.float32)

    i_net = DMCI()
    i_net.load_state_dict(load_state(plan["image_ckpt"], plan["image_sha256"]), strict=True)
    i_net = i_net.eval().to(device)
    for p in i_net.parameters():
        p.requires_grad_(False)
    p_net = DMC(model_structure=ModelStructure.HTL)
    p_net.load_state_dict(load_state(plan["video_ckpt"], plan["video_sha256"]), strict=True)
    p_net = p_net.to(device).train()
    p_net.set_use_ckpt(ph > 256)
    optimizer = torch.optim.AdamW(p_net.parameters(), lr=plan["lr"])
    lpips = load_lpips(plan["lpips_backbone"], device)  # the monitor set always reports LPIPS on V

    # The monitor set: fixed crops and QPs spread over the range, drawn once.
    monitor_rng = np.random.default_rng(plan["seed"] + 7919)
    monitor = []
    for qp_value in plan["monitor_qps"]:
        x, m, _ = scene.batch(monitor_rng, starts, length, plan["monitor_batch"], ph, pw, qp_num)
        monitor.append((x, m, torch.full((x.shape[1],), qp_value, dtype=torch.int32)))

    def evaluate() -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        torch.manual_seed(plan["seed"])  # the same quantization noise every time
        with torch.no_grad():
            for x, m, qp in monitor:
                x, m, qp = x.to(device), m.to(device), qp.to(device)
                _, info = run_sequence(p_net, i_net, x, m, qp, lambda_table.to(device)[qp.long()], lpips,
                                       plan["lpips_weight"], g_frame_delay, measure_lpips=True)
                for k, v in info.items():
                    totals.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in totals.items()}

    state_path = Path(plan["resume_dir"]) / "state.pt" if plan.get("resume_dir") else None
    step, curve = 0, []
    rng = np.random.default_rng(plan["seed"])
    if state_path is not None and state_path.is_file():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        p_net.load_state_dict(saved["p_net"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step, curve = saved["step"], saved["curve"]
        rng = np.random.default_rng(plan["seed"] + step)
        emit({"event": "resumed", "step": step})
    if step == 0:
        curve.append({"step": 0, **evaluate()})
        emit({"event": "monitor", **curve[-1]})

    began, train_seconds, skipped, first = time.time(), 0.0, 0, step
    while step < plan["steps"]:
        t0 = time.time()
        x, m, qp = scene.batch(rng, starts, length, plan["batch"], ph, pw, qp_num)
        x, m, qp = x.to(device, non_blocking=True), m.to(device, non_blocking=True), qp.to(device)
        loss, info = run_sequence(p_net, i_net, x, m, qp, lambda_table.to(device)[qp.long()], lpips,
                                  plan["lpips_weight"], g_frame_delay)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(p_net.parameters(), max_norm=0.2, error_if_nonfinite=False).item()
        if math.isfinite(norm):
            optimizer.step()
        else:
            skipped += 1
        torch.cuda.synchronize()
        train_seconds += time.time() - t0
        step += 1
        if step % plan["log_every"] == 0:
            emit({"event": "train", "step": step, **info, "grad_norm": norm,
                  "seconds_per_step": round(train_seconds / (step - first), 3)})
        if step % plan["monitor_every"] == 0 or step == plan["steps"]:
            curve.append({"step": step, **evaluate(), "elapsed": round(time.time() - began, 1)})
            emit({"event": "monitor", **curve[-1]})
        if state_path is not None and (step % plan["save_every"] == 0 or step == plan["steps"]):
            tmp = state_path.with_suffix(".tmp")
            torch.save({"p_net": p_net.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "curve": curve}, tmp)
            os.replace(tmp, state_path)

    final = out / "video_ft.pth.tar"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in p_net.state_dict().items()}}, final)
    return {"checkpoint": str(final), "checkpoint_sha256": sha256(final), "checkpoint_bytes": final.stat().st_size,
            "parameters": int(sum(p.numel() for p in p_net.parameters())), "steps": step,
            "skipped_nonfinite": skipped, "sequence_frames": length, "sequence_starts": len(starts),
            "train_seconds": round(train_seconds, 1),
            "seconds_per_step": round(train_seconds / max(1, step - first), 3), "resumed_at": first,
            "curve": curve, "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "torch": torch.__version__}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if not str(Path.cwd()).endswith("DCVC") or "src" not in os.listdir("."):
        raise SystemExit("run from the vendored DCVC tree (cwd and PYTHONPATH)")
    result = train(json.loads(args.plan.read_text()))
    args.report.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
