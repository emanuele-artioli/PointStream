"""G5 oracle, arm B: a per-clip model conditioned on a warped reference refreshed every 0.1 s.

Each frame t is decoded from its own latent and from the model's output at its
reference r (the last refresh point before t), warped to t by the dense DIS
flow of the *source* frames (oracle motion, priced as `planes4`). The decoder
upsamples the latent and sees the warped reference at every scale; its head
blends a synthesised image with the warped reference through a predicted
weight, so disocclusions and independent motion are synthesised and the rest
is copied.

`fit` trains it as the decoder runs it: every epoch walks the refresh groups
in time order and takes the references from the current outputs (detached).
With ``fixed`` frames (the refreshes, transmitted by another codec), those are
the outputs at the refresh points and the model codes only the frames between
them; without, the references are the model's own outputs (closed loop), which
leaves the first frame to the latent alone. Latents are fitted directly (an auto-decoder) under uniform-noise
quantization and a learned factorized Laplace prior per channel. The result is
scored after a fresh sequential rollout from rounded latents and 8-bit weights,
and priced as latents (ideal code length under the prior, plus the prior) and
weights (empirical entropy of their 8-bit symbols, plus one scale per tensor).
"""

from __future__ import annotations

import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import numpy as np

LATENT_CHANNELS = 4
LATENT_STRIDE = 12  # latent grid 45×80 at 960×540
SCALES = (3, 2, 2)
BASE_WIDTHS = (32, 24, 16, 12)  # at λ = 200 on 240 frames
REF_CHANNELS = 8
BASE_FRAMES = 240
BASE_LAMB = 200.0
CURVE_EVERY = 30
WEIGHT_BITS = 8
PRIOR_BITS = 32


# ----------------------------------------------------------------- oracle motion (CPU, spawned)


def _flow_chunk(task: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    import cv2

    cv2.setNumThreads(1)
    targets, references = task
    dis = getattr(cv2, "DISOpticalFlow_create")(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = np.empty(targets.shape + (2,), np.float16)
    for i in range(len(targets)):
        out[i] = dis.calc(targets[i], references[i], None)
    return out


def oracle_flows(frames: np.ndarray, refs: list[int | None], workers: int = 0) -> np.ndarray:
    """(n, h, w, 2) flow f with frame_t(x) ≈ frame_r(x + f(x)) for each frame's reference r (zero without one)."""
    import cv2

    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
    flows = np.zeros(gray.shape + (2,), np.float16)
    pairs = [(t, r) for t, r in enumerate(refs) if r is not None]
    if not pairs:
        return flows
    workers = workers or max(1, min(32, (multiprocessing.cpu_count() or 2) - 1))
    chunks = [pairs[i::workers] for i in range(workers)]
    tasks = [(gray[[t for t, _ in c]], gray[[r for _, r in c]]) for c in chunks if c]
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context("spawn")) as pool:
        for chunk, result in zip([c for c in chunks if c], pool.map(_flow_chunk, tasks)):
            for (t, _), f in zip(chunk, result):
                flows[t] = f
    return flows


def refresh_groups(refs: list[int | None]) -> list[list[int]]:
    """Frames grouped by reference, in time order; every group's reference lies in an earlier group."""
    groups: dict[int | None, list[int]] = {}
    for t, r in enumerate(refs):
        groups.setdefault(r, []).append(t)
    ordered = [groups.pop(None)] if None in groups else []
    ordered += [groups[r] for r in sorted(k for k in groups if k is not None)]
    return ordered


# ----------------------------------------------------------------- model


def widths_for(lamb: float, frames: int) -> tuple[int, ...]:
    """Decoder widths: more capacity for more frames and higher λ. The weights are priced and their bits
    are spread over the clip's frames, so the size is part of the rate point (about 110k parameters at
    λ = 200 on 240 frames)."""
    m = math.sqrt(frames / BASE_FRAMES) * (lamb / BASE_LAMB) ** 0.25
    return tuple(max(4, int(round(w * m))) for w in BASE_WIDTHS)


def build_model(widths: tuple[int, ...]) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Conditioned(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(LATENT_CHANNELS, widths[0], 3, padding=1)
            self.refs = nn.ModuleList(nn.Conv2d(4, REF_CHANNELS, 3, padding=1) for _ in range(len(SCALES) + 1))
            self.ups = nn.ModuleList(
                nn.Conv2d(widths[k] + REF_CHANNELS, widths[k + 1] * s * s, 3, padding=1) for k, s in enumerate(SCALES))
            self.head = nn.Conv2d(widths[-1] + REF_CHANNELS, 4, 3, padding=1)
            self.log_scale = nn.Parameter(torch.zeros(LATENT_CHANNELS))  # Laplace prior per channel

        def forward(self, z: Any, warped: Any, valid: Any) -> Any:
            ref_in = torch.cat([warped, valid], 1)
            x = F.gelu(self.stem(z))
            for k, s in enumerate(SCALES):
                r = F.adaptive_avg_pool2d(ref_in, x.shape[-2:])
                x = F.gelu(F.pixel_shuffle(self.ups[k](torch.cat([x, F.gelu(self.refs[k](r))], 1)), s))
            o = self.head(torch.cat([x, F.gelu(self.refs[-1](ref_in))], 1))
            syn = torch.sigmoid(o[:, :3])
            weight = torch.sigmoid(o[:, 3:4]) * valid
            return weight * warped + (1 - weight) * syn

        def bits(self, z_hat: Any) -> Any:
            """Ideal code length of integer-valued latents under the Laplace prior (per element)."""
            b = F.softplus(self.log_scale).view(1, -1, 1, 1) + 1e-3

            def cdf(v: Any) -> Any:
                return 0.5 + 0.5 * torch.sign(v) * (1 - torch.exp(-v.abs() / b))

            p = (cdf(z_hat + 0.5) - cdf(z_hat - 0.5)).clamp_min(1e-9)
            return -torch.log2(p)

    return Conditioned()


def warp(images: Any, flows: Any) -> tuple[Any, Any]:
    """Backward warp of (b, 3, h, w) by (b, h, w, 2) pixel flows; also the in-bounds mask."""
    import torch
    import torch.nn.functional as F

    b, _, h, w = images.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=images.device, dtype=torch.float32),
                            torch.arange(w, device=images.device, dtype=torch.float32), indexing="ij")
    px = xs.unsqueeze(0) + flows[..., 0].float()
    py = ys.unsqueeze(0) + flows[..., 1].float()
    valid = ((px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)).unsqueeze(1).float()
    grid = torch.stack([2 * px / (w - 1) - 1, 2 * py / (h - 1) - 1], dim=-1)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True) * valid, valid


# ----------------------------------------------------------------- fit, quantize, roll out


def entropy_bits(symbols: np.ndarray) -> float:
    """Empirical (zeroth-order) entropy of integer symbols times their count."""
    _, counts = np.unique(symbols, return_counts=True)
    p = counts / counts.sum()
    return float(-(counts * np.log2(p)).sum())


def fit(frames: np.ndarray, keep: np.ndarray, flows: np.ndarray, refs: list[int | None], *, lamb: float,
        epochs: int, device: str, on_epoch: Callable[[int], None] | None = None, seed: int = 0,
        fixed: dict[int, np.ndarray] | None = None) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    n, h, w, _ = frames.shape
    gh, gw = h // LATENT_STRIDE, w // LATENT_STRIDE
    widths = widths_for(lamb, n)
    model = build_model(widths).to(device)
    latents = torch.nn.Embedding(n, LATENT_CHANNELS * gh * gw, sparse=True).to(device)
    torch.nn.init.zeros_(latents.weight)
    src = torch.from_numpy(frames).to(device)  # uint8 (n, h, w, 3)
    mask = torch.from_numpy(keep).to(device)
    flow = torch.from_numpy(flows).to(device)
    groups = refresh_groups(refs)
    steps = epochs * len(groups)
    opt_model = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt_latent = torch.optim.SparseAdam(latents.parameters(), lr=1e-2)
    sched = [torch.optim.lr_scheduler.CosineAnnealingLR(o, steps, eta_min=lr * 0.01)
             for o, lr in ((opt_model, 1e-3), (opt_latent, 1e-2))]
    buffer = torch.zeros((n, 3, h, w), dtype=torch.float16, device=device)
    fixed = fixed or {}
    fixed_t = {t: torch.from_numpy(img).to(device).permute(2, 0, 1).float() / 255.0 for t, img in fixed.items()}
    for t, img in fixed_t.items():
        buffer[t] = img.to(torch.float16)
    coded = [t for t in range(n) if t not in fixed]
    groups = [[t for t in g if t not in fixed] for g in groups]
    groups = [g for g in groups if g]
    steps = epochs * len(groups)
    for s_ in sched:
        s_.T_max = steps
    curve = []

    def target(idx: list[int]) -> tuple[Any, Any]:
        return src[idx].permute(0, 3, 1, 2).float() / 255.0, mask[idx].unsqueeze(1).float()

    def reference(idx: list[int], images: Any) -> tuple[Any, Any]:
        r = refs[idx[0]]
        if r is None:
            return torch.zeros((len(idx), 3, h, w), device=device), torch.zeros((len(idx), 1, h, w), device=device)
        return warp(images[[r] * len(idx)].float(), flow[idx])

    def latent(idx: list[int]) -> Any:
        return latents(torch.tensor(idx, device=device)).view(len(idx), LATENT_CHANNELS, gh, gw)

    for epoch in range(epochs):
        model.train()
        for idx in groups:
            warped, valid = reference(idx, buffer)
            z = latent(idx)
            z_hat = z + (torch.rand_like(z) - 0.5)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(z_hat, warped, valid)
            tgt, keep_t = target(idx)
            mse = (((out.float() - tgt) * keep_t) ** 2).mean()
            bpp = model.bits(z_hat).sum() / (len(idx) * h * w)
            loss = lamb * mse + bpp
            opt_model.zero_grad(set_to_none=True)
            opt_latent.zero_grad(set_to_none=True)
            loss.backward()
            opt_model.step()
            opt_latent.step()
            for s in sched:
                s.step()
            buffer[idx] = out.detach().to(torch.float16)
        if (epoch + 1) % CURVE_EVERY == 0 or epoch + 1 == epochs:
            roll = rollout(model, latent, list(range(n)), refs, flow, (h, w), device, fixed_t)
            curve.append({"epoch": epoch + 1, **measure(roll, src, mask, model, latent, coded, h, w)})
        if on_epoch:
            on_epoch(epoch + 1)
    float_decoded = to_uint8(rollout(model, latent, list(range(n)), refs, flow, (h, w), device, fixed_t))
    weight_bits, tensors = quantize_weights(model)
    with torch.no_grad():
        z_int = torch.round(latents.weight).view(n, LATENT_CHANNELS, gh, gw)
        latent_bits = float(model.bits(z_int[coded]).double().sum()) + LATENT_CHANNELS * PRIOR_BITS
    if device == "cuda":
        torch.cuda.synchronize()
    began = time.perf_counter()
    decoded = rollout(model, lambda idx: z_int[idx], list(range(n)), refs, flow, (h, w), device, fixed_t)
    if device == "cuda":
        torch.cuda.synchronize()
    render_ms = 1000 * (time.perf_counter() - began) / max(1, len(coded))
    params = sum(p.numel() for name, p in model.named_parameters() if name != "log_scale")
    return {
        "decoded": to_uint8(decoded), "decoded_float": float_decoded,
        "bits": {"latents": latent_bits, "weights": weight_bits},
        "render_ms_per_frame": render_ms, "curve": curve, "converged": curve_converged(curve),
        "model": {"widths": list(widths), "latent": [LATENT_CHANNELS, gh, gw], "scales": list(SCALES),
                  "ref_channels": REF_CHANNELS, "parameters": params, "coded_frames": len(coded), "weight_bits_per_parameter":
                  weight_bits / params, "quantized_tensors": tensors, "groups": len(groups)},
    }


def rollout(model: Any, latent: Callable[[list[int]], Any], frames: list[int], refs: list[int | None], flow: Any,
            size: tuple[int, int], device: str, fixed: dict[int, Any] | None = None) -> Any:
    """Decode every frame in time order, each from the decoded output at its reference; ``fixed`` frames
    (transmitted refreshes) are taken as they are."""
    import torch

    h, w = size
    out = torch.zeros((len(frames), 3, h, w), dtype=torch.float32, device=device)
    fixed = fixed or {}
    for t, img in fixed.items():
        out[t] = img
    model.eval()
    with torch.no_grad():
        for group in refresh_groups(refs):
            idx = [t for t in group if t not in fixed]
            if not idx:
                continue
            r = refs[idx[0]]
            if r is None:
                warped = torch.zeros((len(idx), 3, h, w), device=device)
                valid = torch.zeros((len(idx), 1, h, w), device=device)
            else:
                warped, valid = warp(out[[r] * len(idx)], flow[idx])
            z = latent(idx)
            out[idx] = model(torch.round(z), warped, valid).float()
    return out


def to_uint8(images: Any) -> np.ndarray:
    import torch

    return torch.clamp(torch.round(images * 255.0), 0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def measure(images: Any, src: Any, mask: Any, model: Any, latent: Callable[[list[int]], Any], coded: list[int],
            h: int, w: int) -> dict[str, float]:
    """PSNR on the fitted pixels (V) of the frames the model codes, and their latents' bits per pixel, for the
    convergence curve."""
    import torch

    with torch.no_grad():
        decoded = torch.clamp(torch.round(images[coded] * 255.0), 0, 255)
        diff = (decoded - src[coded].permute(0, 3, 1, 2).float()) ** 2
        m = mask[coded].unsqueeze(1).float()
        mse = float((diff * m).sum() / (3 * m.sum()))
        bits = float(sum(model.bits(torch.round(latent([i]))).sum() for i in coded))
    return {"psnr": 10 * math.log10(255.0 ** 2 / max(mse, 1e-12)), "bpp": bits / (len(coded) * h * w)}


def warp_only(fixed: dict[int, np.ndarray], flows: np.ndarray, refs: list[int | None], n: int,
              device: str) -> np.ndarray:
    """Every frame between refreshes as its refresh warped by the oracle flow, nothing coded (border pixels
    clamped, not zeroed)."""
    import torch
    import torch.nn.functional as F

    h, w, _ = next(iter(fixed.values())).shape
    out = np.empty((n, h, w, 3), np.uint8)
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    for t in range(n):
        if t in fixed:
            out[t] = fixed[t]
            continue
        r = refs[t]
        assert r is not None and r in fixed, f"frame {t} has no transmitted reference"
        img = torch.from_numpy(fixed[r]).to(device).permute(2, 0, 1)[None].float()
        f = torch.from_numpy(flows[t].astype(np.float32)).to(device)
        grid = torch.stack([2 * (xs + f[..., 0]) / (w - 1) - 1, 2 * (ys + f[..., 1]) / (h - 1) - 1], -1)[None]
        warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
        out[t] = torch.clamp(torch.round(warped[0]), 0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return out


def curve_converged(curve: list[dict[str, float]]) -> dict[str, Any]:
    """The G5 rule's check between the last two curve points (30 epochs apart)."""
    from experiments.background.g5 import DECISION

    rule = DECISION["converged"]
    if len(curve) < 2:
        return {"checked": False, "reason": f"{len(curve)} curve points"}
    a, b = curve[-2], curve[-1]
    gain = b["psnr"] - a["psnr"]
    saving = 1.0 - b["bpp"] / a["bpp"] if a["bpp"] > 0 else 0.0
    ok = not ((gain > rule["max_gain_db"] and b["bpp"] <= a["bpp"] * 1.0001) or
              (saving > rule["max_rate_saving"] and b["psnr"] >= a["psnr"] - 1e-4))
    return {"checked": True, "converged": ok, "from_epoch": a["epoch"], "to_epoch": b["epoch"],
            "psnr_gain_db": gain, "rate_saving": saving}


def quantize_weights(model: Any) -> tuple[float, int]:
    """Round every decoder tensor to 8-bit symmetric levels in place; their price in bits and the tensor count.
    The prior's scales are priced with the latents."""
    import torch

    bits, tensors = 0.0, 0
    levels = 2 ** (WEIGHT_BITS - 1) - 1
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == "log_scale":
                continue
            peak = float(p.abs().max())
            if peak == 0:
                bits += PRIOR_BITS
                continue
            step = peak / levels
            q = torch.round(p / step)
            bits += entropy_bits(q.cpu().numpy().astype(np.int16).ravel()) + PRIOR_BITS
            p.copy_(q * step)
            tensors += 1
    return bits, tensors
