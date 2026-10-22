"""K6b (hands) 2: zero-shot oracle of a hand renderer (docs/experiments.md, 2026-10-22 K6b (hands) 2).

    python -m experiments.visor.k6bh run --eval-set JSON --archive TAR --masks TAR --mask-record JSON --video MP4 ... \\
        --items IDS --frames N --lpips-backbone PTH --dists-backbone PTH --k1f-stage TAR ... \\
        --foundhand-src TGZ --foundhand-ckpt CKPT --vae CKPT --steps N
    python -m experiments.visor.k6bh validate
    python -m experiments.visor.k6bh report --stage DIR ... --out JSON

Per K1f item: K1f's coded and uncoded 2D keypoints (its published rows, every masked hand fitted), one reference crop
per hand side (the window's first frame where that side is fitted, granted and unbilled), and four renderings of
each fitted hand inside that side's mask (R granted as the matte; Telea fill where no fit exists, as K1c):
``gen`` (FoundHand's released video procedure), ``gen@frame`` (FoundHand per frame from the reference alone), ``ref``
(K6's warp of the reference crop to the coded keypoints) and ``warp`` (K1f's bound, reproduced as a check). Scored on
R exactly as K1f (``h3c.Scorer``); ``report`` compares each variant with K1f's per-item hull at the item's rate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor import k1c
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

FOUNDHAND_REV = "ae0e9808ef1300820d499e707ec4d655da158e6d"
SIZE, LATENT, LATENT_DIM = 256, 32, 4
SCALING = 0.18215  # SD's latent scale, as FoundHand's demos
CFG = 2.5  # the released demos' cfg_scale
CROP_SCALE = 1.25
MIN_CROP = 32  # px
VARIANTS = ("gen", "gen@frame", "ref", "warp")
FRAME_BATCH = 16
WARP_TOLERANCE = 0.002  # LPIPS on R, against K1f's recorded bound
CRITERION = 0.75
DECISION: dict[str, Any] = {
    "item_won": "the variant's LPIPS (DISTS) on R is below K1f's hull at the item's keypoint rate",
    "score": "mean over {LPIPS, DISTS} of the share of the items won, for gen",
    "pass": "score >= 0.75: the zero-shot route holds with the reference granted; next, the reference billed",
    "fail": "score < 0.75: the overfit rung gets its rule and budget",
    "to_user": "ref scores >= 0.75; or the score is on different sides of 0.75 for items FoundHand has seen and has not",
    "check": f"warp reproduces K1f's recorded LPIPS on R within {WARP_TOLERANCE} per item, or nothing is read",
}


# ----------------------------------------------------------------- geometry (CPU)


def crop_box(points: np.ndarray, mask: np.ndarray | None) -> tuple[int, int, int]:
    """(x0, y0, side): a square around the union of the points' box and the mask's box, x CROP_SCALE."""
    xs, ys = [points[:, 0].min(), points[:, 0].max()], [points[:, 1].min(), points[:, 1].max()]
    if mask is not None and mask.any():
        my, mx = np.nonzero(mask)
        xs += [mx.min(), mx.max() + 1]
        ys += [my.min(), my.max() + 1]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    side = max(MIN_CROP, int(np.ceil(CROP_SCALE * max(x1 - x0, y1 - y0))))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2)), side


def crop_pixels(image: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    """The box's pixels, zero outside the image, resized to SIZE (area)."""
    import cv2

    x0, y0, side = box
    out = np.zeros((side, side, *image.shape[2:]), image.dtype)
    h, w = image.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    if image.dtype == bool:
        return cv2.resize(out.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
    return cv2.resize(out, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def to_crop(points: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    x0, y0, side = box
    return (points[:, :2] - np.array([x0, y0], float)) * (SIZE / side)


def paste(canvas: np.ndarray, rendered: np.ndarray, box: tuple[int, int, int], mask: np.ndarray) -> np.ndarray:
    """Write ``rendered`` (SIZE x SIZE, resized back to the box, Lanczos) onto ``canvas`` on ``mask`` within the box.
    Returns the pixels written."""
    import cv2

    x0, y0, side = box
    h, w = canvas.shape[:2]
    big = cv2.resize(rendered, (side, side), interpolation=cv2.INTER_LANCZOS4)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    written = np.zeros(mask.shape, bool)
    if sx1 <= sx0 or sy1 <= sy0:
        return written
    window = mask[sy0:sy1, sx0:sx1]
    canvas[sy0:sy1, sx0:sx1][window] = big[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0][window]
    written[sy0:sy1, sx0:sx1] = window
    return written


def slots(points: np.ndarray, right: bool) -> np.ndarray:
    """FoundHand's 42 keypoints: the true right hand in 0-20, the left in 21-41 (README; the demos un-mirror
    MediaPipe's handedness); the other hand absent (zero, invalid)."""
    out = np.zeros((42, 2))
    if right:
        out[:21] = points
    else:
        out[21:] = points
    return out


def heatmaps(points42: np.ndarray) -> np.ndarray:
    """FoundHand's ``keypoint_heatmap`` (var 1) at the latent size, times ``check_keypoints_validity`` on SIZE."""
    valid = ((points42[:, 0] > 0) & (points42[:, 0] < SIZE) & (points42[:, 1] > 0) & (points42[:, 1] < SIZE)).astype(float)
    pts = points42 * (LATENT / SIZE)
    grid = np.stack(np.meshgrid(np.linspace(0, LATENT - 1, LATENT), np.linspace(0, LATENT - 1, LATENT)), -1)
    diff = grid[None] - pts[:, None, None, :]
    return np.exp(-(diff ** 2).sum(-1) / 2.0) / (2.0 * np.pi) * valid[:, None, None]


# ----------------------------------------------------------------- FoundHand (GPU)


def _load_module(name: str, path: Path, package: bool = False) -> Any:
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py" if package else path,
                                                  submodule_search_locations=[str(path)] if package else None)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FoundHand:
    """FoundHand's DiT-XL/2 (EMA weights) and SD VAE, loaded from the pinned source tree under private module names."""

    def __init__(self, src: Path, ckpt: Path, vae: Path, steps: int, device: str, scratch: Path):
        import torch

        root = scratch / "foundhand"
        if not (root / "FoundHand").is_dir():
            root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(src) as tar:
                tar.extractall(root, filter="data")
        tree = root / "FoundHand"
        vit = _load_module("foundhand_vit", tree / "models" / "vit.py")
        vqvae = _load_module("foundhand_vqvae", tree / "models" / "vqvae.py")
        diffusion = _load_module("foundhand_diffusion", tree / "diffusion", package=True)
        self.device, self.steps = device, steps
        self.diffusion = diffusion.create_diffusion(str(steps))
        model = vit.DiT_XL_2(input_size=LATENT, latent_dim=LATENT_DIM, in_channels=LATENT_DIM + 42 + 1, learn_sigma=True)
        try:
            state = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)["ema_state_dict"]
        except RuntimeError:  # a legacy (non-zip) checkpoint cannot be memory-mapped
            state = torch.load(ckpt, map_location="cpu", weights_only=False)["ema_state_dict"]
        missing, extra = model.load_state_dict(state, strict=False)
        if missing:
            raise SystemExit(f"FoundHand: missing keys {missing[:5]}")
        del state
        self.model = model.to(device).eval().requires_grad_(False)
        auto = vqvae.create_model(3, 3, LATENT_DIM)
        missing_v, extra_v = auto.load_state_dict(torch.load(vae, map_location="cpu", weights_only=False)["state_dict"], strict=False)
        if missing_v:
            raise SystemExit(f"VAE: missing keys {missing_v[:5]}")
        self.vae = auto.to(device).eval().requires_grad_(False)
        attn = self.model.blocks[0].attn
        self.record = {"revision": FOUNDHAND_REV, "steps": steps, "cfg": CFG, "unexpected_keys": len(extra), "vae_unexpected_keys": len(extra_v),
                       "params": int(sum(p.numel() for p in self.model.parameters())), "dtype": "bfloat16 autocast (DiT), float32 (VAE)",
                       "fused_attn": bool(getattr(attn, "fused_attn", False)),
                       "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled())}
        self.record["kernels"], self.record["step_ms"] = self.probe()

    def probe(self) -> tuple[dict[str, Any], float]:
        """The first-run check (AGENTS.md): one guided step profiled (kernels, attention family), then 20 timed."""
        import torch

        from experiments.audit.env_smoke import profile_cuda

        x = torch.randn((2, LATENT_DIM, LATENT, LATENT), device=self.device)
        t = torch.full((2,), 500, device=self.device)
        kw = {"target_cond": torch.zeros((2, 43, LATENT, LATENT), device=self.device),
              "ref_cond": torch.zeros((2, LATENT_DIM + 43, LATENT, LATENT), device=self.device),
              "nvs": torch.tensor([0, 2], device=self.device), "cfg_scale": CFG}
        with torch.inference_mode():
            self._forward(x, t, **kw)
            _, kernels = profile_cuda(lambda: self._forward(x, t, **kw))
            torch.cuda.synchronize()
            began = time.time()
            for _ in range(20):
                self._forward(x, t, **kw)
            torch.cuda.synchronize()
        return kernels, (time.time() - began) / 20 * 1000

    def _forward(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.forward_with_cfg(*args, **kwargs)
        return out.float()

    def encode(self, image: np.ndarray, seed: int) -> Any:
        import torch

        torch.manual_seed(seed)
        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        with torch.inference_mode():
            return SCALING * self.vae.encode(x).sample()

    def cond(self, latent: Any, points42: np.ndarray, mask: np.ndarray | None) -> Any:
        """[latent, 42 heatmaps, mask] (the reference) or [heatmaps, zero mask] (a target, ``latent`` None)."""
        import cv2
        import torch

        hm = torch.tensor(heatmaps(points42), dtype=torch.float, device=self.device)[None]
        m = np.zeros((LATENT, LATENT)) if mask is None else cv2.resize(mask.astype(np.uint8), (LATENT, LATENT), interpolation=cv2.INTER_NEAREST)
        mt = torch.tensor(m, dtype=torch.float, device=self.device)[None, None]
        return torch.cat([hm, mt], 1) if latent is None else torch.cat([latent, hm, mt], 1)

    def sample(self, target: Any, ref: Any | None, refs: list[Any] | None, seed: int) -> tuple[Any, np.ndarray]:
        """FoundHand's demo sampler (``p_sample_loop`` on ``forward_with_cfg``, cfg 2.5, nvs off); ``refs``: the video
        procedure's reference list (one picked at random per step, batch 1). Returns (latents, uint8 images)."""
        import torch

        n = target.shape[0]
        torch.manual_seed(seed)
        random.seed(seed)
        z = torch.randn((n, LATENT_DIM, LATENT, LATENT), device=self.device)
        z = torch.cat([z, z], 0)
        nvs = torch.zeros(n, dtype=torch.int, device=self.device)
        kwargs = {"target_cond": torch.cat([target, torch.zeros_like(target)]), "nvs": torch.cat([nvs, 2 * torch.ones_like(nvs)]),
                  "cfg_scale": CFG}
        if ref is not None:
            kwargs["ref_cond"] = torch.cat([ref, torch.zeros_like(ref)])
        with torch.inference_mode():
            samples = self.diffusion.p_sample_loop(self._forward, z.shape, z, clip_denoised=False, model_kwargs=kwargs,
                                                   ref_conds=refs, progress=False, device=self.device).chunk(2)[0]
            images = torch.clamp(self.vae.decode(samples / SCALING), -1.0, 1.0)
        out = (((images.permute(0, 2, 3, 1).float().cpu().numpy() + 1) / 2) * 255).astype(np.uint8)
        return samples, out


# ----------------------------------------------------------------- run (GPU)


def k1f_rows(tars: list[str]) -> dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]]:
    from experiments.visor.h2 import load_units

    return {m["item"]: (m, a) for m, a in load_units(tars, "k1c")}


def render_item(fh: FoundHand, rows: dict[str, np.ndarray], truth: Any, clip: Any, frames: int, base_seed: int,
                variants: tuple[str, ...]) -> dict[str, Any]:
    """FoundHand's renderings per row (crops and boxes), for ``gen`` and ``gen@frame``; the reference per side."""
    import torch

    from experiments.visor.h3 import HANDS

    out: dict[str, Any] = {"refs": {}, "boxes": {}, "gen": {}, "gen@frame": {}, "seconds": {}}
    order = np.argsort(rows["t"], kind="stable")
    for right in (False, True):
        side = HANDS[1] if right else HANDS[0]
        mine = [int(i) for i in order if bool(rows["right"][i]) == right and int(rows["t"][i]) < frames]
        mine = [i for i in mine if clip.class_mask(int(rows["t"][i]), side).any()]
        if not mine:
            continue
        r0 = mine[0]
        t0 = int(rows["t"][r0])
        ref_mask = clip.class_mask(t0, side)
        box0 = crop_box(rows["uncoded"][r0], ref_mask)
        ref_img = crop_pixels(np.asarray(truth[t0]), box0)
        ref_crop_mask = crop_pixels(ref_mask, box0)
        seed = base_seed * 2 + int(right)
        latent = fh.encode(ref_img, seed)
        src = fh.cond(latent, slots(to_crop(rows["uncoded"][r0], box0), right), ref_crop_mask)
        out["refs"][side] = {"row": r0, "t": t0, "box": box0}
        targets = {}
        for i in mine:
            t = int(rows["t"][i])
            box = crop_box(rows["coded"][i], clip.class_mask(t, side))
            out["boxes"][i] = box
            targets[i] = fh.cond(None, slots(to_crop(rows["coded"][i], box), right), None)
        if "gen" in variants:
            began = time.time()
            temp: list[Any] = []
            for k, i in enumerate(mine):
                samples, image = fh.sample(targets[i], None, [src] + temp, seed * 100003 + k)
                out["gen"][i] = image[0]
                temp = [torch.cat([samples, targets[i]], 1)]  # the demo's last_N_frames = 1
            out["seconds"][f"gen:{side}"] = time.time() - began
        if "gen@frame" in variants:
            began = time.time()
            for first in range(0, len(mine), FRAME_BATCH):
                chunk = mine[first:first + FRAME_BATCH]
                tc = torch.cat([targets[i] for i in chunk])
                _, images = fh.sample(tc, src.repeat(len(chunk), 1, 1, 1), None, seed * 100003 + 50000 + first)
                for i, image in zip(chunk, images):
                    out["gen@frame"][i] = image
            out["seconds"][f"gen@frame:{side}"] = time.time() - began
        out["refs"][side]["image"] = np.asarray(truth[t0]).copy()
        out["refs"][side]["mask"] = ref_mask
    return out


def item_seed(item_id: str) -> int:
    """A seed from the item's id alone, so the samples do not depend on how items are split into jobs."""
    import hashlib

    return int(hashlib.sha256(item_id.encode()).hexdigest()[:7], 16)


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any], fh: FoundHand,
                 k1f_unit: tuple[dict[str, Any], dict[str, np.ndarray]], device: str, publish: Path, scratch: Path) -> None:
    import cv2

    from experiments.tennis.k6 import draw, telea_fill
    from experiments.visor.b2 import load_mask_sets
    from experiments.visor.h2 import save_unit
    from experiments.visor.h3 import FILL, HANDS, rgb_memmap
    from experiments.visor.h3c import Scorer

    began = time.time()
    frames = int(args.frames)
    width, height = decoded["width"], decoded["height"]
    work = scratch / "work" / item["id"]
    work.mkdir(parents=True, exist_ok=True)
    clips, mask_info = load_mask_sets(item, frames, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
    clip = clips[FILL]
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, decoded["full_range"], device)
    meta_k, a_k = k1f_unit
    ones = np.ones((len(a_k["row_t"]), 21, 1))
    rows = {"t": np.asarray(a_k["row_t"]).astype(int), "right": np.asarray(a_k["row_right"]).astype(bool),
            "uncoded": np.concatenate([a_k["uncoded_kp2d"], ones], -1), "coded": np.concatenate([a_k["coded_kp2d"], ones], -1)}
    # Draw order without K1f's joint depths (not published): smaller hands, assumed farther, first.
    size = np.ptp(rows["coded"][:, :, 1], axis=1) + np.ptp(rows["coded"][:, :, 0], axis=1)
    variants = tuple(args.variants.split(","))
    rendered = render_item(fh, rows, truth, clip, frames, item_seed(item["id"]), variants)
    rows_at: dict[int, list[int]] = {}
    for i, t in enumerate(rows["t"]):
        if t < frames:
            rows_at.setdefault(int(t), []).append(i)
    regions = [clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1]) for t in range(frames)]
    outs = {v: np.lib.format.open_memmap(work / f"{v.replace('@', '-')}.npy", mode="w+", dtype=np.uint8, shape=(frames, height, width, 3))
            for v in variants}
    covered = {v: np.zeros(frames, np.int64) for v in variants}
    for t in range(frames):
        truth_t = np.asarray(truth[t])
        if not regions[t].any():
            for v in variants:
                outs[v][t] = truth_t
            continue
        sides = {s: clip.class_mask(t, s) for s in HANDS}
        objects = clip.foreground(t) & ~regions[t]
        here = sorted(rows_at.get(t, []), key=lambda i: size[i])
        for v in variants:
            if v == "warp":
                hands = [(bool(rows["right"][i]), rows["uncoded"][i], rows["coded"][i], -float(size[i])) for i in here]
                frame, mine = k1c.compose(truth_t, sides, objects, hands)
                outs[v][t] = frame
                covered[v][t] = int((mine & regions[t]).sum())
                continue
            canvas = telea_fill(truth_t, regions[t])
            drawn = np.zeros(regions[t].shape, bool)
            for i in here:
                side = HANDS[1] if rows["right"][i] else HANDS[0]
                mask = sides[side]
                if not mask.any() or side not in rendered["refs"]:
                    continue
                if v == "ref":
                    ref = rendered["refs"][side]
                    ys = np.nonzero(ref["mask"].any(1))[0]
                    draw(canvas, drawn, ref["image"], ref["mask"], rows["uncoded"][ref["row"]], rows["coded"][i],
                         k1c.SIGMA_OF_HEIGHT * float(ys.max() - ys.min() + 1), k1c.HAND_SKELETON)
                elif i in rendered[v]:
                    drawn |= paste(canvas, rendered[v][i], rendered["boxes"][i], mask)
            over = drawn & objects
            canvas[over] = truth_t[over]
            outs[v][t] = canvas
            covered[v][t] = int((drawn & regions[t]).sum())
    for v in variants:
        outs[v].flush()
    score = Scorer(nets, truth, regions, device)
    hand = np.array(score.hand, int)
    arrays: dict[str, np.ndarray] = {"pixels_r": np.array([int(r.sum()) for r in regions], np.int64)}
    summary: dict[str, dict[str, float]] = {}
    luma = np.array([0.299, 0.587, 0.114])
    for v in variants:
        s = score(outs[v], upscale=False)
        key = v.replace("@", "_")
        arrays[f"{key}_lpips_r"], arrays[f"{key}_dists_r"], arrays[f"{key}_sse_r"] = s["lpips"], s["dists"], s["sse"]
        arrays[f"{key}_covered_r"] = covered[v]
        flicker, flicker_src = [], []
        for t in hand:
            if t == 0 or not regions[t - 1].any():
                continue
            r = regions[t]
            a = score.composite(outs[v], t, False)[r] @ luma
            b = score.composite(outs[v], t - 1, False)[r] @ luma
            flicker.append(float(np.abs(a - b).mean()))
            flicker_src.append(float(np.abs(np.asarray(truth[t])[r] @ luma - np.asarray(truth[t - 1])[r] @ luma).mean()))
        summary[v] = {"lpips_r": float(np.nanmean(s["lpips"][hand])), "dists_r": float(np.nanmean(s["dists"][hand])),
                      "flicker": float(np.mean(flicker)) if flicker else float("nan"),
                      "covered_share": float(covered[v][hand].sum() / max(1, arrays["pixels_r"][hand].sum()))}
        if flicker_src:
            summary["source"] = {"flicker": float(np.mean(flicker_src))}
    gen_std = [float(np.std(x)) for v in ("gen", "gen@frame") for x in rendered[v].values()]
    sizes = [b[2] for b in rendered["boxes"].values()]
    look = int(hand[len(hand) // 2]) if len(hand) else 0
    if len(hand):
        ys, xs = np.nonzero(regions[look])
        box = (slice(max(0, ys.min() - 64), min(height, ys.max() + 65)), slice(max(0, xs.min() - 64), min(width, xs.max() + 65)))
        strip = np.concatenate([np.asarray(truth[look])[box]] + [score.composite(outs[v], look, False)[box] for v in variants], 1)
        (publish / "frames").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(publish / "frames" / f"{item['id']}-f{look:03d}-source-{'-'.join(v.replace('@', '_') for v in variants)}.png"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        crops = [rendered[v][i] for v in ("gen", "gen@frame") for i in rows_at.get(look, []) if i in rendered[v]]
        if crops:
            cv2.imwrite(str(publish / "frames" / f"{item['id']}-f{look:03d}-crops.png"), cv2.cvtColor(np.concatenate(crops, 1), cv2.COLOR_RGB2BGR))
    meta = {"unit": item["id"], "item": item["id"], "video": item["video"], "frames": frames, "hand_frames": len(hand),
            "rows": int(sum(len(v) for v in rows_at.values())), "summary": summary,
            "k1f": {"kbps": float(meta_k["cover"]["kbps"]) if meta_k.get("cover") else k1c.POSE_KBPS,
                    "warp": meta_k["summary"]["warp"], "frames": int(meta_k["frames"]), "npz_hash_ok": bool(meta_k["npz_hash_ok"])},
            "references": {s: {"t": r["t"], "box": list(r["box"])} for s, r in rendered["refs"].items()},
            "crop_side_px": {"min": int(min(sizes)) if sizes else 0, "median": float(np.median(sizes)) if sizes else 0.0,
                             "max": int(max(sizes)) if sizes else 0},
            "generated_std_min": min(gen_std) if gen_std else None, "render_seconds": rendered["seconds"],
            "mask_set": mask_info, "seconds": round(time.time() - began, 2)}
    save_unit(publish, "k6bh", item["id"], arrays, meta)
    del outs, truth
    os.remove(decoded["path"])
    for f in work.glob("*.npy"):
        f.unlink()


def command_run(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record, model_on_cuda
    from experiments.visor.b2 import select_items
    from experiments.visor.h2 import restore_units
    from experiments.visor.h3b import load_dists
    from experiments.visor.h3c import decode_windows
    from src.codecs.quality import load_lpips, lpips_record

    items = select_items(json.loads(Path(args.eval_set).read_text()), args.items)
    units = k1f_rows(args.k1f_stage)
    if missing := [it["id"] for it in items if it["id"] not in units]:
        raise SystemExit(f"no K1f rows for {missing}")
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "k6bh")
    todo = [it for it in items if it["id"] not in restored]
    windows = decode_windows(args, todo, int(args.frames), scratch)
    if not torch.cuda.is_available():
        raise SystemExit("k6bh run needs a GPU")
    device = "cuda"
    nets: dict[str, Any] = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    loaded = time.time()
    fh = FoundHand(Path(args.foundhand_src), Path(args.foundhand_ckpt), Path(args.vae), int(args.steps), device, scratch)
    load_seconds = time.time() - loaded
    on_cuda = {k: next(v.parameters()).is_cuda for k, v in nets.items()}
    on_cuda["foundhand"], on_cuda["vae"] = model_on_cuda(fh.model), model_on_cuda(fh.vae)
    began = time.time()
    done = len(restored)
    order = [it["id"] for it in items]
    for item, decoded in windows:
        process_item(args, item, decoded, nets, fh, units[item["id"]], device, publish, scratch)
        done += 1
        progress(done)
    import torchmetrics

    write_json(stage_dir() / "k6bh.json", {
        "command": "run", "device": device_record(args.gpu_class), "on_cuda": on_cuda,
        "seconds": round(time.time() - began, 2), "load_seconds": round(load_seconds, 2), "foundhand": fh.record,
        "inputs": {"foundhand_src": file_sha256(Path(args.foundhand_src)), "foundhand_ckpt": file_sha256(Path(args.foundhand_ckpt)),
                   "vae": file_sha256(Path(args.vae)), "k1f_stages": [file_sha256(Path(t)) for t in args.k1f_stage]},
        "lpips": lpips_record(Path(args.lpips_backbone)),
        "dists": {"torchmetrics": torchmetrics.__version__, "backbone_sha256": file_sha256(Path(args.dists_backbone))},
        "settings": {"frames": int(args.frames), "items": order, "variants": args.variants.split(","), "crop_scale": CROP_SCALE,
                     "size": SIZE, "decision": DECISION}, "restored_units": sorted(restored)})
    return 0


# ----------------------------------------------------------------- validation (CPU)


def validate(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    from experiments.visor.h2 import load_units

    units = {m["item"]: (m, a) for m, a in load_units([str(stage / "published.tar")], "k6bh")}
    settings = result["settings"]
    checks = {"every_item": sorted(units) == sorted(settings["items"]), "on_cuda": all(result["on_cuda"].values()),
              "foundhand_loaded": result["foundhand"]["params"] > 6e8, "device_class": bool(result["device"]["class_matches"]),
              "kernels_on_gpu": result["foundhand"]["kernels"]["kernel_launches"] > 0}
    for item, (m, a) in units.items():
        hand = a["pixels_r"] > 0
        checks[f"{item}:hash"] = bool(m["npz_hash_ok"]) and bool(m["k1f"]["npz_hash_ok"])
        for v in settings["variants"]:
            key = v.replace("@", "_")
            checks[f"{item}:{v}:finite"] = bool(np.isfinite(a[f"{key}_lpips_r"][hand]).all() and np.isfinite(a[f"{key}_dists_r"][hand]).all())
            checks[f"{item}:{v}:draws"] = m["summary"][v]["covered_share"] > 0.5
        if "gen" in settings["variants"]:
            checks[f"{item}:gen_not_flat"] = (m["generated_std_min"] or 0) > 2.0
            checks[f"{item}:gen_differs_from_frame"] = ("gen@frame" not in settings["variants"]
                                                       or m["summary"]["gen"]["lpips_r"] != m["summary"]["gen@frame"]["lpips_r"])
        if "warp" in settings["variants"] and settings["frames"] == m["k1f"]["frames"]:
            checks[f"{item}:warp_reproduces_k1f"] = abs(m["summary"]["warp"]["lpips_r"] - m["k1f"]["warp"]["lpips_r"]) <= WARP_TOLERANCE
    return checks


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage) if args.stage else stage_dir()
    result = json.loads((stage / "k6bh.json").read_text())
    checks = {k: bool(v) for k, v in validate(stage, result).items()}
    report = {"passed": all(checks.values()), "checks": checks}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def score_items(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    if not rows:
        return {"items": 0}
    lp = float(np.mean([r[variant]["lpips_won"] for r in rows]))
    di = float(np.mean([r[variant]["dists_won"] for r in rows]))
    return {"items": len(rows), "lpips_share": lp, "dists_share": di, "score": (lp + di) / 2, "passes": (lp + di) / 2 >= CRITERION}


def command_report(args: argparse.Namespace) -> int:
    from experiments.background.g5 import envelope
    from experiments.visor.h2 import load_units
    from experiments.visor.k1 import at_rate
    from experiments.visor.k1f import G_VIDEOS

    hulls = {m["item"]: m for m, _ in load_units(args.k1f_stage, "k1f")}
    rows: list[dict[str, Any]] = []
    sources = []
    variants: list[str] = []
    for stage in map(Path, args.stage):
        result = json.loads((stage / "k6bh.json").read_text())
        checks = validate(stage, result)
        if not all(checks.values()) or result["settings"]["frames"] != 240:
            raise SystemExit(f"{stage} fails validation or is not a full stage: {[k for k, v in checks.items() if not v]}")
        sources.append({"stage": str(stage), "sha256": file_sha256(stage / "published.tar")})
        variants = result["settings"]["variants"]
        for m, _ in load_units([str(stage / "published.tar")], "k6bh"):
            kbps = m["k1f"]["kbps"]
            row: dict[str, Any] = {"item": m["item"], "video": m["video"], "kbps": kbps, "g_overlap": m["video"] in G_VIDEOS,
                                   "foundhand_seen": True, "crop_side_px": m["crop_side_px"], "bound": m["k1f"]["warp"]}
            for v in variants:
                row[v] = {"flicker": m["summary"][v]["flicker"]}
                for metric in ("lpips", "dists"):
                    hull = envelope([(p["kbps"], -p[metric]) for p in hulls[m["item"]]["points"]])
                    q = at_rate(hull, kbps)
                    value = m["summary"][v][f"{metric}_r"]
                    row[v][metric] = value
                    row[v][f"{metric}_hull"] = None if q is None else -q
                    row[v][f"{metric}_won"] = bool(value < (-q if q is not None else -hull[0][1]))
            row["source_flicker"] = m["summary"].get("source", {}).get("flicker")
            rows.append(row)
    groups = {"all": rows, "clean_of_g": [r for r in rows if not r["g_overlap"]]}
    out = {"inputs": {"stages": sources, "k1f_stages": [file_sha256(Path(t)) for t in args.k1f_stage],
                      "code_sha256": file_sha256(Path(__file__))},
           "decision_rule": DECISION, "rows": rows,
           "groups": {g: {v: score_items(rs, v) for v in variants} for g, rs in groups.items()},
           "overlap": "FoundHand-10M lists VISOR; its file lists are on Globus only, so every item counts as seen"}
    write_json(Path(args.out), out)
    for r in rows:
        print(r["item"], f"{r['kbps']:.2f} kbps", " ".join(
            f"{v}: L {r[v]['lpips']:.3f}/{r[v]['lpips_hull']:.3f} D {r[v]['dists']:.3f}/{r[v]['dists_hull']:.3f}" for v in variants))
    print(json.dumps(out["groups"], indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for flag in ("--eval-set", "--archive", "--masks", "--mask-record", "--items", "--lpips-backbone", "--dists-backbone",
                 "--foundhand-src", "--foundhand-ckpt", "--vae"):
        run.add_argument(flag, required=True)
    run.add_argument("--video", action="append", default=[])
    run.add_argument("--k1f-stage", action="append", required=True, help="K1f's published.tar (rows, bound and hull)")
    run.add_argument("--frames", default="240")
    run.add_argument("--steps", default="50", help="FoundHand's respaced sampling steps (released demos: 250)")
    run.add_argument("--variants", default=",".join(VARIANTS))
    run.add_argument("--gpu-class", default="RTX 6000 Ada", help="the GPU class the smoke asserts (model-GPU table)")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default="")
    rep = sub.add_parser("report")
    rep.add_argument("--stage", action="append", required=True)
    rep.add_argument("--k1f-stage", action="append", required=True)
    rep.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
