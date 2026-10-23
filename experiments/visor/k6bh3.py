"""K6b (hands) 3: the overfit oracle, FoundHand fitted on the window's own frames (docs/experiments.md, 2026-10-22).

    python -m experiments.visor.k6bh3 run --mode pilot|fit <k6bh run's inputs> --fit-seconds S --eval-frames N
    python -m experiments.visor.k6bh3 validate
    python -m experiments.visor.k6bh3 report --pilot DIR --stage DIR ... --k1f-stage TAR ... --out JSON

Per item: rung 2's crops (K1f's rows, a square around the keypoints and the side's mask, x1.25, 256²) of every fitted
hand in the fitting frames, encoded by SD's VAE once; FoundHand's DiT-XL/2 (EMA weights) fine-tuned on (reference,
target) pairs of the same hand with its own diffusion loss (``training_losses``; uncoded keypoints), conditional
dropout as its sampler's unconditional branch. Judged as rung 2's ``gen``: its video procedure from the window's first
fitted frame, on the coded keypoints, composited inside R and scored on R (``h3c.Scorer``).

* ``pilot`` (P01_107, development): fit on frames 0-179 for ``fit-seconds`` of training, judging ``gen`` on the
  held-out frames from 180 at 0%, 25%, 50% and 100% of it (the learning curve).
* ``fit`` (the scored items): fit on all 240 frames, then judge ``gen`` and reproduce K1f's ``warp``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor import k1c, k6bh
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

EVAL_FROM = 180  # pilot: fit on frames 0-179, judge from 180
FRACTIONS = (0.0, 0.25, 0.5, 1.0)  # of the fit budget: the probes at each, the pilot's judging from 0.25
JUDGED = (0.25, 0.5, 1.0)
BATCH = 8
LR = 1e-5  # the released checkpoint's
P_UNCOND = 0.1  # the sampler's unconditional branch (zero conditions, nvs 2), kept trained
P_NO_REF_MASK = 0.5  # the video procedure's later references carry no mask
PROBE_PAIRS = 16
GPU_CAP_MIB = 27 * 1024
TRAIN_SEED = 20261022
HULL_MINUTES = 45.0  # the pilot's horizon: the curve must reach the hull within this much fitting per item
CRITERION = 0.75
DECISION: dict[str, Any] = {
    "model": "FoundHand's EMA DiT fine-tuned per item (full, AdamW lr 1e-5, batch 8, bf16 autocast), its training_losses",
    "pilot": "P01_107, fit on frames 0-179 for 30 min of training; gen judged on frames 180-239 at 25/50/100% (and 0%); the "
             "scaled fits run only if LPIPS on R, extrapolated in log time, reaches K1c's hull at P01_107's rate within 45 min",
    "item_won": "gen's LPIPS (DISTS) on R below K1f's hull at the item's keypoint rate (the fitted weights free: an oracle)",
    "score": "mean over {LPIPS, DISTS} of the share of the 3 items won (P06_108, P22_107, P25_09)",
    "pass": "score >= 0.75: rung 4 (a fit on other shots of the same video, billed)",
    "fail": "score < 0.75: the generative route closes for hands at the keypoint rate; K1f's bound stays a bound",
}


# ----------------------------------------------------------------- data (CPU)


def hand_crops(rows: dict[str, np.ndarray], truth: Any, clip: Any, frames: range, key: str) -> list[dict[str, Any]]:
    """Rung 2's crop of every fitted, masked hand in ``frames`` with its ``key`` keypoints: image, mask and joints in
    the crop."""
    from experiments.visor.h3 import HANDS

    out = []
    for i in np.argsort(rows["t"], kind="stable"):
        t, right = int(rows["t"][i]), bool(rows["right"][i])
        if t not in frames:
            continue
        mask = clip.class_mask(t, HANDS[1] if right else HANDS[0])
        if not mask.any():
            continue
        box = k6bh.crop_box(rows[key][i], mask)
        out.append({"row": int(i), "t": t, "right": right, "box": box, "image": k6bh.crop_pixels(np.asarray(truth[t]), box),
                    "mask": k6bh.crop_pixels(mask, box), "kp": k6bh.to_crop(rows[key][i], box)})
    return out


def reference_pairs(sides: np.ndarray, rng: np.random.Generator, picks: np.ndarray) -> np.ndarray:
    """For each picked target, another crop of the same hand side (itself when it is the side's only one)."""
    out = np.empty(len(picks), int)
    for n, i in enumerate(picks):
        same = np.nonzero(sides == sides[i])[0]
        others = same[same != i]
        out[n] = int(rng.choice(others)) if len(others) else int(i)
    return out


def log_time_fit(seconds: list[float], values: list[float]) -> tuple[float, float]:
    """Least squares of value = a + b log(seconds) (the learning curve), as (a, b)."""
    x = np.log(np.asarray(seconds, float))
    b, a = np.polyfit(x, np.asarray(values, float), 1)
    return float(a), float(b)


def minutes_to_reach(seconds: list[float], values: list[float], target: float) -> float | None:
    """Minutes of fitting at which the log-time line reaches ``target`` (lower is better); None if it never falls."""
    a, b = log_time_fit(seconds, values)
    if b >= 0:
        return None if a + b * np.log(max(seconds)) > target else max(seconds) / 60
    return float(np.exp((target - a) / b) / 60)


LUMA = np.array([0.299, 0.587, 0.114])
LOOKS = 5  # frames saved for the visual check


def look_frames(hand: list[int], count: int = LOOKS) -> list[int]:
    """Evenly spaced hand frames for the visual check (content-blind: by position only)."""
    if not hand:
        return []
    picks = np.linspace(0, len(hand) - 1, min(count, len(hand)) + 2)[1:-1] if len(hand) > count else np.arange(len(hand))
    return sorted({int(hand[int(round(i))]) for i in picks})


def flicker(composite: Any, regions: list[np.ndarray], hand: list[int]) -> float:
    """Rung 2's flicker: the mean absolute luma change inside R between consecutive frames (``composite(t)`` gives frame
    t), over the hand frames whose previous frame has R too."""
    values = []
    for t in hand:
        if t == 0 or not regions[t - 1].any():
            continue
        r = regions[t]
        values.append(float(np.abs(composite(t)[r] @ LUMA - composite(t - 1)[r] @ LUMA).mean()))
    return float(np.mean(values)) if values else float("nan")


# ----------------------------------------------------------------- fitting (GPU)


class Fitter:
    """The crops as FoundHand's conditions, and the fit itself."""

    def __init__(self, fh: k6bh.FoundHand, crops: list[dict[str, Any]], device: str, seed: int) -> None:
        import cv2
        import torch

        self.fh, self.device = fh, device
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        images = torch.from_numpy(np.stack([c["image"] for c in crops])).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        with torch.inference_mode():
            self.latent = torch.cat([k6bh.SCALING * fh.vae.encode(images[i:i + 16]).sample() for i in range(0, len(crops), 16)]).clone()
        self.heat = torch.tensor(np.stack([k6bh.heatmaps(k6bh.slots(c["kp"], c["right"])) for c in crops]), dtype=torch.float, device=device)
        self.mask = torch.tensor(np.stack([cv2.resize(c["mask"].astype(np.uint8), (k6bh.LATENT, k6bh.LATENT),
                                                      interpolation=cv2.INTER_NEAREST) for c in crops]), dtype=torch.float, device=device)[:, None]
        self.sides = np.array([c["right"] for c in crops])

    def conds(self, target: np.ndarray, ref: np.ndarray, keep_mask: Any, uncond: Any) -> dict[str, Any]:
        import torch

        tc = torch.cat([self.heat[target], torch.zeros_like(self.mask[target])], 1)
        rc = torch.cat([self.latent[ref], self.heat[ref], self.mask[ref] * keep_mask[:, None, None, None]], 1)
        drop = uncond[:, None, None, None]
        return {"target_cond": tc * (1 - drop), "ref_cond": rc * (1 - drop), "nvs": (2 * uncond).long()}

    def model_fn(self, x: Any, t: Any, **kw: Any) -> Any:
        import torch

        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16):
            return self.fh.model(x, t, **kw).float()


def fit(fitter: Fitter, diffusion: Any, seconds: float, marks: list[float], on_mark: Any, probe: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune for ``seconds`` of training (judging time excluded), calling ``on_mark(fraction, steps, seconds)`` at
    each fraction in ``marks``. Returns the loss trace and the fixed-probe losses at the marks."""
    import torch

    from experiments.visor.k1f import release

    model = fitter.fh.model
    model.pos_embed.requires_grad_(False)  # the sin-cos embedding stays as released
    params = [p for n, p in model.named_parameters() if n != "pos_embed"]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    trace: list[tuple[int, float]] = []
    fixed: list[tuple[float, float]] = []
    trained, steps, pending = 0.0, 0, sorted(marks)
    n = len(fitter.sides)

    def mark(fraction: float) -> None:
        model.eval()
        fixed.append((fraction, probe_loss(fitter, diffusion, probe)))
        on_mark(fraction, steps, trained)
        release()
        model.train()

    while pending and pending[0] <= 0.0:
        mark(pending.pop(0))
    model.train()
    while pending:
        began = time.time()
        picks = fitter.rng.integers(0, n, BATCH)
        refs = reference_pairs(fitter.sides, fitter.rng, picks)
        keep = torch.tensor(fitter.rng.random(BATCH) >= P_NO_REF_MASK, dtype=torch.float, device=fitter.device)
        uncond = torch.tensor(fitter.rng.random(BATCH) < P_UNCOND, dtype=torch.float, device=fitter.device)
        t = torch.from_numpy(fitter.rng.integers(0, diffusion.num_timesteps, BATCH)).to(fitter.device)
        losses = diffusion.training_losses(fitter.model_fn, fitter.latent[picks], t, model_kwargs=fitter.conds(picks, refs, keep, uncond))
        loss = losses["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        trained += time.time() - began
        steps += 1
        if steps % 10 == 0:
            trace.append((steps, float(loss.detach())))
        while pending and trained >= pending[0] * seconds:
            mark(pending.pop(0))
    del opt
    model.eval().requires_grad_(False)
    release()
    return {"steps": steps, "trained_seconds": trained, "trace": trace, "fixed": fixed}


def make_probe(fitter: Fitter, seed: int) -> dict[str, Any]:
    """Fixed (target, reference, timestep, noise) pairs for a loss that is comparable across the fit."""
    import torch

    rng = np.random.default_rng(seed)
    n = len(fitter.sides)
    picks = rng.choice(n, min(PROBE_PAIRS, n), replace=False)
    refs = reference_pairs(fitter.sides, rng, picks)
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((len(picks), k6bh.LATENT_DIM, k6bh.LATENT, k6bh.LATENT), generator=g).to(fitter.device)
    t = torch.linspace(50, 950, len(picks)).round().long().to(fitter.device)
    ones = torch.ones(len(picks), device=fitter.device)
    return {"picks": picks, "refs": refs, "noise": noise, "t": t, "keep": ones, "uncond": 0 * ones}


def probe_loss(fitter: Fitter, diffusion: Any, probe: dict[str, Any]) -> float:
    import torch

    with torch.no_grad():
        losses = diffusion.training_losses(fitter.model_fn, fitter.latent[probe["picks"]], probe["t"],
                                           model_kwargs=fitter.conds(probe["picks"], probe["refs"], probe["keep"], probe["uncond"]),
                                           noise=probe["noise"])
    return float(losses["mse"].mean())


def sample_probe(fitter: Fitter, probe: dict[str, Any], lpips_net: Any, seed: int) -> float:
    """The smoke's "a sample changes towards the target": FoundHand's sampler on the probe's training pairs (each from its
    reference), and the mean LPIPS of the decoded crops against the target crops."""
    import torch

    tc = torch.cat([fitter.heat[probe["picks"]], torch.zeros_like(fitter.mask[probe["picks"]])], 1)
    rc = torch.cat([fitter.latent[probe["refs"]], fitter.heat[probe["refs"]], fitter.mask[probe["refs"]]], 1)
    _, images = fitter.fh.sample(tc, rc, None, seed)
    with torch.inference_mode():
        target = fitter.fh.vae.decode(fitter.latent[probe["picks"]] / k6bh.SCALING).clamp(-1, 1)
        made = torch.from_numpy(images).to(fitter.device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return float(lpips_net(made, target).mean())


# ----------------------------------------------------------------- judging (GPU)


def render_gen(fh: k6bh.FoundHand, rows: dict[str, np.ndarray], truth: Any, clip: Any, ref_frames: range, frames: range,
               seed: int) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, int, int]]]:
    """Rung 2's ``gen`` (the video procedure: the side's first fitted frame in ``ref_frames`` as reference, then the
    previous generated frame), on the coded keypoints of the rows in ``frames``, both sides' chains sampled in
    lockstep (one batch). Returns crops and boxes per row."""
    import torch

    from experiments.visor.h3 import HANDS

    chains: list[dict[str, Any]] = []
    order = np.argsort(rows["t"], kind="stable")
    for right in (False, True):
        side = HANDS[1] if right else HANDS[0]
        fitted = [int(i) for i in order if bool(rows["right"][i]) == right and clip.class_mask(int(rows["t"][i]), side).any()]
        refs = [i for i in fitted if int(rows["t"][i]) in ref_frames]
        mine = [i for i in fitted if int(rows["t"][i]) in frames]
        if not refs or not mine:
            continue
        r0 = refs[0]
        t0 = int(rows["t"][r0])
        mask0 = clip.class_mask(t0, side)
        box0 = k6bh.crop_box(rows["uncoded"][r0], mask0)
        latent = fh.encode(k6bh.crop_pixels(np.asarray(truth[t0]), box0), seed * 2 + int(right))
        src = fh.cond(latent, k6bh.slots(k6bh.to_crop(rows["uncoded"][r0], box0), right), k6bh.crop_pixels(mask0, box0))
        boxes = {i: k6bh.crop_box(rows["coded"][i], clip.class_mask(int(rows["t"][i]), side)) for i in mine}
        targets = {i: fh.cond(None, k6bh.slots(k6bh.to_crop(rows["coded"][i], boxes[i]), right), None) for i in mine}
        chains.append({"src": src, "rows": mine, "boxes": boxes, "targets": targets, "prev": None})
    crops: dict[int, np.ndarray] = {}
    boxes_out: dict[int, tuple[int, int, int]] = {}
    for c in chains:
        boxes_out.update(c["boxes"])
    for k in range(max((len(c["rows"]) for c in chains), default=0)):
        live = [c for c in chains if k < len(c["rows"])]
        target = torch.cat([c["targets"][c["rows"][k]] for c in live])
        refs = [torch.cat([c["src"] for c in live])]
        if all(c["prev"] is not None for c in live):
            refs.append(torch.cat([c["prev"] for c in live]))
        samples, images = fh.sample(target, None, refs, seed * 100003 + k)
        for j, c in enumerate(live):
            i = c["rows"][k]
            crops[i] = images[j]
            c["prev"] = torch.cat([samples[j:j + 1], c["targets"][i]], 1)  # the demo's last_N_frames = 1
    return crops, boxes_out


def compose_gen(truth: Any, clip: Any, rows: dict[str, np.ndarray], crops: dict[int, np.ndarray], boxes: dict[int, tuple[int, int, int]],
                frames: range, out: Any) -> np.ndarray:
    """Rung 2's composite: Telea in R, each generated crop pasted on its side's mask (smaller hands first), objects kept.
    Frames outside ``frames`` are the source. Returns R's drawn pixels per frame."""
    from experiments.tennis.k6 import telea_fill
    from experiments.visor.h3 import HANDS

    size = np.ptp(rows["coded"][:, :, 1], axis=1) + np.ptp(rows["coded"][:, :, 0], axis=1)
    at: dict[int, list[int]] = {}
    for i in crops:
        at.setdefault(int(rows["t"][i]), []).append(i)
    covered = np.zeros(len(out), np.int64)
    for t in range(len(out)):
        truth_t = np.asarray(truth[t])
        region = clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1])
        if t not in frames or not region.any():
            out[t] = truth_t
            continue
        canvas = telea_fill(truth_t, region)
        drawn = np.zeros(region.shape, bool)
        for i in sorted(at.get(t, []), key=lambda i: size[i]):
            mask = clip.class_mask(t, HANDS[1] if rows["right"][i] else HANDS[0])
            drawn |= k6bh.paste(canvas, crops[i], boxes[i], mask)
        objects = clip.foreground(t) & ~region
        over = drawn & objects
        canvas[over] = truth_t[over]
        out[t] = canvas
        covered[t] = int((drawn & region).sum())
    return covered


def judge(fh: k6bh.FoundHand, nets: dict[str, Any], rows: dict[str, np.ndarray], truth: Any, clip: Any, ref_frames: range,
          frames: range, seed: int, work: Path, device: str) -> dict[str, Any]:
    """``gen`` rendered, composited and scored on R over ``frames``."""
    from experiments.visor.h3 import HANDS
    from experiments.visor.h3c import Scorer
    from experiments.visor.k1f import release

    began = time.time()
    crops, boxes = render_gen(fh, rows, truth, clip, ref_frames, frames, seed)
    render_seconds = time.time() - began
    n = len(truth)
    out = np.lib.format.open_memmap(work / "gen.npy", mode="w+", dtype=np.uint8, shape=truth.shape)
    covered = compose_gen(truth, clip, rows, crops, boxes, frames, out)
    regions = [(clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1])) if t in frames else np.zeros(truth.shape[1:3], bool)
               for t in range(n)]
    score = Scorer(nets, truth, regions, device)
    s = score(out, upscale=False)
    release()
    hand = np.array(score.hand, int)
    pixels = np.array([int(r.sum()) for r in regions], np.int64)
    std = [float(np.std(c)) for c in crops.values()]
    return {"lpips": s["lpips"], "dists": s["dists"], "sse": s["sse"], "covered": covered, "pixels_r": pixels, "out": out,
            "score": score, "regions": regions,
            "summary": {"lpips_r": float(np.nanmean(s["lpips"][hand])), "dists_r": float(np.nanmean(s["dists"][hand])),
                        "flicker": flicker(lambda t: score.composite(out, t, False), regions, score.hand),
                        "flicker_source": flicker(lambda t: np.asarray(truth[t]), regions, score.hand),
                        "covered_share": float(covered[hand].sum() / max(1, pixels[hand].sum())), "hand_frames": int(len(hand)),
                        "crops": len(crops), "generated_std_min": min(std) if std else None, "render_seconds": round(render_seconds, 2)}}


def save_looks(publish: Path, item_id: str, truth: Any, gen: Any, warp: Any, regions: list[np.ndarray], hand: list[int]) -> list[int]:
    """The visual check: at evenly spaced hand frames, source | fitted gen | K1f's warp around R (+-64 px), as PNGs."""
    import cv2

    looks = look_frames(hand)
    folder = publish / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    height, width = regions[0].shape
    for t in looks:
        ys, xs = np.nonzero(regions[t])
        box = (slice(max(0, ys.min() - 64), min(height, ys.max() + 65)), slice(max(0, xs.min() - 64), min(width, xs.max() + 65)))
        strip = np.concatenate([np.asarray(truth[t])[box], np.asarray(gen[t])[box], np.asarray(warp[t])[box]], 1)
        cv2.imwrite(str(folder / f"{item_id}-f{t:03d}-source-gen-warp.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return looks


# ----------------------------------------------------------------- run (GPU)


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any],
                 k1f_unit: tuple[dict[str, Any], dict[str, np.ndarray]], device: str, publish: Path, scratch: Path) -> None:
    from experiments.visor.b2 import load_mask_sets
    from experiments.visor.h2 import save_unit
    from experiments.visor.h3 import FILL, HANDS, rgb_memmap
    from experiments.visor.h3c import Scorer

    began = time.time()
    frames = int(args.frames)
    work = scratch / "work" / item["id"]
    work.mkdir(parents=True, exist_ok=True)
    clips, mask_info = load_mask_sets(item, frames, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
    clip = clips[FILL]
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, decoded["width"], decoded["height"], decoded["full_range"], device)
    meta_k, a_k = k1f_unit
    ones = np.ones((len(a_k["row_t"]), 21, 1))
    rows = {"t": np.asarray(a_k["row_t"]).astype(int), "right": np.asarray(a_k["row_right"]).astype(bool),
            "uncoded": np.concatenate([a_k["uncoded_kp2d"], ones], -1), "coded": np.concatenate([a_k["coded_kp2d"], ones], -1)}
    pilot = args.mode == "pilot"
    eval_n = int(args.eval_frames)
    fit_frames = range(0, EVAL_FROM) if pilot else range(0, frames)
    eval_frames = range(EVAL_FROM, min(frames, EVAL_FROM + eval_n)) if pilot else range(0, min(frames, eval_n))
    fh = nets["foundhand"]
    fh.model.load_state_dict(nets["foundhand_state"])  # each item starts from the released EMA weights
    crops = hand_crops(rows, truth, clip, fit_frames, "uncoded")
    seed = k6bh.item_seed(item["id"])
    fitter = Fitter(fh, crops, device, TRAIN_SEED + seed)
    probe = make_probe(fitter, seed)
    marks = list(FRACTIONS) if pilot else [0.0, 1.0]
    curve: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    last: dict[str, Any] = {}  # the latest judgement: its frames, for the visual check

    def on_mark(fraction: float, steps: int, trained: float) -> None:
        point: dict[str, Any] = {"fraction": fraction, "steps": steps, "trained_seconds": round(trained, 2),
                                 "sample_lpips_train": sample_probe(fitter, probe, nets["lpips"], seed)}
        if (pilot and fraction in JUDGED) or fraction == 1.0:
            j = judge(fh, nets, rows, truth, clip, fit_frames, eval_frames, seed, work, device)
            point.update(j["summary"])
            last.clear()
            last.update(j)
            key = f"gen_{int(round(100 * fraction)):03d}"
            for k in ("lpips", "dists", "sse", "covered"):
                arrays[f"{key}_{k}_r"] = j[k]
            arrays["pixels_r"] = j["pixels_r"]
        curve.append(point)

    trace = fit(fitter, nets["diffusion"], float(args.fit_seconds), marks, on_mark, probe)
    summary: dict[str, Any] = {"gen": {k: curve[-1][k] for k in ("lpips_r", "dists_r", "covered_share", "flicker")},
                               "source": {"flicker": curve[-1]["flicker_source"]}}
    looks: list[int] = []
    if not pilot:  # K1f's bound (the check, at its 240 frames) and the visual check's frames
        warp = np.lib.format.open_memmap(work / "warp.npy", mode="w+", dtype=np.uint8, shape=truth.shape)
        rows_at: dict[int, list[int]] = {}
        for i, t in enumerate(rows["t"]):
            rows_at.setdefault(int(t), []).append(i)
        size = np.ptp(rows["coded"][:, :, 1], axis=1) + np.ptp(rows["coded"][:, :, 0], axis=1)
        regions = [clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1]) for t in range(frames)]
        for t in range(frames):
            truth_t = np.asarray(truth[t])
            if not regions[t].any():
                warp[t] = truth_t
                continue
            here = sorted(rows_at.get(t, []), key=lambda i: size[i])
            warp[t], _ = k1c.compose(truth_t, {s: clip.class_mask(t, s) for s in HANDS}, clip.foreground(t) & ~regions[t],
                                     [(bool(rows["right"][i]), rows["uncoded"][i], rows["coded"][i], -float(size[i])) for i in here])
        score = Scorer(nets, truth, regions, device)
        s = score(warp, upscale=False)
        hand = np.array(score.hand, int)
        summary["warp"] = {"lpips_r": float(np.nanmean(s["lpips"][hand])), "dists_r": float(np.nanmean(s["dists"][hand]))}
        arrays["warp_lpips_r"], arrays["warp_dists_r"] = s["lpips"], s["dists"]
        summary["warp"]["flicker"] = flicker(lambda t: score.composite(warp, t, False), regions, score.hand)
        looks = save_looks(publish, item["id"], truth, last["out"], warp, last["regions"], last["score"].hand)
    arrays["loss_trace"] = np.array(trace["trace"], float).reshape(-1, 2)
    meta = {"unit": item["id"], "item": item["id"], "video": item["video"], "mode": args.mode, "frames": frames,
            "fit_frames": [fit_frames.start, fit_frames.stop], "eval_frames": [eval_frames.start, eval_frames.stop],
            "crops": len(crops), "fit_seconds": float(args.fit_seconds), "look_frames": looks, "steps": trace["steps"],
            "trained_seconds": round(trace["trained_seconds"], 2), "fixed_loss": trace["fixed"], "curve": curve, "summary": summary,
            "k1f": {"kbps": float(meta_k["cover"]["kbps"]) if meta_k.get("cover") else k1c.POSE_KBPS, "warp": meta_k["summary"]["warp"],
                    "frames": int(meta_k["frames"]), "npz_hash_ok": bool(meta_k["npz_hash_ok"])},
            "mask_set": mask_info, "seconds": round(time.time() - began, 2)}
    save_unit(publish, "k6bh3", item["id"], arrays, meta)
    os.remove(decoded["path"])
    for f in work.glob("*.npy"):
        f.unlink()


def command_run(args: argparse.Namespace) -> int:
    from experiments.visor.b2 import select_items
    from experiments.visor.h2 import restore_units
    from experiments.visor.h3c import decode_windows

    items = select_items(json.loads(Path(args.eval_set).read_text()), args.items)
    units = k6bh.k1f_rows(args.k1f_stage)
    if missing := [it["id"] for it in items if it["id"] not in units]:
        raise SystemExit(f"no K1f rows for {missing}")
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "k6bh3")
    todo = [it for it in items if it["id"] not in restored]
    windows = decode_windows(args, todo, int(args.frames), scratch)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    from experiments.audit.env_smoke import device_record, model_on_cuda
    from experiments.visor.h3b import load_dists
    from src.codecs.quality import load_lpips, lpips_record

    if not torch.cuda.is_available():
        raise SystemExit("k6bh3 run needs a GPU")
    device = "cuda"
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_CAP_MIB / total_mib))
    nets: dict[str, Any] = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    fh = k6bh.FoundHand(Path(args.foundhand_src), Path(args.foundhand_ckpt), Path(args.vae), int(args.steps), device, scratch)
    nets["foundhand"] = fh
    nets["foundhand_state"] = {k: v.detach().clone().cpu() for k, v in fh.model.state_dict().items()}
    import sys

    nets["diffusion"] = sys.modules["foundhand_diffusion"].create_diffusion("")  # 1000 training timesteps
    on_cuda = {"lpips": next(nets["lpips"].parameters()).is_cuda, "dists": next(nets["dists"].parameters()).is_cuda,
               "foundhand": model_on_cuda(fh.model), "vae": model_on_cuda(fh.vae)}
    began = time.time()
    done = len(restored)
    for item, decoded in windows:
        process_item(args, item, decoded, nets, units[item["id"]], device, publish, scratch)
        done += 1
        progress(done)
    import torchmetrics

    write_json(stage_dir() / "k6bh3.json", {
        "command": "run", "device": device_record(args.gpu_class), "on_cuda": on_cuda, "seconds": round(time.time() - began, 2),
        "foundhand": fh.record, "gpu_cap_mib": GPU_CAP_MIB, "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
        "inputs": {"foundhand_src": file_sha256(Path(args.foundhand_src)), "foundhand_ckpt": file_sha256(Path(args.foundhand_ckpt)),
                   "vae": file_sha256(Path(args.vae)), "k1f_stages": [file_sha256(Path(t)) for t in args.k1f_stage]},
        "lpips": lpips_record(Path(args.lpips_backbone)),
        "dists": {"torchmetrics": torchmetrics.__version__, "backbone_sha256": file_sha256(Path(args.dists_backbone))},
        "settings": {"mode": args.mode, "frames": int(args.frames), "items": [it["id"] for it in items], "fit_seconds": float(args.fit_seconds),
                     "eval_frames": int(args.eval_frames), "steps": int(args.steps), "batch": BATCH, "lr": LR, "p_uncond": P_UNCOND,
                     "p_no_ref_mask": P_NO_REF_MASK, "fractions": FRACTIONS, "decision": DECISION}, "restored_units": sorted(restored)})
    return 0


# ----------------------------------------------------------------- validation (CPU)


def validate(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    from experiments.visor.h2 import load_units

    units = {m["item"]: (m, a) for m, a in load_units([str(stage / "published.tar")], "k6bh3")}
    settings = result["settings"]
    checks = {"every_item": sorted(units) == sorted(settings["items"]), "on_cuda": all(result["on_cuda"].values()),
              "device_class": bool(result["device"]["class_matches"]), "foundhand_loaded": result["foundhand"]["params"] > 6e8}
    for item, (m, a) in units.items():
        fixed = [v for _, v in m["fixed_loss"]]
        samples = [p["sample_lpips_train"] for p in m["curve"]]
        checks[f"{item}:hash"] = bool(m["npz_hash_ok"]) and bool(m["k1f"]["npz_hash_ok"])
        checks[f"{item}:steps"] = m["steps"] >= 200  # the rule's smoke: 200 steps
        checks[f"{item}:loss_falls"] = len(fixed) >= 2 and fixed[-1] < fixed[0]
        checks[f"{item}:sample_moves_to_target"] = len(samples) >= 2 and samples[-1] < samples[0]
        checks[f"{item}:loss_finite"] = bool(np.isfinite(a["loss_trace"]).all()) and len(a["loss_trace"]) > 0
        judged = [p for p in m["curve"] if "lpips_r" in p]
        checks[f"{item}:judged"] = bool(judged) and all(np.isfinite(p["lpips_r"]) and np.isfinite(p["dists_r"]) and p["covered_share"] > 0.5
                                                        and (p["generated_std_min"] or 0) > 2.0 for p in judged)
        if m["mode"] == "pilot":
            checks[f"{item}:curve_points"] = [p["fraction"] for p in judged] == list(JUDGED)
            checks[f"{item}:held_out"] = m["fit_frames"][1] <= m["eval_frames"][0]
        else:
            checks[f"{item}:looks_saved"] = len(m.get("look_frames", [])) > 0 and saved_looks(stage, item) == len(m["look_frames"])
            checks[f"{item}:flicker"] = bool(np.isfinite(m["summary"]["gen"]["flicker"]) and np.isfinite(m["summary"]["source"]["flicker"]))
        if m["mode"] == "fit" and settings["frames"] == m["k1f"]["frames"]:
            checks[f"{item}:warp_reproduces_k1f"] = abs(m["summary"]["warp"]["lpips_r"] - m["k1f"]["warp"]["lpips_r"]) <= k6bh.WARP_TOLERANCE
    return checks


def saved_looks(stage: Path, item: str) -> int:
    import tarfile

    with tarfile.open(stage / "published.tar") as tar:
        return sum(1 for n in tar.getnames() if n.startswith(f"publish/frames/{item}-f") and n.endswith(".png"))


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage) if args.stage else stage_dir()
    result = json.loads((stage / "k6bh3.json").read_text())
    checks = {k: bool(v) for k, v in validate(stage, result).items()}
    report = {"passed": all(checks.values()), "checks": checks}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def command_report(args: argparse.Namespace) -> int:
    from experiments.background.g5 import envelope
    from experiments.visor.h2 import load_units
    from experiments.visor.k1f import G_VIDEOS
    from experiments.visor.k6bh2c import won

    hulls = {m["item"]: m for m, _ in load_units(args.k1f_stage, "k1f")}
    out: dict[str, Any] = {"decision_rule": DECISION, "inputs": {"k1f_stages": [file_sha256(Path(t)) for t in args.k1f_stage],
                                                                 "code_sha256": file_sha256(Path(__file__))}}
    for stage in map(Path, args.pilot):
        result = json.loads((stage / "k6bh3.json").read_text())
        checks = validate(stage, result)
        if not all(checks.values()) or result["settings"]["mode"] != "pilot" or result["settings"]["fit_seconds"] < 1800:
            raise SystemExit(f"{stage} is not a valid full pilot: {[k for k, v in checks.items() if not v]}")
        for m, _ in load_units([str(stage / "published.tar")], "k6bh3"):
            kbps = m["k1f"]["kbps"]
            hull = {mt: envelope([(p["kbps"], -p[mt]) for p in hulls[m["item"]]["points"]]) for mt in ("lpips", "dists")}
            bar = {mt: won(0.0, hull[mt], kbps)[1] for mt in ("lpips", "dists")}
            fitted = [p for p in m["curve"] if p["fraction"] > 0]
            secs = [p["trained_seconds"] for p in fitted]
            reach: dict[str, float | None] = {}
            for mt in ("lpips", "dists"):
                level = bar[mt]
                if level is not None:
                    reach[mt] = minutes_to_reach(secs, [p[f"{mt}_r"] for p in fitted], level)
            out["pilot"] = {"item": m["item"], "stage": str(stage), "sha256": file_sha256(stage / "published.tar"), "kbps": kbps,
                            "hull_at_rate": bar, "curve": m["curve"], "fixed_loss": m["fixed_loss"],
                            "log_time_fit": {mt: log_time_fit(secs, [p[f"{mt}_r"] for p in fitted]) for mt in ("lpips", "dists")},
                            "minutes_to_reach_hull": reach,
                            "go": (reach.get("lpips") or float("inf")) <= HULL_MINUTES}
    rows = []
    for stage in map(Path, args.stage):
        result = json.loads((stage / "k6bh3.json").read_text())
        checks = validate(stage, result)
        if not all(checks.values()) or result["settings"]["mode"] != "fit" or result["settings"]["frames"] != 240:
            raise SystemExit(f"{stage} is not a valid full fit: {[k for k, v in checks.items() if not v]}")
        for m, _ in load_units([str(stage / "published.tar")], "k6bh3"):
            kbps = m["k1f"]["kbps"]
            row: dict[str, Any] = {"item": m["item"], "kbps": kbps, "g_overlap": m["video"] in G_VIDEOS, "steps": m["steps"],
                                   "trained_seconds": m["trained_seconds"], "stage": str(stage), "sha256": file_sha256(stage / "published.tar")}
            for mt in ("lpips", "dists"):
                item_hull = envelope([(p["kbps"], -p[mt]) for p in hulls[m["item"]]["points"]])
                value = m["summary"]["gen"][f"{mt}_r"]
                ok, at, _ = won(value, item_hull, kbps)
                row.update({mt: value, f"{mt}_hull": at, f"{mt}_won": ok, f"warp_{mt}": m["summary"]["warp"][f"{mt}_r"]})
            rows.append(row)
    if rows:
        lp = float(np.mean([r["lpips_won"] for r in rows]))
        di = float(np.mean([r["dists_won"] for r in rows]))
        out["rows"] = rows
        out["score"] = {"items": len(rows), "lpips_share": lp, "dists_share": di, "score": (lp + di) / 2, "passes": (lp + di) / 2 >= CRITERION}
    write_json(Path(args.out), out)
    print(json.dumps({k: v for k, v in out.items() if k != "decision_rule"}, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for flag in ("--eval-set", "--archive", "--masks", "--mask-record", "--items", "--lpips-backbone", "--dists-backbone",
                 "--foundhand-src", "--foundhand-ckpt", "--vae"):
        run.add_argument(flag, required=True)
    run.add_argument("--mode", choices=("pilot", "fit"), required=True)
    run.add_argument("--video", action="append", default=[])
    run.add_argument("--k1f-stage", action="append", required=True)
    run.add_argument("--frames", default="240")
    run.add_argument("--fit-seconds", default="1800", help="seconds of training (judging excluded)")
    run.add_argument("--eval-frames", default="30", help="frames judged (pilot: from 180)")
    run.add_argument("--steps", default="50", help="FoundHand's sampling steps (rung 2's)")
    run.add_argument("--gpu-class", default="RTX 6000 Ada")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default="")
    rep = sub.add_parser("report")
    rep.add_argument("--pilot", action="append", default=[])
    rep.add_argument("--stage", action="append", default=[])
    rep.add_argument("--k1f-stage", action="append", required=True)
    rep.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
