"""K1c: K6's bound on egocentric hands, the outline granted (docs/experiments.md, 2026-10-21 K1c).

    python -m experiments.visor.k1c run --eval-set JSON --archive DIR --masks DIR --mask-record JSON --video MP4 ... \\
        --items IDS --frames N --lpips-backbone PTH --dists-backbone PTH --poses TAR --poses TAR --pose-code JSON \\
        --mano-right PKL
    python -m experiments.visor.k1c validate
    python -m experiments.visor.k1c report --stage DIR --h3d H3D-COMPONENT-REPORT.json --out JSON

Per pilot item: the window and its fill masks (H3c's loading), H2's WiLoR rows coded at the 100 ms choice and
projected (as ``h3c.item_judge``), and K6's skeleton warp (``k6.draw``) with the 21-joint hand skeleton, moving each
fitted hand's own pixels (its side's class mask) from the uncoded to the coded joints over a Telea fill of R. Scored
on R exactly as H3c and H3d scored their points (``h3c.Scorer``), and over S (the drawing kept outside R).
``report`` compares each item's bound with the per-item hull of SVT-AV1 + DCVC-UF (H3d's bar) and DCVC-UF + G.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

#: OpenPose / MANO 21-joint hand: wrist (0) to each finger's base, then along the finger (thumb 1-4 ... pinky 17-20).
HAND_SKELETON: tuple[tuple[int, int], ...] = tuple(
    pair for base in (1, 5, 9, 13, 17) for pair in ((0, base), (base, base + 1), (base + 1, base + 2), (base + 2, base + 3)))
SIGMA_OF_HEIGHT = 0.08  # K6's constant, of the moved mask's height
BUDGET = "100"  # H2's choice named for H3 (1.93 kbps)
POSE_KBPS = 1.9270436424939223  # K1's recorded rate of that choice (H2's pooled VISOR rate)
VARIANTS = ("warp", "warp@uncoded")
DECISION: dict[str, Any] = {
    "metric": "LPIPS on R (h3c.Scorer), mean over the window's frames with R non-empty",
    "envelope": "per item, g5.envelope of H3d's bar (SVT-AV1's eleven points, DCVC-UF off the shelf) and DCVC-UF + G "
                "(component arm G-L), on R; K1's joint hull",
    "pass_item": "warp's LPIPS on R below the hull's LPIPS at the keypoint rate",
    "pass": "at least 3 of 4 items pass: hands get K6b's ladder; otherwise hands are closed for every renderer",
    "to_user": "LPIPS over S or DISTS on R moves the count of passing items across 3",
    "min_items": 3,
}
#: K1d (docs/experiments.md, 2026-10-21): K1c's rule with every masked hand fitted.
DECISION_K1D: dict[str, Any] = {
    **DECISION,
    "change": "WiLoR on the fill-set mask's box for every frame and side with a mask (box sides >= 16 px) and no H2 row; "
              "the item's WiLoR tracks coded again with H2's 100 ms choice",
    "rate": "H2's pooled rate x coded rows after / before, per item; the hull is read at each item's rate",
}
MIN_BOX = 16  # px, the shorter side of a mask box WiLoR is run on
WILOR_BATCH = 32
ROW_KEYS = ("t", "right", "box", "wilor_rotvec", "wilor_betas", "wilor_transl", "wilor_kp2d")


# ----------------------------------------------------------------- run (GPU for the judges)


def item_keypoints(args: argparse.Namespace, item_id: str, frames: int) -> dict[str, Any]:
    """H2's WiLoR rows in the window: frame, side, projected uncoded and coded joints (21, 3; confidence 1), depth."""
    from experiments.visor.h2 import CodingSet, pinhole
    from experiments.visor.h3b import item_poses

    meta, a = args.pose_units[item_id]
    focal, centre = float(meta["focal"]), tuple(meta["principal"])
    n = int(meta["hands"])
    coded = item_poses(meta, a, args.mano, args.combos, args.basis)["wilor"]["coded_joints"]
    data = CodingSet("visor", "wilor", [(meta, a)], args.mano)
    uncoded = args.mano.joints(data.rot, data.betas, data.transl, data.left) if n else np.zeros((0, 21, 3))

    def with_conf(j3: np.ndarray) -> np.ndarray:
        return np.concatenate([pinhole(j3, focal, centre), np.ones((*j3.shape[:-1], 1))], -1)

    rows = [r for r in range(n) if int(a["t"][r]) < frames]
    return {"rows": rows, "t": np.asarray(a["t"])[rows].astype(int), "right": np.asarray(a["right"])[rows].astype(bool),
            "uncoded": with_conf(uncoded[rows]) if rows else np.zeros((0, 21, 3)),
            "coded": with_conf(coded[rows]) if rows else np.zeros((0, 21, 3)),
            "depth": coded[rows][:, :, 2].mean(1) if rows else np.zeros(0),
            "wilor_kp2d": np.asarray(a["wilor_kp2d"])[rows] if rows else np.zeros((0, 21, 2))}


def mask_box_rows(wilor: Any, truth: Any, clip: Any, frames: int, meta: dict[str, Any], a: dict[str, np.ndarray],
                  hand_box: bool = False) -> dict[str, np.ndarray]:
    """K1d: WiLoR on the fill-set mask's box for every frame and side with a mask and no H2 row, as ``h2.run_visor``
    stores its rows (rotation, shape, full-frame translation at H2's focal, projected joints)."""
    from experiments.visor import mano_np
    from experiments.visor.h2 import full_translation, pinhole
    from experiments.visor.h3 import HANDS

    have = {(int(t), bool(r)) for t, r in zip(a["t"], a["right"])}
    spec: list[tuple[int, bool, np.ndarray]] = []
    for t in range(frames):
        for right, side in ((False, HANDS[0]), (True, HANDS[1])):
            if (t, right) in have:
                continue
            m = clip.class_mask(t, side)
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
            if min(box[2] - box[0], box[3] - box[1]) >= MIN_BOX:
                spec.append((t, right, box))
    if not spec:
        return {k: np.asarray(a[k])[:0] for k in ROW_KEYS}
    def run(boxes: list[tuple[int, bool, np.ndarray]]) -> dict[str, np.ndarray]:
        fits: list[dict[str, np.ndarray]] = []
        for first in range(0, len(boxes), WILOR_BATCH):
            chunk = boxes[first:first + WILOR_BATCH]
            fits.append(wilor.forward([wilor.item(np.asarray(truth[t]), box, right) for t, right, box in chunk]))
        return {k: np.concatenate([f[k] for f in fits], 0) for k in fits[0]}

    fit = run(spec)
    first_kp2d = fit["kp2d"]
    if hand_box:  # K1e: cut the mask at the first fit's wrist line (H1's split) and fit again on the hand part's box
        from experiments.visor.h1 import wrist_split

        kp = fit["kp2d"]
        second = []
        for i, (t, right, box) in enumerate(spec):
            hand, _ = wrist_split(clip.class_mask(t, HANDS[1] if right else HANDS[0]), kp[i][0], kp[i][9])
            ys, xs = np.nonzero(hand)
            cut = (np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32) if len(ys) else box)
            second.append((t, right, cut if min(cut[2] - cut[0], cut[3] - cut[1]) >= MIN_BOX else box))
        spec = second
        fit = run(spec)
    height, width = (int(v) for v in meta["image_shape"][:2])
    focal, centre = float(meta["focal"]), tuple(meta["principal"])
    transl = full_translation(fit, np.array([width, height]), focal)
    return {"t": np.array([t for t, _, _ in spec], int), "right": np.array([r for _, r, _ in spec], bool),
            "box": np.stack([b for _, _, b in spec]).astype(float), "wilor_rotvec": mano_np.rotvec_of(fit["rotmats"].astype(float)),
            "wilor_betas": fit["betas"], "wilor_transl": transl,
            "wilor_kp2d": np.stack([pinhole(fit["joints"][i] + transl[i], focal, centre) for i in range(len(spec))]),
            "first_kp2d": first_kp2d}


def covered_keypoints(args: argparse.Namespace, meta: dict[str, Any], a: dict[str, np.ndarray], frames: int) -> dict[str, Any]:
    """K1d: as ``item_keypoints`` on H2's rows plus the mask-box rows, WiLoR's tracks alone coded (H3b's coding)."""
    from experiments.visor import mano_np
    from experiments.visor import pose_coding as pc
    from experiments.visor.h2 import CodingSet, pinhole

    n = int(meta["hands"])
    focal, centre = float(meta["focal"]), tuple(meta["principal"])
    data = CodingSet("visor", "wilor", [(meta, a)], args.mano)
    coded = np.zeros((n, 21, 3))
    info = data.units[0]
    for tr in data.tracks:
        symbols, sent = pc.encode(tr["x"], args.combos["wilor"], tr["fps"], args.basis)
        x = pc.decode(symbols, sent, len(tr["x"]), args.combos["wilor"], args.basis)
        rot = mano_np.rodrigues(x[:, :48].reshape(-1, 16, 3))
        transl = pc.root_transl(x[:, 48:51], info["focal"], info["centre"])
        coded[tr["rows"]] = args.mano.joints(rot, np.repeat(tr["betas"][None], len(rot), 0), transl, np.full(len(rot), tr["left"]))

    def with_conf(j3: np.ndarray) -> np.ndarray:
        return np.concatenate([pinhole(j3, focal, centre), np.ones((*j3.shape[:-1], 1))], -1)

    rows = [r for r in range(n) if int(a["t"][r]) < frames]
    return {"rows": rows, "t": np.asarray(a["t"])[rows].astype(int), "right": np.asarray(a["right"])[rows].astype(bool),
            "uncoded": with_conf(data.uncoded[rows]), "coded": with_conf(coded[rows]), "depth": coded[rows][:, :, 2].mean(1),
            "wilor_kp2d": np.asarray(a["wilor_kp2d"])[rows], "new": np.asarray(a["new"])[rows].astype(bool),
            "tracks": len(data.tracks)}


def compose(truth_t: np.ndarray, sides: dict[str, np.ndarray], objects: np.ndarray, hands: list[tuple[bool, np.ndarray, np.ndarray, float]]
            ) -> tuple[np.ndarray, np.ndarray]:
    """The warp's frame: R Telea-filled, then each fitted hand's side mask moved from its source to its target joints
    (farther first), never over objects. Returns (frame, drawn)."""
    from experiments.tennis.k6 import draw, telea_fill
    from experiments.visor.h3 import HANDS

    region = sides[HANDS[0]] | sides[HANDS[1]]
    canvas = telea_fill(truth_t, region)
    drawn = np.zeros(region.shape, bool)
    for right, source, target, _ in sorted(hands, key=lambda h: -h[3]):
        mask = sides[HANDS[1] if right else HANDS[0]]
        if not mask.any():
            continue
        ys = np.nonzero(mask.any(1))[0]
        draw(canvas, drawn, truth_t, mask, source, target, SIGMA_OF_HEIGHT * float(ys.max() - ys.min() + 1), HAND_SKELETON)
    over = drawn & objects
    canvas[over] = truth_t[over]
    drawn &= ~objects
    return canvas, drawn


def lpips_over(net: Any, frames: Any, truth: Any, masks: list[np.ndarray], ts: list[int], device: str, batch: int) -> np.ndarray:
    """LPIPS (full-frame spatial map) averaged over each frame's mask."""
    import torch

    out = np.full(len(masks), np.nan)
    for first in range(0, len(ts), batch):
        chunk = ts[first:first + batch]

        def tensor(xs: list[np.ndarray]) -> Any:
            return torch.from_numpy(np.stack(xs)).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0

        with torch.inference_mode():
            spatial = net(tensor([np.asarray(frames[t]) for t in chunk]), tensor([np.asarray(truth[t]) for t in chunk]))[:, 0]
        for i, t in enumerate(chunk):
            m = torch.from_numpy(masks[t]).to(device)
            out[t] = float((spatial[i] * m).sum()) / max(1, int(masks[t].sum()))
    return out


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any], device: str,
                 publish: Path, scratch: Path, first: bool) -> None:
    import cv2

    from experiments.visor.b2 import load_mask_sets
    from experiments.visor.h2 import save_unit
    from experiments.visor.h3 import FILL, HANDS, LPIPS_BATCH, rgb_memmap
    from experiments.visor.h3c import Scorer

    began = time.time()
    frames = int(args.frames)
    width, height, full_range = decoded["width"], decoded["height"], decoded["full_range"]
    work = scratch / "work" / item["id"]
    work.mkdir(parents=True, exist_ok=True)
    clips, mask_info = load_mask_sets(item, frames, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
    clip = clips[FILL]
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, full_range, device)
    cover: dict[str, Any] = {}
    first_new: np.ndarray | None = None
    if args.cover:
        meta_p, a_p = args.pose_units[item["id"]]
        old = {k: np.asarray(a_p[k]) for k in ROW_KEYS}
        new = mask_box_rows(nets["wilor"], truth, clip, frames, meta_p, old, hand_box=args.hand_box)
        first_new = new.pop("first_kp2d", None)
        merged = {k: np.concatenate([old[k], new[k].astype(old[k].dtype)]) for k in ROW_KEYS}
        merged["new"] = np.concatenate([np.zeros(len(old["t"]), bool), np.ones(len(new["t"]), bool)])
        kp = covered_keypoints(args, {**meta_p, "hands": len(merged["t"])}, merged, frames)
        rows_before = int((old["t"] < frames).sum())
        cover = {"rows_h2": rows_before, "rows_new": int(len(new["t"])), "tracks": kp["tracks"],
                 "kbps": POSE_KBPS * (rows_before + len(new["t"])) / max(1, rows_before)}
    else:
        kp = item_keypoints(args, item["id"], frames)
    rows_at: dict[int, list[int]] = {}
    for i, t in enumerate(kp["t"]):
        rows_at.setdefault(int(t), []).append(i)
    regions = [clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1]) for t in range(frames)]
    outs = {v: np.lib.format.open_memmap(work / f"{v.replace('@', '-')}.npy", mode="w+", dtype=np.uint8, shape=(frames, height, width, 3))
            for v in VARIANTS}
    drawn = {v: [np.zeros((height, width), bool) for _ in range(frames)] for v in VARIANTS}
    stats = {k: np.zeros(frames, np.int64) for k in ("r_px", "fitted_r_px", "unfitted_r_px", "missed", "spill", "drawn", "hands")}
    identity_max = 0
    for t in range(frames):
        truth_t = np.asarray(truth[t])
        if not regions[t].any():
            for v in VARIANTS:
                outs[v][t] = truth_t
            continue
        sides = {s: clip.class_mask(t, s) for s in HANDS}
        objects = clip.foreground(t) & ~regions[t]
        rows = rows_at.get(t, [])
        fitted_sides = {HANDS[1] if kp["right"][i] else HANDS[0] for i in rows}
        for v in VARIANTS:
            hands = [(bool(kp["right"][i]), kp["uncoded"][i], kp["uncoded" if v == "warp@uncoded" else "coded"][i], float(kp["depth"][i]))
                     for i in rows]
            frame, mine = compose(truth_t, sides, objects, hands)
            outs[v][t] = frame
            drawn[v][t] = mine
        fitted = np.zeros_like(regions[t])
        for s in fitted_sides:
            fitted |= sides[s]
        identity_max = max(identity_max, int(np.abs(outs["warp@uncoded"][t][fitted & drawn["warp@uncoded"][t]].astype(int)
                                                     - truth_t[fitted & drawn["warp@uncoded"][t]].astype(int)).max(initial=0)))
        d = drawn["warp"][t]
        stats["r_px"][t] = int(regions[t].sum())
        stats["fitted_r_px"][t] = int(fitted.sum())
        stats["unfitted_r_px"][t] = int((regions[t] & ~fitted).sum())
        stats["missed"][t] = int((fitted & ~d).sum())
        stats["spill"][t] = int((d & ~regions[t]).sum())
        stats["drawn"][t] = int(d.sum())
        stats["hands"][t] = len(rows)
    for v in VARIANTS:
        outs[v].flush()
    score = Scorer(nets, truth, regions, device)
    arrays: dict[str, np.ndarray] = {"pixels_r": np.array([int(r.sum()) for r in regions], np.int64), **stats}
    s_masks = [regions[t] | drawn["warp"][t] | drawn["warp@uncoded"][t] for t in range(frames)]
    for v in VARIANTS:
        scored = score(outs[v], upscale=False)
        key = v.replace("@", "_")
        arrays[f"{key}_lpips_r"], arrays[f"{key}_sse_r"], arrays[f"{key}_dists_r"] = scored["lpips"], scored["sse"], scored["dists"]
        arrays[f"{key}_lpips_s"] = lpips_over(nets["lpips"], outs[v], truth, s_masks, score.hand, device, LPIPS_BATCH)
    # what the coded joints moved: mean 2D distance to the uncoded and to WiLoR's own 2D joints (px)
    arrays["coded_shift_px"] = np.linalg.norm(kp["coded"][:, :, :2] - kp["uncoded"][:, :, :2], axis=-1).mean(1)
    arrays["uncoded_vs_wilor_px"] = np.linalg.norm(kp["uncoded"][:, :, :2] - kp["wilor_kp2d"], axis=-1).mean(1)
    if args.cover:
        arrays["new_row"] = kp["new"]
        arrays["row_t"], arrays["row_right"] = kp["t"], kp["right"]
        arrays["uncoded_kp2d"], arrays["coded_kp2d"] = kp["uncoded"][:, :, :2].astype(np.float32), kp["coded"][:, :, :2].astype(np.float32)
        if first_new is not None and len(first_new):  # the mask-box fits (K1d's) in place of the new rows' second fits
            arrays["first_kp2d"] = arrays["uncoded_kp2d"].copy()
            arrays["first_kp2d"][kp["new"]] = first_new.astype(np.float32)
    look = score.hand[len(score.hand) // 2] if score.hand else 0
    if score.hand:
        ys, xs = np.nonzero(s_masks[look])
        box = (slice(max(0, ys.min() - 64), min(height, ys.max() + 65)), slice(max(0, xs.min() - 64), min(width, xs.max() + 65)))
        row = np.concatenate([np.asarray(truth[look])[box], outs["warp"][look][box], outs["warp@uncoded"][look][box]], 1)
        frames_dir = publish / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frames_dir / f"{item['id']}-f{look:03d}-source-warp-uncoded.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    checks: dict[str, Any] = {"identity_max_abs": identity_max}
    if first and score.hand:
        from experiments.visor.h3b import dists_crops
        t = score.hand[0]
        checks["dists_identity"] = dists_crops(nets["dists"], [np.asarray(truth[t])], np.asarray(truth[t]), regions[t], device)[0]
        checks["lpips_identity"] = float(lpips_over(nets["lpips"], truth, truth, regions, [t], device, 1)[t])
    hand = np.array(score.hand, int)
    summary = {v: {"lpips_r": float(np.nanmean(arrays[f"{v.replace('@', '_')}_lpips_r"][hand])),
                   "lpips_s": float(np.nanmean(arrays[f"{v.replace('@', '_')}_lpips_s"][hand])),
                   "dists_r": float(np.nanmean(arrays[f"{v.replace('@', '_')}_dists_r"][hand]))} for v in VARIANTS}
    meta = {"unit": item["id"], "item": item["id"], "video": item["video"], "fps": float(item["fps"]), "frames": frames,
            "hand_frames": len(score.hand), "rows": len(kp["rows"]), "look_frame": look, "summary": summary, "cover": cover or None,
            "decode": {k: v for k, v in decoded.items() if k != "path"}, "mask_set": mask_info, "checks": checks,
            "pose_unit_hash_ok": bool(args.pose_units[item["id"]][0]["npz_hash_ok"]), "seconds": round(time.time() - began, 2)}
    save_unit(publish, "k1c", item["id"], arrays, meta)
    del outs, truth
    os.remove(decoded["path"])
    for f in work.glob("*.npy"):
        f.unlink()


def command_run(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record
    from experiments.visor import mano_np
    from experiments.visor import pose_coding as pc
    from experiments.visor.b2 import select_items
    from experiments.visor.h2 import load_units, restore_units
    from experiments.visor.h3b import load_dists
    from experiments.visor.h3c import combo_at, decode_windows
    from src.codecs.quality import load_lpips, lpips_record

    items = select_items(json.loads(Path(args.eval_set).read_text()), args.items)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "k1c")
    todo = [it for it in items if it["id"] not in restored]
    windows = decode_windows(args, todo, int(args.frames), scratch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not args.allow_cpu:
        raise SystemExit("k1c run expects a GPU for the judges (--allow-cpu for a local check)")
    code = json.loads(Path(args.pose_code).read_text())
    kbps = float(code["choices"]["wilor"][BUDGET]["visor_kbps"])
    combo = combo_at(code, "wilor", BUDGET)
    args.combos = {"wilor": combo, "hamer": combo}  # only WiLoR's joints are read (as h3c.item_judge)
    args.mano = mano_np.Mano.load(args.mano_right)
    args.basis = pc.Basis(args.mano.hands_mean, args.mano.hands_components)
    args.pose_units = {meta["unit"]: (meta, a) for meta, a in load_units(args.poses, "visor")}
    if missing := [it["id"] for it in items if it["id"] not in args.pose_units]:
        raise SystemExit(f"no H2 poses for {missing}")
    nets: dict[str, Any] = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    on_cuda = {k: next(v.parameters()).is_cuda for k, v in nets.items()}
    if args.cover:  # K1d: WiLoR on the mask boxes H2 has no row for
        from experiments.audit.env_smoke import model_on_cuda
        from experiments.visor.h2 import Regressor, mano_dir

        args.mano_dir = mano_dir(args, scratch)
        nets["wilor"] = Regressor("wilor", args)
        on_cuda["wilor"] = model_on_cuda(nets["wilor"].model)
    began = time.time()
    done = len(restored)
    for number, (item, decoded) in enumerate(windows):
        process_item(args, item, decoded, nets, device, publish, scratch, first=number == 0)
        done += 1
        progress(done)
    import torchmetrics

    write_json(stage_dir() / "k1c.json", {
        "command": "run", "device": device_record(torch.cuda.get_device_name(0)) if device == "cuda" else {"device": "cpu"},
        "on_cuda": on_cuda, "seconds": round(time.time() - began, 2), "lpips": lpips_record(Path(args.lpips_backbone)),
        "dists": {"torchmetrics": torchmetrics.__version__, "backbone_sha256": file_sha256(Path(args.dists_backbone))},
        "pose": {"budget_ms": BUDGET, "combo": pc.combo_key(combo), "visor_kbps": kbps, "code_sha256": file_sha256(Path(args.pose_code)),
                 "poses_sha256": [file_sha256(Path(x)) for x in args.poses], "mano_sha256": file_sha256(Path(args.mano_right))},
        "settings": {"frames": int(args.frames), "items": [it["id"] for it in items], "skeleton": HAND_SKELETON,
                     "sigma_of_height": SIGMA_OF_HEIGHT,
                     "cover": bool(args.cover), "hand_box": bool(args.hand_box), "decision": DECISION_K1D if args.cover else DECISION,
                     "wilor": nets["wilor"].load if args.cover else None},
        "restored_units": sorted(restored)})
    return 0


# ----------------------------------------------------------------- validation (CPU)


def validate(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    from experiments.visor.h2 import load_units

    units = load_units([str(stage / "published.tar")], "k1c")
    by_item = {m["item"]: (m, a) for m, a in units}
    checks: dict[str, bool] = {
        "every_item": sorted(by_item) == sorted(result["settings"]["items"]),
        "on_cuda": all(result["on_cuda"].values()),
        "pose_kbps": abs(result["pose"]["visor_kbps"] - POSE_KBPS) < 1e-6,
        "pose_hashes": all(m["pose_unit_hash_ok"] for m, _ in units),
        "skeleton": [list(p) for p in result["settings"]["skeleton"]] == [list(p) for p in HAND_SKELETON] and len(HAND_SKELETON) == 20,
    }
    for item, (m, a) in by_item.items():
        hand = a["pixels_r"] > 0
        checks[f"{item}:frames"] = len(a["pixels_r"]) == m["frames"] == result["settings"]["frames"]
        checks[f"{item}:hand_frames"] = int(hand.sum()) == m["hand_frames"] > 0
        checks[f"{item}:fits"] = m["rows"] > 0 and int(a["hands"].sum()) > 0
        checks[f"{item}:identity"] = m["checks"]["identity_max_abs"] <= 1  # bilinear at integer positions: the source
        checks[f"{item}:scores_finite"] = all(bool(np.isfinite(a[f"{v}_lpips_r"][hand]).all() and np.isfinite(a[f"{v}_lpips_s"][hand]).all())
                                              for v in ("warp", "warp_uncoded"))
        checks[f"{item}:scores_off_hand_nan"] = bool(np.isnan(a["warp_lpips_r"][~hand]).all())
        checks[f"{item}:summary"] = all(abs(m["summary"][v]["lpips_r"] - float(np.nanmean(a[f"{v.replace('@', '_')}_lpips_r"][hand]))) < 1e-9
                                        for v in VARIANTS)
        checks[f"{item}:coding_moves"] = 1.0 < float(np.mean(a["coded_shift_px"])) < 60.0
        checks[f"{item}:uncoded_near_wilor"] = float(np.median(a["uncoded_vs_wilor_px"])) < 15.0
        checks[f"{item}:pixel_split"] = bool((a["fitted_r_px"] + a["unfitted_r_px"] == a["r_px"]).all()
                                             and (a["missed"] <= a["fitted_r_px"]).all())
        if result["settings"].get("cover"):
            c = m["cover"]
            checks[f"{item}:cover_rows"] = c["rows_new"] == int(a["new_row"].sum()) and len(a["new_row"]) == m["rows"]
            checks[f"{item}:cover_kbps"] = abs(c["kbps"] - POSE_KBPS * (c["rows_h2"] + c["rows_new"]) / max(1, c["rows_h2"])) < 1e-9
            checks[f"{item}:cover_unfitted"] = float(a["unfitted_r_px"][hand].sum() / a["r_px"][hand].sum()) < 0.01
        if "lpips_identity" in m["checks"]:
            checks[f"{item}:lpips_identity"] = m["checks"]["lpips_identity"] < 1e-6
            checks[f"{item}:dists_identity"] = abs(m["checks"]["dists_identity"]) < 1e-4
    return checks


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage) if args.stage else stage_dir()
    result = json.loads((stage / "k1c.json").read_text())
    checks = {k: bool(v) for k, v in validate(stage, result).items()}
    report = {"passed": all(checks.values()), "checks": checks}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def item_verdict(warp: dict[str, float], hull: list[tuple[float, float]], kbps: float) -> dict[str, Any]:
    from experiments.visor.k1 import at_rate, rate_for

    q = at_rate(hull, kbps)
    if q is None:
        raise ValueError(f"the hull does not reach {kbps} kbps")
    row: dict[str, Any] = {"hull_lpips_at_rate": -q}
    for metric in ("lpips_r", "lpips_s"):
        need = rate_for(hull, -warp[metric])
        row[metric] = warp[metric]
        row[f"{metric}_below_hull"] = bool(warp[metric] < -q)
        row[f"{metric}_hull_kbps_at_quality"] = need
        row[f"{metric}_saving"] = None if need is None else 1.0 - kbps / need
        if need is None:
            row[f"{metric}_beyond"] = "better than every point" if -warp[metric] > hull[-1][1] else "worse than every point"
    return row


def motion(a: dict[str, np.ndarray], rows: np.ndarray, key: str = "uncoded_kp2d") -> dict[str, float | None]:
    """Median over the given rows of the uncoded joints' frame-to-frame speed and acceleration (px, mean over the 21
    joints), along each side's consecutive frames; acceleration is the jitter measure (K1e)."""
    speed: list[float] = []
    accel: list[float] = []
    for right in (False, True):
        idx = np.nonzero(a["row_right"] == right)[0]
        idx = idx[np.argsort(a["row_t"][idx])]
        kp, t = a[key][idx], a["row_t"][idx]
        for i in range(1, len(idx)):
            if t[i] - t[i - 1] == 1 and rows[idx[i]]:
                speed.append(float(np.linalg.norm(kp[i] - kp[i - 1], axis=-1).mean()))
                if i >= 2 and t[i - 1] - t[i - 2] == 1:
                    accel.append(float(np.linalg.norm(kp[i] - 2 * kp[i - 1] + kp[i - 2], axis=-1).mean()))
    return {"speed_px": float(np.median(speed)) if speed else None, "accel_px": float(np.median(accel)) if accel else None}


def command_report(args: argparse.Namespace) -> int:
    from experiments.background.g5 import envelope
    from experiments.visor.h2 import load_units

    stage = Path(args.stage)
    result = json.loads((stage / "k1c.json").read_text())
    checks = validate(stage, result)
    if not all(checks.values()):
        raise SystemExit(f"stage fails validation: {[k for k, v in checks.items() if not v]}")
    h3d = json.loads(Path(args.h3d).read_text())
    kbps = float(result["pose"]["visor_kbps"])
    items: dict[str, Any] = {}
    for m, a in load_units([str(stage / "published.tar")], "k1c"):
        item = m["item"]
        bar = h3d["bar"][item]
        g = h3d["component"]["arms"]["G-L"]["items"][item]["front"]
        hull = envelope([(p["kbps"], -p["lpips"]) for p in g + bar])
        dists_hull = envelope([(p["kbps"], -p["dists"]) for p in g + bar]) if all("dists" in p for p in g + bar) else None
        item_kbps = float(m["cover"]["kbps"]) if m.get("cover") else kbps
        verdict = item_verdict(m["summary"]["warp"], hull, item_kbps)
        verdict["kbps"] = item_kbps
        if m.get("cover"):
            verdict["cover"] = m["cover"]
            new = a["new_row"].astype(bool)
            verdict["coded_shift_px_h2_rows"] = float(np.mean(a["coded_shift_px"][~new])) if (~new).any() else None
            verdict["coded_shift_px_new_rows"] = float(np.mean(a["coded_shift_px"][new])) if new.any() else None
            if "uncoded_kp2d" in a:
                verdict["motion_h2_rows"], verdict["motion_new_rows"] = motion(a, ~new), motion(a, new)
                verdict["motion_all_rows"] = motion(a, np.ones_like(new))
                if "first_kp2d" in a:
                    verdict["motion_new_rows_first_pass"] = motion(a, new, "first_kp2d")
                verdict["h2_share_of_rows"] = float((~new).mean())
                verdict["hand_px_mean"] = float(a["r_px"][a["pixels_r"] > 0].mean())
        if dists_hull is not None:
            from experiments.visor.k1 import at_rate
            dq = at_rate(dists_hull, item_kbps)
            verdict["dists_r"] = m["summary"]["warp"]["dists_r"]
            verdict["dists_hull_at_rate"] = None if dq is None else -dq
            verdict["dists_r_below_hull"] = None if dq is None else bool(m["summary"]["warp"]["dists_r"] < -dq)
        hand = a["pixels_r"] > 0
        fitted = a["fitted_r_px"][hand].sum()
        items[item] = {
            **verdict, "uncoded": m["summary"]["warp@uncoded"], "hull": hull,
            "unfitted_share_of_r": float(a["unfitted_r_px"][hand].sum() / a["r_px"][hand].sum()),
            "missed_share_of_fitted": float(a["missed"][hand].sum() / max(1, fitted)),
            "spill_share_of_drawn": float(a["spill"][hand].sum() / max(1, a["drawn"][hand].sum())),
            "coded_shift_px": float(np.mean(a["coded_shift_px"])), "hand_frames": m["hand_frames"]}
    count = {k: sum(bool(v.get(k)) for v in items.values()) for k in ("lpips_r_below_hull", "lpips_s_below_hull", "dists_r_below_hull")}
    rule = DECISION_K1D if result["settings"].get("cover") else DECISION
    passes = count["lpips_r_below_hull"] >= rule["min_items"]
    escalate = any((count[k] >= rule["min_items"]) != passes for k in ("lpips_s_below_hull", "dists_r_below_hull")
                   if any(v.get(k) is not None for v in items.values()))
    out = {"inputs": {"stage": str(stage), "published_sha256": file_sha256(stage / "published.tar"),
                      "h3d": {"path": args.h3d, "sha256": file_sha256(Path(args.h3d))}, "code_sha256": file_sha256(Path(__file__))},
           "checks": checks, "kbps": kbps, "items": items, "items_below_hull": count, "decision_rule": rule,
           "decision": {"passes": passes, "to_user": escalate,
                        "reading": ("hands get K6b's ladder" if passes else "hands closed for every renderer")
                        + (" (to the user: S or DISTS crosses the bar differently)" if escalate else "")}}
    write_json(Path(args.out), out)
    for item, row in items.items():
        print(item, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("hull", "uncoded")},
              "uncoded", {k: round(v, 4) for k, v in row["uncoded"].items()})
    print("below hull:", count, "decision:", out["decision"])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for flag in ("--eval-set", "--archive", "--masks", "--mask-record", "--items", "--lpips-backbone", "--dists-backbone",
                 "--pose-code", "--mano-right"):
        run.add_argument(flag, required=True)
    run.add_argument("--video", action="append", default=[])
    run.add_argument("--poses", action="append", required=True, help="H2's VISOR published.tar (both jobs)")
    run.add_argument("--frames", default="240")
    run.add_argument("--allow-cpu", action="store_true")
    run.add_argument("--cover", action="store_true", help="K1d: WiLoR on every masked hand H2 has no row for")
    run.add_argument("--hand-box", action="store_true", help="K1e: with --cover, refit on the hand part of the mask box")
    run.add_argument("--mano-left", default="")
    run.add_argument("--wilor-checkpoint", default="")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default="")
    rep = sub.add_parser("report")
    rep.add_argument("--stage", required=True)
    rep.add_argument("--h3d", required=True)
    rep.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
