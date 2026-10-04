"""Train a hand generator on RTMW-l joints inside SAM crops.

Factory 001 (clip 1 and clip 3) and factory 2 are separate runs. Aisle and
look frames stay out. The last sampled second of each folder is held out.
Training stops when held-out appearance loss stops falling. Matte loss is
not the stopping rule.

A short smoke compares SPADE with the pix2pix UNet on the same crops.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from demo.models.hand_objective import hand_step_loss
from demo.models.unet_generator import HandPix2PixUNet, HandSPADEUNet
from demo.pipeline.foreground_segmenter import letterbox_crop
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand, render_skeleton_on_canvas

DEST = Path("/home/itec/emanuele/Datasets/pointstream-demo")


def letterbox_alpha(mask: np.ndarray, bbox: list[int], target_size: int = 256) -> np.ndarray:
    gray = mask if mask.ndim == 2 else mask[:, :, 0]
    packed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas, _ = letterbox_crop(packed, bbox, target_size=target_size)
    return canvas[:, :, 0]


def matte_bgr(crop: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    out = np.zeros_like(crop)
    a = alpha.astype(np.float32)
    if a.max() > 1:
        a = a / 255.0
    a = a[:, :, None]
    return np.where(a > 0, crop, out)


def soft_edge_alpha(binary_u8: np.ndarray, falloff_px: float = 2.0) -> np.ndarray:
    mask = (binary_u8 > 127).astype(np.uint8)
    if int(mask.max()) == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    half = max(float(falloff_px), 1e-3) / 2.0
    dist_in = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    alpha = np.zeros(mask.shape, dtype=np.float32)
    alpha[dist_in >= half] = 1.0
    inner = (mask > 0) & (dist_in < half)
    alpha[inner] = 0.5 + 0.5 * (dist_in[inner] / half)
    outer = (mask == 0) & (dist_out > 0) & (dist_out < half)
    alpha[outer] = 0.5 * (1.0 - dist_out[outer] / half)
    return alpha
FACTORIES = {
    "factory001": [
        DEST / "clip_01_factory001_worker001_00001/f000000-f035129",
        DEST / "clip_03_factory001_worker001_00000/f000000-f012629",
    ],
    "factory002": [DEST / "factory002_worker001_00000/f000000-f035129"],
}


def load_hands(folder: Path) -> tuple[list[dict], set[int]]:
    rows = json.loads((folder / "sam_poses" / "rtmw-l.json").read_text())
    frames = []
    held = set()
    usable = [row for row in rows if not row["aisle"] and not row["look"]]
    if not usable:
        return [], held
    last_second = max(int(row["frame_idx"]) // 30 for row in usable)
    for row in rows:
        second = int(row["frame_idx"]) // 30
        if row["aisle"] or row["look"]:
            continue
        if second == last_second:
            held.add(int(row["frame_idx"]))
        chosen = [hand for hand in row["hands"] if hand.get("selected")]
        if chosen:
            frames.append((folder, row["file"], int(row["frame_idx"]), chosen, int(row["frame_idx"]) in held))
    return frames, held


def crop_sample(folder: Path, file: str, frame: int, hand: dict, anchor: np.ndarray, scene: str) -> dict | None:
    image = cv2.imread(str(folder / "original" / file))
    mask = cv2.imread(str(folder / "masks" / f"{Path(file).stem}.png"), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return None
    box = [int(v) for v in hand["box"]]
    alpha = letterbox_alpha(mask, box, target_size=256)
    if int(np.count_nonzero(alpha > 8)) < 200:
        return None
    target, _ = letterbox_crop(image, box, target_size=256)
    target = matte_bgr(target, alpha)
    pose = FrameHandPose(frame_idx=frame, hands=[SingleHand(
        handedness=str(hand["side"]),
        confidence=float(hand["confidence"]),
        bbox=box,
        landmarks_norm=hand["landmarks_norm"],
        landmarks_pixel=hand["landmarks_pixel"],
    )])
    skeleton = render_skeleton_on_canvas(pose, 256, 256, crop_bbox=box)
    return {
        "appearance_crop": anchor,
        "skeleton_crop": skeleton,
        "target_crop": target,
        "target_alpha": soft_edge_alpha(alpha),
        "frame_idx": frame,
        "handedness": hand["side"],
        "bbox": box,
        "scene": scene,
    }


def build(factory: str, limit: int | None) -> tuple[list[dict], list[dict]]:
    records = []
    folders = FACTORIES[factory] if factory != "all" else [folder for group in FACTORIES.values() for folder in group]
    for folder in folders:
        records.extend(load_hands(folder)[0])
    train_recs = [rec for rec in records if not rec[4]]
    hold_recs = [rec for rec in records if rec[4]]
    anchors: dict[str, tuple[float, np.ndarray]] = {}
    for folder, file, _frame, hands, held in train_recs:
        if held:
            continue
        image = cv2.imread(str(folder / "original" / file))
        mask = cv2.imread(str(folder / "masks" / f"{Path(file).stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        for hand in hands:
            box = [int(v) for v in hand["box"]]
            alpha = letterbox_alpha(mask, box, target_size=256)
            if int(np.count_nonzero(alpha > 8)) < 200:
                continue
            crop, _ = letterbox_crop(image, box, target_size=256)
            crop = matte_bgr(crop, alpha)
            side = f"{folder.parent.name}:{hand['side']}"
            score = float(hand["confidence"])
            if side not in anchors or score > anchors[side][0]:
                anchors[side] = (score, crop)
    if not anchors:
        raise RuntimeError(f"no appearance anchor for {factory}")
    fallback = next(iter(anchors.values()))[1]

    def materialize(recs: list) -> list[dict]:
        samples = []
        for folder, file, frame, hands, _held in recs:
            for hand in hands:
                anchor = anchors.get(f"{folder.parent.name}:{hand['side']}", (0.0, fallback))[1]
                sample = crop_sample(folder, file, frame, hand, anchor, folder.parent.name)
                if sample is not None:
                    samples.append(sample)
                    if limit is not None and len(samples) >= limit:
                        return samples
        return samples

    return materialize(train_recs), materialize(hold_recs)


class HandCrops(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        def tensor(image: np.ndarray) -> torch.Tensor:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return torch.from_numpy((rgb.astype(np.float32) / 127.5) - 1.0).permute(2, 0, 1)
        target = tensor(item["target_crop"])
        alpha = torch.from_numpy(np.asarray(item["target_alpha"], dtype=np.float32)).unsqueeze(0)
        return {
            "input": torch.cat([tensor(item["appearance_crop"]), tensor(item["skeleton_crop"])], dim=0),
            "target": torch.cat([target, alpha], dim=0),
        }


def appearance(model, loader, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["input"].to(device))[:, :3]
            target = batch["target"].to(device)
            rgb = target[:, :3]
            alpha = target[:, 3:4]
            weight = alpha.sum().clamp(min=1.0)
            total += float((pred - rgb).abs().mul(alpha).sum() / weight)
            count += 1
    return total / max(1, count)


def run_epochs(model, loader, hold, device, epochs: int, patience: int) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    best = float("inf")
    stale = 0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            opt.zero_grad()
            outputs = model(inputs)
            if outputs.shape[1] == 4 and targets.shape[1] == 4:
                loss, _, _ = hand_step_loss(outputs, targets, None)
            else:
                loss = (outputs[:, :3] - targets[:, :3]).abs().mean()
            loss.backward()
            opt.step()
        score = appearance(model, hold, device)
        history.append(score)
        print(f"epoch {epoch} holdout_appearance {score:.4f}", flush=True)
        if score < best - 1e-3:
            best = score
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_holdout_appearance": best, "epochs": len(history), "history": history}


def make_model(name: str) -> torch.nn.Module:
    if name == "spade":
        return HandSPADEUNet(out_channels=4)
    if name == "pix2pix":
        return HandPix2PixUNet(out_channels=3)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--factory", choices=tuple(FACTORIES) + ("all",), action="append")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke-limit", type=int, default=64)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--model", choices=("spade", "pix2pix"), default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    factories = args.factory or ["all"]
    names = [args.model] if args.model else ["spade", "pix2pix"]
    args.out.mkdir(parents=True, exist_ok=True)
    trained = {}
    for name in names:
        for factory in factories:
            train_s, hold_s = build(factory, None)
            print(name, factory, "train", len(train_s), "hold", len(hold_s), flush=True)
            model = make_model(name).to(device)
            loader = DataLoader(HandCrops(train_s), batch_size=8, shuffle=True)
            hold = DataLoader(HandCrops(hold_s), batch_size=8)
            result = run_epochs(model, loader, hold, device, args.epochs, args.patience)
            scenes = {}
            for scene in sorted({sample["scene"] for sample in hold_s}):
                subset = [sample for sample in hold_s if sample["scene"] == scene]
                scenes[scene] = appearance(model, DataLoader(HandCrops(subset), batch_size=8), device)
            path = args.out / f"{factory}_{name}.pt"
            torch.save({"model": name, "factory": factory, "state_dict": model.state_dict(), "best_holdout_appearance": result["best_holdout_appearance"]}, path)
            trained[f"{factory}_{name}"] = {"checkpoint": str(path), "scenes": scenes, **result}
            print("WROTE", path, scenes, flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report = {
        "schema": "pointstream.rtmw_hand_generator.v2",
        "note": "SPADE and pix2pix were trained from scratch on RTMW-l crops. FoundHand is scored separately from its pretrained weights. ControlNet and AnimateAnyone can be trained from scratch as pose-conditioned denoisers; that is a different model, not a reason they cannot learn hands.",
        "trained": trained,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", args.out / "report.json", flush=True)


if __name__ == "__main__":
    main()
