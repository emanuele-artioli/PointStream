"""LPIPS and inference latency of the trained hand generators on hold-out crops.

AV1 rate, distortion, and decode time come from the encodes already written
beside those crops. Generator rate is the keypoint packet for the same
10-second span. The skeleton is the one that packet decodes to.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from demo.experiments.hand_packet_rate import decode_segment_delta, segment_delta_bytes
from demo.experiments.train_rtmw_hands import (
    FACTORIES,
    letterbox_alpha,
    load_hands,
    make_model,
    matte_bgr,
)
from demo.pipeline.foreground_segmenter import letterbox_crop
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand, render_skeleton_on_canvas

HOLD = Path("/home/itec/emanuele/pointstream-data/jobs/hand-rd/holdout")
TRAIN = Path("/home/itec/emanuele/pointstream-data/jobs/hand-rd/train")
CROP = 256
FPS = 30.0
CLIPS = (
    ("Clip 1", "clip_01_factory001_worker001_00001_last10s", "factory001", "clip_01_factory001_worker001_00001"),
    ("Clip 3", "clip_03_factory001_worker001_00000_last10s", "factory001", "clip_03_factory001_worker001_00000"),
    ("Factory 2", "factory002_worker001_00000_last10s", "factory002", "factory002_worker001_00000"),
)


def _tensor(image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy((rgb.astype(np.float32) / 127.5) - 1.0).permute(2, 0, 1)


def _bgr_from_pred(pred: torch.Tensor) -> np.ndarray:
    rgb = pred.detach().float().clamp(-1, 1).cpu().numpy()
    rgb = ((rgb + 1.0) * 127.5).round().astype(np.uint8)
    return cv2.cvtColor(np.transpose(rgb, (1, 2, 0)), cv2.COLOR_RGB2BGR)


def build_anchors(factory: str) -> dict[str, np.ndarray]:
    anchors: dict[str, tuple[float, np.ndarray]] = {}
    for folder in FACTORIES[factory]:
        for folder_path, file, _frame, hands, held in load_hands(folder)[0]:
            if held:
                continue
            image = cv2.imread(str(folder_path / "original" / file))
            mask = cv2.imread(str(folder_path / "masks" / f"{Path(file).stem}.png"), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                continue
            for hand in hands:
                box = [int(v) for v in hand["box"]]
                alpha = letterbox_alpha(mask, box, target_size=CROP)
                if int(np.count_nonzero(alpha > 8)) < 200:
                    continue
                crop, _ = letterbox_crop(image, box, target_size=CROP)
                crop = matte_bgr(crop, alpha)
                key = f"{folder_path.parent.name}:{hand['side']}"
                score = float(hand["confidence"])
                if key not in anchors or score > anchors[key][0]:
                    anchors[key] = (score, crop)
    if not anchors:
        raise RuntimeError(f"no appearance anchor for {factory}")
    return {key: value[1] for key, value in anchors.items()}


def _hand_pose(hand: dict, points: list | np.ndarray) -> FrameHandPose:
    return FrameHandPose(
        frame_idx=0,
        hands=[SingleHand(
            handedness=str(hand["side"]),
            confidence=float(hand.get("confidence", 1.0)),
            bbox=[int(v) for v in hand["box"]],
            landmarks_norm=[[0, 0, 0]] * 21,
            landmarks_pixel=[[float(p[0]), float(p[1])] for p in points],
        )],
    )


def clip_samples(clip_dir: Path, scene: str, anchors: dict[str, np.ndarray], bits: int) -> list[dict]:
    frames_dir = clip_dir / "frames"
    masks_dir = clip_dir / "masks"
    poses = json.loads((clip_dir / "poses.json").read_text())
    decoded = decode_segment_delta(segment_delta_bytes(poses, bits))
    fallback = next(iter(anchors.values()))
    samples = []
    paths = sorted(frames_dir.glob("*.jpg"))
    for index, path in enumerate(paths):
        if index >= len(poses):
            break
        image = cv2.imread(str(path))
        mask = cv2.imread(str(masks_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        hands = sorted(poses[index], key=lambda item: str(item.get("side", "")))
        points = decoded[index] if index < len(decoded) else []
        for hand_index, hand in enumerate(hands):
            pts = points[hand_index] if hand_index < len(points) else hand["landmarks_pixel"]
            box = [int(v) for v in hand["box"]]
            alpha = letterbox_alpha(mask, box, target_size=CROP)
            if int(np.count_nonzero(alpha > 8)) < 200:
                continue
            target, _ = letterbox_crop(image, box, target_size=CROP)
            target = matte_bgr(target, alpha)
            anchor = anchors.get(f"{scene}:{hand['side']}", fallback)
            skeleton = render_skeleton_on_canvas(_hand_pose(hand, pts), CROP, CROP, crop_bbox=box)
            samples.append({"anchor": anchor, "skeleton": skeleton, "target": target})
    return samples


def score_model(model, samples: list[dict], device: torch.device, lpips_fn) -> dict:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(samples), 8):
            batch = samples[start:start + 8]
            inputs = torch.stack([
                torch.cat([_tensor(item["anchor"]), _tensor(item["skeleton"])], dim=0) for item in batch
            ]).to(device)
            out = model(inputs)
            for row in out:
                preds.append(_bgr_from_pred(row[:3]))
    scores = []
    with torch.no_grad():
        for start in range(0, len(samples), 8):
            batch = samples[start:start + 8]
            images = preds[start:start + 8]
            refs, recs = [], []
            for item, pred in zip(batch, images):
                for dest, image in ((refs, item["target"]), (recs, pred)):
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
                    dest.append(torch.from_numpy(rgb).permute(2, 0, 1))
            out = lpips_fn(torch.stack(refs).to(device), torch.stack(recs).to(device))
            scores.extend(float(v) for v in out.view(-1).tolist())
    # Batch-1 latency after the weights are warm.
    timed = samples[: min(48, len(samples))]
    with torch.no_grad():
        warm = torch.stack([
            torch.cat([_tensor(item["anchor"]), _tensor(item["skeleton"])], dim=0) for item in timed[:4]
        ]).to(device)
        for _ in range(3):
            model(warm[:1])
        torch.cuda.synchronize()
        started = time.perf_counter()
        for item in timed:
            row = torch.cat([_tensor(item["anchor"]), _tensor(item["skeleton"])], dim=0).unsqueeze(0).to(device)
            model(row)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return {
        "crops": len(samples),
        "lpips": float(np.mean(scores)) if scores else None,
        "infer_ms": 1000.0 * elapsed / max(1, len(timed)),
        "timed_crops": len(timed),
    }


def av1_decode_ms(clip_dir: Path, ffmpeg: str, n_frames: int) -> list[dict]:
    rows = []
    for path in sorted((clip_dir / "av1").glob("seg300_0_*_crf63.mp4")):
        rung = path.name.split("_crf")[0].split("_")[-1]
        started = time.perf_counter()
        res = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        elapsed = time.perf_counter() - started
        if res.returncode != 0:
            continue
        rows.append({
            "file": path.name,
            "rung": rung,
            "decode_s": round(elapsed, 3),
            "decode_ms": 1000.0 * elapsed / max(1, n_frames),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="/opt/local/bin/ffmpeg")
    parser.add_argument("--bits", default="8,6,4")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    lpips_fn.eval()
    bits_list = tuple(int(part) for part in args.bits.split(",") if part)
    hold_report = json.loads((HOLD / "report.json").read_text())
    by_stem = {Path(clip["clip"]).stem: clip for clip in hold_report["clips"]}
    anchors: dict[str, dict[str, np.ndarray]] = {}
    payload = {"schema": "pointstream.holdout_generator_rd.v1", "device": str(device), "clips": []}
    for label, stem, factory, scene in CLIPS:
        if factory not in anchors:
            print("anchors", factory, flush=True)
            anchors[factory] = build_anchors(factory)
        clip_dir = HOLD / stem
        existing = by_stem[stem]
        packet_kbps = {
            (row["technique"], row["segment_frames"]): row["kbps"]
            for row in existing["packets"]
        }
        models = {}
        for name in ("spade", "pix2pix"):
            ckpt = torch.load(TRAIN / f"{factory}_{name}.pt", map_location=device, weights_only=False)
            model = make_model(name).to(device)
            model.load_state_dict(ckpt["state_dict"])
            per_bits = []
            for bits in bits_list:
                print(label, name, "bits", bits, flush=True)
                samples = clip_samples(clip_dir, scene, anchors[factory], bits)
                scored = score_model(model, samples, device, lpips_fn)
                technique = f"segment_delta_u{bits}_zlib"
                scored["bits"] = bits
                scored["kbps_10s"] = packet_kbps.get((technique, 300))
                scored["kbps_1s"] = packet_kbps.get((technique, 30))
                per_bits.append(scored)
                print(" ", scored, flush=True)
            models[name] = per_bits
            del model
            torch.cuda.empty_cache()
        payload["clips"].append({
            "label": label,
            "factory": factory,
            "av1": existing["av1"],
            "sam_fps": existing["sam"]["fps_excluding_load"],
            "pose_ms": existing["pose"]["ms_per_frame"],
            "combined_fps": existing["combined_fps_excluding_load"],
            "models": models,
            "av1_decode": av1_decode_ms(clip_dir, args.ffmpeg, int(existing["frames"])),
        })
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
