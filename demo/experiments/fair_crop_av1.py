"""AV1 of the hold-out hand crops at square sizes of the 256 crop.

The site ladder's 240p step scales a 256 crop up to 426x240. That is not a
smaller picture of the hand. SVT-AV1 refuses a side under 64, so this encodes
64, 80, 96, 128, 160, 192, and 256 with CRF 63 preset 7, then scores LPIPS
after scaling back to 256. Each side video keeps one crop per frame, the last
detection of that side, and the generators are scored on those same crops.
A contact sheet puts the reference, both generators, and a few of those
sizes on the same frames.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from demo.experiments.holdout_hand_rd import kbps
from demo.experiments.score_holdout_generators import (
    CLIPS,
    CROP,
    FPS,
    HOLD,
    TRAIN,
    _bgr_from_pred,
    _hand_pose,
    _tensor,
    build_anchors,
)
from demo.experiments.hand_packet_rate import decode_segment_delta, segment_delta_bytes
from demo.experiments.train_rtmw_hands import letterbox_alpha, make_model, matte_bgr
from demo.pipeline.foreground_segmenter import letterbox_crop
from demo.pipeline.hand_keypoints import render_skeleton_on_canvas
from demo.pipeline.maps.av1_crf import encode_av1_crf

# SVT-AV1 rejects width or height under 64.
SIZES = (64, 80, 96, 128, 160, 192, 256)
SHEET_SIZES = (64, 96, 128, 256)


def side_tracks(clip_dir: Path, scene: str, anchors: dict[str, np.ndarray]) -> tuple[dict[str, list], list[dict]]:
    poses = json.loads((clip_dir / "poses.json").read_text())
    decoded = decode_segment_delta(segment_delta_bytes(poses, 8))
    paths = sorted((clip_dir / "frames").glob("*.jpg"))
    n = min(len(paths), len(poses))
    tracks: dict[str, list] = {"Left": [None] * n, "Right": [None] * n}
    samples = []
    fallback = next(iter(anchors.values()))
    for index in range(n):
        image = cv2.imread(str(paths[index]))
        mask = cv2.imread(str(clip_dir / "masks" / f"{paths[index].stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        hands = sorted(poses[index], key=lambda item: str(item.get("side", "")))
        points = decoded[index] if index < len(decoded) else []
        chosen: dict[str, dict] = {}
        for hand_index, hand in enumerate(hands):
            side = hand["side"] if hand["side"] in tracks else "Left"
            box = [int(v) for v in hand["box"]]
            alpha = letterbox_alpha(mask, box, target_size=CROP)
            if int(np.count_nonzero(alpha > 8)) < 200:
                continue
            target, _ = letterbox_crop(image, box, target_size=CROP)
            target = matte_bgr(target, alpha)
            pts = points[hand_index] if hand_index < len(points) else hand["landmarks_pixel"]
            anchor = anchors.get(f"{scene}:{hand['side']}", fallback)
            skeleton = render_skeleton_on_canvas(_hand_pose(hand, pts), CROP, CROP, crop_bbox=box)
            chosen[side] = {"index": index, "side": side, "anchor": anchor, "skeleton": skeleton, "target": target}
        for side, sample in chosen.items():
            tracks[side][index] = sample["target"]
            samples.append(sample)
    return tracks, samples


def write_video(frames: list[np.ndarray | None], dest: Path, ffmpeg: str) -> None:
    blank = np.zeros((CROP, CROP, 3), np.uint8)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{CROP}x{CROP}",
        "-r", f"{FPS:.6f}", "-i", "-", "-c:v", "ffv1", str(dest),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(blank if frame is None else frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(proc.stderr.read().decode("utf-8", errors="replace")[-800:] if proc.stderr else "video")


def decode_square(path: Path, size: int, n_frames: int, ffmpeg: str) -> list[np.ndarray]:
    res = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode()[-400:])
    frame_bytes = size * size * 3
    frames = []
    for index in range(n_frames):
        chunk = res.stdout[index * frame_bytes:(index + 1) * frame_bytes]
        if len(chunk) < frame_bytes:
            break
        frame = np.frombuffer(chunk, np.uint8).reshape(size, size, 3)
        if size != CROP:
            frame = cv2.resize(frame, (CROP, CROP), interpolation=cv2.INTER_AREA)
        frames.append(frame.copy())
    return frames


def lpips_mean(lpips_fn, refs: list[np.ndarray], recs: list[np.ndarray], device: torch.device) -> float:
    scores = []
    with torch.no_grad():
        for start in range(0, len(refs), 8):
            batch_r, batch_d = [], []
            for ref, rec in zip(refs[start:start + 8], recs[start:start + 8]):
                for dest, image in ((batch_r, ref), (batch_d, rec)):
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
                    dest.append(torch.from_numpy(rgb).permute(2, 0, 1))
            out = lpips_fn(torch.stack(batch_r).to(device), torch.stack(batch_d).to(device))
            scores.extend(float(v) for v in out.view(-1).tolist())
    return float(np.mean(scores)) if scores else 0.0


def predict(model, samples: list[dict], device: torch.device) -> list[np.ndarray]:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), 8):
            batch = samples[start:start + 8]
            inputs = torch.stack([
                torch.cat([_tensor(item["anchor"]), _tensor(item["skeleton"])], 0) for item in batch
            ]).to(device)
            for row in model(inputs):
                preds.append(_bgr_from_pred(row[:3]))
    return preds


def label(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (CROP, 22), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def rescore_encoded(args: argparse.Namespace) -> None:
    """LPIPS for generators and existing square AV1 files on the same crops.

    The encoded side video keeps one crop per frame, the last detection of
    that side. Generator scores that used every detection are not the same set.
    """
    device = torch.device("cpu")
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    previous = json.loads((HOLD.parent / "compare" / "report.json").read_text())
    by_label = {clip["label"]: clip for clip in previous["clips"]}
    anchors: dict[str, dict] = {}
    payload = json.loads(args.out.read_text())
    for clip, (label_name, stem, factory, scene) in zip(payload["clips"], CLIPS):
        cache = args.out.parent / f"anchors_{factory}.npz"
        if factory not in anchors:
            loaded = np.load(cache)
            anchors[factory] = {key.replace("|", ":"): loaded[key] for key in loaded.files}
        clip_dir = HOLD / stem
        _tracks, samples = side_tracks(clip_dir, scene, anchors[factory])
        work = args.out.parent / "fair" / stem
        n_frames = clip["frames"]
        size = 64
        refs, recs = [], []
        decoded: dict[str, list[np.ndarray]] = {}
        for side in ("Left", "Right"):
            dest = work / f"{side}_{size}.mp4"
            if dest.is_file():
                decoded[side] = decode_square(dest, size, n_frames, args.ffmpeg)
        for item in samples:
            frames = decoded.get(item["side"])
            if frames and item["index"] < len(frames):
                refs.append(item["target"])
                recs.append(frames[item["index"]])
        check = next(row for row in clip["av1"] if int(row["size"]) == size)
        check["lpips_matched"] = lpips_mean(lpips_fn, refs, recs, device)
        check["hands_matched"] = len(refs)
        print(label_name, "av1-64", check["lpips"], check["lpips_matched"], len(refs), flush=True)
        for name in ("pix2pix", "spade"):
            ckpt = torch.load(TRAIN / f"{factory}_{name}.pt", map_location=device, weights_only=False)
            model = make_model(name).to(device)
            model.load_state_dict(ckpt["state_dict"])
            images = predict(model, samples, device)
            score = lpips_mean(lpips_fn, [item["target"] for item in samples], images, device)
            packet = next(row for row in by_label[label_name]["models"][name] if row["bits"] == 8)
            clip["models"][name] = {"lpips": score, "kbps": packet["kbps_10s"], "infer_ms": packet["infer_ms"], "hands": len(samples)}
            print(label_name, name, score, len(samples), flush=True)
            del model
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", args.out, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="/opt/local/bin/ffmpeg")
    parser.add_argument("--rescore-encoded", action="store_true")
    args = parser.parse_args()
    if args.rescore_encoded:
        rescore_encoded(args)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    previous = json.loads((HOLD.parent / "compare" / "report.json").read_text())
    by_label = {clip["label"]: clip for clip in previous["clips"]}
    anchors: dict[str, dict] = {}
    payload = {"schema": "pointstream.fair_crop_av1.v1", "sizes": list(SIZES), "note": "Square sizes of the 256 crop. CRF 63 preset 7. LPIPS after scale-back to 256.", "clips": []}
    sheet_dir = args.out.parent / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for label_name, stem, factory, scene in CLIPS:
        if factory not in anchors:
            cache = args.out.parent / f"anchors_{factory}.npz"
            if cache.is_file():
                loaded = np.load(cache)
                anchors[factory] = {key.replace("|", ":"): loaded[key] for key in loaded.files}
                print("anchors cached", factory, flush=True)
            else:
                print("anchors", factory, flush=True)
                anchors[factory] = build_anchors(factory)
                np.savez_compressed(cache, **{key.replace(":", "|"): value for key, value in anchors[factory].items()})
        clip_dir = HOLD / stem
        tracks, samples = side_tracks(clip_dir, scene, anchors[factory])
        n_frames = len(next(iter(tracks.values())))
        work = args.out.parent / "fair" / stem
        work.mkdir(parents=True, exist_ok=True)
        raws = {}
        for side, frames in tracks.items():
            if not any(frame is not None for frame in frames):
                continue
            raw = work / f"{side}.mkv"
            write_video(frames, raw, args.ffmpeg)
            raws[side] = frames
        rows = []
        decoded_by_size: dict[int, dict[str, list[np.ndarray]]] = {}
        for size in SIZES:
            total = 0
            recs = []
            refs = []
            decoded_by_size[size] = {}
            for side, frames in raws.items():
                src = work / f"{side}.mkv"
                dest = work / f"{side}_{size}.mp4"
                scale = None if size == CROP else f"{size}:{size}"
                encode_av1_crf(src, dest, scale=scale, ffmpeg=args.ffmpeg)
                total += dest.stat().st_size
                decoded = decode_square(dest, size, n_frames, args.ffmpeg)
                decoded_by_size[size][side] = decoded
                for index, frame in enumerate(frames):
                    if frame is not None and index < len(decoded):
                        refs.append(frame)
                        recs.append(decoded[index])
            score = lpips_mean(lpips_fn, refs, recs, device)
            rows.append({"size": size, "bytes": total, "kbps": kbps(total, n_frames), "lpips": score, "hands": len(refs)})
            print(label_name, size, rows[-1], flush=True)
        models = {}
        preds = {}
        for name in ("pix2pix", "spade"):
            ckpt = torch.load(TRAIN / f"{factory}_{name}.pt", map_location=device, weights_only=False)
            model = make_model(name).to(device)
            model.load_state_dict(ckpt["state_dict"])
            images = predict(model, samples, device)
            preds[name] = images
            score = lpips_mean(lpips_fn, [item["target"] for item in samples], images, device)
            packet = next(row for row in by_label[label_name]["models"][name] if row["bits"] == 8)
            models[name] = {"lpips": score, "kbps": packet["kbps_10s"], "infer_ms": packet["infer_ms"]}
            del model
            torch.cuda.empty_cache()
        present = [item["index"] for item in samples]
        picks = []
        if present:
            for fraction in (0.15, 0.5, 0.85):
                target = present[int(fraction * (len(present) - 1))]
                picks.append(next(item for item in samples if item["index"] == target))
        panels = []
        by_key = {(item["index"], item["side"]): (item, preds["pix2pix"][i], preds["spade"][i]) for i, item in enumerate(samples)}
        for item in picks:
            key = (item["index"], item["side"])
            _, pix, spade = by_key[key]
            row_imgs = [
                label(item["target"], f"ref f{item['index']}"),
                label(pix, "pix2pix"),
                label(spade, "SPADE"),
            ]
            for size in SHEET_SIZES:
                frame = decoded_by_size[size][item["side"]][item["index"]]
                row_imgs.append(label(frame, f"AV1 {size}"))
            panels.append(np.concatenate(row_imgs, axis=1))
        if panels:
            sheet = np.concatenate(panels, axis=0)
            cv2.imwrite(str(sheet_dir / f"{stem}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        payload["clips"].append({"label": label_name, "frames": n_frames, "av1": rows, "models": models})
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
