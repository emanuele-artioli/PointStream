"""Score pretrained FoundHand against the RTMW-crop SPADE and pix2pix checkpoints.

FoundHand is the gesture-transfer path from the Chaerin5/FoundHand demo:
a reference crop, target 2D keypoints as heatmaps, 250 denoising steps.
The score is the same masked L1, in [-1, 1], on a few held-out crops from
each scene.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from demo.experiments.train_rtmw_hands import (
    FACTORIES,
    crop_sample,
    letterbox_alpha,
    letterbox_crop,
    load_hands,
    make_model,
    matte_bgr,
)
SPACE = Path("/home/itec/emanuele/pointstream-data/weights/foundhand/space")
WEIGHTS = Path("/home/itec/emanuele/pointstream-data/weights/foundhand/weights")


def crop_points(points: np.ndarray, box: list[int], size: int = 256) -> np.ndarray:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    scale = float(size) / max(bw, bh)
    new_w = int(round(bw * scale))
    new_h = int(round(bh * scale))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    mapped = np.zeros_like(points, dtype=np.float32)
    mapped[:, 0] = (points[:, 0] - x1) * scale + pad_x
    mapped[:, 1] = (points[:, 1] - y1) * scale + pad_y
    return mapped


def holdout_examples(per_scene: int) -> list[dict]:
    chosen = []
    folders = [folder for group in FACTORIES.values() for folder in group]
    anchors: dict[str, np.ndarray] = {}
    pending = []
    best: dict[str, tuple] = {}
    for folder in folders:
        scene = folder.parent.name
        count = 0
        print(f"index {scene}", flush=True)
        for folder, file, frame, hands, held in load_hands(folder)[0]:
            if not held:
                for hand in hands:
                    key = f"{scene}:{hand['side']}"
                    confidence = float(hand["confidence"])
                    if key not in best or confidence > best[key][0]:
                        best[key] = (confidence, folder, file, hand)
                continue
            if count >= per_scene:
                continue
            pending.append((scene, folder, file, frame, hands[0]))
            count += 1
    for key, (confidence, folder, file, hand) in best.items():
        image = cv2.imread(str(folder / "original" / file))
        mask = cv2.imread(str(folder / "masks" / f"{Path(file).stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        box = [int(v) for v in hand["box"]]
        alpha = letterbox_alpha(mask, box)
        if int(np.count_nonzero(alpha > 8)) < 200:
            continue
        crop, _ = letterbox_crop(image, box)
        anchors[key] = (confidence, matte_bgr(crop, alpha))
    fallback = next(iter(anchors.values()))[1]
    for scene, folder, file, frame, hand in pending:
        anchor = anchors.get(f"{scene}:{hand['side']}", (0.0, fallback))[1]
        sample = crop_sample(folder, file, frame, hand, anchor, scene)
        if sample is None:
            continue
        box = [int(v) for v in hand["box"]]
        points = crop_points(np.asarray(hand["landmarks_pixel"], dtype=np.float32), box)
        packed = np.zeros((42, 2), dtype=np.float32)
        if str(hand["side"]).lower().startswith("l"):
            packed[21:] = points
        else:
            packed[:21] = points
        sample["keypoints"] = packed
        sample["anchor_bgr"] = anchor
        chosen.append(sample)
    return chosen


class HandDiffOpts:
    """Stub so the Drive checkpoint, pickled from the training script, can load."""


def load_foundhand(device: torch.device):
    sys.path.insert(0, str(SPACE))
    import vit
    import vqvae
    from diffusion import create_diffusion

    ckpt = WEIGHTS / "foundhand.ckpt"
    vae_path = WEIGHTS / "vae-ft-mse-840000-ema-pruned.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if not vae_path.is_file():
        raise FileNotFoundError(vae_path)
    diffusion = create_diffusion("250")
    model = vit.DiT_XL_2(input_size=32, latent_dim=4, in_channels=4 + 42 + 1, learn_sigma=True).to(device)
    state = torch.load(ckpt, map_location="cpu")
    state = state.get("ema_state_dict", state.get("model_state_dict", state))
    missing, _extra = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"FoundHand checkpoint missed {len(missing)} keys, first {missing[:5]}")
    model.eval()
    autoencoder = vqvae.create_model(3, 3, 4).eval().requires_grad_(False)
    vae_state = torch.load(vae_path, map_location="cpu")["state_dict"]
    vae_missing, _ = autoencoder.load_state_dict(vae_state, strict=False)
    if vae_missing:
        raise RuntimeError(f"VAE missed {len(vae_missing)} keys")
    autoencoder = autoencoder.to(device).eval()
    return model, autoencoder, diffusion


def foundhand_image(model, autoencoder, diffusion, sample: dict, device: torch.device) -> np.ndarray:
    from utils import check_keypoints_validity, keypoint_heatmap, scale_keypoint

    image_transform = Compose([
        ToTensor(),
        Resize((256, 256)),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    rgb = cv2.cvtColor(sample["anchor_bgr"], cv2.COLOR_BGR2RGB)
    keypts = sample["keypoints"]
    mask = (sample["target_alpha"] > 0.05).astype(np.float32)
    image = image_transform(rgb).to(device)[None]
    valid = check_keypoints_validity(keypts, (256, 256))
    heatmaps = torch.tensor(
        keypoint_heatmap(scale_keypoint(keypts, (256, 256), (32, 32)), (32, 32), var=1.0) * valid[:, None, None],
        dtype=torch.float,
        device=device,
    )[None]
    mask_t = torch.tensor(
        cv2.resize(mask, (32, 32), interpolation=cv2.INTER_NEAREST),
        dtype=torch.float,
        device=device,
    )[None, None]
    with torch.no_grad():
        latent = 0.18215 * autoencoder.encode(image).sample()
    ref_cond = torch.cat([latent, heatmaps, mask_t], 1)
    target_cond = torch.cat([heatmaps, torch.zeros_like(heatmaps[:, :1])], 1)
    z = torch.randn(1, 4, 32, 32, device=device)
    nvs = torch.zeros(1, dtype=torch.int, device=device)
    z = torch.cat([z, z], 0)
    model_kwargs = dict(
        target_cond=torch.cat([target_cond, torch.zeros_like(target_cond)]),
        ref_cond=torch.cat([ref_cond, torch.zeros_like(ref_cond)]),
        nvs=torch.cat([nvs, 2 * torch.ones_like(nvs)]),
        cfg_scale=3.5,
    )
    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=None,
            device=device,
        )
        samples, _ = samples.chunk(2)
        decoded = autoencoder.decode(samples / 0.18215).clamp(-1, 1)
    return decoded[0]


def masked_l1(pred: torch.Tensor, target_rgb: np.ndarray, alpha: np.ndarray) -> float:
    rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB).astype(np.float32)
    tgt = torch.from_numpy(rgb / 127.5 - 1.0).permute(2, 0, 1).to(pred.device)
    weight = torch.from_numpy(alpha.astype(np.float32)).to(pred.device)[None]
    return float((pred[:3] - tgt).abs().mul(weight).sum() / weight.sum().clamp(min=1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--per-scene", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = holdout_examples(args.per_scene)
    print("examples", len(examples), flush=True)
    model, autoencoder, diffusion = load_foundhand(device)
    generators = {}
    for name in ("spade", "pix2pix"):
        path = args.checkpoints / f"all_{name}.pt"
        net = make_model(name).to(device)
        net.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        net.eval()
        generators[name] = net
    rows = []
    for sample in examples:
        pred_fh = foundhand_image(model, autoencoder, diffusion, sample, device)
        app = torch.from_numpy(cv2.cvtColor(sample["appearance_crop"], cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1).permute(2, 0, 1)
        skel = torch.from_numpy(cv2.cvtColor(sample["skeleton_crop"], cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1).permute(2, 0, 1)
        batch = torch.cat([app, skel], 0)[None].to(device)
        scores = {"foundhand": masked_l1(pred_fh, sample["target_crop"], sample["target_alpha"])}
        with torch.no_grad():
            for name, net in generators.items():
                pred = net(batch)[0]
                scores[name] = masked_l1(pred, sample["target_crop"], sample["target_alpha"])
        row = {"scene": sample["scene"], "frame": sample["frame_idx"], **scores}
        rows.append(row)
        print(row, flush=True)
    summary = {}
    for name in ("foundhand", "spade", "pix2pix"):
        summary[name] = float(np.mean([row[name] for row in rows]))
    dest = args.out if args.out.suffix == ".json" else args.out / "foundhand_compare.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"per_frame": rows, "mean_masked_l1": summary, "lower_is_better": True}
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print("MEAN", summary, flush=True)


if __name__ == "__main__":
    main()
