# ruff: noqa: E402 - the sys.path bootstrap below MUST run before the first
# `from src...` import: the pinned env's editable install hard-maps `src` to
# MAIN's tree, so without it a git worktree silently imports the wrong code.
import os
import sys
from pathlib import Path as _Path

# Run as `python scripts/train_controlnet.py`, sys.path[0] is `scripts/` and the
# repo root is nowhere on the path. The pinned env carries an editable install
# whose finder hard-maps `src` -> /home/itec/emanuele/pointstream/src, so from a
# git worktree `from src... import ...` silently resolved against MAIN's tree and
# this branch's own modules looked missing. Put this checkout first.
_REPO_ROOT = str(_Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import glob
import json
import logging
import random
import shutil
import time
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from accelerate import Accelerator
from diffusers import (
    ControlNetModel,
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from src.components.metrics.lpips import LpipsMetric
from src.shared.training.stop import StopBounds, TaskStopRule
from src.shared.training.task_eval import (
    ItemScore,
    mean_scores,
    score_item,
    static_copy_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STOP_EVAL_N_CLIPS = 4
STOP_EVAL_OFFSET = 8
STOP_EVAL_SEED = 42
STOP_EVAL_CANVAS = 512
STOP_EVAL_STEPS = 4
CONDITION_TO_VARIANT = {
    "pose": "pose",
    "pose-racket": "pose",
    "canny": "canny",
    "seg": "seg",
    "ip-adapter": "ip-adapter",
}

# Derived directories and sidecars under a scene dir -- never training items.
DERIVED_SUFFIXES = ("_skeleton", "_canny", "_caption", "_pose_racket", "_pose_body")


def _letterbox(image, mask, canvas: int = STOP_EVAL_CANVAS):
    from src.components.generation._numpy import prepare_letterboxed

    mask_u8 = (np.asarray(mask, dtype=bool).astype("uint8") * 255)
    prepared = prepare_letterboxed(
        image, None, canvas, canvas, extras={"mask": mask_u8}
    )
    boxed_mask = np.asarray(prepared["mask"])
    if boxed_mask.ndim == 3:
        boxed_mask = boxed_mask[..., 0]
    return prepared["appearance"], boxed_mask > 0


def load_stop_samples(n_clips: int = STOP_EVAL_N_CLIPS, offset: int = STOP_EVAL_OFFSET):
    """A handful of coding-task crops. Stopping signal, not a result."""
    from experiments.probe.clips import list_clips, load_coding_sample

    clips = [clip for clip in list_clips() if clip.n_frames > offset]
    if len(clips) < 2:
        raise RuntimeError(
            f"task-stop eval needs at least 2 probe clips with >{offset} frames; "
            f"found {len(clips)}"
        )
    chosen = clips[:n_clips]
    return [load_coding_sample(clip, 0, offset) for clip in chosen]


def measure_stop_floors(samples, lpips: LpipsMetric) -> StopBounds:
    """Static-copy floor and unrelated-image null, on the stop-eval items."""
    copies: list[ItemScore] = []
    unrelated: list[ItemScore] = []
    for index, sample in enumerate(samples):
        appearance, mask = _letterbox(sample.appearance_rgb, sample.object_mask)
        target, target_mask = _letterbox(sample.reference_rgb, sample.object_mask)
        copies.append(
            static_copy_scores(
                appearance, target, target_mask, key=sample.key, lpips=lpips
            )
        )
        donor = samples[(index + 1) % len(samples)]
        donor_app, _ = _letterbox(donor.appearance_rgb, donor.object_mask)
        lpips_value, psnr, n_pixels = score_item(
            target, donor_app, target_mask, lpips=lpips
        )
        unrelated.append(
            ItemScore(
                key=f"{sample.key}<-{donor.key}",
                lpips=lpips_value,
                psnr=psnr,
                n_mask_pixels=n_pixels,
            )
        )
    copy_mean = mean_scores(copies)
    null_mean = mean_scores(unrelated)
    return StopBounds(
        floor_psnr=copy_mean.psnr,
        floor_lpips=copy_mean.lpips,
        null_lpips=null_mean.lpips,
        source=(
            f"stop-eval static-copy n={copy_mean.n} offset={STOP_EVAL_OFFSET} "
            f"seed={STOP_EVAL_SEED}; unrelated n={null_mean.n}"
        ),
    )


def score_stop_generations(samples, predictions, lpips: LpipsMetric):
    from src.components.generation._numpy import as_hwc

    items: list[ItemScore] = []
    for sample, predicted in zip(samples, predictions, strict=True):
        target, mask = _letterbox(sample.reference_rgb, sample.object_mask)
        pred = as_hwc(predicted)
        if pred.shape[:2] != target.shape[:2]:
            pred, _ = _letterbox(pred, sample.object_mask)
        lpips_value, psnr, n_pixels = score_item(target, pred, mask, lpips=lpips)
        items.append(
            ItemScore(
                key=sample.key, lpips=lpips_value, psnr=psnr, n_mask_pixels=n_pixels
            )
        )
    return mean_scores(items)


def generate_stop_predictions(
    samples,
    *,
    checkpoint: str,
    variant: str,
    device: str,
    steps: int,
):
    """Drive the coding task through the live generator. Not a citable eval."""
    from experiments.probe.run import _coding_bundle
    from src.components.generation.controlnet import ControlNetGenerator
    from src.contracts.conditioning import GenerationParams

    generator = ControlNetGenerator(variant=variant, checkpoint=checkpoint, steps=steps)
    params = GenerationParams(steps=steps)
    images = []
    for sample in samples:
        bundle = _coding_bundle(sample)
        images.append(
            generator.generate(
                bundle, seed=STOP_EVAL_SEED, device=device, params=params
            )
        )
    return images


def pad_to_square(img, fill=0):
    w, h = img.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2
    pad_right = max_dim - w - pad_left
    pad_bottom = max_dim - h - pad_top
    import torchvision.transforms.functional as TF
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill, padding_mode='constant')

class ControlNetDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        condition_type: str,
        target_size: int = 512,
        tokenizer=None,
        include_reference: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.condition_type = condition_type
        self.target_size = target_size
        self.tokenizer = tokenizer
        self.include_reference = include_reference
        self.items = []
        if self.condition_type == "ip-adapter" and not self.include_reference:
            raise ValueError(
                "ip-adapter is an image-embedding pathway. Pass --include-reference "
                "so appearance enters the adapter, not the pose control. "
                "Painting the reference into the control image is the pose-ref "
                "recipe that already failed (PLAN.md §2.4)."
            )
        self.track_to_colors: dict[str, list[str]] = {}
        
        search_pattern = os.path.join(str(self.root_dir), "*", "segmentations", "scene_*", "track_*")
        all_tracks = glob.glob(search_pattern)
        
        for track_dir_str in all_tracks:
            # Skip derived directories and sidecar files -- only primary track
            # dirs are training items. Missing a suffix here makes the loader
            # treat e.g. `track_0036_pose_body` as a track and hunt for
            # `track_0036_pose_body_pose_body`.
            if track_dir_str.endswith(DERIVED_SUFFIXES) or not os.path.isdir(track_dir_str):
                continue

            track_dir = Path(track_dir_str)
            
            # Read caption if available, otherwise fallback
            caption_path = track_dir.parent / f"{track_dir.name}_caption.json"
            prompt = "photorealistic tennis player, broadcast sports shot"
            if caption_path.exists():
                with open(caption_path, "r") as f:
                    cdata = json.load(f)
                    prompt = cdata.get("caption", prompt)
                    
            color_frames = sorted(list(track_dir.glob("frame_*.png")))
            parts = track_dir.parts
            unique_track_id = f"{parts[-4]}_{parts[-2]}_{parts[-1]}"
            if self.include_reference and len(color_frames) < 2:
                continue

            # Identify the condition directory.
            # `pose` selects the body-only variant, which the decoder reproduces
            # for free and bit-identically; `pose-racket` matches the legacy
            # checkpoints but needs racket geometry the wire format does not
            # carry. The legacy `_skeleton` tree is NOT selectable here -- its
            # filenames are positional and pairing it wrecked this dataset.
            if self.condition_type == "pose":
                cond_dir = track_dir.with_name(f"{track_dir.name}_pose_body")
            elif self.condition_type == "pose-racket":
                cond_dir = track_dir.with_name(f"{track_dir.name}_pose_racket")
            elif self.condition_type == "canny":
                cond_dir = track_dir.with_name(f"{track_dir.name}_canny")
            elif self.condition_type == "ip-adapter":
                # Appearance goes through the image-embedding adapter, pose
                # through ControlNet. This used to share the seg branch with
                # cond_dir = None, which is why the registry entry was fiction.
                cond_dir = track_dir.with_name(f"{track_dir.name}_pose_body")
            elif self.condition_type == "seg":
                cond_dir = None
            else:
                raise ValueError(f"Unknown condition type: {self.condition_type}")

            # Pair POSITIONALLY, matching src/shared/tennis_dataset.py.
            # Pairing by filename silently produced garbage: `_skeleton` frames
            # were named by position while colour frames carry the absolute
            # source frame id, so across the training view 32.7% of items were
            # paired with the WRONG pose and 22.9% were dropped without a word.
            # A count mismatch now raises instead of quietly shrinking the
            # dataset -- a training set that silently loses a quarter of its
            # data is indistinguishable from one that trained fine.
            if cond_dir is None:
                for color_path in color_frames:
                    self.items.append({
                        "image_path": str(color_path),
                        "cond_path": None,
                        "prompt": prompt,
                        "track_id": unique_track_id,
                    })
                    self.track_to_colors.setdefault(unique_track_id, []).append(str(color_path))
                continue

            if not cond_dir.exists():
                raise FileNotFoundError(
                    f"Condition directory missing for {track_dir}: {cond_dir}. "
                    f"Run scripts/process_dataset.py's '{self.condition_type}' stage first."
                )

            cond_frames = sorted(cond_dir.glob("frame_*.png"))
            if len(cond_frames) != len(color_frames):
                raise ValueError(
                    f"Frame-count mismatch for {track_dir.name}: {len(color_frames)} colour frames "
                    f"vs {len(cond_frames)} '{self.condition_type}' frames in {cond_dir.name}. "
                    "Colour and condition sequences must correspond one-to-one; regenerate the "
                    "condition stage rather than training on a partial pairing."
                )

            for color_path, cond_path in zip(color_frames, cond_frames):
                self.items.append({
                    "image_path": str(color_path),
                    "cond_path": str(cond_path),
                    "prompt": prompt,
                    "track_id": unique_track_id,
                })
                self.track_to_colors.setdefault(unique_track_id, []).append(str(color_path))

        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.cond_transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        img = Image.open(item["image_path"])
        
        seg_mask = None
        if self.condition_type == "seg":
            if img.mode == 'RGBA':
                alpha = img.split()[-1]
                seg_mask = Image.merge("RGB", (alpha, alpha, alpha))
            else:
                seg_mask = Image.new("RGB", img.size, (255, 255, 255))
        
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, (0, 0, 0, 255))
            img = Image.alpha_composite(background, img).convert("RGB")
        else:
            img = img.convert("RGB")
            
        img = pad_to_square(img, fill=0)
        image_tensor = self.transform(img)
        
        if self.condition_type in ("pose", "pose-racket", "canny", "ip-adapter"):
            cond_img = Image.open(item["cond_path"]).convert("RGB")
            cond_img = pad_to_square(cond_img, fill=0)
            cond_tensor = self.cond_transform(cond_img)
        elif self.condition_type == "seg":
            cond_img = pad_to_square(seg_mask, fill=0)
            cond_tensor = self.cond_transform(cond_img)
        else:
            raise ValueError(f"Unknown condition type: {self.condition_type}")
            
        tokens = self.tokenizer(
            item["prompt"], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze(0)
        
        sample = {
            "pixel_values": image_tensor,
            "conditioning_pixel_values": cond_tensor,
            "input_ids": tokens
        }
        if self.include_reference:
            track_id = item["track_id"]
            colors = self.track_to_colors[track_id]
            candidates = [path for path in colors if path != item["image_path"]]
            ref_path = random.choice(candidates or colors)
            ref_img = Image.open(ref_path)
            if ref_img.mode == "RGBA":
                background = Image.new("RGBA", ref_img.size, (0, 0, 0, 255))
                ref_img = Image.alpha_composite(background, ref_img).convert("RGB")
            else:
                ref_img = ref_img.convert("RGB")
            ref_img = pad_to_square(ref_img, fill=0)
            sample["reference_pixel_values"] = self.cond_transform(ref_img)
        return sample


def compose_pose_on_appearance_tensor(
    pose: torch.Tensor, appearance: torch.Tensor, *, threshold: float = 8.0 / 255.0
) -> torch.Tensor:
    """Tensor twin of ``compose_pose_on_appearance``. CHW, values in [0, 1]."""
    mask = pose.amax(dim=1, keepdim=True) > threshold
    return torch.where(mask, pose, appearance)


def controlnet_cond_for_batch(
    batch: dict,
    *,
    condition_type: str,
    include_reference: bool,
    weight_dtype,
):
    """Pose/canny/seg tensor that ControlNet sees. IP-Adapter never paints the reference here."""
    pose = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
    if condition_type == "ip-adapter":
        return pose
    if include_reference:
        return compose_pose_on_appearance_tensor(
            pose, batch["reference_pixel_values"].to(dtype=weight_dtype)
        )
    return pose


def collect_ip_adapter_parameters(unet) -> list:
    """The ~22M adapter: image projection plus IP-Attn to_k_ip / to_v_ip."""
    params = []
    proj = getattr(unet, "encoder_hid_proj", None)
    if proj is not None:
        params.extend(proj.parameters())
    for proc in getattr(unet, "attn_processors", {}).values():
        if hasattr(proc, "to_k_ip"):
            params.extend(proc.parameters())
    return params


def load_ip_adapter_state_dict(
    repo: str = "h94/IP-Adapter",
    subfolder: str = "models",
    weight_name: str = "ip-adapter_sd15.bin",
):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo, filename=f"{subfolder}/{weight_name}", local_files_only=True
    )
    return torch.load(path, map_location="cpu")


def attach_ip_adapter(unet, state_dict) -> None:
    """Install h94 processors on a frozen UNet, then unfreeze only the adapter."""
    unet.requires_grad_(False)
    unet._load_ip_adapter_weights([state_dict])
    for param in collect_ip_adapter_parameters(unet):
        param.requires_grad = True


def export_ip_adapter_state_dict(unet) -> dict:
    """Write the h94 ``image_proj`` / ``ip_adapter`` layout so inference can reload."""
    ip_adapter = {}
    key_id = 1
    for proc in unet.attn_processors.values():
        if not hasattr(proc, "to_k_ip"):
            continue
        ip_adapter[f"{key_id}.to_k_ip.weight"] = proc.to_k_ip[0].weight.detach().cpu().contiguous()
        ip_adapter[f"{key_id}.to_v_ip.weight"] = proc.to_v_ip[0].weight.detach().cpu().contiguous()
        key_id += 2
    proj = unet.encoder_hid_proj
    inner = proj.image_projection_layers[0] if hasattr(proj, "image_projection_layers") else proj
    # Write the ORIGINAL h94 layout, not the diffusers-internal one.
    # `_convert_ip_adapter_image_proj_to_diffusers` branches on `proj.weight`
    # and renames proj -> image_embeds itself. Saving the converted keys makes
    # that branch miss, so it falls through to the IP-Adapter-Full branch and
    # dies on KeyError: 'proj.0.weight'. Undo the rename on the way out.
    image_proj = {}
    for key, value in inner.state_dict().items():
        out_key = "proj" + key[len("image_embeds"):] if key.startswith("image_embeds") else key
        image_proj[out_key] = value.detach().cpu().contiguous()
    if "proj.weight" not in image_proj:
        raise RuntimeError(
            "exported image_proj lacks 'proj.weight'; diffusers would mis-detect the "
            f"projection type. Keys: {sorted(image_proj)}"
        )
    return {"image_proj": image_proj, "ip_adapter": ip_adapter}


def encode_reference_image_embeds(
    image_encoder, feature_extractor, reference, *, device, dtype
):
    """CLIP image embeds for IP-Adapter. ``reference`` is BCHW in [0, 1]."""
    from torchvision.transforms.functional import to_pil_image

    pils = [to_pil_image(frame.detach().float().cpu().clamp(0, 1)) for frame in reference]
    pixel_values = feature_extractor(images=pils, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    with torch.no_grad():
        embeds = image_encoder(pixel_values).image_embeds
    # MultiIPAdapterImageProjection wants each list entry as
    # [batch, num_images, embed_dim]. Handing it the bare [batch, embed_dim]
    # makes it read embed_dim as num_images, flatten to a 1-D tensor and hit
    # `mat1 and mat2 shapes cannot be multiplied (1x2048 and 1024x3072)`.
    # Inference builds the same 3-D shape (pipeline_controlnet.py:
    # `image_embeds.append(single_image_embeds[None, :])`).
    embeds = embeds.unsqueeze(1)
    if embeds.ndim != 3 or embeds.shape[1] != 1:
        raise RuntimeError(
            f"image embeds must be [batch, 1, embed_dim]; got {tuple(embeds.shape)}"
        )
    return [embeds]


def assert_reference_enters_controlnet(
    controlnet, batch, *, weight_dtype, timesteps, noisy_latents, encoder_hidden_states
) -> float:
    """Drive two forwards that differ only in the reference; residuals must move."""
    pose = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
    reference = batch["reference_pixel_values"].to(dtype=weight_dtype)
    composed = compose_pose_on_appearance_tensor(pose, reference)
    blank_ref = torch.zeros_like(reference)
    composed_blank = compose_pose_on_appearance_tensor(pose, blank_ref)
    if torch.equal(composed, composed_blank):
        raise RuntimeError(
            "composed control equals pose-on-black; the reference never reached the canvas."
        )
    with torch.no_grad():
        _, mid_ref = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=composed,
            return_dict=False,
        )
        _, mid_blank = controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=composed_blank,
            return_dict=False,
        )
    delta = float((mid_ref.float() - mid_blank.float()).abs().mean().cpu())
    logging.info("reference-enters-controlnet residual_delta=%.6g", delta)
    if delta <= 1e-6:
        raise RuntimeError(
            f"ControlNet residuals did not change when the reference was zeroed "
            f"(delta={delta}). Appearance is not an input."
        )
    return delta


def main():
    parser = argparse.ArgumentParser(description="Train ControlNet")
    parser.add_argument("--data-root", type=str, default="assets/dataset", help="Dataset root")
    parser.add_argument("--condition-type", type=str,
                        choices=["pose", "pose-racket", "canny", "seg", "ip-adapter"], required=True,
                        help="pose = body-only skeleton (decoder-reproducible for free); "
                             "pose-racket = body + racket (needs racket geometry in ActorPacket)")
    parser.add_argument("--model-id", type=str, default="assets/weights/stable-diffusion-v1-5")
    parser.add_argument("--controlnet-model-id", type=str, default=None, help="Path to pre-trained ControlNet to fine-tune")
    parser.add_argument("--from-scratch", action="store_true", help="Initialize ControlNet from scratch (no fine-tuning)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="assets/weights/custom-controlnet")
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Sample a same-track reference frame and paint it under the pose (BP8).",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps.")
    parser.add_argument(
        "--checkpoint-every-minutes",
        type=int,
        default=60,
        help="Wall-clock checkpoint interval. Long jobs must checkpoint at least hourly.",
    )
    parser.add_argument(
        "--progress-every-minutes",
        type=int,
        default=10,
        help="Append a progress line at least this often so a hang is visible.",
    )
    parser.add_argument(
        "--smoke-check-reference",
        action="store_true",
        help="Before training, drive two ControlNet forwards that differ only in the reference.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-task-stop",
        action="store_true",
        help="Disable the coding-task stop rule. Default is on: a run that cannot "
        "clear the static-copy floor stops by epoch 3–4 instead of burning GPU hours.",
    )
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=2000,
        help="Extra coding-task evals inside an epoch. 0 = epoch-end only. "
        "Mid-epoch evals cannot stop before --min-stop-epochs.",
    )
    parser.add_argument("--task-eval-steps", type=int, default=STOP_EVAL_STEPS)
    parser.add_argument("--task-eval-clips", type=int, default=STOP_EVAL_N_CLIPS)
    args = parser.parse_args()

    if (
        args.include_reference
        and args.controlnet_model_id is None
        and not args.from_scratch
        and args.condition_type != "ip-adapter"
    ):
        args.controlnet_model_id = "assets/weights/pose-controlnet/checkpoint-epoch-10"
        logging.info(
            "include-reference defaults to fine-tuning the tennis pose ControlNet at %s",
            args.controlnet_model_id,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
    )

    logging.info(f"Loading models from {args.model_id}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    if args.from_scratch:
        logging.info("Initializing ControlNet from scratch using UNet config.")
        controlnet = ControlNetModel.from_unet(unet)
    elif args.controlnet_model_id:
        logging.info(f"Loading pre-trained ControlNet from {args.controlnet_model_id} for fine-tuning.")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_id)
    else:
        defaults = {
            "pose": "assets/weights/control_v11p_sd15_openpose",
            "pose-racket": "assets/weights/control_v11p_sd15_openpose",
            "canny": "lllyasviel/control_v11p_sd15_canny",
            "seg": "lllyasviel/control_v11p_sd15_seg",
            "ip-adapter": "assets/weights/control_v11p_sd15_openpose"
        }
        cnet_id = defaults[args.condition_type]
        logging.info(f"Loading default pre-trained ControlNet {cnet_id} for fine-tuning.")
        controlnet = ControlNetModel.from_pretrained(cnet_id)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    image_encoder = None
    feature_extractor = None
    if args.condition_type == "ip-adapter":
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        logging.info("Attaching h94 IP-Adapter; ControlNet stays frozen stock OpenPose.")
        attach_ip_adapter(unet, load_ip_adapter_state_dict())
        controlnet.requires_grad_(False)
        controlnet.eval()
        adapter_params = collect_ip_adapter_parameters(unet)
        n_adapter = sum(p.numel() for p in adapter_params)
        logging.info("IP-Adapter trainable parameters: %s", n_adapter)
        if not (10_000_000 <= n_adapter <= 40_000_000):
            raise RuntimeError(
                f"IP-Adapter should be ~22M trainable params; got {n_adapter}. "
                "The optimiser is not looking at the adapter."
            )
        optimizer = torch.optim.AdamW(adapter_params, lr=args.lr)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", local_files_only=True
        )
        image_encoder.requires_grad_(False)
        # The h94 snapshot ships the vision encoder's config.json and weights but
        # no preprocessor_config.json, and this host is offline. Build the same
        # processor diffusers builds at inference time
        # (loaders/ip_adapter.py: CLIPImageProcessor(size=..., crop_size=...) from
        # image_encoder.config.image_size) so train and inference preprocess
        # identically. Those defaults are the standard OpenAI CLIP mean/std, which
        # is also what assets/weights/stable-diffusion-v1-5/feature_extractor holds.
        clip_image_size = image_encoder.config.image_size
        feature_extractor = CLIPImageProcessor(
            size=clip_image_size, crop_size=clip_image_size
        )
    else:
        controlnet.train()
        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    dataset = ControlNetDataset(
        args.data_root,
        args.condition_type,
        tokenizer=tokenizer,
        include_reference=args.include_reference,
    )
    if args.include_reference and len(dataset) == 0:
        raise RuntimeError("include-reference yielded an empty dataset; tracks need at least 2 colour frames.")
    logging.info("Dataset size %s include_reference=%s", len(dataset), args.include_reference)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    if args.condition_type == "ip-adapter":
        unet, controlnet, optimizer, dataloader = accelerator.prepare(
            unet, controlnet, optimizer, dataloader
        )
    else:
        controlnet, optimizer, dataloader = accelerator.prepare(
            controlnet, optimizer, dataloader
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if args.condition_type != "ip-adapter":
        unet.to(accelerator.device, dtype=weight_dtype)
    if image_encoder is not None:
        image_encoder.to(accelerator.device, dtype=weight_dtype)

    if (
        args.include_reference
        and args.controlnet_model_id is None
        and not args.from_scratch
        and args.condition_type != "ip-adapter"
    ):
        logging.info(
            "include-reference fine-tunes the pose ControlNet with appearance under the skeleton. "
            "Pass --controlnet-model-id to choose weights; default OpenPose is used otherwise."
        )

    global_step = 0
    last_ckpt_wall = time.monotonic()
    last_progress_wall = time.monotonic()
    started_wall = time.monotonic()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    stop_rule: TaskStopRule | None = None
    stop_samples = None
    stop_lpips: LpipsMetric | None = None
    if not args.no_task_stop:
        if accelerator.is_main_process:
            logging.info("Measuring coding-task floors before the first training step.")
            stop_lpips = LpipsMetric(device=str(accelerator.device))
            stop_samples = load_stop_samples(n_clips=args.task_eval_clips)
            bounds = measure_stop_floors(stop_samples, stop_lpips)
            stop_rule = TaskStopRule(bounds, Path(args.output_dir))
            logging.info(
                "task-stop bounds floor_lpips=%.4f floor_psnr=%.2f null_lpips=%.4f n=%s",
                bounds.floor_lpips,
                bounds.floor_psnr,
                bounds.null_lpips,
                len(stop_samples),
            )
        accelerator.wait_for_everyone()

    def _save(tag: str) -> None:
        if not accelerator.is_main_process:
            return
        dest = os.path.join(args.output_dir, tag)
        logging.info("Saving checkpoint to %s", dest)
        Path(dest).mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(controlnet).save_pretrained(dest)
        if args.condition_type == "ip-adapter":
            torch.save(
                export_ip_adapter_state_dict(accelerator.unwrap_model(unet)),
                os.path.join(dest, "ip-adapter.bin"),
            )

    def _broadcast_stop(stop: bool) -> bool:
        if accelerator.num_processes == 1:
            return stop
        flag = torch.tensor([1.0 if stop else 0.0], device=accelerator.device)
        flag = accelerator.reduce(flag, reduction="max")
        return bool(flag.item() >= 0.5)

    def _run_task_eval(epoch_1: int, *, kind: str, step: int | None = None) -> bool:
        stop = False
        if (
            not args.no_task_stop
            and accelerator.is_main_process
            and stop_rule is not None
            and stop_samples is not None
            and stop_lpips is not None
        ):
            tag = f"checkpoint-epoch-{epoch_1}"
            ckpt = os.path.join(args.output_dir, tag)
            if not Path(ckpt).is_dir():
                _save(tag)
            variant = CONDITION_TO_VARIANT[args.condition_type]
            predictions = generate_stop_predictions(
                stop_samples,
                checkpoint=ckpt,
                variant=variant,
                device=str(accelerator.device),
                steps=args.task_eval_steps,
            )
            scores = score_stop_generations(stop_samples, predictions, stop_lpips)
            eval_kind = "mid" if kind == "mid" else "epoch"
            decision = stop_rule.observe(
                epoch=epoch_1,
                lpips=scores.lpips,
                psnr=scores.psnr,
                step=step,
                kind=eval_kind,
            )
            logging.info(
                "task-eval epoch=%s kind=%s lpips=%.4f (floor %.4f) psnr=%.2f "
                "(floor %.2f) n=%s best=%s stop=%s %s",
                epoch_1,
                eval_kind,
                scores.lpips,
                stop_rule.bounds.floor_lpips,
                scores.psnr,
                stop_rule.bounds.floor_psnr,
                scores.n,
                decision.keep_as_best,
                decision.stop,
                decision.reason,
            )
            if decision.keep_as_best:
                best = Path(args.output_dir) / "checkpoint-best"
                if best.exists():
                    shutil.rmtree(best)
                shutil.copytree(ckpt, best)
            stop = decision.stop
        accelerator.wait_for_everyone()
        return _broadcast_stop(stop)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(tqdm(dataloader, disable=not accelerator.is_local_main_process)):
            trained = unet if args.condition_type == "ip-adapter" else controlnet
            with accelerator.accumulate(trained):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                controlnet_image = controlnet_cond_for_batch(
                    batch,
                    condition_type=args.condition_type,
                    include_reference=args.include_reference,
                    weight_dtype=weight_dtype,
                )

                if args.smoke_check_reference and global_step == 0:
                    if not args.include_reference:
                        raise ValueError("--smoke-check-reference requires --include-reference")
                    if args.condition_type == "ip-adapter":
                        raise ValueError(
                            "--smoke-check-reference paints the reference into ControlNet; "
                            "that is the pose-ref recipe. IP-Adapter appearance is the CLIP path."
                        )
                    delta = assert_reference_enters_controlnet(
                        controlnet,
                        batch,
                        weight_dtype=weight_dtype,
                        timesteps=timesteps,
                        noisy_latents=noisy_latents,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                    logging.info("smoke-check-reference passed residual_delta=%.6g", delta)
                    if accelerator.is_main_process:
                        dest = Path(args.output_dir)
                        dest.mkdir(parents=True, exist_ok=True)
                        pose_cpu = batch["conditioning_pixel_values"].detach().float().cpu()
                        ref_cpu = batch["reference_pixel_values"].detach().float().cpu()
                        composed_cpu = compose_pose_on_appearance_tensor(pose_cpu, ref_cpu)

                        def _chw_to_png(tensor: torch.Tensor, path: Path) -> None:
                            array = (
                                tensor[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
                            )
                            Image.fromarray(array).save(path)

                        _chw_to_png(pose_cpu, dest / "smoke-pose.png")
                        _chw_to_png(ref_cpu, dest / "smoke-reference.png")
                        _chw_to_png(composed_cpu, dest / "smoke-composed.png")
                        (dest / "smoke-check.json").write_text(
                            json.dumps({"residual_delta": delta, "passed": True}, indent=2)
                            + "\n"
                        )
                    if args.max_steps is None:
                        accelerator.wait_for_everyone()
                        return

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                unet_kwargs = {}
                if args.condition_type == "ip-adapter":
                    if image_encoder is None or feature_extractor is None:
                        raise RuntimeError("ip-adapter training is missing the CLIP image encoder")
                    unet_kwargs["added_cond_kwargs"] = {
                        "image_embeds": encode_reference_image_embeds(
                            image_encoder,
                            feature_extractor,
                            batch["reference_pixel_values"],
                            device=accelerator.device,
                            dtype=weight_dtype,
                        )
                    }

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                    **unet_kwargs,
                )[0]

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            now = time.monotonic()
            if accelerator.is_local_main_process and now - last_progress_wall >= args.progress_every_minutes * 60:
                logging.info(
                    "progress step=%s epoch=%s/%s loss=%.5f elapsed_min=%.1f",
                    global_step,
                    epoch + 1,
                    args.epochs,
                    float(loss.detach().float().cpu()),
                    (now - started_wall) / 60.0,
                )
                last_progress_wall = now
            if accelerator.is_main_process and now - last_ckpt_wall >= args.checkpoint_every_minutes * 60:
                accelerator.wait_for_everyone()
                _save(f"checkpoint-step-{global_step}")
                last_ckpt_wall = now
            if (
                args.eval_every_steps
                and global_step > 0
                and global_step % args.eval_every_steps == 0
            ):
                if _run_task_eval(epoch + 1, kind="mid", step=global_step):
                    logging.info("task-stop fired at step %s", global_step)
                    return
            if args.max_steps is not None and global_step >= args.max_steps:
                logging.info("Reached --max-steps %s", args.max_steps)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save(f"checkpoint-step-{global_step}")
                return

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save(f"checkpoint-epoch-{epoch+1}")
        if _run_task_eval(epoch + 1, kind="epoch"):
            logging.info("task-stop fired at epoch %s: %s", epoch + 1, getattr(stop_rule, "stop_reason", ""))
            return

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.info(f"Saving ControlNet to {args.output_dir}")
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)
        if args.condition_type == "ip-adapter":
            torch.save(
                export_ip_adapter_state_dict(accelerator.unwrap_model(unet)),
                os.path.join(args.output_dir, "ip-adapter.bin"),
            )


if __name__ == "__main__":
    main()
