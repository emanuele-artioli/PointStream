import torch
from pathlib import Path
from PIL import Image
from typing import Any
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load our newly added helpers from eval_checkpoint
from scripts.eval_checkpoint import (
    load_manifest, clip_condition_frame_paths,
    load_clip_tensor, load_seg_clip_tensor, resolve_reference_frame_path, load_image_rgb01
)
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

dataset_root = Path("assets/probe_set/training_view")
manifest = load_manifest(Path("assets/probe_set/manifest.json"))

clip: dict[str, Any] = {
    "video": "alcaraz_perricard",
    "scene": "scene_002",
    "track": "track_0001",
    "frame_ids": [0]
}
print(f"Testing alignment for hardcoded clip: {clip['video']}/{clip['scene']}/{clip['track']}")

frame_ids = sorted(clip["frame_ids"])
# Just pick the first frame
color_paths = [dataset_root / clip["video"] / "segmentations" / clip["scene"] / clip["track"] / f"frame_{frame_ids[0]:06d}.png"]
pose_paths = clip_condition_frame_paths(dataset_root, clip, "skeleton")[:1]
canny_paths = clip_condition_frame_paths(dataset_root, clip, "canny")[:1]

ref_path = resolve_reference_frame_path(dataset_root, clip)

pose_tensor = load_clip_tensor(pose_paths, 512)[0]
canny_tensor = load_clip_tensor(canny_paths, 512)[0]
seg_tensor = load_seg_clip_tensor(color_paths, 512)[0]
ref_tensor = load_image_rgb01(ref_path, 512)

out_dir = Path("outputs/debug_align")
out_dir.mkdir(parents=True, exist_ok=True)

# Function to save tensor [3, H, W] in [0,1] to PNG
def save_tensor(t: torch.Tensor, name: str):
    img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(out_dir / name)

save_tensor(pose_tensor, "01_pose.png")
save_tensor(canny_tensor, "02_canny.png")
save_tensor(seg_tensor, "03_seg.png")
save_tensor(ref_tensor, "04_ref.png")

# Also dump a composited version showing they align
# Let's average them
avg = (pose_tensor + canny_tensor + seg_tensor + ref_tensor) / 4.0
save_tensor(avg, "05_composite.png")

print("Dumped alignment check to outputs/debug_align/")

print("Testing MultiControlNetPipeline initialization...")

device = "cuda:0"
paths = [
    "assets/weights/pose-controlnet",
    "assets/weights/custom-controlnet",
    "assets/weights/seg-controlnet",
    "assets/weights/ip-adapter-controlnet"
]
controlnets = [ControlNetModel.from_pretrained(p, torch_dtype=torch.float16, local_files_only=True).to(device) for p in paths]
print("Loaded ControlNets.")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "assets/weights/stable-diffusion-v1-5", 
    controlnet=controlnets, 
    safety_checker=None, 
    torch_dtype=torch.float16,
    local_files_only=True
).to(device)
print(f"Pipeline on {pipe.device}")

ref_pil = Image.fromarray((ref_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
pose_pil = Image.fromarray((pose_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
canny_pil = Image.fromarray((canny_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
seg_pil = Image.fromarray((seg_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

print("Running pipeline...")
gen = torch.Generator(device=device).manual_seed(0)
result = pipe(
    "photorealistic tennis player, broadcast sports shot", 
    image=[pose_pil, canny_pil, seg_pil, ref_pil], 
    num_inference_steps=20, 
    generator=gen,
    controlnet_conditioning_scale=[1.0, 1.0, 1.0, 1.0]
).images[0]
print("Done!")
result.save("outputs/debug_align/06_gen.png")
