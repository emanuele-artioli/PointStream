import os
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TennisSkeletonDataset(Dataset):
    """Paired skeleton (+ optional reference) → colour dataset for Pix2Pix and Spade4Tennis.

    Directory layout:
        <root>/<subset>/           ← colour crops
        <root>/output_task_1_<subset>/skeleton_*.png  ← skeleton images
    """

    def __init__(self, root_dir: str, subset: str = "all", transform=None, include_reference: bool = False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.include_reference = include_reference

        if subset.lower() == "all":
            self.subsets = ["djokovic_front", "djokovic_back", "federer_front", "federer_back"]
        else:
            self.subsets = [subset]

        self.items: list[tuple[str, Path, Path]] = []
        self.subset_to_colors: dict[str, list[Path]] = {s: [] for s in self.subsets}

        for s in self.subsets:
            color_dir = self.root_dir / s
            skeleton_dir = self.root_dir / f"output_task_1_{s}"

            if not color_dir.exists() or not skeleton_dir.exists():
                continue

            for f in os.listdir(skeleton_dir):
                if f.startswith("skeleton_") and f.endswith(".png"):
                    color_name = f.replace("skeleton_", "")
                    color_path = color_dir / color_name
                    if color_path.exists():
                        skeleton_path = skeleton_dir / f
                        self.items.append((s, color_path, skeleton_path))
                        self.subset_to_colors[s].append(color_path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        subset, color_path, skeleton_path = self.items[idx]

        color_img = Image.open(color_path).convert("RGB")
        skeleton_img = Image.open(skeleton_path).convert("RGB")

        if self.include_reference:
            # Pick a random reference image from the same subset
            ref_color_path = random.choice(self.subset_to_colors[subset])
            ref_img = Image.open(ref_color_path).convert("RGB")
            
            if self.transform:
                combined = Image.new("RGB", (color_img.width * 3, color_img.height))
                combined.paste(skeleton_img, (0, 0))
                combined.paste(ref_img, (color_img.width, 0))
                combined.paste(color_img, (color_img.width * 2, 0))

                combined_tensor = self.transform(combined)

                _, h, w = combined_tensor.shape
                third_w = w // 3
                skeleton_tensor = combined_tensor[:, :, :third_w]
                ref_tensor = combined_tensor[:, :, third_w:third_w * 2]
                color_tensor = combined_tensor[:, :, third_w * 2:]
            else:
                to_tensor = transforms.ToTensor()
                color_tensor = to_tensor(color_img)
                skeleton_tensor = to_tensor(skeleton_img)
                ref_tensor = to_tensor(ref_img)
        else:
            if self.transform:
                combined = Image.new("RGB", (color_img.width * 2, color_img.height))
                combined.paste(skeleton_img, (0, 0))
                combined.paste(color_img, (color_img.width, 0))

                combined_tensor = self.transform(combined)

                _, h, w = combined_tensor.shape
                half_w = w // 2
                skeleton_tensor = combined_tensor[:, :, :half_w]
                color_tensor = combined_tensor[:, :, half_w:]
            else:
                to_tensor = transforms.ToTensor()
                color_tensor = to_tensor(color_img)
                skeleton_tensor = to_tensor(skeleton_img)

        # Normalise to [-1, 1]
        skeleton_tensor = (skeleton_tensor - 0.5) * 2.0
        color_tensor = (color_tensor - 0.5) * 2.0
        
        if self.include_reference:
            ref_tensor = (ref_tensor - 0.5) * 2.0
            return skeleton_tensor, ref_tensor, color_tensor

        return skeleton_tensor, color_tensor
