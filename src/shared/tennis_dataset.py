import os
import random
import glob
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

def pad_to_square(img, fill=0):
    """Pads an image with a constant fill color to make it square without stretching."""
    w, h = img.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2
    pad_right = max_dim - w - pad_left
    pad_bottom = max_dim - h - pad_top
    return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill, padding_mode='constant')

class TennisSkeletonDataset(Dataset):
    """Paired skeleton (+ optional reference) → colour dataset for Pix2Pix and Spade4Tennis.

    Directory layout:
        assets/dataset/<video>/segmentations/scene_XXX/track_YYY/frame_ZZZ.png
        assets/dataset/<video>/segmentations/scene_XXX/track_YYY_skeleton/frame_ZZZ.png
    """

    def __init__(self, root_dir: str, target_size: int = 512, transform=None, include_reference: bool = False):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.transform = transform
        self.include_reference = include_reference

        # Items are tuples of (color_path, skeleton_path, track_id)
        self.items: list[tuple[Path, Path, str]] = []
        
        # Map track_id to a list of valid color paths in that track (for reference frame sampling)
        self.track_to_colors: dict[str, list[Path]] = {}

        # Parse the new directory structure
        # root_dir is usually assets/dataset
        # We look for */segmentations/scene_*/track_* (excluding *_skeleton)
        
        search_pattern = os.path.join(str(self.root_dir), "*", "segmentations", "scene_*", "track_*")
        all_tracks = glob.glob(search_pattern)
        
        for track_dir_str in all_tracks:
            if track_dir_str.endswith("_skeleton"):
                continue
                
            track_dir = Path(track_dir_str)
            skel_dir = track_dir.with_name(f"{track_dir.name}_skeleton")
            
            if not skel_dir.exists():
                continue
                
            # Create a unique track ID spanning video and scene
            # track_dir parts: .../dataset/<video>/segmentations/<scene>/<track>
            parts = track_dir.parts
            video_name = parts[-4]
            scene_name = parts[-2]
            track_name = parts[-1]
            unique_track_id = f"{video_name}_{scene_name}_{track_name}"
            
            color_frames = sorted(track_dir.glob("frame_*.png"))
            skel_frames = sorted(skel_dir.glob("frame_*.png"))
            
            if len(color_frames) < 2 and self.include_reference:
                # We need at least 2 frames if we want to pick a different reference frame
                continue
                
            if unique_track_id not in self.track_to_colors:
                self.track_to_colors[unique_track_id] = []
                
            # Pair them sequentially by order, accommodating missing frames at the tail if extractor stopped early
            min_len = min(len(color_frames), len(skel_frames))
            for i in range(min_len):
                color_path = color_frames[i]
                skel_path = skel_frames[i]
                self.items.append((color_path, skel_path, unique_track_id))
                self.track_to_colors[unique_track_id].append(color_path)

        # Base transform for converting to tensor and resizing
        self.base_transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.items)

    def _process_image(self, img_path: Path) -> torch.Tensor:
        """Loads, pads to square with black, resizes to target_size, and converts to tensor."""
        img: Image.Image = Image.open(img_path)
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, (0, 0, 0, 255))
            img = Image.alpha_composite(background, img).convert("RGB")
        else:
            img = img.convert("RGB")
        img = pad_to_square(img, fill=0)
        tensor = self.base_transform(img)
        return tensor

    def __getitem__(self, idx: int):
        color_path, skeleton_path, track_id = self.items[idx]

        color_tensor = self._process_image(color_path)
        skeleton_tensor = self._process_image(skeleton_path)
        
        # Apply data augmentation transformations if provided (e.g., random flip)
        # Note: self.transform must be a transform that accepts and returns tensors
        if self.transform:
            # We stack them to ensure same random transforms (like flipping) are applied to all
            if self.include_reference:
                ref_color_path = random.choice(self.track_to_colors[track_id])
                ref_tensor = self._process_image(ref_color_path)
                
                stacked = torch.cat([skeleton_tensor, ref_tensor, color_tensor], dim=0) # [9, H, W]
                stacked = self.transform(stacked)
                skeleton_tensor = stacked[0:3]
                ref_tensor = stacked[3:6]
                color_tensor = stacked[6:9]
            else:
                stacked = torch.cat([skeleton_tensor, color_tensor], dim=0) # [6, H, W]
                stacked = self.transform(stacked)
                skeleton_tensor = stacked[0:3]
                color_tensor = stacked[3:6]
        else:
            if self.include_reference:
                ref_color_path = random.choice(self.track_to_colors[track_id])
                ref_tensor = self._process_image(ref_color_path)

        # Normalize to [-1, 1]
        skeleton_tensor = (skeleton_tensor - 0.5) * 2.0
        color_tensor = (color_tensor - 0.5) * 2.0
        
        if self.include_reference:
            ref_tensor = (ref_tensor - 0.5) * 2.0
            return skeleton_tensor, ref_tensor, color_tensor

        return skeleton_tensor, color_tensor
