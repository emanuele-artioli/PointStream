"""Pinned: still used by pix2pix/spade training, eval_checkpoint, and process-dataset tooling. Not retired in BP15.

Directory layout uses two naming conventions in one track group — pair by
position in the sorted frame lists, never by reconstructing a filename.
"""
from __future__ import annotations

import os
import random
import glob
import sqlite3  # noqa: F401
from pathlib import Path
from typing import Any
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

    Directory layout (two naming conventions in one track group):
        crop / _canny / _pose_body / _pose_racket: ``frame_{global_source_id}.png``
        ``_skeleton``: ``frame_{track_local_index}.png`` (zero-based)

    Colour and skeleton files do **not** share names. A track that starts at
    source frame 29 has crop ``frame_000029.png`` and skeleton
    ``frame_000000.png`` for the same instant. Pair them by position in the
    sorted ``frame_*.png`` lists, never by reconstructing a filename.
    """

    #: Directory suffixes that are derived outputs, not tracks.
    DERIVED_SUFFIXES = ("_skeleton", "_canny", "_pose_racket", "_pose_body")

    #: Selectable pose-condition variants (see scripts/process_dataset.py).
    #: `pose_body` is the one the decoder can reproduce for free and
    #: bit-identically; `pose_racket` matches what the legacy checkpoints were
    #: trained on but needs racket geometry the wire format does not carry.
    #: `skeleton` is the legacy tree, kept only for reproducing old runs -- its
    #: filenames are positional, so never pair it by name.
    CONDITION_SUFFIXES = {
        "pose_body": "_pose_body",
        "pose_racket": "_pose_racket",
        "skeleton": "_skeleton",
    }

    REFERENCE_MODES = ("first", "keyframe", "offset", "random", "deterministic")

    def __init__(
        self,
        root_dir: str | Path = "assets/dataset",
        target_size: int = 512,
        transform=None,
        include_reference: bool = False,
        condition: str = "pose_body",
        reference_mode: str = "first",
        keyframe_interval: int = 16,
        reference_offset: int = 1,
    ):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.transform = transform
        self.include_reference = include_reference
        if condition not in self.CONDITION_SUFFIXES:
            raise ValueError(
                f"Unknown condition {condition!r}; expected one of {sorted(self.CONDITION_SUFFIXES)}"
            )
        self.condition = condition
        cond_suffix = self.CONDITION_SUFFIXES[condition]
        if reference_mode not in self.REFERENCE_MODES:
            raise ValueError(
                f"Unknown reference_mode {reference_mode!r}; expected one of {sorted(self.REFERENCE_MODES)}"
            )
        self.reference_mode = reference_mode
        self.keyframe_interval = max(1, keyframe_interval)
        self.reference_offset = max(1, reference_offset)

        # Items are tuples of (color_path, condition_path, track_id)
        self.items: list[tuple[Path, Path, str]] = []
        self.item_track_indices: list[int] = []

        # Map track_id to a list of valid color paths in that track (for reference frame sampling)
        self.track_to_colors: dict[str, list[Path]] = {}

        # Parse the new directory structure
        # root_dir is usually assets/dataset
        # We look for */segmentations/scene_*/track_* (excluding derived dirs)

        search_pattern = os.path.join(str(self.root_dir), "*", "segmentations", "scene_*", "track_*")
        all_tracks = sorted(glob.glob(search_pattern))

        for track_dir_str in all_tracks:
            if track_dir_str.endswith(self.DERIVED_SUFFIXES):
                continue

            track_dir = Path(track_dir_str)
            skel_dir = track_dir.with_name(f"{track_dir.name}{cond_suffix}")

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
                self.item_track_indices.append(i)
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

    def _select_reference_path(
        self,
        track_id: str,
        track_pos: int = 0,
        item_idx: int = 0,
    ) -> Path:
        colors = self.track_to_colors[track_id]
        if not colors:
            raise RuntimeError(f"No color frames available for track {track_id!r}")
        if self.reference_mode == "first":
            return colors[0]
        elif self.reference_mode == "keyframe":
            ref_idx = (track_pos // self.keyframe_interval) * self.keyframe_interval
            ref_idx = min(ref_idx, len(colors) - 1)
            return colors[ref_idx]
        elif self.reference_mode == "offset":
            ref_idx = max(0, track_pos - self.reference_offset)
            ref_idx = min(ref_idx, len(colors) - 1)
            return colors[ref_idx]
        elif self.reference_mode == "deterministic":
            # Stable first reference policy matching transmitted keyframe availability;
            # replaces the historical target-copy shortcut (colors[idx % len(colors)]).
            return colors[0]
        else:  # "random"
            return random.choice(colors)

    def get_reference_info(self, idx: int) -> dict[str, Any]:
        """Inspect and record source and selected reference paths and IDs for sample idx."""
        color_path, skel_path, track_id = self.items[idx]
        track_pos = self.item_track_indices[idx] if idx < len(self.item_track_indices) else 0
        ref_color_path = self._select_reference_path(track_id, track_pos=track_pos, item_idx=idx)
        return {
            "track_id": track_id,
            "track_pos": track_pos,
            "source_path": color_path,
            "reference_path": ref_color_path,
            "source_id": color_path.name,
            "reference_id": ref_color_path.name,
            "is_target_match": bool(color_path == ref_color_path),
            "reference_mode": self.reference_mode,
        }

    def __getitem__(self, idx: int):
        color_path, skeleton_path, track_id = self.items[idx]
        track_pos = self.item_track_indices[idx] if idx < len(self.item_track_indices) else 0

        color_tensor = self._process_image(color_path)
        skeleton_tensor = self._process_image(skeleton_path)
        
        # Apply data augmentation transformations if provided (e.g., random flip)
        # Note: self.transform must be a transform that accepts and returns tensors
        if self.transform:
            # We stack them to ensure same random transforms (like flipping) are applied to all
            if self.include_reference:
                ref_color_path = self._select_reference_path(track_id, track_pos=track_pos, item_idx=idx)
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
                ref_color_path = self._select_reference_path(track_id, track_pos=track_pos, item_idx=idx)
                ref_tensor = self._process_image(ref_color_path)

        # Normalize to [-1, 1]
        skeleton_tensor = (skeleton_tensor - 0.5) * 2.0
        color_tensor = (color_tensor - 0.5) * 2.0
        
        if self.include_reference:
            ref_tensor = (ref_tensor - 0.5) * 2.0
            return skeleton_tensor, ref_tensor, color_tensor

        return skeleton_tensor, color_tensor
