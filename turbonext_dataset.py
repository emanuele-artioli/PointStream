from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _collect_pairs(images_dir: Path, poses_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    if not images_dir.exists() or not poses_dir.exists():
        return pairs

    pose_files = {p.name: p for p in sorted(poses_dir.glob("*.jpg"))}
    pose_files.update({p.name: p for p in sorted(poses_dir.glob("*.png"))})

    for image_path in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
        pose_path = pose_files.get(image_path.name)
        if pose_path is None:
            continue
        pairs.append((image_path, pose_path))
    return pairs


class VideoPoseDataset(Dataset):
    def __init__(self, root: str, image_dir_name: str = "images", pose_dir_name: str = "poses") -> None:
        self.root = Path(root)
        self.image_dir_name = image_dir_name
        self.pose_dir_name = pose_dir_name
        self.samples: List[Tuple[Path, Path, Path]] = []

        self._scan()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.pose_transform = transforms.Compose(
            [
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

    def _scan(self) -> None:
        for video_dir in sorted(self.root.iterdir()):
            if not video_dir.is_dir():
                continue
            for scene_dir in sorted(video_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue

                images_dir = scene_dir / self.image_dir_name
                poses_dir = scene_dir / self.pose_dir_name
                if images_dir.exists():
                    pairs = _collect_pairs(images_dir, poses_dir)
                    if pairs:
                        ref_image = pairs[0][0]
                        for image_path, pose_path in pairs:
                            self.samples.append((image_path, pose_path, ref_image))
                        continue

                crops_dir = scene_dir / "crops"
                poses_dir = scene_dir / self.pose_dir_name
                if crops_dir.exists() and poses_dir.exists():
                    for id_dir in sorted(crops_dir.iterdir()):
                        if not id_dir.is_dir():
                            continue
                        pose_id_dir = poses_dir / id_dir.name
                        if not pose_id_dir.exists():
                            continue
                        pairs = _collect_pairs(id_dir, pose_id_dir)
                        if not pairs:
                            continue
                        ref_image = pairs[0][0]
                        for image_path, pose_path in pairs:
                            self.samples.append((image_path, pose_path, ref_image))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        image_path, pose_path, ref_path = self.samples[idx]

        image = self._load_image(image_path)
        pose = self._load_image(pose_path)
        ref = self._load_image(ref_path)

        image_tensor = self.image_transform(image)
        pose_tensor = self.pose_transform(pose)
        ref_tensor = self.image_transform(ref)

        return {
            "pixel_values": image_tensor,
            "pose_images": pose_tensor,
            "ref_image": ref_tensor,
            "image_path": str(image_path),
        }
