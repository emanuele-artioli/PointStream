"""HaMeR / Hand-Texture-Module second judge (OpenPose-21 projected to the image).

Requires the Hand-Texture-Module checkout plus MANO_RIGHT.pkl. Person/hand
boxes come from RTMPose whole-body so we do not need Detectron2/ViTDet.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand

logger = logging.getLogger(__name__)

HTM_ROOT = Path(os.environ.get("HTM_ROOT", "/home/itec/emanuele/Datasets/HaMeR/Hand-Texture-Module"))
MANO_RIGHT = Path(os.environ.get("MANO_RIGHT", "/home/itec/emanuele/Datasets/MANO/mano_v1_2/models/MANO_RIGHT.pkl"))

_MODEL = None


def _load_hamer():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if str(HTM_ROOT) not in sys.path:
        sys.path.insert(0, str(HTM_ROOT))
    import torch
    from hamer.models import load_hamer
    from hamer.utils.renderer import cam_crop_to_full

    ckpt = HTM_ROOT / "_DATA/hamer_ckpts/checkpoints/texture_supervised_hamer_weights.ckpt"
    if not ckpt.exists():
        ckpt = HTM_ROOT / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
    if not ckpt.exists():
        matches = list(HTM_ROOT.rglob("*.ckpt"))
        if not matches:
            raise FileNotFoundError(f"No HaMeR checkpoint under {HTM_ROOT}")
        ckpt = matches[0]
    os.environ.setdefault("MANO_MODEL_PATH", str(MANO_RIGHT.parent))
    model, model_cfg = load_hamer(str(ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    _MODEL = (model, model_cfg, device, cam_crop_to_full)
    logger.info("Loaded HaMeR from %s on %s", ckpt, device)
    return _MODEL


def extract_hamer(video_path: Path, max_frames: int | None = None) -> list[FrameHandPose]:
    """Run texture-supervised HaMeR on RTM whole-body hand boxes."""
    import torch
    from demo.evaluation.pose_backends import extract_rtm_wholebody_hands
    from demo.pipeline.background_codec import read_video_frames_robust
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to

    model, model_cfg, device, cam_crop_to_full = _load_hamer()
    frames = read_video_frames_robust(video_path, max_frames=max_frames)
    boxes_by_frame = extract_rtm_wholebody_hands(video_path, max_frames=max_frames)
    poses: list[FrameHandPose] = []
    for idx, frame in enumerate(frames):
        if (frame.shape[1], frame.shape[0]) != (1920, 1080):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        hands_in = boxes_by_frame[idx].hands if idx < len(boxes_by_frame) else []
        if not hands_in:
            poses.append(FrameHandPose(frame_idx=idx, hands=[]))
            continue
        bboxes = np.array([h.bbox for h in hands_in], dtype=np.float32)
        is_right = np.array([1 if h.handedness.lower().startswith("r") else 0 for h in hands_in], dtype=np.int32)
        dataset = ViTDetDataset(model_cfg, frame, bboxes, is_right, rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        out_hands: list[SingleHand] = []
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
            kps = out["pred_keypoints_2d"].detach().cpu().numpy()
            rights = batch["right"].detach().cpu().numpy()
            for i in range(kps.shape[0]):
                pts = kps[i]
                if pts.max() <= 1.5:
                    pts = pts.copy()
                    pts[:, 0] *= 1920.0
                    pts[:, 1] *= 1080.0
                xs, ys = pts[:21, 0], pts[:21, 1]
                x1, y1 = int(max(0, xs.min() - 8)), int(max(0, ys.min() - 8))
                x2, y2 = int(min(1920, xs.max() + 8)), int(min(1080, ys.max() + 8))
                lms_px = [[float(p[0]), float(p[1])] for p in pts[:21]]
                lms_nm = [[p[0] / 1920.0, p[1] / 1080.0, 0.0] for p in lms_px]
                side = "Right" if int(rights[i]) == 1 else "Left"
                out_hands.append(SingleHand(side, 0.9, [x1, y1, x2, y2], lms_nm, lms_px))
        poses.append(FrameHandPose(frame_idx=idx, hands=out_hands))
        if idx % 30 == 0:
            logger.info("%s hamer frame %s/%s hands=%s", video_path.name, idx, len(frames), len(out_hands))
    return poses
