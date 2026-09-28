"""Compare hand models on clip 1 as encoders and as judges of each other.

Eight frames. Each model emits 21 image-plane joints. The encoder measure is the
shipped keypoint packet plus, for MANO models, a quantized pose packet. The judge
measure is pairwise joint agreement on the reference frames. HOPformer's object
head is not run: this clip has no EPIC object mesh.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.pipeline.hand_keypoints import HAND_CONNECTIONS, FINGER_COLORS, FrameHandPose, SingleHand

logger = logging.getLogger("hand_model_compare")

CLIP = Path("/home/itec/emanuele/Datasets/Egocentric-10K/curated/clip_01_factory035_close_hands.mp4")
HOP_ROOT = Path("/home/itec/emanuele/pointstream-data/third_party/HOPformer")
WILOR_DIR = Path("/home/itec/emanuele/pointstream-data/weights/wilor/pretrained_models")
HOP_CKPT = Path("/home/itec/emanuele/pointstream-data/weights/hopformer/epic_epoch125.ckpt")
DELTA_SRC = Path("/home/itec/emanuele/pointstream-data/third_party/deltadorsal/src")
DINOV3_ROOT = Path("/home/itec/emanuele/pointstream-data/third_party/dinov3")
DINO_PTH = Path(
    "/home/itec/emanuele/pointstream-data/weights/deltadorsal/dinov3_vitl16_from_deltadorsal-8aa4cbdd.pth"
)
DELTA_HEADS = Path("/home/itec/emanuele/pointstream-data/weights/deltadorsal/heads.pt")
MANO_RIGHT = Path("/home/itec/emanuele/Datasets/MANO/mano_v1_2/models/MANO_RIGHT.pkl")
FRAME_W = 1920
FRAME_H = 1080
NET = 224
FPS = 30.0

MANO_JOINT_SRC = np.array([0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9], dtype=np.int64)
OPENPOSE_DST = np.array([0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], dtype=np.int64)
TIP_VERTS = np.array([743, 333, 443, 554, 671], dtype=np.int64)
TIP_DST = np.array([4, 8, 12, 16, 20], dtype=np.int64)


def payload_kbps(packets: list[bytes], fps: float = FPS) -> float:
    if not packets:
        return 0.0
    return (sum(len(packet) for packet in packets) * 8) / (len(packets) / fps) / 1000.0


def quantize_axis_angle(pose: np.ndarray) -> tuple[bytes, np.ndarray]:
    flat = np.asarray(pose, dtype=np.float32).reshape(-1)
    scale = float(max(np.max(np.abs(flat)), 1e-6))
    codes = np.clip(np.round(flat / scale * 127.0), -127, 127).astype(np.int8)
    recovered = codes.astype(np.float32) / 127.0 * scale
    return struct.pack("<f", scale) + codes.tobytes(), recovered


def network_to_frame(pixels: np.ndarray, frame_w: int = FRAME_W, frame_h: int = FRAME_H) -> np.ndarray:
    mapped = np.asarray(pixels, dtype=np.float64).copy()
    mapped[:, 0] *= frame_w / NET
    mapped[:, 1] *= frame_h / NET
    return mapped


def openpose_from_mano(joints16: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints16, dtype=np.float64)
    verts = np.asarray(vertices, dtype=np.float64)
    out = np.zeros((21, 3), dtype=np.float64)
    out[OPENPOSE_DST] = joints[MANO_JOINT_SRC]
    out[TIP_DST] = verts[TIP_VERTS]
    return out


def fit_joints_to_bbox(joints_xyz: np.ndarray, bbox: list[int]) -> np.ndarray:
    xy = np.asarray(joints_xyz, dtype=np.float64)[:, :2].copy()
    xy[:, 1] *= -1.0
    xy -= xy[0]
    span = max(float(np.ptp(xy, axis=0).max()), 1e-6)
    x1, y1, x2, y2 = bbox
    side = max(int(x2) - int(x1), int(y2) - int(y1), 1)
    scale = 0.8 * side / span
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    pixels = xy * scale
    pixels[:, 0] += center[0]
    pixels[:, 1] += center[1]
    return pixels


def pose_from_pixels(frame_idx: int, hands: list[tuple[str, np.ndarray, list[int]]]) -> FrameHandPose:
    built: list[SingleHand] = []
    for side, pixels, bbox in hands:
        pts = np.asarray(pixels, dtype=np.float64).reshape(21, 2)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        norms = [[float(p[0]) / FRAME_W, float(p[1]) / FRAME_H, 0.0] for p in pts]
        built.append(
            SingleHand(side, 0.9, [x1, y1, x2, y2], norms, [[float(p[0]), float(p[1])] for p in pts])
        )
    return FrameHandPose(frame_idx=frame_idx, hands=built)


def encoder_report(poses: list[FrameHandPose]) -> dict[str, float]:
    from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks
    from demo.pipeline.keypoint_compressor import KeypointCompressor

    packets = [KeypointCompressor.compress_frame(pose, FRAME_W, FRAME_H) for pose in poses]
    decoded = [
        FrameHandPose(index, KeypointCompressor.decompress_frame(packet, FRAME_W, FRAME_H))
        for index, packet in enumerate(packets)
    ]
    scores = score_pose_tracks(poses, decoded)
    return {
        "keypoint_kbps": payload_kbps(packets),
        "roundtrip_mpjpe_px": scores["mpjpe_pixels"],
        "roundtrip_pck50_all_gt": scores["pck50_all_gt"],
        "mean_packet_bytes": float(np.mean([len(packet) for packet in packets])) if packets else 0.0,
    }


def sample_clip(source: Path, dest: Path, count: int) -> None:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate,duration",
            "-of", "json", str(source),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    rate = stream.get("r_frame_rate", "30/1")
    num, den = rate.split("/")
    fps = float(num) / float(den)
    frames = int(stream.get("nb_frames") or 0)
    if frames <= 0:
        frames = max(1, int(float(stream.get("duration") or 1.0) * fps))
    if frames <= count:
        indices = list(range(frames))
    else:
        indices = [int(round(i * (frames - 1) / (count - 1))) for i in range(count)]
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source), "-vf", f"select='{select}'",
            "-vsync", "vfr", str(dest),
        ],
        check=True, capture_output=True,
    )
    logger.info("sampled frames %s from %s into %s", indices, frames, dest)


def draw_pose(frame: np.ndarray, pose: FrameHandPose, label: str) -> np.ndarray:
    import cv2

    canvas = frame.copy()
    for hand in pose.hands:
        points = hand.landmarks_pixel
        for bone_index, (start, end) in enumerate(HAND_CONNECTIONS):
            p1 = (int(points[start][0]), int(points[start][1]))
            p2 = (int(points[end][0]), int(points[end][1]))
            cv2.line(canvas, p1, p2, FINGER_COLORS[min(bone_index // 4, 4)], 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
    cv2.rectangle(canvas, (0, 0), (420, 36), (0, 0, 0), -1)
    cv2.putText(canvas, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def square_crop(frame: np.ndarray, bbox: list[int], size: int) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1, 1.0) * 1.2
    half = side / 2.0
    left, top = int(round(center_x - half)), int(round(center_y - half))
    right, bottom = int(round(center_x + half)), int(round(center_y + half))
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    crop = frame[top:bottom, left:right]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def imagenet(rgb: np.ndarray):
    import torch

    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32)).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _epic_intrinsics():
    import torch

    focal = 5000.0
    fx = focal * (NET / 854.0)
    fy = focal * (NET / 480.0)
    matrix = torch.tensor(
        [[fx, 0.0, NET / 2.0], [0.0, fy, NET / 2.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    return matrix.unsqueeze(0)


def load_hopformer(device: str):
    import torch

    hop = str(HOP_ROOT)
    if hop not in sys.path:
        sys.path.insert(0, hop)
    previous = os.getcwd()
    os.chdir(hop)
    try:
        import common.object_tensor_epic as epic_objects

        class _DummyObjects:
            def __init__(self, *args, **kwargs):
                self.dev = torch.device("cpu")

            def to(self, target):
                self.dev = target
                return self

            def forward(self, *args, **kwargs):
                raise RuntimeError("EPIC object meshes are not used for the hand comparison")

        epic_objects.ObjectTensorsEPIC = _DummyObjects
        import src.models.wilor as wilor_pkg
        import src.models.wilor.models as wilor_models
        import src.models.wilor.models.wilor as wilor_mod

        sys.modules.setdefault("wilor", wilor_pkg)
        sys.modules.setdefault("wilor.models", wilor_models)
        sys.modules.setdefault("wilor.models.wilor", wilor_mod)
        from src.models.transformer_sf.model import TransformerSF

        args = Namespace(
            backbone="vit-g",
            focal_length=1000.0,
            img_res=NET,
            decoder_dim=512,
            decoder_depth=12,
            queries="per_joint",
            dataset="epic",
            freeze_backbone=True,
            without_arti=True,
            wilor_default_dir=str(WILOR_DIR),
        )
        model = TransformerSF("vit-g", args.focal_length, args.img_res, args)
        blob = torch.load(HOP_CKPT, map_location="cpu", weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        stripped = {}
        for key, value in state.items():
            name = key[6:] if key.startswith("model.") else key
            if name.startswith("arti_head.object_tensors"):
                continue
            stripped[name] = value
        missing, unexpected = model.load_state_dict(stripped, strict=False)
        hand_missing = [key for key in missing if not key.startswith("arti_head.object_tensors")]
        if hand_missing:
            raise RuntimeError(f"HOPformer hand weights missing: {hand_missing[:12]}")
        logger.info("HOPformer loaded missing=%s unexpected=%s", len(missing), len(unexpected))
        model.to(device).eval()
        return model
    finally:
        os.chdir(previous)


def hopformer_poses(model, frames: list[np.ndarray], boxes: list[FrameHandPose], device: str) -> tuple[list[FrameHandPose], list[bytes]]:
    import cv2
    import torch

    poses: list[FrameHandPose] = []
    packets: list[bytes] = []
    intrinsics = _epic_intrinsics().to(device)
    for index, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scene = cv2.resize(rgb, (NET, NET), interpolation=cv2.INTER_AREA)
        found = boxes[index].hands if index < len(boxes) else []
        crops = {"Left": None, "Right": None}
        box_of = {"Left": None, "Right": None}
        for hand in found:
            side = "Right" if hand.handedness.lower().startswith("r") else "Left"
            crops[side] = square_crop(frame, hand.bbox, 256)
            box_of[side] = hand.bbox
        left = crops["Left"]
        right = crops["Right"]
        if left is None:
            left = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            left = cv2.flip(cv2.cvtColor(left, cv2.COLOR_BGR2RGB), 1)
        if right is None:
            right = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        batch_img = imagenet(scene).unsqueeze(0).to(device)
        left_t = imagenet(left).unsqueeze(0).to(device)
        right_t = imagenet(right).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.backbone(batch_img)
            batched = torch.cat([left_t, right_t], dim=0)
            hand_feats = model.hand_backbone.get_vit_features({"img": batched})
            projected = model.jwem_projection(hand_feats)
            left_proj, right_proj = projected.chunk(2, dim=0)
            tokens = features.view(1, model.feature_dim, -1).permute(0, 2, 1)
            combined = torch.cat((left_proj, right_proj), dim=1)
            params_r, params_l, _arti = model.head(combined, tokens)
            out_r = model.mano_r(rotmat=params_r.pose, shape=params_r.shape, K=intrinsics, cam=params_r.root)
            out_l = model.mano_l(rotmat=params_l.pose, shape=params_l.shape, K=intrinsics, cam=params_l.root)
        hands = []
        pose_bytes = bytearray()
        for side, output, params in (("Right", out_r, params_r), ("Left", out_l, params_l)):
            if box_of[side] is None:
                continue
            joints = output["joints3d." + ("r" if side == "Right" else "l")][0].detach().cpu().numpy()
            vertices = output["vertices." + ("r" if side == "Right" else "l")][0].detach().cpu().numpy()
            cam = output["cam_t." + ("r" if side == "Right" else "l")][0].detach().cpu().numpy()
            posed = openpose_from_mano(joints, vertices)
            posed = posed + cam.reshape(1, 3)
            pixels = _project(posed, intrinsics[0].detach().cpu().numpy())
            full = network_to_frame(pixels)
            hands.append((side, full, list(box_of[side])))
            from common.rot import matrix_to_axis_angle

            axis = matrix_to_axis_angle(params.pose[0].detach().cpu()).reshape(-1).numpy()
            packet, _ = quantize_axis_angle(axis)
            pose_bytes.extend(packet)
        poses.append(pose_from_pixels(index, hands))
        packets.append(bytes(pose_bytes))
    return poses, packets


def _project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = np.clip(points[:, 2:3], 1e-6, None)
    homog = points / depth
    pixels = homog[:, :2] * np.array([intrinsics[0, 0], intrinsics[1, 1]]) + intrinsics[:2, 2]
    return pixels


def load_deltadorsal(device: str):
    import torch

    for path in (str(DINOV3_ROOT), str(DELTA_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from dinov3.hub.backbones import dinov3_vitl16
    from models.deltadorsalnet import ChangeEncoder, ResidualPoseHead
    from models.mano_wrapper import MANOPoseOnly

    vit = dinov3_vitl16(pretrained=True, weights=str(DINO_PTH))
    change = ChangeEncoder(1024, mid_dim=256, use_delta=True, use_ft_f0=True)
    head = ResidualPoseHead(z_dim=256, out_dim=45, use_prior=False, gated=False, pool="avg")
    rest = torch.load(DELTA_HEADS, map_location="cpu", weights_only=True)
    change.load_state_dict({key.split(".", 2)[-1]: value for key, value in rest.items() if key.startswith("backbone.change.")})
    head.load_state_dict({key[len("head.") :]: value for key, value in rest.items() if key.startswith("head.")})
    mano = MANOPoseOnly(model_path=str(MANO_RIGHT), is_rhand=True, use_pca=False, flat_hand_mean=True)
    for module in (vit, change, head, mano):
        module.to(device).eval()
    return vit, change, head, mano


def _delta_features(vit, image):
    layers = vit.get_intermediate_layers(image, n=range(24), reshape=True, norm=True)
    return layers[-1]


def deltardorsal_poses(bundle, frames: list[np.ndarray], boxes: list[FrameHandPose], device: str) -> tuple[list[FrameHandPose], list[bytes]]:
    import cv2
    import torch

    vit, change, head, mano = bundle
    sys.path.insert(0, str(DELTA_SRC))
    from utils.mano_utils import mano_to_openpose

    poses: list[FrameHandPose] = []
    packets: list[bytes] = []
    bases: dict[str, np.ndarray] = {}
    for index, frame in enumerate(frames):
        found = boxes[index].hands if index < len(boxes) else []
        hands = []
        pose_bytes = bytearray()
        for hand in found:
            side = "Right" if hand.handedness.lower().startswith("r") else "Left"
            crop = square_crop(frame, hand.bbox, 512)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            flipped = side == "Left"
            if flipped:
                rgb = cv2.flip(rgb, 1)
            if side not in bases:
                bases[side] = rgb
            current = imagenet(rgb).unsqueeze(0).to(device)
            base = imagenet(bases[side]).unsqueeze(0).to(device)
            with torch.no_grad():
                latent = change(_delta_features(vit, current), _delta_features(vit, base))
                theta, _delta, _gate = head(latent, torch.zeros(1, 45, device=device))
                betas = torch.zeros(1, 10, device=device)
                mano_out = mano(betas, theta)
                joints = mano_to_openpose(mano_out.joints, mano_out.vertices)[0].detach().cpu().numpy()
            if flipped:
                joints = joints.copy()
                joints[:, 0] *= -1.0
            pixels = fit_joints_to_bbox(joints, hand.bbox)
            hands.append((side, pixels, list(hand.bbox)))
            packet, _ = quantize_axis_angle(theta[0].detach().cpu().numpy())
            pose_bytes.extend(packet)
        poses.append(pose_from_pixels(index, hands))
        packets.append(bytes(pose_bytes))
    return poses, packets


def read_frames(path: Path) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (FRAME_W, FRAME_H):
            frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def montage(frames: list[np.ndarray], named: dict[str, list[FrameHandPose]], dest: Path) -> None:
    import cv2

    labels = list(named)
    panels = []
    for index, frame in enumerate(frames):
        row = []
        for label in labels:
            panel = draw_pose(frame, named[label][index], label)
            row.append(cv2.resize(panel, (480, 270), interpolation=cv2.INTER_AREA))
        panels.append(np.concatenate(row, axis=1))
    image = np.concatenate(panels, axis=0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), image)


def judge_matrix(named: dict[str, list[FrameHandPose]]) -> dict[str, dict[str, float]]:
    from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks

    matrix: dict[str, dict[str, float]] = {}
    for reference_name, reference in named.items():
        matrix[reference_name] = {}
        for other_name, other in named.items():
            scores = score_pose_tracks(reference, other)
            matrix[reference_name][other_name] = {
                "mpjpe_px": scores["mpjpe_pixels"],
                "detection_rate": scores["detection_rate"],
                "pck50_all_gt": scores["pck50_all_gt"],
                "handedness_agreement": scores["handedness_agreement"],
            }
    return matrix


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=CLIP)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sampled = out / "clip1_sample.mp4"
    sample_clip(args.video, sampled, args.frames)
    frames = read_frames(sampled)

    from demo.evaluation.pose_backends import extract_dwpose_hands, extract_rtm_hand, extract_rtm_wholebody_hands
    from demo.evaluation.hamer_backend import extract_hamer

    logger.info("running DW-Pose, RTMPose, HaMeR")
    dwpose = extract_dwpose_hands(sampled)
    rtm = extract_rtm_hand(sampled)
    boxes = extract_rtm_wholebody_hands(sampled)
    hamer = extract_hamer(sampled)

    logger.info("running DeltaDorsal")
    delta_bundle = load_deltadorsal(device)
    delta, delta_packets = deltardorsal_poses(delta_bundle, frames, boxes, device)
    del delta_bundle
    if device == "cuda":
        torch.cuda.empty_cache()

    logger.info("running HOPformer")
    hop_model = load_hopformer(device)
    hop, hop_packets = hopformer_poses(hop_model, frames, boxes, device)
    del hop_model
    if device == "cuda":
        torch.cuda.empty_cache()

    named = {
        "dwpose": dwpose,
        "rtmpose": rtm,
        "hamer": hamer,
        "deltadorsal": delta,
        "hopformer": hop,
    }
    encoders = {name: encoder_report(poses) for name, poses in named.items()}
    encoders["deltadorsal"]["mano_kbps"] = payload_kbps(delta_packets)
    encoders["hopformer"]["mano_kbps"] = payload_kbps(hop_packets)
    montage_path = out / "montage.png"
    montage(frames, named, montage_path)
    report = {
        "clip": str(args.video),
        "frames": len(frames),
        "device": device,
        "encoders": encoders,
        "judges": judge_matrix(named),
        "notes": {
            "crops": "DeltaDorsal and HOPformer hand crops use RTM whole-body boxes.",
            "deltadorsal_2d": "Canonical MANO joints, X mirrored for left hands, then scaled and translated onto the box. No camera is predicted.",
            "deltadorsal_base": "The first sampled crop of each hand is the neutral base.",
            "hopformer_2d": "Projected with the EPIC training intrinsics (focal 5000 scaled by 224/854 and 224/480) and mapped from the 224 network image back to 1920x1080.",
            "hopformer_object": "The object head was not run. This clip has no EPIC object mesh.",
            "hopformer_checkpoint": str(HOP_CKPT),
        },
        "montage": str(montage_path),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("wrote %s", out / "report.json")


if __name__ == "__main__":
    main()
