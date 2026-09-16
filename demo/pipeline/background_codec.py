"""Background codec: downscaled low-rate stream and static plate modes for egocentric background."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from demo.pipeline.foreground_segmenter import mask_out_hands
from demo.pipeline.hand_keypoints import FrameHandPose


class BackgroundCodec:
    """Encodes the infilled/downscaled background stream using SVT-AV1 at ultra-low bitrates."""

    def __init__(
        self,
        downscale_factor: float = 0.5,
        target_bitrate_kbps: int = 250,
        preset: int = 7,
        encoder: str = "libsvtav1",
        mask_hands: bool = False,
        scale_resolution: tuple[int, int] | None = None,
    ) -> None:
        self.downscale_factor = downscale_factor
        self.target_bitrate_kbps = target_bitrate_kbps
        self.preset = preset
        self.encoder = encoder
        self.mask_hands = mask_hands
        self.scale_resolution = scale_resolution

    def prepare_background_video(
        self,
        video_path: Path,
        poses: list[FrameHandPose],
        output_mp4: Path,
        max_frames: int | None = None,
    ) -> tuple[Path, int]:
        """Encodes the downscaled background stream.

        Note: mask_hands is False by default. Empirical benchmarking confirmed that
        pixel-domain blurring saves <2% rate (6 kbps) while producing boundary blur halos
        that destroy downstream palm detection. Clean downscaled inter-frame encoding
        preserves natural wrist/forearm contours without rate penalty.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.scale_resolution is not None:
            bg_w, bg_h = self.scale_resolution
        else:
            bg_w = int(round(w * self.downscale_factor))
            bg_h = int(round(h * self.downscale_factor))
        # Ensure dimensions are even (required by video codecs)
        bg_w = bg_w if bg_w % 2 == 0 else bg_w + 1
        bg_h = bg_h if bg_h % 2 == 0 else bg_h + 1

        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        tmp_raw = output_mp4.with_suffix(".raw.mp4")
        writer = cv2.VideoWriter(
            str(tmp_raw),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (bg_w, bg_h),
        )

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.mask_hands and frame_idx < len(poses) and poses[frame_idx].hands:
                frame_to_scale, _ = mask_out_hands(frame, poses[frame_idx].hands)
            else:
                frame_to_scale = frame
            downscaled = cv2.resize(frame_to_scale, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
            writer.write(downscaled)
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break

        cap.release()
        writer.release()

        # Encode with SVT-AV1 / ffmpeg with explicit preset
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(tmp_raw),
            "-c:v", self.encoder,
            "-b:v", f"{self.target_bitrate_kbps}k",
            "-preset", str(self.preset),
            "-pix_fmt", "yuv420p",
            str(output_mp4),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if tmp_raw.exists():
            tmp_raw.unlink()

        if not output_mp4.exists():
            raise RuntimeError(f"FFmpeg encoding failed: {res.stderr.decode('utf-8', errors='ignore')}")

        file_size = output_mp4.stat().st_size
        return output_mp4, file_size

    def decode_background_frames(
        self,
        encoded_mp4: Path,
        target_w: int,
        target_h: int,
    ) -> list[np.ndarray]:
        """Decodes and upscales background frames to native resolution (Lanczos).

        Regardless of the encoding resolution (180p–1080p), decoded frames are
        always returned at (target_w × target_h) so that neural hand compositing
        operates at full 1080p spatial resolution.
        """
        return read_video_frames_robust(
            encoded_mp4,
            target_w=target_w,
            target_h=target_h,
            upsample_interpolation=cv2.INTER_LANCZOS4,
        )


def read_video_frames_robust(
    video_path: Path,
    target_w: int | None = None,
    target_h: int | None = None,
    max_frames: int | None = None,
    upsample_interpolation: int = cv2.INTER_LINEAR,
) -> list[np.ndarray]:
    """Decodes video frames using cv2 with an automatic fallback to ffmpeg rawvideo pipe for AV1."""
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    if cap.isOpened():
        ret, test_f = cap.read()
        if ret and test_f is not None:
            if target_w and target_h and (test_f.shape[1] != target_w or test_f.shape[0] != target_h):
                test_f = cv2.resize(test_f, (target_w, target_h), interpolation=upsample_interpolation)
            frames.append(test_f)
            while cap.isOpened():
                if max_frames and len(frames) >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if target_w and target_h and (frame.shape[1] != target_w or frame.shape[0] != target_h):
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=upsample_interpolation)
                frames.append(frame)
            cap.release()
            if len(frames) > 0:
                return frames
    cap.release()

    # Fallback to FFmpeg rawvideo pipe
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(video_path)
    ]
    probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if not probe_res.stdout.strip():
        return []

    parts = probe_res.stdout.strip().split("x")
    orig_w, orig_h = int(parts[0]), int(parts[1])

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if max_frames:
        cmd.extend(["-vframes", str(max_frames)])
    cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "-"])

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_bytes = p.stdout.read()
    p.wait()

    if not raw_bytes:
        return []

    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, orig_h, orig_w, 3)
    decoded: list[np.ndarray] = []
    for f in arr:
        if target_w and target_h and (orig_w != target_w or orig_h != target_h):
            f = cv2.resize(f, (target_w, target_h), interpolation=upsample_interpolation)
        decoded.append(f)
    return decoded


class BackgroundPlateCodec:
    """Encodes periodic infilled background plates (keyframes every N frames) to test static keyframe vs continuous inter-frame AV1."""

    def __init__(
        self,
        plate_interval_frames: int = 60,
        quality: int = 80,
    ) -> None:
        self.plate_interval_frames = plate_interval_frames
        self.quality = quality

    def prepare_background_plates(
        self,
        video_path: Path,
        poses: list[FrameHandPose],
        output_dir: Path,
        max_frames: int | None = None,
    ) -> tuple[list[Path], int]:
        """Extracts and compresses a static infilled background keyframe every plate_interval_frames."""
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        plate_paths: list[Path] = []
        total_bytes = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.plate_interval_frames == 0:
                hands = poses[frame_idx].hands if frame_idx < len(poses) else []
                masked_frame, _ = mask_out_hands(frame, hands)
                plate_path = output_dir / f"plate_{frame_idx:06d}.jpg"
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                _, enc_buf = cv2.imencode(".jpg", masked_frame, encode_param)
                plate_path.write_bytes(enc_buf.tobytes())
                plate_paths.append(plate_path)
                total_bytes += len(enc_buf)
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break

        cap.release()
        return plate_paths, total_bytes

    def decode_background_frames(
        self,
        plate_paths: list[Path],
        total_frames: int,
        target_w: int,
        target_h: int,
    ) -> list[np.ndarray]:
        """Reconstructs the full background sequence by holding the most recent keyframe plate."""
        loaded_plates: dict[int, np.ndarray] = {}
        for p in plate_paths:
            stem_idx = int(p.stem.split("_")[1])
            img = cv2.imread(str(p))
            if img is not None and (img.shape[1] != target_w or img.shape[0] != target_h):
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            loaded_plates[stem_idx] = img

        plate_indices = sorted(loaded_plates.keys())
        decoded_frames: list[np.ndarray] = []
        curr_plate = loaded_plates[plate_indices[0]] if plate_indices else np.zeros((target_h, target_w, 3), dtype=np.uint8)

        for i in range(total_frames):
            if i in loaded_plates:
                curr_plate = loaded_plates[i]
            decoded_frames.append(curr_plate)

        return decoded_frames


