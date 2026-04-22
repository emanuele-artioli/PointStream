# https://github.com/IDEA-Research/DWPose
# Adapted for PointStream – model paths are constructor arguments.

from pathlib import Path

import cv2
import numpy as np

try:
    import torch  # noqa: F401 – ensure CUDA libs are loaded before ONNX Runtime
except Exception:
    torch = None

import onnxruntime as ort

from .onnxdet import inference_detector
from .onnxpose import inference_pose

_DEFAULT_DET = Path("/home/itec/emanuele/Models/DWPose/yolox_l.onnx")
_DEFAULT_POSE = Path("/home/itec/emanuele/Models/DWPose/dw-ll_ucoco_384.onnx")


class Wholebody:
    """ONNX-based whole-body pose estimator (body + hands + face).

    Returns 134 keypoints per detected person:
      0-17   body (OpenPose order, with neck at index 1)
      18-23  feet
      24-91  face (68 landmarks)
      92-112 left hand (21 landmarks)
      113-133 right hand (21 landmarks)
    """

    def __init__(self, device="cuda:0", det_model_path=None, pose_model_path=None):
        det_path = str(det_model_path or _DEFAULT_DET)
        pose_path = str(pose_model_path or _DEFAULT_POSE)

        self.session_det = _create_session(det_path, try_cuda=(device != "cpu"))
        self.session_pose = _create_session(pose_path, try_cuda=(device != "cpu"))

    def __call__(self, oriImg):
        """Run detection + pose estimation on a BGR numpy image.

        Returns:
            keypoints: (N, 134, 2) pixel coordinates.
            scores:    (N, 134) confidence values.
        """
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

        # Compute neck as midpoint of left/right shoulder
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3,
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

        # Re-order body keypoints from MMPose to OpenPose convention
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        return keypoints_info[..., :2], keypoints_info[..., 2]


def _create_session(onnx_path, try_cuda=True):
    """Create an ONNXRuntime InferenceSession.

    Notes on provider assignment:
    - ONNX Runtime may intentionally place some shape/control nodes on CPU even when
      CUDA is enabled (this is expected and often faster).
    - We explicitly register providers in priority order (CUDA -> CPU) so node placement
      and fallback behavior are deterministic.
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if not try_cuda:
        return ort.InferenceSession(
            path_or_bytes=onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    try:
        return ort.InferenceSession(
            path_or_bytes=onnx_path,
            sess_options=sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception:
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        try:
            return ort.InferenceSession(
                path_or_bytes=onnx_path,
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as exc:
            import warnings
            warnings.warn(
                f"CUDA EP failed for ONNXRuntime; falling back to CPU. Error: {exc}. "
                "Install a matching onnxruntime-gpu wheel for GPU support."
            )
            return ort.InferenceSession(
                path_or_bytes=onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
