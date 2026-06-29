import cv2
import numpy as np

def get_iou(boxA: tuple | list, boxB: tuple | list) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_global_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, float]:
    """Estimate global camera motion between two frames using phase correlation."""
    prev_small = cv2.resize(prev_gray, (256, 256)).astype(np.float32)
    curr_small = cv2.resize(curr_gray, (256, 256)).astype(np.float32)
    
    # create hanning window to reduce edge artifacts
    hanning = cv2.createHanningWindow((256, 256), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(prev_small, curr_small, hanning)
    
    scale_x = curr_gray.shape[1] / 256.0
    scale_y = curr_gray.shape[0] / 256.0
    return float(dx * scale_x), float(dy * scale_y)
