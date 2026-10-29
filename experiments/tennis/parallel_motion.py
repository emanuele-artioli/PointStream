"""CPU encoder fit and delivered float32 homographies for paid background reuse."""
import struct
import zlib

import cv2
import numpy as np


def encode_motion(matrices: np.ndarray) -> bytes:
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError('Finite homographies required')
    return b'HOM1' + struct.pack('<I', len(matrices)) + zlib.compress(matrices.astype('<f4').tobytes(), 9)


def decode_motion(data: bytes, frames: int) -> np.ndarray:
    if len(data) < 9 or data[:4] != b'HOM1' or struct.unpack('<I', data[4:8])[0] != frames:
        raise ValueError('Wrong motion packet cohort')
    raw = zlib.decompress(data[8:])
    if len(raw) != frames * 9 * 4:
        raise ValueError('Incomplete motion packet')
    matrices = np.frombuffer(raw, '<f4').reshape(frames, 3, 3).astype(np.float64)
    if not np.isfinite(matrices).all() or np.any(np.abs(np.linalg.det(matrices)) < 1e-8):
        raise ValueError('Invalid delivered homography')
    return matrices


def warp_reference(reference: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    height, width = 544, 960
    if reference.shape != (height * 3 // 2, width) or matrix.shape != (3, 3):
        raise ValueError('Native reference geometry required')
    chroma = reference[height:].reshape(2, height // 2, width // 2)
    uv_matrix = np.diag([.5, .5, 1.]) @ matrix @ np.diag([2., 2., 1.])
    planes = [cv2.warpPerspective(reference[:height], matrix, (width, height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(16.,))]
    planes += [cv2.warpPerspective(p, uv_matrix, (width // 2, height // 2),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128.,))
               for p in chroma]
    return np.concatenate([p.ravel() for p in planes]).reshape(reference.shape)


def fit_motion(reference: np.ndarray, source: np.ndarray, regions: np.ndarray) -> tuple[bytes, dict[str, int]]:
    cv2.setRNGSeed(20261028)
    orb = getattr(cv2, 'ORB_create')(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    key_ref, desc_ref = orb.detectAndCompute(reference[:544], None)
    matrices = []
    fitted = 0
    for i, frame in enumerate(source):
        mask = ~cv2.resize(regions[i].astype(np.uint8), (960, 544), interpolation=cv2.INTER_NEAREST).astype(bool)
        key, desc = orb.detectAndCompute(frame[:544], mask.astype(np.uint8) * 255)
        matrix = np.eye(3)
        if desc_ref is not None and desc is not None and len(desc) >= 2:
            matches = [p[0] for p in matcher.knnMatch(desc_ref, desc, k=2)
                       if len(p) == 2 and p[0].distance < .75 * p[1].distance]
            if len(matches) >= 20:
                origin = np.asarray([key_ref[m.queryIdx].pt for m in matches], dtype=np.float32)
                target = np.asarray([key[m.trainIdx].pt for m in matches], dtype=np.float32)
                candidate, inliers = cv2.findHomography(origin, target, cv2.RANSAC, 2.)
                if candidate is not None and inliers is not None and int(inliers.sum()) >= 20:
                    predicted = cv2.perspectiveTransform(origin[:, None], candidate)[:, 0]
                    errors = np.linalg.norm(predicted-target, axis=1)[inliers.ravel().astype(bool)]
                    if (np.isfinite(candidate).all() and abs(candidate[2, 2]) > 1e-8
                            and abs(np.linalg.det(candidate)) > 1e-8 and float(np.median(errors)) <= 1.5):
                        matrix = candidate / candidate[2, 2]
                        fitted += 1
        matrices.append(matrix)
    return encode_motion(np.stack(matrices)), {'fitted': fitted, 'identity_fallbacks': len(matrices)-fitted}
