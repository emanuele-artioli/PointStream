"""Steered residual encoding: cropped actor residual and band-limited background.

Incorporates the key findings from Presley (/home/itec/emanuele/presley):
1. Avoid black-masked full frames: Masking a 4K frame with black blocks introduces
   severe 1-pixel step edge discontinuities in transform/DCT coding, inflating
   residual size to 17.8 kB. Bounding boxes are ~74% background.
2. Cropped Actor Residual: Extracting and encoding only the tight bounding box
   crop (e.g. 700x600) eliminates artificial boundary discontinuities, reducing
   residual size to <= 4.5 kB (77% reduction).
3. Morphological Dilation (MORPH_ELLIPSE): Hard foreground protection (fg_protect)
   dilates actor masks to prevent segmentation flutter from cutting off limbs/rackets.
4. Passthrough Compositing: Client applies residual strictly inside the cropped
   bounding box, keeping background canvas pixels untouched.
5. Band-Limited Background Target: If background residual bits are spent, the ground
   truth background is band-limited (0.5x down/up) before differencing, eliminating
   high-frequency court noise bits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Tuple

import cv2
import numpy as np

from src.pipeline.residual.lossy import decode_lossy, encode_lossy

DEFAULT_DILATION_RADIUS: Final[int] = 8


@dataclass(frozen=True)
class CroppedResidualPayload:
    """Compressed wire payload for a tight cropped actor residual."""

    payload: bytes
    bbox: Tuple[int, int, int, int]  # (y1, y2, x1, x2)
    original_shape: Tuple[int, int, int]  # (H, W, C)
    codec: str = "webp"
    quality: int = 75
    mode: str = "clipped"
    offset: float = 128.0

    @property
    def byte_count(self) -> int:
        return len(self.payload)


class ActorMaskProcessor:
    """Applies elliptical morphology to actor masks for hard foreground protection."""

    def __init__(self, dilation_radius: int = DEFAULT_DILATION_RADIUS) -> None:
        if dilation_radius < 0:
            raise ValueError(f"dilation_radius must be non-negative, got {dilation_radius}")
        self.dilation_radius = int(dilation_radius)

    def dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """Dilate binary mask with an elliptical structuring element.

        Args:
            mask: (H, W) uint8 or bool mask (non-zero is foreground).

        Returns:
            (H, W) uint8 mask with expanded foreground boundary.
        """
        if self.dilation_radius == 0:
            return (mask > 0).astype(np.uint8) * 255
        k = 2 * self.dilation_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = (mask > 0).astype(np.uint8) * 255
        return cv2.dilate(binary, kernel)

    def compute_tight_bbox(
        self,
        mask: np.ndarray,
        pad: int = 8,
        even_align: bool = True,
    ) -> Tuple[int, int, int, int] | None:
        """Compute bounding box (y1, y2, x1, x2) covering foreground mask.

        Args:
            mask: (H, W) mask.
            pad: Extra pixel padding around boundary.
            even_align: If True, forces width and height to even integers.

        Returns:
            (y1, y2, x1, x2) or None if mask is empty.
        """
        yy, xx = np.where(mask > 0)
        if len(yy) == 0:
            return None

        h, w = mask.shape[:2]
        y1 = max(0, int(yy.min()) - pad)
        y2 = min(h, int(yy.max()) + 1 + pad)
        x1 = max(0, int(xx.min()) - pad)
        x2 = min(w, int(xx.max()) + 1 + pad)

        if even_align:
            # Even extent adjustment
            if (y2 - y1) % 2 != 0:
                if y2 < h:
                    y2 += 1
                elif y1 > 0:
                    y1 -= 1
                else:
                    y2 -= 1
            if (x2 - x1) % 2 != 0:
                if x2 < w:
                    x2 += 1
                elif x1 > 0:
                    x1 -= 1
                else:
                    x2 -= 1

        return int(y1), int(y2), int(x1), int(x2)


class CroppedActorResidualEncoder:
    """Encodes residual difference strictly within a tight actor crop."""

    def __init__(
        self,
        codec: str = "webp",
        quality: int = 75,
        mode: str = "clipped",
        dilation_radius: int = DEFAULT_DILATION_RADIUS,
    ) -> None:
        if codec not in ("webp", "jpeg", "png"):
            raise ValueError(f"unsupported residual patch codec: {codec}")
        self.codec = codec
        self.quality = int(quality)
        self.mode = mode
        self.dilation_radius = int(dilation_radius)
        self.mask_processor = ActorMaskProcessor(dilation_radius=self.dilation_radius)

    def encode_frame(
        self,
        target_frame: np.ndarray,
        reconstructed_frame: np.ndarray,
        actor_mask: np.ndarray,
    ) -> CroppedResidualPayload | None:
        """Compute difference, crop to actor bbox, and encode.

        Args:
            target_frame: (H, W, 3) uint8 pristine reference.
            reconstructed_frame: (H, W, 3) uint8 decoded canvas.
            actor_mask: (H, W) binary actor mask.

        Returns:
            CroppedResidualPayload with compressed bytes, or None if mask is empty.
        """
        if target_frame.shape != reconstructed_frame.shape:
            raise ValueError(
                f"target shape {target_frame.shape} and reconstructed shape "
                f"{reconstructed_frame.shape} mismatch"
            )

        # 1. Dilate mask with elliptical structuring element
        dilated_mask = self.mask_processor.dilate_mask(actor_mask)

        # 2. Compute tight bounding box
        bbox = self.mask_processor.compute_tight_bbox(dilated_mask, pad=8, even_align=True)
        if bbox is None:
            return None

        y1, y2, x1, x2 = bbox

        # 3. Extract cropped difference
        target_crop = target_frame[y1:y2, x1:x2].astype(np.int16)
        recon_crop = reconstructed_frame[y1:y2, x1:x2].astype(np.int16)
        signed_diff = target_crop - recon_crop

        # 4. Map to uint8 via lossy representation
        offset = 128.0
        encoded_uint8 = encode_lossy(signed_diff, mode=self.mode, offset=offset)

        # 5. Compress patch with image codec
        if self.codec == "webp":
            ext = ".webp"
            params = [cv2.IMWRITE_WEBP_QUALITY, self.quality]
        elif self.codec == "jpeg":
            ext = ".jpg"
            params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        elif self.codec == "png":
            ext = ".png"
            params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
        else:
            raise ValueError(f"unsupported codec: {self.codec}")

        success, encoded = cv2.imencode(ext, encoded_uint8, params)
        if not success:
            raise RuntimeError(f"Failed to compress residual patch with {self.codec}")

        return CroppedResidualPayload(
            payload=encoded.tobytes(),
            bbox=bbox,
            original_shape=(int(target_frame.shape[0]), int(target_frame.shape[1]), int(target_frame.shape[2])),
            codec=self.codec,
            quality=self.quality,
            mode=self.mode,
            offset=offset,
        )


class CroppedActorResidualDecoder:
    """Decodes cropped residual payload and composites onto reconstructed canvas."""

    def __init__(self) -> None:
        pass

    def apply(
        self,
        reconstructed_frame: np.ndarray,
        payload: CroppedResidualPayload,
    ) -> np.ndarray:
        """Decode crop and add to reconstructed frame at bbox coordinates.

        Args:
            reconstructed_frame: (H, W, 3) uint8 canvas.
            payload: CroppedResidualPayload.

        Returns:
            Composited (H, W, 3) uint8 image. Pixels outside bbox are untouched.
        """
        h, w = reconstructed_frame.shape[:2]
        y1, y2, x1, x2 = payload.bbox
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w or y1 >= y2 or x1 >= x2:
            raise ValueError(
                f"crop bbox {payload.bbox} exceeds frame bounds ({h}, {w})"
            )

        buf = np.frombuffer(payload.payload, dtype=np.uint8)
        decoded_uint8 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded_uint8 is None:
            raise ValueError("corrupted residual bitstream: cannot decode image")

        # Map uint8 back to signed diff
        signed_diff = decode_lossy(
            decoded_uint8,
            mode=payload.mode,
            offset=payload.offset,
        )

        out = reconstructed_frame.copy()
        recon_crop = out[y1:y2, x1:x2].astype(np.int16)
        corrected_crop = np.clip(recon_crop + signed_diff, 0, 255).astype(np.uint8)
        out[y1:y2, x1:x2] = corrected_crop
        return out


class BandLimitedBackgroundResidual:
    """Encodes background residual against a smooth, band-limited ground truth target."""

    def __init__(self, downscale_factor: float = 0.5, quality: int = 50) -> None:
        if not 0.0 < downscale_factor <= 1.0:
            raise ValueError(f"downscale_factor must be in (0, 1], got {downscale_factor}")
        self.downscale_factor = float(downscale_factor)
        self.quality = int(quality)

    def band_limit_target(
        self,
        target_frame: np.ndarray,
        actor_mask: np.ndarray,
    ) -> np.ndarray:
        """Band-limit the background region of the target frame.

        Args:
            target_frame: (H, W, 3) uint8 pristine reference.
            actor_mask: (H, W) mask (actor pixels are kept untouched).

        Returns:
            (H, W, 3) uint8 target with smoothed, band-limited background.
        """
        h, w = target_frame.shape[:2]
        scaled_w = max(2, int(round(w * self.downscale_factor)))
        scaled_h = max(2, int(round(h * self.downscale_factor)))

        # Downsample with INTER_AREA, then upsample back
        downsampled = cv2.resize(target_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)

        # Composite: retain pristine actor where mask > 0, smoothed background where mask == 0
        out = upsampled.copy()
        actor_binary = actor_mask > 0
        out[actor_binary] = target_frame[actor_binary]
        return out

    def compute_residual(
        self,
        target_frame: np.ndarray,
        reconstructed_frame: np.ndarray,
        actor_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute background residual against band-limited target.

        Args:
            target_frame: (H, W, 3) uint8 pristine reference.
            reconstructed_frame: (H, W, 3) uint8 canvas.
            actor_mask: (H, W) actor mask.

        Returns:
            (H, W, 3) signed int16 difference, with 0 inside actor region.
        """
        bl_target = self.band_limit_target(target_frame, actor_mask)
        diff = bl_target.astype(np.int16) - reconstructed_frame.astype(np.int16)
        # Steer away from actor: actor is handled by CroppedActorResidual
        diff[actor_mask > 0] = 0
        return diff
