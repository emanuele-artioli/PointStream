"""Bounding-box geometry used by detection, selection and tracking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates, ``(x1, y1, x2, y2)``."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Box is inverted: {(self.x1, self.y1, self.x2, self.y2)}")

    @classmethod
    def from_xyxy(cls, xyxy: list[float] | tuple[float, ...]) -> Box:
        if len(xyxy) != 4:
            raise ValueError(f"xyxy needs 4 values, got {xyxy!r}")
        return cls(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    def iou(self, other: Box) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def clip(self, width: int, height: int) -> Box:
        if width < 1 or height < 1:
            raise ValueError(f"clip needs a positive frame size, got {width}x{height}")
        x1 = min(max(self.x1, 0.0), float(width - 1))
        y1 = min(max(self.y1, 0.0), float(height - 1))
        x2 = min(max(self.x2, x1 + 1.0), float(width))
        y2 = min(max(self.y2, y1 + 1.0), float(height))
        return Box(x1, y1, x2, y2)

    def padded(self, ratio: float, width: int, height: int) -> Box:
        """Expand by `ratio` of width/height, then clip to the frame."""
        if ratio < 0.0:
            raise ValueError(f"pad ratio must be >= 0, got {ratio}")
        pad_x = self.width * ratio
        pad_y = self.height * ratio
        return Box(
            self.x1 - pad_x, self.y1 - pad_y, self.x2 + pad_x, self.y2 + pad_y
        ).clip(width, height)
