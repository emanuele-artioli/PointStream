from __future__ import annotations

from pathlib import Path

import numpy as np
from src.shared.config import PointstreamConfig
import pytest
import torch

from src.components.generation import torch_dtype as td
from src.transport.panorama_encoder import (
    JpegPanoramaEncoder,
    PngPanoramaEncoder,
    build_panorama_encoder,
)


def test_dtype_helpers() -> None:
    assert td.parse_gpu_dtype("fp16") == torch.float16


def test_panorama_encoder_build_and_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 255

    jpeg_encoder = build_panorama_encoder("jpeg")
    assert isinstance(jpeg_encoder, JpegPanoramaEncoder)
    jpeg_path = jpeg_encoder.encode(image, tmp_path / "pano_jpeg")
    assert jpeg_path.suffix == ".jpg"
    assert jpeg_path.exists()

    png_encoder = build_panorama_encoder("png")
    assert isinstance(png_encoder, PngPanoramaEncoder)
    png_path = png_encoder.encode(image, tmp_path / "pano_png")
    assert png_path.suffix == ".png"
    assert png_path.exists()

    with pytest.raises(ValueError, match=r"expected \[H, W, 3\]"):
        jpeg_encoder.encode(np.zeros((12, 16), dtype=np.uint8), tmp_path / "bad")

    try:
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            build_panorama_encoder("jpeg", config=PointstreamConfig(panorama_jpeg_quality="bad"))  # type: ignore[arg-type]
    except ImportError:
        pass

    with pytest.raises(ValueError, match="Unsupported panorama encoder"):
        build_panorama_encoder("webp")
