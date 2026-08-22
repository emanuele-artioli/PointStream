"""Per-object encoded video. The classical codec answer applied per crop."""

from __future__ import annotations

from src.contracts.capabilities import MOTION_ENCODED_VIDEO
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.objectstream import EncodedVideoMotion


class EncodedVideoMotionEncoder:
    """Describe how the object crop is encoded. Actual bytes are measured later."""

    kind = MOTION_ENCODED_VIDEO

    def __init__(
        self,
        codec_name: str = "av1",
        width: int = 64,
        height: int = 64,
        rate_control: RateControl = RateControl.CRF,
        rate: int | None = 35,
        preset: str | None = "8",
        pix_fmt: str = "yuv420p",
        measured_bytes_per_frame: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.request = EncodeRequest(
            codec_name=codec_name,
            rate_control=rate_control,
            rate=rate,
            preset=preset,
            pix_fmt=pix_fmt,
        )
        self.measured_bytes_per_frame = measured_bytes_per_frame

    def encode(self, measured_bytes_per_frame: int | None = None) -> EncodedVideoMotion:
        self.request.validate()
        measured = (
            measured_bytes_per_frame
            if measured_bytes_per_frame is not None
            else self.measured_bytes_per_frame
        )
        return EncodedVideoMotion(
            request=self.request,
            width=self.width,
            height=self.height,
            measured_bytes_per_frame=measured,
        )
