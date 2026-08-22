"""ROI map formatting, pixel-domain degradation, matched-rate hygiene.

The silent-wrong-answer cases: an AV1 map with too-large offsets (regions
converge), a matched-QP comparison reported as a rate saving, and a pixel arm
that touches the salient blocks it was supposed to protect.
"""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from src.components.codec.encode import match_rate, search_qp
from src.components.codec.roi import (
    AV1_BLOCK,
    AV1_MAX_SEGMENTS,
    BlockRoiMap,
    addroi_filter,
    degrade_luma,
    write_kvazaar,
    write_svtav1,
)
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.errors import CodecConstraintError


def _map(*, inside: int, outside: int = 0, size: int = 128, block: int = AV1_BLOCK) -> BlockRoiMap:
    return BlockRoiMap.centred(
        size,
        size,
        inside_offset=inside,
        outside_offset=outside,
        fraction=0.5,
        block_size=block,
    )


class TestMapFormatting:
    def test_svtav1_map_is_one_line_per_frame_row_major(self, tmp_path: Path) -> None:
        roi = _map(inside=-30, outside=0)
        path = tmp_path / "roi.txt"
        write_svtav1(roi, path, frames=2)
        lines = path.read_text(encoding="ascii").splitlines()
        assert len(lines) == 2
        frame0 = [int(tok) for tok in lines[0].split()]
        assert frame0[0] == 0
        offsets = frame0[1:]
        assert len(offsets) == roi.blocks_wide * roi.blocks_high
        # 128x128 at 64x64 is 2x2; fraction 0.5 -> one centred block.
        assert offsets.count(-30) == 1
        assert offsets.count(0) == 3
        assert lines[1].startswith("1 ")

    def test_kvazaar_map_is_int32_size_then_int8_deltas(self, tmp_path: Path) -> None:
        roi = _map(inside=-8, outside=4, block=64)
        path = tmp_path / "roi.bin"
        write_kvazaar(roi, path, frames=1)
        data = path.read_bytes()
        cols, rows = struct.unpack_from("ii", data)
        assert (cols, rows) == (roi.blocks_wide, roi.blocks_high)
        deltas = np.frombuffer(data, dtype=np.int8, offset=8)
        assert deltas.tolist() == list(roi.offsets)

    def test_addroi_filter_scales_qp_offset_and_skips_a_zero_inside(self) -> None:
        active = _map(inside=-26)
        text = addroi_filter(active, 128, 128)
        assert text.startswith("addroi=")
        assert "-0." in text or "-1" in text  # negative qoffset = better
        silent = _map(inside=0)
        assert addroi_filter(silent, 128, 128) == ""


class TestAv1OffsetTraps:
    def test_offsets_past_the_convergence_zone_are_rejected(self, tmp_path: Path) -> None:
        """-120/+60 made both regions lose quality. That is not a stronger map."""
        roi = _map(inside=-121, outside=0)
        with pytest.raises(CodecConstraintError, match="q_index"):
            write_svtav1(roi, tmp_path / "roi.txt", frames=1)

    def test_more_than_eight_distinct_offsets_are_quantised_to_the_segment_cap(
        self, tmp_path: Path
    ) -> None:
        # 3x3 at 64px covering 192x192, nine different offsets — over the cap.
        offsets = tuple(range(-4, 5))  # 9 values
        roi = BlockRoiMap(block_size=64, blocks_wide=3, blocks_high=3, offsets=offsets)
        path = tmp_path / "roi.txt"
        write_svtav1(roi, path, frames=1)
        values = [int(tok) for tok in path.read_text(encoding="ascii").split()[1:]]
        assert len(set(values)) <= AV1_MAX_SEGMENTS
        assert 0 in set(values)


class TestPixelDomainArm:
    def test_only_positive_offset_blocks_are_degraded(self) -> None:
        luma = np.arange(16, dtype=np.uint8).reshape(4, 4)
        # 2x2 blocks of 2px. Bottom-right is non-salient (+8); the rest is protected.
        roi = BlockRoiMap(
            block_size=2,
            blocks_wide=2,
            blocks_high=2,
            offsets=(0, 0, 0, 8),
            col0=1,
            col1=2,
            row0=1,
            row1=2,
            inside_offset=8,
        )
        out = degrade_luma(luma, roi)
        assert np.array_equal(out[:2, :], luma[:2, :])
        assert np.array_equal(out[2:, :2], luma[2:, :2])
        assert not np.array_equal(out[2:, 2:], luma[2:, 2:])

    def test_negative_offset_is_left_intact(self) -> None:
        luma = np.full((4, 4), 100, dtype=np.uint8)
        roi = BlockRoiMap(block_size=4, blocks_wide=1, blocks_high=1, offsets=(-12,))
        assert np.array_equal(degrade_luma(luma, roi), luma)


class TestMatchedRateHygiene:
    def test_search_qp_lands_on_the_oracle_target(self) -> None:
        # Higher QP -> fewer bytes. Closed form: bytes = 10000 - 100*qp.
        def encode_at_qp(qp: int) -> int:
            return 10000 - 100 * qp

        qp, size = search_qp(encode_at_qp, 7000, 1, 51, tolerance=0.01)
        assert qp == 30
        assert size == 7000

    def test_match_rate_refuses_mixed_rate_control_before_touching_disk(self, tmp_path: Path) -> None:
        source = tmp_path / "missing.y4m"
        roi = _map(inside=-30)
        with pytest.raises(CodecConstraintError, match="rate_control"):
            match_rate(
                source,
                tmp_path,
                EncodeRequest("av1", RateControl.CRF, 35, preset="8"),
                EncodeRequest("av1", RateControl.QP, 35, preset="8"),
                roi=roi,
            )
        assert not source.exists()
