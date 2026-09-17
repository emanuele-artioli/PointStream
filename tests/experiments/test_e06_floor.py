"""Lossless E06 floor packs: RLE/XOR roundtrip, thin envelope, ledger, fail-closed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tier.e06_floor import (
    MODE_INTRA,
    MODE_XOR_KEY,
    decode_mask_stack,
    encode_mask_stack,
    pack_mask_rle,
    pack_thin_placement,
    physical_ledger,
    required_decode_fields,
    unpack_floor_pack,
)
from experiments.tier.e06_pack import pack_lossless_compact
from experiments.tier.e06_transport import (
    PREDICTOR_BBOX_RESIZE,
    reconstruct_standalone,
    serialize_setting,
)
from tests.experiments.test_e06_probe import _jpeg_background, _toy


def test_rle_intra_and_xor_roundtrip() -> None:
    stack = np.zeros((8, 12, 10), dtype=np.uint8)
    for index in range(8):
        stack[index, 2:5, 1 + index : 4 + index] = 1
        if index > 0:
            stack[index, 6, 3] = 1
    intra = encode_mask_stack(stack, mode=MODE_INTRA)
    xor = encode_mask_stack(stack, mode=MODE_XOR_KEY)
    np.testing.assert_array_equal(decode_mask_stack(intra), stack)
    np.testing.assert_array_equal(decode_mask_stack(xor), stack)


def test_truncated_rle_fails_closed() -> None:
    stack = np.zeros((2, 4, 4), dtype=np.uint8)
    stack[0, 1, 1] = 1
    blob = encode_mask_stack(stack, mode=MODE_INTRA)
    with pytest.raises(ValueError, match="truncated|underfill|corrupt"):
        decode_mask_stack(blob[:-1])


def test_floor_packs_match_parent_pixels_and_reconcile_t(tmp_path: Path) -> None:
    frames, masks = _toy()
    view, _b = _jpeg_background(frames)
    parent = serialize_setting(
        background=view,
        frames=frames,
        masks=masks,
        predictor=PREDICTOR_BBOX_RESIZE,
        residual=None,
    )
    compact = pack_lossless_compact(parent)
    parent_frames = reconstruct_standalone(compact)
    for packed in (
        pack_mask_rle(compact, mode=MODE_INTRA),
        pack_mask_rle(compact, mode=MODE_XOR_KEY),
        pack_thin_placement(compact),
    ):
        ledger = physical_ledger(packed)
        assert ledger["transport_total"] == len(packed)
        assert ledger["reconciled"] is True
        envelope = unpack_floor_pack(packed)
        required_decode_fields(envelope)
        packed_frames = reconstruct_standalone(packed)
        np.testing.assert_array_equal(parent_frames, packed_frames)
        (tmp_path / "t.bin").write_bytes(packed)
        assert (tmp_path / "t.bin").stat().st_size == ledger["transport_total"]


def test_thin_metadata_does_not_repeat_masks() -> None:
    frames, masks = _toy()
    view, _b = _jpeg_background(frames)
    compact = pack_lossless_compact(
        serialize_setting(
            background=view,
            frames=frames,
            masks=masks,
            predictor=PREDICTOR_BBOX_RESIZE,
            residual=None,
        )
    )
    packed = pack_thin_placement(compact)
    import zipfile
    import io

    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        names = set(archive.namelist())
        assert "metadata.npy" not in names
        thin = json.loads(archive.read("thin_meta.json").decode("utf-8"))
    assert thin["background"]["homographies"] == []
    for item in thin["placements"]:
        assert "bbox" not in item
        assert "mask_wire" not in item


def test_physical_ledger_sums_to_file_length() -> None:
    frames, masks = _toy()
    view, _b = _jpeg_background(frames)
    packed = pack_mask_rle(
        pack_lossless_compact(
            serialize_setting(
                background=view,
                frames=frames,
                masks=masks,
                predictor=PREDICTOR_BBOX_RESIZE,
                residual=None,
            )
        ),
        mode=MODE_INTRA,
    )
    ledger = physical_ledger(packed)
    assert ledger["panorama"] + ledger["actor_reference"] + ledger["residual"] + ledger[
        "metadata"
    ] + ledger["unallocated_H"] == len(packed)
