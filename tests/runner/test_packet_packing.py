"""Opt-in transport packing: exact wire identity, lossy flags and client parity."""

import io
import json
import zipfile

import numpy as np
import pytest

from src.runner.client import (
    ClientPlacement,
    reconstruct_serialized_client,
    serialize_client_request,
)
from src.runner.mask_wire import decode_mask
from src.runner.packet_packing import (
    MANIFEST,
    pack_client_envelope,
    packing_info,
    unpack_client_envelope,
)


def envelope(mask=None, *, frames=3, placements=None):
    if mask is None:
        mask = np.zeros((5, 7), dtype=np.uint8)
        mask[1, 1] = 1
    if placements is None:
        placements = (
            ClientPlacement(
                crop=np.full((5, 7, 3), 180, dtype=np.uint8),
                bbox=(0, 0, 7, 5),
                mask=mask,
                frame_index=0,
            ),
        )
    return serialize_client_request(
        background=None, frame_count=frames, height=5, width=7, placements=placements
    )


def members(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def rewrite(payload, *, metadata=None, extra=None, spec=None):
    files = members(payload)
    if metadata is not None:
        out = io.BytesIO()
        np.save(out, np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
        files["metadata.npy"] = out.getvalue()
    if spec is not None:
        files[MANIFEST] = json.dumps(spec).encode()
    for key, array in (extra or {}).items():
        out = io.BytesIO()
        np.save(out, array, allow_pickle=False)
        files[key + ".npy"] = out.getvalue()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return out.getvalue()


def meta(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return json.loads(arrays["metadata"].tobytes())


@pytest.mark.parametrize("codec", ["psm1", "rle"])
def test_lossless_client_parity_all_frames_and_deterministic_bytes(codec):
    original = envelope()
    packed = pack_client_envelope(original, mask_codec=codec)
    assert packed == pack_client_envelope(original, mask_codec=codec)
    restored = unpack_client_envelope(packed)
    before = reconstruct_serialized_client(original)
    after = reconstruct_serialized_client(restored)
    assert after.shape == (3, 5, 7, 3)
    assert np.array_equal(after, before)
    assert np.all(after[1:] == 0)  # Missing placements remain emitted black frames.
    info = packing_info(packed)
    assert info["complete_file_bytes"] == len(packed)
    assert info["physical_member_bytes"] + info["zip_framing_bytes"] == len(packed)
    assert info["packing"]["lossy_masks"] is False


@pytest.mark.parametrize("scale", [2, 4, 8])
@pytest.mark.parametrize("codec", ["psm1", "rle"])
def test_lossy_nearest_masks_preserve_odd_original_dimensions(scale, codec):
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 1] = 1
    mask[4, 6] = 1
    packed = pack_client_envelope(envelope(mask), mask_scale=scale, mask_codec=codec)
    restored = unpack_client_envelope(packed)
    with np.load(io.BytesIO(restored), allow_pickle=False) as arrays:
        decoded = decode_mask(arrays["mask_0"].tobytes())
    expected = np.repeat(np.repeat(mask[::scale, ::scale], scale, axis=0), scale, axis=1)[:5, :7]
    assert decoded.shape == mask.shape
    assert np.array_equal(decoded, expected)
    info = packing_info(packed)
    assert info["packing"]["lossy_masks"] is True
    assert info["packing"]["predictor_pixels_may_change"] is True
    assert info["packing"]["changed_mask_pixels"] == np.count_nonzero(mask != expected)
    assert meta(restored)["placements"][0]["mask_wire"]["shape"] == [5, 7]
    assert reconstruct_serialized_client(restored).shape == (3, 5, 7, 3)


@pytest.mark.parametrize("codec", ["psm1", "rle"])
def test_empty_masks_and_multiple_objects_do_not_assume_one_mask_per_frame(codec):
    empty = np.zeros((5, 7), dtype=np.uint8)
    placements = tuple(
        ClientPlacement(
            crop=np.ones((5, 7, 3), dtype=np.uint8),
            bbox=(0, 0, 7, 5),
            mask=empty,
            frame_index=i % 2,
            object_id=str(i),
        )
        for i in range(5)
    )
    original = envelope(frames=2, placements=placements)
    packed = pack_client_envelope(original, mask_codec=codec, mask_scale=4)
    restored = unpack_client_envelope(packed)
    assert len(meta(restored)["placements"]) == 5
    assert np.array_equal(
        reconstruct_serialized_client(restored), np.zeros((2, 5, 7, 3), dtype=np.uint8)
    )
    assert packing_info(packed)["packing"]["changed_mask_pixels"] == 0


@pytest.mark.parametrize("batch", [False, True])
def test_native_members_and_geometry_metadata_are_preserved(batch):
    original = envelope()
    metadata = meta(original)
    metadata["background"] = {
        "geometry_header": "123456abcdef",
        "wire_header_keys": ["background_header_0"],
    }
    extra = {
        key: np.arange(17, dtype=np.uint8)
        for key in [
            "background_payload_0",
            "encoded_crop_99",
            "ref_player:0",
            "residual_bitstream",
            "background_header_0",
        ]
    }
    original = rewrite(original, metadata=metadata, extra=extra)
    packed = pack_client_envelope(original, mask_codec="rle", batch_masks=batch)
    restored = unpack_client_envelope(packed)
    before, after = members(original), members(restored)
    for key in extra:
        assert before[key + ".npy"] == after[key + ".npy"]
    assert meta(restored)["background"] == metadata["background"]
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        assert archive.getinfo("residual_bitstream.npy").compress_type == zipfile.ZIP_STORED
        assert archive.getinfo("metadata.npy").compress_type == zipfile.ZIP_DEFLATED


def test_lossy_mask_really_changes_delivered_pixels():
    original = envelope()
    packed = pack_client_envelope(original, mask_scale=2)
    assert not np.array_equal(
        reconstruct_serialized_client(original),
        reconstruct_serialized_client(unpack_client_envelope(packed)),
    )


def test_residual_on_lossy_input_rejected_but_lossless_allowed():
    original = envelope()
    metadata = meta(original)
    metadata["residual"]["present"] = True
    original = rewrite(
        original, metadata=metadata, extra={"residual_bitstream": np.array([1, 2], dtype=np.uint8)}
    )
    pack_client_envelope(original, mask_scale=1)
    with pytest.raises(ValueError, match="recomputed correction"):
        pack_client_envelope(original, mask_scale=2)


@pytest.mark.parametrize("scale", [0, 3, 16, True])
def test_unsupported_mask_scales_rejected(scale):
    with pytest.raises(ValueError, match="unsupported"):
        pack_client_envelope(envelope(), mask_scale=scale)


def test_unsupported_formats_and_frame_count_rejected():
    with pytest.raises(ValueError):
        unpack_client_envelope(b"not an NPZ")
    original = envelope()
    metadata = meta(original)
    metadata["schema"] = 2
    with pytest.raises(ValueError, match="schema-1"):
        pack_client_envelope(rewrite(original, metadata=metadata))
    metadata["schema"] = 1
    metadata["placements"][0]["frame_index"] = 3
    with pytest.raises(ValueError, match="frame count"):
        pack_client_envelope(rewrite(original, metadata=metadata))
    packed = pack_client_envelope(original)
    spec = packing_info(packed)["packing"]
    spec["version"] = 99
    with pytest.raises(ValueError, match="unsupported"):
        unpack_client_envelope(rewrite(packed, spec=spec))


def test_mask_stack_keeps_empty_frames_and_rejects_count_mismatch():
    volume = np.zeros((3, 5, 7), dtype=np.uint8)
    volume[0, 0, 0] = 1
    volume[2, 4, 6] = 1
    original = envelope(volume)
    for codec in ["psm1", "rle"]:
        restored = unpack_client_envelope(pack_client_envelope(original, mask_codec=codec))
        with np.load(io.BytesIO(restored), allow_pickle=False) as arrays:
            assert np.array_equal(decode_mask(arrays["mask_0"].tobytes()), volume)
    with pytest.raises(ValueError, match="stack frame count"):
        pack_client_envelope(envelope(volume[:2]))


def test_unpacked_ordinary_envelope_is_validated_identity():
    original = envelope()
    assert unpack_client_envelope(original) == original
    metadata = meta(original)
    metadata["placements"][0]["mask_wire"]["shape"] = [6, 7]
    with pytest.raises(ValueError, match="shape"):
        unpack_client_envelope(rewrite(original, metadata=metadata))


@pytest.mark.parametrize("mode", [0, 1])
def test_rle_wire_parity_with_retained_e06(mode):
    from experiments.tier import e06_floor
    from src.runner import mask_rle

    masks = np.zeros((5, 7, 9), dtype=np.uint8)
    masks[0, 1:4, 2:5] = 1
    masks[1] = masks[0]
    masks[2, ::2, ::2] = 1
    masks[4] = 1
    retained = e06_floor.encode_mask_stack(masks, mode=mode)
    installed = mask_rle.encode_mask_stack(masks, mode=mode)
    assert installed == retained
    np.testing.assert_array_equal(mask_rle.decode_mask_stack(retained), masks)
    np.testing.assert_array_equal(e06_floor.decode_mask_stack(installed), masks)


def test_installed_adapter_without_experiments(tmp_path):
    # A fresh interpreter gets only the installed src tree and rejects any
    # accidental experimental import, including a lazy RLE import.
    import pathlib
    import shutil
    import subprocess
    import sys

    source = pathlib.Path(__file__).resolve().parents[2] / "src"
    shutil.copytree(source, tmp_path / "src")
    (tmp_path / "input.npz").write_bytes(envelope())
    script = """
import importlib.abc
import pathlib
import sys
class RejectExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'experiments' or fullname.startswith('experiments.'):
            raise AssertionError('installed adapter imported experiments')
sys.meta_path.insert(0, RejectExperiments())
from src.runner.packet_packing import pack_client_envelope, unpack_client_envelope
from src.runner.client import reconstruct_serialized_client
payload = pathlib.Path('input.npz').read_bytes()
for scale in (1, 2):
    packed = pack_client_envelope(payload, mask_codec='rle', mask_scale=scale)
    frames = reconstruct_serialized_client(unpack_client_envelope(packed))
    assert frames.shape == (3, 5, 7, 3)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("codec", ["psm1", "rle"])
@pytest.mark.parametrize("scale", [1, 2, 4, 8])
def test_batch_mask_storage_sparse_multiobject_parity_and_overhead(codec, scale):
    from src.runner.packet_packing import MASK_BLOB

    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[::2, ::2] = 1
    placements = tuple(
        ClientPlacement(
            crop=np.full((5, 7, 3), 180, dtype=np.uint8),
            bbox=(0, 0, 7, 5),
            mask=mask,
            frame_index=i % 2,
            object_id=str(i),
        )
        for i in range(48)
    )
    original = envelope(frames=3, placements=placements)
    per_mask = pack_client_envelope(original, mask_codec=codec, mask_scale=scale)
    batch = pack_client_envelope(original, mask_codec=codec, mask_scale=scale, batch_masks=True)
    assert batch == pack_client_envelope(
        original, mask_codec=codec, mask_scale=scale, batch_masks=True
    )
    assert len(batch) < len(per_mask)
    batch_members = members(batch)
    assert MASK_BLOB in batch_members
    assert not any(name.startswith("mask_") for name in batch_members)
    assert meta(batch) == meta(original)  # Original metadata and geometry remain intact.
    np.testing.assert_array_equal(
        reconstruct_serialized_client(unpack_client_envelope(batch)),
        reconstruct_serialized_client(unpack_client_envelope(per_mask)),
    )
    assert packing_info(batch)["zip_framing_bytes"] < packing_info(per_mask)["zip_framing_bytes"]
    assert packing_info(batch)["complete_file_bytes"] == len(batch)
    assert packing_info(batch)["packing"]["version"] == 2
    # Non-mask arrays, including crop/native packet bytes, remain byte-identical.
    for name, value in members(original).items():
        if not name.startswith("mask_"):
            assert batch_members[name] == value


def test_batch_empty_inventory_and_corrupt_ranges():
    from src.runner.packet_packing import MASK_BLOB

    original = envelope(placements=())
    batch = pack_client_envelope(original, batch_masks=True, mask_codec="rle")
    assert members(batch)[MASK_BLOB] == b""
    assert reconstruct_serialized_client(unpack_client_envelope(batch)).shape == (3, 5, 7, 3)
    batch = pack_client_envelope(envelope(), batch_masks=True)
    spec = packing_info(batch)["packing"]
    spec["masks"]["mask_0"]["offset"] = 1
    with pytest.raises(ValueError, match="range"):
        unpack_client_envelope(rewrite(batch, spec=spec))


def test_corrupt_rle_header_and_frame_modes_fail_before_allocation():
    from src.runner.mask_rle import MODE_INTRA, decode_mask_stack, encode_mask_stack
    import struct

    payload = bytearray(encode_mask_stack(np.zeros((1, 2, 2), dtype=np.uint8), mode=MODE_INTRA))
    payload[13] = 0
    with pytest.raises(ValueError, match="frame mode"):
        decode_mask_stack(payload)
    payload[13] = 1
    struct.pack_into("<HHH", payload, 6, 65535, 65535, 65535)
    with pytest.raises(ValueError, match="excessive"):
        decode_mask_stack(payload)


def test_packed_mask_expansion_limit_rejects_forged_dimensions():
    original = envelope()
    packed = pack_client_envelope(original)
    spec = packing_info(packed)["packing"]
    # Match the coded shape to the forged dimensions: the limit must be checked
    # before expanding even a valid small mask payload.
    spec["masks"]["mask_0"]["original_shape"] = [1000000000, 1000000000]
    with pytest.raises(ValueError, match="pixel limit"):
        unpack_client_envelope(rewrite(packed, spec=spec))


def test_npy_header_cannot_trigger_oversized_allocation():
    from src.runner.packet_packing import _load_array

    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, {"descr": "|u1", "fortran_order": False, "shape": (2**40,)}
    )
    with pytest.raises(ValueError, match="array exceeds"):
        _load_array(stream.getvalue())


def test_psm1_header_cannot_trigger_oversized_allocation():
    from src.runner.packet_packing import _decode_psm1
    from src.runner.mask_wire import encode_mask
    import struct

    blob = bytearray(encode_mask(np.zeros((2, 2), dtype=np.uint8)))
    struct.pack_into("<II", blob, 10, 2**31, 2**31)
    with pytest.raises(ValueError, match="pixel limit"):
        _decode_psm1(bytes(blob))
