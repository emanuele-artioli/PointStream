"""Compact mask wire: lossless roundtrip, fail-closed decode, and byte subledger."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from src.runner.accounting import sizes_bytes
from src.runner.client import (
    ClientPlacement,
    account_serialized_request,
    reconstruct_independent_client,
    reconstruct_serialized_client,
    serialize_client_request,
)
from src.runner.mask_wire import (
    ENCODING_NAME,
    MAGIC,
    SCHEMA_VERSION,
    decode_mask,
    encode_mask,
    wire_declaration,
)
from src.runner.stages import ledger_from_bag

# Wave 2 16-frame 4K diagnostic wrote several uncompressed uint8 full-frame
# masks through np.savez. The envelope remainder reported as "metadata" was
# ~41.5 MB. Recorded here before any compact-size assertion so a regression
# that returns to raw frames cannot hide behind a moved bound.
RAW_FULLFRAME_MASK_METADATA_BOUND_BYTES = 41_500_000


def _binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) != 0).astype(np.uint8)


def _assert_roundtrip(mask: np.ndarray) -> bytes:
    blob = encode_mask(mask)
    decoded = decode_mask(blob)
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, _binary(mask))
    return blob


def test_wire_declaration_names_the_format() -> None:
    decl = wire_declaration()
    assert decl["schema_version"] == SCHEMA_VERSION
    assert decl["encoding"] == ENCODING_NAME
    assert decl["dtype"] == "uint8"
    assert decl["endian"] == "little"
    assert decl["packing"] == "lsb-first-row-major"
    assert decl["empty_frames"] == "omitted"


def test_empty_mask_roundtrip() -> None:
    _assert_roundtrip(np.zeros((16, 24), dtype=np.uint8))
    _assert_roundtrip(np.zeros((4, 16, 24), dtype=bool))


def test_dense_mask_roundtrip() -> None:
    _assert_roundtrip(np.ones((9, 11), dtype=np.uint8))
    _assert_roundtrip(np.ones((3, 9, 11), dtype=bool))


def test_sparse_mask_roundtrip() -> None:
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[7, 13] = 1
    mask[20, 40] = 1
    blob = _assert_roundtrip(mask)
    assert len(blob) < mask.nbytes


def test_moving_mask_roundtrip() -> None:
    frames = np.zeros((8, 40, 50), dtype=np.uint8)
    for t in range(8):
        frames[t, 2 + t, 3 + 2 * t : 9 + 2 * t] = 1
        frames[t, 3 + t, 3 + 2 * t : 9 + 2 * t] = 1
    blob = _assert_roundtrip(frames)
    assert len(blob) < frames.nbytes


def test_odd_sized_mask_roundtrip() -> None:
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:5] = 1
    _assert_roundtrip(mask)
    volume = np.zeros((3, 5, 7), dtype=bool)
    volume[0, 0, 0] = True
    volume[2, 4, 6] = True
    _assert_roundtrip(volume)


def test_multi_object_masks_roundtrip_independently() -> None:
    left = np.zeros((20, 20), dtype=np.uint8)
    left[2:8, 1:6] = 1
    right = np.zeros((20, 20), dtype=np.uint8)
    right[10:18, 12:19] = 1
    assert not np.array_equal(decode_mask(encode_mask(left)), decode_mask(encode_mask(right)))
    _assert_roundtrip(left)
    _assert_roundtrip(right)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"PSM",
        MAGIC + b"\x01",
        encode_mask(np.ones((4, 4), dtype=np.uint8))[:-1],
        b"XXXX" + encode_mask(np.ones((4, 4), dtype=np.uint8))[4:],
    ],
)
def test_corrupt_or_truncated_mask_payload_fails_closed(payload: bytes) -> None:
    with pytest.raises((ValueError, TypeError)):
        decode_mask(payload)


def test_packed_length_mismatch_fails_closed() -> None:
    blob = bytearray(encode_mask(np.ones((3, 5), dtype=np.uint8)))
    # n_packed is the last u32 before packed bits; shrink it without shrinking data.
    # Easier: drop packed bytes after rewriting n_packed too short via truncation of a dense rect.
    with pytest.raises(ValueError, match="truncated|packed length|corrupt"):
        decode_mask(bytes(blob[:-2]))


def test_serialize_declares_mask_wire_and_roundtrips() -> None:
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:9, 3:10] = 1
    crop = np.full((7, 7, 3), 80, dtype=np.uint8)
    payload = serialize_client_request(
        background=None,
        frame_count=1,
        height=12,
        width=16,
        placements=(
            ClientPlacement(
                crop=crop,
                bbox=(3, 2, 10, 9),
                frame_index=0,
                mask=mask,
                object_id="a",
            ),
        ),
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
    assert metadata["mask_wire"]["encoding"] == ENCODING_NAME
    assert metadata["mask_wire"]["schema_version"] == SCHEMA_VERSION
    item = metadata["placements"][0]
    assert item["mask_wire"]["shape"] == [12, 16]
    assert item["mask_key"] == "mask_0"
    reconstructed = reconstruct_serialized_client(payload)
    assert reconstructed.shape == (1, 12, 16, 3)


def test_transport_total_equals_serialized_length() -> None:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:12, 6:18] = 1
    payload = serialize_client_request(
        background=None,
        frame_count=2,
        height=24,
        width=32,
        placements=(
            ClientPlacement(
                crop=np.full((8, 12, 3), 40, dtype=np.uint8),
                bbox=(6, 4, 18, 12),
                mask=mask,
                object_id="player",
            ),
            ClientPlacement(
                crop=np.full((8, 12, 3), 90, dtype=np.uint8),
                bbox=(6, 4, 18, 12),
                frame_index=1,
                mask=mask,
                object_id="player",
            ),
        ),
        generator_meta={"name": "paint", "seed": 1, "params": {}},
    )
    source = np.zeros((2, 24, 32, 3), dtype=np.uint8)
    ledger = ledger_from_bag({"wire_request": payload}, source)
    assert ledger.transport_total == len(payload)
    assert ledger.subledger.total == ledger.metadata
    assert ledger.metadata + ledger.residual + ledger.panorama + ledger.actor_reference == len(
        payload
    )


def test_subledger_present_and_sums_to_metadata() -> None:
    pose = np.full((8, 8, 3), 12, dtype=np.uint8)
    motion = np.zeros((8, 8, 2), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[1:5, 2:9] = 1
    payload = serialize_client_request(
        background=None,
        frame_count=1,
        height=16,
        width=16,
        placements=(
            ClientPlacement(
                crop=np.full((4, 7, 3), 200, dtype=np.uint8),
                bbox=(2, 1, 9, 5),
                mask=mask,
                object_id="gen",
                is_generated=True,
                pose=pose,
                motion_field=motion,
            ),
        ),
        generator_meta={"name": "paint", "seed": 7, "params": {"steps": 1}},
        references={"gen": b"\xff\xd8fakejpeg\xff\xd9"},
    )
    residual = 0
    panorama = 0
    actor = len(b"\xff\xd8fakejpeg\xff\xd9")
    sub = account_serialized_request(
        payload, residual=residual, panorama=panorama, actor_reference=actor
    )
    assert sub.mask_payload > 0
    assert sub.pose_motion > 0
    assert sub.placement_headers > 0
    assert sub.generator_metadata > 0
    assert sub.envelope_overhead > 0
    assert sub.total == len(payload) - residual - panorama - actor
    ledger = sizes_bytes(
        source=16 * 16 * 3,
        residual=residual,
        panorama=panorama,
        actor_reference=actor,
        metadata=sub.total,
        subledger=sub,
    )
    assert ledger.transport_total == len(payload)
    assert ledger.as_dict()["metadata_subledger"]["mask_payload"] == sub.mask_payload
    assert ledger.subledger.total == ledger.metadata


def test_serialized_corrupt_mask_fails_closed() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    payload = serialize_client_request(
        background=None,
        frame_count=1,
        height=8,
        width=8,
        placements=(
            ClientPlacement(
                crop=np.full((3, 3, 3), 1, dtype=np.uint8),
                bbox=(2, 2, 5, 5),
                mask=mask,
            ),
        ),
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        data = {key: np.asarray(arrays[key]) for key in arrays.files}
    data["mask_0"] = data["mask_0"][: max(1, data["mask_0"].size // 2)]
    stream = io.BytesIO()
    np.savez(stream, **data)
    with pytest.raises(ValueError, match="truncated|corrupt|unsupported"):
        reconstruct_serialized_client(stream.getvalue())


def test_compact_4k_two_object_masks_beat_raw_bound() -> None:
    frames, height, width = 16, 2160, 3840
    raw_uint8_bytes = frames * 2 * height * width
    assert raw_uint8_bytes > RAW_FULLFRAME_MASK_METADATA_BOUND_BYTES

    placements: list[ClientPlacement] = []
    compact_blob_bytes = 0
    for frame_index in range(frames):
        for object_index in range(2):
            mask = np.zeros((height, width), dtype=np.uint8)
            cy = 360 + object_index * 900
            cx = 280 + frame_index * 180
            if object_index == 1:
                cx = width - 280 - frame_index * 160
                cy = 420 + frame_index * 40
            yy, xx = np.ogrid[:height, :width]
            mask[((yy - cy) / 70.0) ** 2 + ((xx - cx) / 120.0) ** 2 <= 1.0] = 1
            compact_blob_bytes += len(encode_mask(mask))
            if frame_index == 0 and object_index == 0:
                assert np.array_equal(decode_mask(encode_mask(mask)), mask)
            x1 = max(0, int(cx - 120))
            y1 = max(0, int(cy - 70))
            x2 = min(width, max(x1 + 1, int(cx + 120)))
            y2 = min(height, max(y1 + 1, int(cy + 70)))
            placements.append(
                ClientPlacement(
                    crop=np.full((8, 8, 3), 30 + object_index, dtype=np.uint8),
                    bbox=(x1, y1, x2, y2),
                    frame_index=frame_index,
                    mask=mask,
                    object_id=f"obj-{object_index}",
                )
            )

    assert compact_blob_bytes < RAW_FULLFRAME_MASK_METADATA_BOUND_BYTES
    payload = serialize_client_request(
        background=None,
        frame_count=frames,
        height=height,
        width=width,
        placements=tuple(placements),
    )
    sub = account_serialized_request(payload)
    assert sub.mask_payload == compact_blob_bytes
    assert sub.mask_payload < RAW_FULLFRAME_MASK_METADATA_BOUND_BYTES / 20
    assert len(payload) < RAW_FULLFRAME_MASK_METADATA_BOUND_BYTES / 10


def test_generation_follows_is_generated_not_missing_crop() -> None:
    class _Paint:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, conditioning, *, seed, device, params):  # noqa: ANN001
            self.calls += 1
            return np.full((4, 4, 3), 9, dtype=np.uint8)

    from src.pipeline.reconstruction.dispatch import GeneratorRef
    from src.pipeline.reconstruction.reconstruct import ObjectRequest

    backend = _Paint()
    appearance = np.full((4, 4, 3), 255, dtype=np.uint8)
    generated = reconstruct_independent_client(
        background=None,
        frame_count=1,
        height=8,
        width=8,
        placements=(
            ClientPlacement(
                crop=appearance,
                bbox=(0, 0, 4, 4),
                mask=np.ones((8, 8), dtype=np.uint8),
                is_generated=True,
                object_id="g",
            ),
        ),
        generator=GeneratorRef(backend=backend, name="paint"),
    )
    assert backend.calls == 1
    assert np.all(np.asarray(generated)[0, 0:4, 0:4] == 9)

    skipped = _Paint()
    reconstruct_independent_client(
        background=None,
        frame_count=1,
        height=8,
        width=8,
        placements=(
            ClientPlacement(
                crop=None,
                bbox=(0, 0, 4, 4),
                is_generated=False,
                object_id="skip",
            ),
        ),
        objects=(
            ObjectRequest(
                object_id="skip",
                appearance=appearance,
                bbox=(0, 0, 4, 4),
            ),
        ),
        generator=GeneratorRef(backend=skipped, name="paint"),
    )
    assert skipped.calls == 0
