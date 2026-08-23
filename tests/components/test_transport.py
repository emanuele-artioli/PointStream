"""Transport: serialization is not the medium.

A network backend has to reuse the serializer. If packing lives inside
DiskTransport.send, it cannot.
"""

from __future__ import annotations

from pathlib import Path

import msgpack
import pytest

from src.components.temporal.policy import ConfigurableTemporalPolicy
from src.components.transport import REGISTRY as TRANSPORT
from src.components.transport.disk import DiskTransport
from src.components.transport.medium import MemoryMedium
from src.components.transport.payload import (
    SCHEDULE_KEY,
    ChunkPayload,
    dump_schedule,
    load_schedule,
)
from src.components.transport.serialize import PayloadSerializer
from src.contracts.config import PointstreamConfig, TemporalConfig, validate_backends
from src.contracts.errors import ConfigError


def _payload(tmp_schedule=None) -> ChunkPayload:
    motion = [0.0] * 8
    planned = tmp_schedule or ConfigurableTemporalPolicy(
        TemporalConfig(
            metadata_sparsity=True,
            generation_sparsity=False,
            pipeline_sparsity=False,
            keyframe_interval=4,
            delta_threshold=20.0,
        )
    ).plan(frame_count=8, object_ids=["p1"], motion=motion)
    return ChunkPayload(
        chunk_id="c0",
        schedule=planned,
        blobs={
            "panorama.jpg": b"\xff\xd8fake-jpeg",
            "actor_0001.jpg": b"\xff\xd8crop",
            "residual.mp4": b"ftyp",
        },
        extra={"source": "unit-test"},
    )


def test_a_second_medium_reuses_the_serializer(tmp_path: Path) -> None:
    """Disk and memory both move the same packed bytes. Packing is not in either."""
    serializer = PayloadSerializer()
    payload = _payload()
    bundle = serializer.dumps(payload)

    disk = DiskTransport(root=tmp_path / "disk", serializer=serializer)
    memory = DiskTransport(
        root=tmp_path / "unused", serializer=serializer, medium=MemoryMedium()
    )
    disk.put_bundle(payload.chunk_id, bundle)
    memory.put_bundle(payload.chunk_id, bundle)

    assert disk.get_bundle(payload.chunk_id).metadata_bytes == bundle.metadata_bytes
    assert memory.get_bundle(payload.chunk_id).metadata_bytes == bundle.metadata_bytes
    assert disk.receive(payload.chunk_id).blobs == memory.receive(payload.chunk_id).blobs
    assert serializer is disk.serializer is memory.serializer


def test_jpeg_and_residual_sidecars_round_trip(tmp_path: Path) -> None:
    transport = DiskTransport(root=tmp_path)
    sent = _payload()
    transport.send(sent)
    got = transport.receive(sent.chunk_id)
    assert got.blobs["panorama.jpg"] == sent.blobs["panorama.jpg"]
    assert got.blobs["residual.mp4"] == sent.blobs["residual.mp4"]
    assert got.extra["source"] == "unit-test"
    chunk_dir = tmp_path / f"chunk_{sent.chunk_id}"
    assert (chunk_dir / "metadata.msgpack").is_file()
    assert (chunk_dir / "blobs" / "panorama.jpg").is_file()


def test_schedule_bytes_are_identical_after_disk_round_trip(tmp_path: Path) -> None:
    """Bit-identity of the decision, not of pixels."""
    sent = _payload()
    transport = DiskTransport(root=tmp_path)
    transport.send(sent)
    got = transport.receive(sent.chunk_id)
    assert msgpack.packb(dump_schedule(got.schedule), use_bin_type=True) == msgpack.packb(
        dump_schedule(sent.schedule), use_bin_type=True
    )


def test_default_transport_is_registered_disk() -> None:
    config = PointstreamConfig()
    assert config.transport == "disk"
    validate_backends(config, registries={"transport": TRANSPORT})
    assert TRANSPORT.has("disk")
    backend = TRANSPORT.build("disk")
    assert isinstance(backend, DiskTransport)


def test_unknown_transport_is_rejected() -> None:
    """A typo must not silently fall through to whatever DiskTransport used to be."""
    config = PointstreamConfig(transport="pigeon")
    with pytest.raises(ConfigError, match="Unknown transport"):
        validate_backends(config, registries={"transport": TRANSPORT})


def test_a_path_shaped_blob_name_is_rejected() -> None:
    """A slash would write outside the chunk directory."""
    payload = _payload()
    sneaky = ChunkPayload(
        chunk_id=payload.chunk_id,
        schedule=payload.schedule,
        blobs={"../outside.jpg": b"no"},
    )
    with pytest.raises(ValueError, match="plain filename"):
        PayloadSerializer().dumps(sneaky)


def test_missing_chunk_is_rejected(tmp_path: Path) -> None:
    transport = DiskTransport(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="No payload found"):
        transport.receive("does-not-exist")


def test_metadata_without_a_schedule_is_rejected() -> None:
    """The decoder cannot reconstruct the decision from config. It has to be here."""
    serializer = PayloadSerializer()
    packed = msgpack.packb({"chunk_id": "c0", "blob_names": [], "extra": {}}, use_bin_type=True)
    from src.components.transport.serialize import SerializedBundle

    with pytest.raises(ValueError, match=SCHEDULE_KEY):
        serializer.loads(SerializedBundle(metadata_bytes=packed, blobs=()))


def test_an_unknown_schedule_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported temporal-schedule schema"):
        load_schedule({"schema": "pointstream.temporal-schedule.v0", "decisions": []})
