"""Appearance must condition generation, not convert it into a paste.

The pre-fix path wrote decoded JPEG crops into ``supplied_crop`` and then
inferred ``is_generated`` from that field being empty. Appearance-on runs
therefore serialized pasted-reference placements and the client generator
never produced the scored pixels. This module is the regression for that
interaction; existing runner tests omit appearance and cannot catch it.
"""

from __future__ import annotations

from typing import Any
import hashlib
import io
import json

import numpy as np
import pytest

from src.contracts.codecs import RateControl
from src.contracts.conditioning import ConditioningBundle
from src.contracts.config import GeneratorConfig, PointstreamConfig, ResidualConfig
from src.contracts.lattice import (
    STAGE_APPEARANCE,
    STAGE_DETECTION,
    STAGE_GENERATION,
    STAGE_RESIDUAL,
    StageLattice,
)
from src.pipeline.reconstruction import GeneratorRef, ObjectRequest, bit_identical
from src.runner import lattice_config_from, run
from src.runner.client import (
    ClientPlacement,
    reconstruct_serialized_client,
    serialize_client_request,
)
from src.runner.generation_identity import config_identity_digest


def _clip(value: int, *, frames: int = 1, size: int = 16) -> np.ndarray:
    return np.full((frames, size, size, 3), value, dtype=np.uint8)


def _pose(fill: int, *, size: int = 8) -> np.ndarray:
    return np.full((3, size, size), fill, dtype=np.uint8)


def _object(*, size: int = 8, pose_fill: int = 10, appearance_fill: int = 200) -> ObjectRequest:
    appearance = np.full((size, size, 3), appearance_fill, dtype=np.uint8)
    mask = np.ones((size, size), dtype=bool)
    pose = _pose(pose_fill, size=size)
    return ObjectRequest(
        object_id="player",
        appearance=appearance,
        bbox=(0, 0, size, size),
        mask=mask,
        frame_index=0,
        conditioning=ConditioningBundle(
            appearance=appearance,
            pose=pose,
            mask=mask,
            bbox=(0, 0, size, size),
            frame_index=0,
            object_id="player",
        ),
    )


class _SeededPaint:
    """Tiny deterministic generator: seed, appearance, pose, and offset."""

    def __init__(self, offset: int = 0) -> None:
        self.offset = int(offset)

    def generate(self, conditioning: Any, *, seed: int, device: Any, params: Any) -> np.ndarray:
        bbox = conditioning.bbox or (0, 0, 8, 8)
        height = max(1, int(bbox[3] - bbox[1]))
        width = max(1, int(bbox[2] - bbox[0]))
        value = (int(seed) + self.offset) % 180
        if conditioning.appearance is not None:
            value = (value + int(np.mean(np.asarray(conditioning.appearance)))) % 180
        if conditioning.pose is not None:
            value = (value + int(np.mean(np.asarray(conditioning.pose)))) % 180
        return np.full((height, width, 3), value, dtype=np.uint8)


def _ref(offset: int = 0, name: str = "paint") -> GeneratorRef:
    return GeneratorRef(backend=_SeededPaint(offset), name=name)


def _config(*stages: str, residual: ResidualConfig | None = None) -> PointstreamConfig:
    kwargs: dict[str, Any] = {
        "lattice": lattice_config_from(StageLattice.of(*stages)),
    }
    if STAGE_GENERATION in stages:
        kwargs["generator"] = GeneratorConfig(backend="paint")
    if residual is not None:
        kwargs["residual"] = residual
    return PointstreamConfig(**kwargs)


def _lossy_residual() -> ResidualConfig:
    return ResidualConfig(
        codec="avc",
        rate_control=RateControl.CRF,
        rate=51,
        block_size=8,
        block_threshold=0.0,
        background_downscale=1,
    )


def _frame_hash(frames: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def _wire_metadata(wire: bytes) -> dict[str, Any]:
    with np.load(io.BytesIO(wire), allow_pickle=False) as arrays:
        return json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())


def _rewrite_arrays(wire: bytes, mutate: Any) -> bytes:
    with np.load(io.BytesIO(wire), allow_pickle=False) as arrays:
        payload = {key: np.array(arrays[key]) for key in arrays.files}
    mutate(payload)
    stream = io.BytesIO()
    np.savez(stream, **payload)
    return stream.getvalue()


def test_appearance_plus_generation_is_not_a_paste() -> None:
    """Bounds: paste of JPEG-200 sits near 200 in the box; the fake generator
    paints ``(seed + appearance + pose) % 180``, so ~27–117, never a paste of
    the reference. Alarm if the two delivered clips hash equal — that is the
    supplied_crop inference bug."""

    source = _clip(0, frames=1, size=16)
    objects = ((_object(),),)
    generated = run(
        _config(STAGE_DETECTION, STAGE_APPEARANCE, STAGE_GENERATION),
        [source],
        generator=_ref(),
        objects=objects,
    )
    pasted = run(
        _config(STAGE_DETECTION, STAGE_APPEARANCE),
        [source],
        objects=objects,
    )
    assert not bit_identical(generated.delivered_frames, pasted.delivered_frames)
    assert _frame_hash(generated.delivered_frames) != _frame_hash(pasted.delivered_frames)

    generated_wire = generated.chunks[0].bag["wire_request"]
    assert isinstance(generated_wire, bytes)
    wire_meta = _wire_metadata(generated_wire)
    assert wire_meta["generator"] is not None
    assert wire_meta["generator"]["checkpoint_sha256"] == "injected:paint"
    assert wire_meta["references"], "appearance bytes must still travel as conditioning"


def test_shuffled_and_foreign_pose_change_delivered_frames() -> None:
    """Correct pose mean 10 vs inverted 245 vs foreign 80. The generator adds
    pose mean, so delivered hashes must move. Alarm if they do not: pose never
    reached the client generator."""

    source = _clip(0, frames=1, size=16)
    ref = _ref()
    result = run(
        _config(STAGE_DETECTION, STAGE_APPEARANCE, STAGE_GENERATION),
        [source],
        generator=ref,
        objects=((_object(pose_fill=10),),),
    )
    wire = result.chunks[0].bag["wire_request"]
    assert isinstance(wire, bytes)
    baseline = reconstruct_serialized_client(wire, generator=ref)
    baseline_hash = _frame_hash(np.asarray(baseline))

    def _invert_pose(payload: dict[str, np.ndarray]) -> None:
        found = False
        for key, value in payload.items():
            if key.startswith("pose_"):
                payload[key] = np.ascontiguousarray(255 - np.asarray(value, dtype=np.uint8))
                found = True
        assert found, "serialized client has no pose array"

    shuffled = reconstruct_serialized_client(_rewrite_arrays(wire, _invert_pose), generator=ref)
    assert _frame_hash(np.asarray(shuffled)) != baseline_hash

    foreign = run(
        _config(STAGE_DETECTION, STAGE_APPEARANCE, STAGE_GENERATION),
        [source],
        generator=_ref(),
        objects=((_object(pose_fill=80),),),
    )
    assert _frame_hash(foreign.delivered_frames) != _frame_hash(result.delivered_frames)


def test_fresh_serialized_client_matches_run_delivered_frames() -> None:
    """A client that sees only the envelope (no source arrays) must reproduce
    the scored delivered clip."""

    source = _clip(40, frames=1, size=16)
    ref = _ref()
    result = run(
        _config(STAGE_DETECTION, STAGE_APPEARANCE, STAGE_GENERATION),
        [source],
        generator=ref,
        objects=((_object(),),),
    )
    wire = result.chunks[0].bag["wire_request"]
    assert isinstance(wire, bytes)
    fresh = reconstruct_serialized_client(wire, generator=ref)
    assert bit_identical(np.asarray(fresh), result.delivered_frames)


def test_generator_perturbation_changes_hashes_quality_and_residual() -> None:
    """Offset 0 vs 90. Source is 0; generated box fill moves by tens of grey
    levels, so residual energy, reconstruction quality, and (lossy) delivered
    hashes must all move. Alarm if any stays put: the scored path ignored the
    generator."""

    source = _clip(0, frames=1, size=16)
    objects = ((_object(),),)
    stages = (STAGE_DETECTION, STAGE_APPEARANCE, STAGE_GENERATION, STAGE_RESIDUAL)
    config = _config(*stages, residual=_lossy_residual())
    baseline = run(config, [source], generator=_ref(0), objects=objects)
    perturbed = run(config, [source], generator=_ref(90), objects=objects)

    assert _frame_hash(baseline.delivered_frames) != _frame_hash(perturbed.delivered_frames)
    assert not bit_identical(baseline.delivered_frames, perturbed.delivered_frames)
    assert baseline.delivered_quality.whole_frame() != perturbed.delivered_quality.whole_frame()
    assert baseline.quality.whole_frame() != perturbed.quality.whole_frame()
    assert baseline.sizes.residual != perturbed.sizes.residual
    assert baseline.sizes.as_dict()["residual"] != perturbed.sizes.as_dict()["residual"]


def test_no_duplicate_paste_and_generate_placements() -> None:
    source = _clip(0, frames=1, size=16)
    result = run(
        _config(STAGE_DETECTION, STAGE_APPEARANCE, STAGE_GENERATION),
        [source],
        generator=_ref(),
        objects=((_object(),),),
    )
    wire = result.chunks[0].bag["wire_request"]
    assert isinstance(wire, bytes)
    meta = _wire_metadata(wire)
    keys = [(item["object_id"], item["frame_index"]) for item in meta["placements"]]
    assert keys == list(dict.fromkeys(keys))
    assert meta["placements"], "expected at least one placement"
    assert all(item["is_generated"] for item in meta["placements"])
    assert not any(item.get("crop_key") for item in meta["placements"])
    assert not any(item.get("encoded_crop_key") for item in meta["placements"])

    reconstructed = reconstruct_serialized_client(
        wire,
        generator=_ref(),
    )
    assert reconstructed is not None


def test_missing_checkpoint_identity_fails_closed() -> None:
    placement = ClientPlacement(
        bbox=(0, 0, 8, 8),
        frame_index=0,
        object_id="player",
        is_generated=True,
        pose=_pose(10),
    )
    wire = serialize_client_request(
        background=None,
        frame_count=1,
        height=16,
        width=16,
        placements=(placement,),
        generator_meta={
            "name": "paint",
            "seed": 1337,
            "params": {},
            "capabilities": [],
            "requires": [],
        },
    )
    with pytest.raises(ValueError, match="identity"):
        reconstruct_serialized_client(wire, generator=_ref())


def test_unknown_checkpoint_identity_fails_closed() -> None:
    placement = ClientPlacement(
        bbox=(0, 0, 8, 8),
        frame_index=0,
        object_id="player",
        is_generated=True,
        pose=_pose(10),
    )
    digest = "ab" * 32
    meta = {
        "name": "paint",
        "seed": 1337,
        "params": {},
        "capabilities": [],
        "requires": [],
        "checkpoint_id": f"ghost.pt:{digest}",
        "checkpoint_sha256": digest,
    }
    meta["config_identity"] = config_identity_digest(meta)
    wire = serialize_client_request(
        background=None,
        frame_count=1,
        height=16,
        width=16,
        placements=(placement,),
        generator_meta=meta,
    )
    with pytest.raises(ValueError, match="[Uu]nknown|[Mm]ismatched checkpoint"):
        reconstruct_serialized_client(wire)
    with pytest.raises(ValueError, match="[Mm]ismatched checkpoint"):
        reconstruct_serialized_client(wire, generator=_ref())
