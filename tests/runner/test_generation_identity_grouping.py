from __future__ import annotations

import numpy as np
import pytest

from src.contracts.capabilities import CAP_TEMPORAL_SEQUENCE
from src.contracts.conditioning import ConditioningBundle
from src.pipeline.reconstruction.dispatch import GeneratorRef
from src.runner.generation import dispatch_by_object_identity


class _TemporalByIdentity:
    required = ()

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, int]]] = []

    def generate(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("sequence backend should be called as a sequence")

    def generate_sequence(self, bundles, *, seed, device, params):  # noqa: ANN001
        _ = (seed, device, params)
        self.calls.append([(str(item.object_id), int(item.frame_index)) for item in bundles])
        return tuple(
            np.full((4, 4, 3), ord(str(item.object_id)[0]) + int(item.frame_index), dtype=np.uint8)
            for item in bundles
        )


def _bundle(object_id: str, frame_index: int) -> ConditioningBundle:
    return ConditioningBundle(
        appearance=np.zeros((3, 4, 4), dtype=np.uint8),
        object_id=object_id,
        object_class="player",
        frame_index=frame_index,
    )


def test_temporal_generation_is_grouped_and_sorted_by_stable_object_identity() -> None:
    backend = _TemporalByIdentity()
    generator = GeneratorRef(
        backend=backend,
        capabilities=frozenset({CAP_TEMPORAL_SEQUENCE}),
    )
    bundles = (
        _bundle("p", 1),
        _bundle("q", 0),
        _bundle("p", 0),
        _bundle("q", 1),
    )
    crops, decisions = dispatch_by_object_identity(generator, bundles, seed=1)
    assert backend.calls == [[("p", 0), ("p", 1)], [("q", 0), ("q", 1)]]
    assert crops[0][0, 0, 0] == ord("p") + 1
    assert crops[2][0, 0, 0] == ord("p")
    assert crops[1][0, 0, 0] == ord("q")
    assert len(decisions) == 2


def test_temporal_generation_requires_object_identity() -> None:
    generator = GeneratorRef(
        backend=_TemporalByIdentity(),
        capabilities=frozenset({CAP_TEMPORAL_SEQUENCE}),
    )
    with pytest.raises(ValueError, match="requires an explicit object_id"):
        dispatch_by_object_identity(
            generator,
            (ConditioningBundle(appearance=np.zeros((3, 4, 4), dtype=np.uint8)),),
            seed=1,
        )
