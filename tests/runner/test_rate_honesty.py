"""A mixed coded/raw ledger must not present a compression ratio (BP24).

`PLAN.md` §3: a path reports its ratio only when no component in it is still
raw. A total that mixes a coded bitstream with an array size looks like a rate
and is not — and a raw number is more dangerous once its neighbours are real,
because the total stops being obviously wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts.lattice import ART_APPEARANCE_PAYLOAD
from src.runner.accounting import SizesBytes, sizes_bytes
from src.runner.stages import _actor_bytes_exact, _encoded_cost


def test_ratio_is_reported_when_every_part_is_coded() -> None:
    ledger = sizes_bytes(source=1_000_000, residual=2_000, panorama=8_000)
    out = ledger.as_dict()
    assert ledger.is_rate is True
    assert out["transport_to_source_ratio"] == pytest.approx(10_000 / 1_000_000)
    assert "not_a_rate" not in out


def test_a_raw_part_withholds_the_ratio_and_says_which() -> None:
    ledger = sizes_bytes(
        source=1_000_000, residual=2_000, panorama=8_000, raw_parts=("panorama",)
    )
    out = ledger.as_dict()
    assert ledger.is_rate is False
    assert out["transport_to_source_ratio"] is None
    assert out["raw_parts"] == ["panorama"]
    assert "panorama" in out["not_a_rate"]


def test_summing_chunks_keeps_a_raw_part_raw() -> None:
    """The laundering case: one clean chunk must not absolve a raw one."""
    clean = sizes_bytes(source=10, residual=1)
    dirty = sizes_bytes(source=10, residual=1, raw_parts=("residual",))
    total = clean + dirty
    assert total.raw_parts == ("residual",)
    assert total.is_rate is False
    assert total.as_dict()["transport_to_source_ratio"] is None


def test_summing_two_clean_chunks_stays_a_rate() -> None:
    total = sizes_bytes(source=10, residual=1) + sizes_bytes(source=10, residual=1)
    assert total.is_rate is True
    assert total.as_dict()["transport_to_source_ratio"] is not None


def test_raw_parts_rejects_a_name_that_is_not_a_component() -> None:
    """A typo must fail loudly rather than silently disarming the guard."""
    with pytest.raises(ValueError, match="unknown components"):
        sizes_bytes(source=10, residual=1, raw_parts=("panoramas",))


def test_duplicate_raw_parts_are_recorded_once() -> None:
    total = SizesBytes(source=10, raw_parts=("residual",)) + SizesBytes(
        source=10, raw_parts=("residual",)
    )
    assert total.raw_parts == ("residual",)


# ---------------------------------------------------------------------------
# `actor_reference` — coded, packed, or a stand-in?
#
# BP24 listed it raw unconditionally, with the reason recorded: appearance
# reported a measured size and nobody had shown it was a transmitted one.
# `outputs/bp24-ladder/appearance-cost.json` settled it per backend by driving
# each one — `compressed-image` returns a real JPEG bitstream whose size moves
# with quality (1,448 / 2,020 / 7,732 B at q20 / q60 / q95) and which decodes
# back to the crop at MAE 2.83; `image-embedding` and `diffusion-latent` return
# a packed float16 buffer whose length equals the declared cost exactly.
#
# All three are wire costs, so `actor_reference` clears. What must not happen is
# it clearing *by default*: a payload that does not state `exact` has to keep
# withholding the ratio, or the ledger silently regains a raw part the next time
# an appearance backend is added.
# ---------------------------------------------------------------------------


class _BareDescriptor:
    """An encode result with a size and no stated cost. The dangerous shape."""

    measured_bytes = 4096


def test_a_descriptor_with_no_stated_cost_is_not_a_wire_cost() -> None:
    size, exact, basis = _encoded_cost(_BareDescriptor())
    assert size == 4096
    assert exact is False
    assert basis


def test_the_shipped_appearance_backends_state_a_wire_cost() -> None:
    """Driven, not read off the code: each backend encodes a real crop."""
    from src.components.appearance.compressed import CompressedImageAppearance
    from src.components.appearance.embedding import ImageEmbeddingAppearance
    from src.components.appearance.latent import DiffusionLatentAppearance

    rng = np.random.default_rng(11)
    ramp = np.linspace(0, 255, 64, dtype=np.float32)
    crop = np.clip(
        ramp[:, None, None] + ramp[None, :, None] / 2.0 + rng.normal(0, 2.0, (64, 64, 3)),
        0,
        255,
    ).astype(np.uint8)

    for backend in (
        CompressedImageAppearance(quality=90),
        ImageEmbeddingAppearance(),
        DiffusionLatentAppearance(),
    ):
        descriptor, payload = backend.encode(crop)
        size, exact, _ = _encoded_cost((descriptor, payload))
        assert exact is True, f"{backend.kind} does not state a wire cost"
        # The declared number and the buffer must be the same bytes. A mismatch
        # means the ledger would count something other than what is sent.
        assert size == len(payload) == descriptor.cost().byte_count


def test_the_jpeg_appearance_size_moves_with_quality() -> None:
    """A flag existing is not a feature working: drive the knob, see it move."""
    from src.components.appearance.compressed import CompressedImageAppearance

    ramp = np.linspace(0, 255, 64, dtype=np.float32)
    crop = np.clip(
        ramp[:, None, None] + ramp[None, :, None] / 2.0, 0, 255
    ).astype(np.uint8)
    backend = CompressedImageAppearance(quality=90)
    sizes = [len(backend.encode(crop, quality=q)[1]) for q in (20, 60, 95)]
    assert sizes[0] < sizes[1] < sizes[2]


def test_actor_reference_stays_raw_when_the_payload_does_not_state_exact() -> None:
    """The default is 'not a wire cost'. Defaulting the other way is the bug."""
    assert _actor_bytes_exact({ART_APPEARANCE_PAYLOAD: {"byte_count": 500}}) is False
    assert _actor_bytes_exact({}) is False
    assert (
        _actor_bytes_exact({ART_APPEARANCE_PAYLOAD: {"byte_count": 500, "exact": True}})
        is True
    )
