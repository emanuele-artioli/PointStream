"""A mixed coded/raw ledger must not present a compression ratio (BP24).

`PLAN.md` §3: a path reports its ratio only when no component in it is still
raw. A total that mixes a coded bitstream with an array size looks like a rate
and is not — and a raw number is more dangerous once its neighbours are real,
because the total stops being obviously wrong.
"""

from __future__ import annotations

import pytest

from src.runner.accounting import SizesBytes, sizes_bytes


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
