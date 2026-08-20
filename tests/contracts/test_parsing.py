"""Tests for strict config parsing.

The behaviours worth pinning here are the ones whose absence caused real
damage: a typo that silently did nothing, a knob documented but unreachable,
and a value coerced into shape rather than rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
from pathlib import Path

import pytest

from src.contracts import parsing
from src.contracts.errors import ConfigError, ConfigKeyError, ConfigValueError


class Mode(enum.Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True)
class Detector:
    backend: str = "yolo"
    model: str = "yolo26n.pt"


@dataclass(frozen=True)
class Residual:
    codec: str = "av1"
    crf: int = 35
    pix_fmt: str | None = None


@dataclass(frozen=True)
class Sample:
    detector: Detector = field(default_factory=Detector)
    residual: Residual = field(default_factory=Residual)
    seed: int = 1337
    mode: Mode = Mode.FAST
    metrics: tuple[str, ...] = ()
    weights: Path | None = None


@dataclass(frozen=True)
class Demanding:
    required: int
    optional: int = 0


class TestKeysAreChecked:
    def test_nested_values_reach_their_fields(self) -> None:
        built = parsing.build(
            Sample, {"detector": {"backend": "sam3"}, "residual": {"crf": 28}}
        )
        assert built.detector.backend == "sam3"
        assert built.residual.crf == 28
        assert built.residual.codec == "av1", "untouched fields keep their defaults"

    def test_both_spellings_reach_the_same_field(self) -> None:
        kebab = parsing.build(Sample, {"residual": {"pix-fmt": "yuv420p"}})
        snake = parsing.build(Sample, {"residual": {"pix_fmt": "yuv420p"}})
        assert kebab == snake

    def test_unknown_key_is_rejected_not_dropped(self) -> None:
        """The defect this whole module exists for.

        Three knobs sat in the shipped config for months, documented and
        commented, with no backing field. Their readers used
        `getattr(config, name, default)` and so always got the default, which
        made them silently inert.
        """
        with pytest.raises(ConfigError) as caught:
            parsing.build(Sample, {"detector": {"backend": "yolo"}, "canny-lower-threshold": "auto"})
        assert any(isinstance(problem, ConfigKeyError) for problem in caught.value.problems)
        assert "canny-lower-threshold" in str(caught.value)

    def test_unknown_key_suggests_the_intended_one(self) -> None:
        with pytest.raises(ConfigError) as caught:
            parsing.build(Sample, {"detecter": {}})
        assert "Did you mean 'detector'?" in str(caught.value)

    def test_unknown_nested_key_names_its_section(self) -> None:
        with pytest.raises(ConfigError) as caught:
            parsing.build(Sample, {"residual": {"crff": 28}})
        assert "under 'residual'" in str(caught.value)

    def test_every_problem_is_reported_at_once(self) -> None:
        """Reporting one problem per run is how people stop running validation."""
        with pytest.raises(ConfigError) as caught:
            parsing.build(
                Sample,
                {"detecter": {}, "residual": {"crff": 1}, "seed": "many"},
            )
        assert len(caught.value.problems) == 3


class TestValuesAreCheckedNotCoerced:
    def test_text_is_not_accepted_as_a_number(self) -> None:
        with pytest.raises(ConfigError) as caught:
            parsing.build(Sample, {"seed": "1337"})
        assert isinstance(caught.value.problems[0], ConfigValueError)

    def test_bool_is_not_accepted_as_an_int(self) -> None:
        """`bool` subclasses `int`, so an unguarded check turns `true` into 1."""
        with pytest.raises(ConfigError):
            parsing.build(Sample, {"residual": {"crf": True}})

    def test_enum_comes_from_its_value(self) -> None:
        assert parsing.build(Sample, {"mode": "slow"}).mode is Mode.SLOW

    def test_bad_enum_value_lists_the_legal_ones(self) -> None:
        with pytest.raises(ConfigError) as caught:
            parsing.build(Sample, {"mode": "medium"})
        assert "'fast'" in str(caught.value) and "'slow'" in str(caught.value)

    def test_optional_field_accepts_null(self) -> None:
        assert parsing.build(Sample, {"residual": {"pix_fmt": None}}).residual.pix_fmt is None

    def test_non_optional_field_rejects_null(self) -> None:
        with pytest.raises(ConfigError):
            parsing.build(Sample, {"seed": None})

    def test_list_becomes_a_tuple(self) -> None:
        assert parsing.build(Sample, {"metrics": ["psnr", "vmaf"]}).metrics == ("psnr", "vmaf")

    def test_scalar_where_a_list_belongs_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            parsing.build(Sample, {"metrics": "psnr"})

    def test_path_is_built_from_text(self) -> None:
        assert parsing.build(Sample, {"weights": "assets/weights"}).weights == Path("assets/weights")

    def test_missing_required_field_is_reported(self) -> None:
        with pytest.raises(ConfigError) as caught:
            parsing.build(Demanding, {"optional": 2})
        assert "required" in str(caught.value)

    def test_non_dataclass_is_a_programming_error(self) -> None:
        with pytest.raises(TypeError):
            parsing.build(dict, {})


class TestRoundTrip:
    def test_rendering_and_rebuilding_preserves_the_config(self) -> None:
        original = parsing.build(Sample, {"detector": {"backend": "sam3"}, "seed": 7})
        assert parsing.build(Sample, parsing.to_mapping(original)) == original

    def test_rendering_uses_config_spelling(self) -> None:
        rendered = parsing.to_mapping(parsing.build(Sample, {}))
        assert "pix-fmt" in rendered["residual"]

    def test_rendered_keys_are_all_accepted(self) -> None:
        """Generating the shipped config from the schema is what keeps a
        documented key from outliving the field that backs it."""
        parsing.require_known_keys(Sample, parsing.to_mapping(parsing.build(Sample, {})))

    def test_flat_keys_walks_into_sections(self) -> None:
        keys = parsing.flat_keys(Sample)
        assert "detector.backend" in keys
        assert "residual.crf" in keys
        assert "detector" not in keys, "sections themselves are not leaf keys"


class TestMerge:
    def test_sections_merge_rather_than_replace(self) -> None:
        merged = parsing.merge({"residual": {"codec": "av1", "crf": 35}}, {"residual": {"crf": 28}})
        assert merged["residual"] == {"codec": "av1", "crf": 28}

    def test_spellings_do_not_produce_two_entries(self) -> None:
        merged = parsing.merge({"pix-fmt": "yuv420p"}, {"pix_fmt": "yuv444p"})
        assert merged == {"pix_fmt": "yuv444p"}

    def test_scalars_replace(self) -> None:
        assert parsing.merge({"seed": 1}, {"seed": 2})["seed"] == 2


class TestKeyCheckingWithoutBuilding:
    def test_reports_unknown_keys_past_an_invalid_value(self) -> None:
        """A bad value in one section must not hide a typo in another."""
        with pytest.raises(ConfigError) as caught:
            parsing.require_known_keys(Sample, {"seed": "not-a-number", "detecter": {}})
        assert len(caught.value.problems) == 1
        assert "detecter" in str(caught.value)

    def test_accepts_a_config_it_recognises(self) -> None:
        parsing.require_known_keys(Sample, {"detector": {"backend": "sam3"}, "seed": 7})
