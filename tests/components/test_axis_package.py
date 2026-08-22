"""The per-axis registry table exists so parallel streams do not share one dict."""

from src.components import all_registries, describe_all

EXPECTED = frozenset(
    {
        "appearance",
        "background",
        "codec",
        "detector",
        "domain",
        "generator",
        "metric",
        "motion",
        "pose",
        "rigid",
        "scene",
        "segmenter",
        "selection",
        "temporal",
        "tracking",
        "transport",
    }
)


def test_every_axis_has_its_own_registry() -> None:
    tables = all_registries()
    assert set(tables) == EXPECTED
    for axis, registry in tables.items():
        assert registry.axis == axis


def test_listing_every_axis_does_not_require_a_backend_to_be_registered() -> None:
    text = describe_all()
    for axis in EXPECTED:
        assert f"{axis}:" in text
