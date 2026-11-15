import numpy as np
import pytest

from experiments.tennis.k6b import checked_generated_crops, unit_name


def fixture():
    wid, player = "match/001", "near"
    name = unit_name(wid, player)
    mask = np.ones((8, 8), dtype=bool)
    regions = [{player: mask.copy()}, {player: mask.copy()}]
    units = {name: {"t": np.array([0, 1]), "box": np.array([[0, 0, 8, 8], [0, 0, 8, 8]])}}
    original = {"t": np.array([0, 1]), "crops": np.zeros((2, 2, 2, 3), dtype=np.uint8)}
    fitted = {"t": np.array([0, 1]), "crops": np.ones((2, 2, 2, 3), dtype=np.uint8)}
    generated = {(name, "original"): original, (name, "finetuned_tennis"): fitted}
    return wid, player, name, regions, units, generated


def check(data, legacy=False):
    wid, player, _, regions, units, generated = data
    return checked_generated_crops(wid, [player], ("original", "finetuned_tennis"), [0, 1],
                                   regions, units, generated, legacy)


def test_missing_fitted_unit_or_tick_requires_explicit_legacy():
    data = fixture()
    name, generated = data[2], data[-1]
    del generated[(name, "finetuned_tennis")]
    with pytest.raises(ValueError, match="missing generated crop"):
        check(data)
    crops, fallback = check(data, legacy=True)
    assert len(fallback) == 2 and all(item["source_profile"] == "original" for item in fallback)
    assert not crops[("finetuned_tennis", "near", 0)][0].any()
    generated[(name, "finetuned_tennis")] = {"t": np.array([0]), "crops": np.ones((1, 2, 2, 3))}
    with pytest.raises(ValueError, match="t=1"):
        check(data)
    _, fallback = check(data, legacy=True)
    assert [item["t"] for item in fallback] == [1]


def test_missing_both_and_missing_prepared_unit_fail_even_in_legacy():
    data = fixture()
    name, generated = data[2], data[-1]
    generated.clear()
    with pytest.raises(ValueError, match="missing generated crop"):
        check(data, legacy=True)
    data[4].pop(name)
    with pytest.raises(ValueError, match="missing prepared unit"):
        check(data, legacy=True)


@pytest.mark.parametrize("field", ["duplicate_generated", "short_generated", "duplicate_box", "short_box", "missing_box", "misses_region"])
def test_malformed_or_missing_joins_fail(field):
    data = fixture()
    name, units, generated = data[2], data[4], data[5]
    if field == "duplicate_generated":
        generated[(name, "original")]["t"] = np.array([0, 0])
    elif field == "short_generated":
        generated[(name, "original")]["crops"] = np.zeros((1, 2, 2, 3))
    elif field == "duplicate_box":
        units[name]["t"] = np.array([0, 0])
    elif field == "short_box":
        units[name]["box"] = units[name]["box"][:1]
    elif field == "missing_box":
        units[name]["t"] = np.array([0, 2])
    else:
        units[name]["box"][1] = [20, 20, 8, 8]
    with pytest.raises(ValueError):
        check(data)


def test_empty_region_does_not_require_a_crop():
    data = fixture()
    data[3][1]["near"][:] = False
    data[-1][(data[2], "finetuned_tennis")]["t"] = np.array([0])
    data[-1][(data[2], "finetuned_tennis")]["crops"] = np.ones((1, 2, 2, 3))
    crops, fallback = check(data)
    assert len(crops) == 2 and not fallback
