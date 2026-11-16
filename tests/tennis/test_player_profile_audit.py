import copy

import pytest

from experiments.tennis.player_profile_audit import validate_join


def facts():
    h, a, n, c = (x * 64 for x in "abcd")
    return {
        "prepared_sha256": h, "judge_prepared_sha256": h,
        "judge_generated_sha256": [a],
        "windows": [{"id": "m/w", "judged": [0, 3], "units": ["u"], "profiles": ["match"], "adapted_profiles": ["match"], "match": "m"}],
        "generated": [{"unit": "u", "profile": "match", "archive_sha256": a, "npz_sha256": n,
                       "receipt_npz_sha256": n, "source": "job/publish/u@match.npz", "ticks": [0, 3]}],
        "mask_checks": {"u": {"0": True, "3": True}},
        "fits": [{"source": "fit/job", "unit": "u", "profile": "match", "match": "m",
                  "generation_checkpoint_sha256": c, "fit_checkpoint_sha256": c}],
    }


def test_complete_join():
    assert validate_join(facts())["judged_unit_profile_ticks"] == 2


@pytest.mark.parametrize("mutation", [
    lambda d: d["windows"][0].update(units=[]),
    lambda d: d["fits"].clear(),
    lambda d: d["generated"][0].update(ticks=[0]),
    lambda d: d["generated"][0].update(profile="original"),
    lambda d: d["generated"][0].update(receipt_npz_sha256="f" * 64),
    lambda d: d["mask_checks"]["u"].pop("3"),
    lambda d: d["fits"][0].update(match="other"),
    lambda d: d["fits"][0].update(fit_checkpoint_sha256="f" * 64),
    lambda d: d["fits"][0].update(generation_base_sha256="a" * 64),
    lambda d: d["generated"].append(copy.deepcopy(d["generated"][0])),
])
def test_rejects_incomplete_or_misattributed_evidence(mutation):
    data = facts()
    mutation(data)
    with pytest.raises(ValueError):
        validate_join(data)
