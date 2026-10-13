import numpy as np
import pytest

from experiments.visor import h1
from experiments.visor import h4b

SHAPE = (120, 160)


def texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (SHAPE[0] // 8, SHAPE[1] // 8, 3), dtype=np.uint8)
    return np.kron(small, np.ones((8, 8, 1), np.uint8))


def scene(looks: list[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """A still object whose look alternates per frame (looks[t] picks a texture): a turned-back object."""
    frames = np.stack([texture(s) for s in looks])
    mask = np.zeros(SHAPE, bool)
    mask[30:90, 40:120] = True
    return frames, {t: mask for t in range(len(looks))}


def run(coder: h4b.Coder, frames: np.ndarray, masks: dict[int, np.ndarray]) -> h4b.Coder:
    grays = [f.mean(2).astype(np.uint8) for f in frames]
    track = h4b.TrackScene(frames, grays, masks)
    for t in sorted(masks):
        if not coder.views:
            coder._add(t, None)
        else:
            coder.step(t, np.eye(3) if t - 1 in masks else None, track)
    return coder


def test_a_bank_reuses_views_of_an_object_turned_back_and_forth():
    frames, masks = scene([1, 2, 1, 2, 1, 2, 1, 2])
    single = run(h4b.Coder("single", 30.0, 1, False), frames, masks)
    bank = run(h4b.Coder("bank", 30.0, None, False), frames, masks)
    assert single.held == 0 and len(single.new_views) == 8
    assert bank.new_views == [0, 1] and bank.held == 6
    four = run(h4b.Coder("bank-4", 30.0, 4, False), frames, masks)
    assert four.new_views == [0, 1]
    for c in (single, bank, four):
        assert c.held + len(c.new_views) == c.frames == 8
    s = bank.summary(7, 50.0)
    assert s["intervals_s"] == [0.02, 0.14] and s["reached"]["chained"] == 6


def test_capacity_evicts_the_least_recently_used_view():
    frames, masks = scene([1, 2, 3, 1])  # three looks, then the first again
    two = run(h4b.Coder("bank-2", 30.0, 2, False), frames, masks)
    assert two.new_views == [0, 1, 2, 3]  # view 0 was evicted when 2 arrived
    three = run(h4b.Coder("bank-3", 30.0, 3, False), frames, masks)
    assert three.new_views == [0, 1, 2] and three.held == 1


def test_direct_registration_recovers_a_view_after_the_chain_breaks():
    frames, masks = scene([1, 1, 2, 1])
    masks = dict(masks)
    del masks[2]  # the object is absent at frame 2: every chain breaks
    frames = frames.copy()
    chained = run(h4b.Coder("bank", 30.0, None, False), frames, masks)
    assert chained.new_views == [0, 3]
    direct = run(h4b.Coder("bank", 30.0, None, True), frames, masks)
    assert direct.new_views == [0] and direct.reached["direct"] == 1


def test_single_without_registration_is_h1s_reference_chain():
    rng = np.random.default_rng(7)
    looks = list(rng.integers(1, 3, 20))
    frames, masks = scene(looks)
    masks = {t: m for t, m in masks.items() if t != 9}
    chain = h1.ReferenceChain(30.0, 50.0)
    for t in sorted(masks):
        if chain.reference is None or t - 1 not in masks:
            chain.start(t)
        else:
            chain.step(t, np.eye(3), frames, lambda r: masks[r], masks[t])
    mine = run(h4b.Coder("single", 30.0, 1, False), frames, masks)
    assert mine.new_views == chain.refreshes and mine.held == chain.held
    assert mine.summary(max(masks), 50.0)["intervals_s"] == chain.summary(max(masks))["reference_life_s"]


def row(handled_views: list[int], frames: int = 100, held: int | None = None) -> dict:
    coders = {}
    for th in h1.HOLD_PSNR:
        for name in h4b.CODERS:
            views = handled_views if name.startswith("bank") else list(range(0, frames, 2))
            coders[f"{name}@{th:g}"] = {"frames": frames, "held": frames - len(views), "new_views": views,
                                        "intervals_s": [(b - a) / 50 for a, b in zip(views, views[1:] + [frames])],
                                        "reached": {"chained": frames - len(views), "direct": 0}}
    return {"id": "x", "frames": frames, "fps": 50.0,
            "tracks": {"pan": {"kind": "handled_object", "frames": frames, "coders": coders}}}


def test_decision_passes_only_when_the_bank_holds_long_and_beats_single_plus_direct():
    good = h4b.summarize([row([0, 40, 80])], "handled_object")
    assert good["bank@30"]["held_share"] == pytest.approx(0.97)
    assert good["bank_over_single_direct@30"] == pytest.approx(3 / 50)
    assert h4b.verdict(good)["verdict"] == "passes"
    bad = h4b.summarize([row(list(range(0, 100, 3)))], "handled_object")
    assert h4b.verdict(bad)["verdict"] == "fails"


def test_h1_agreement_counts_exact_and_close_tracks():
    r = row([0, 40, 80])
    chains = {"items": {"x": {"pan": {"30": {"held": 50, "references": 50}, "33.75": {"held": 49, "references": 51}}}}}
    a = h4b.h1_agreement([r], chains)
    assert a["track_thresholds"] == 2 and a["exact_share"] == 0.5 and a["within_2pct_share"] == 1.0
