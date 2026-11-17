"""Cost admission must use measured worker and native campaign clocks."""
import pytest

from experiments.tennis.player_requalification import reference_smoke_cost_gate


def test_reference_cost_gate_uses_both_caps() -> None:
    receipt = {"load_seconds": 24.0}
    timing = {"worker_wall_seconds": 100.0}
    gate = reference_smoke_cost_gate(receipt, timing, {"campaign_clock": {"budget_end": 600}}, 100)
    assert gate["forecast_seconds"] == 310
    with pytest.raises(ValueError, match="forecast"):
        reference_smoke_cost_gate(receipt, timing, {"campaign_clock": {"budget_end": 469}}, 100)
    with pytest.raises(ValueError, match="forecast"):
        reference_smoke_cost_gate(receipt, {"worker_wall_seconds": 200}, {"campaign_clock": {"budget_end": 1000}}, 100)
    with pytest.raises(ValueError, match="shorter"):
        reference_smoke_cost_gate(receipt, {"worker_wall_seconds": 20}, {"campaign_clock": {"budget_end": 1000}}, 100)


def test_full_audit_rejects_changed_repeat(tmp_path, monkeypatch) -> None:
    import numpy as np

    from experiments.tennis import player_reference_review as review

    smoke, full = tmp_path / "smoke", tmp_path / "full"
    smoke.mkdir()
    full.mkdir()
    for folder in (smoke, full):
        (folder / "receipt.json").write_text("{}")
        np.savez(folder / f"{review.UNITS[0]}@tennis_only.npz", t=[0, 1], box=[1], crops=[2, 3])
    monkeypatch.setattr(review, "resolved", lambda *args: {"reference_pilot": {"quality": 70},
        "units": [{"unit": u} for u in review.UNITS]})
    monkeypatch.setattr(review, "validate_output", lambda *args: {"plan": {"units": [{"unit": review.UNITS[0]}]}})
    args = (tmp_path / "config", "hash", tmp_path, tmp_path, smoke, full)
    assert review.audit(*args)["input_identities_and_output_arrays_bit_exact"]
    np.savez(full / f"{review.UNITS[0]}@tennis_only.npz", t=[0, 1], box=[1], crops=[2, 4])
    with pytest.raises(ValueError, match="differ"):
        review.audit(*args)


@pytest.mark.parametrize("qualities", [(70,), (70, 40)])
def test_review_hides_variants_and_preserves_labels(tmp_path, monkeypatch, qualities) -> None:
    import json

    import numpy as np

    from experiments.tennis import player_reference_review as review

    units = review.UNITS
    entries = [{"unit": u, "run": 0, "seed": i, "t": [0], "pose_sha256": "p", "box_sha256": "b"} for i, u in enumerate(units)]
    plan = {"units": entries, "source": {"window": "s"}, "settings": {"steps": 30}}
    configs = {}
    variants = []
    for name in ("lossless", *(str(q) for q in qualities)):
        folder = tmp_path / name
        folder.mkdir()
        config_path = folder / "config.json"
        config_path.write_text("{}")
        (folder / "receipt.json").write_text(json.dumps({"plan": plan}))
        inputs = []
        for unit in units:
            p = folder / f"{unit}.npz"
            np.savez(p, reference=np.zeros((2, 2, 3), dtype=np.uint8), pose=[1])
            np.savez(folder / f"{unit}@tennis_only.npz", crops=np.zeros((24, 2, 2, 3), dtype=np.uint8))
            inputs.append({"unit": unit, "path": str(p)})
        configs[str(config_path)] = {"units": inputs, "reference_pilot": {"units": [{"encoded_ref1_bytes": 1}]}}
        if name != "lossless":
            variants.append((config_path, name, folder, folder, folder))
    monkeypatch.setattr(review, "resolved", lambda config, *args: configs[str(config)])
    monkeypatch.setattr(review, "validate_output", lambda *args: {"plan": plan})
    monkeypatch.setattr(review, "audit", lambda config, digest, *args: {"quality": int(digest)})
    monkeypatch.setattr(review, "source_crops", lambda *args: ([np.zeros((24, 2, 2, 3), dtype=np.uint8)] * 2, [0, 0]))
    monkeypatch.setattr(review, "encode", lambda crops, path, fps: path.write_bytes(b"media"))
    from experiments.tennis import player_requalification
    monkeypatch.setattr(player_requalification, "make_plan", lambda *args: (plan, [{}, {}]))
    source = tmp_path / "source"
    source.write_bytes(b"source")
    output = tmp_path / "review"
    control = tmp_path / "lossless"
    review.build(control / "config.json", "hash", control, source, control, variants, source, output, 42)
    page = (output / "review.html").read_text()
    assert "Player 1" in page and "Player 2" in page
    assert "quality-70" not in page and "quality-40" not in page and "366161" not in page
    key = json.loads((output / "reveal.json").read_text())
    assert key["completed_qualities"] == list(qualities)
    assert key["complete_candidate_family"] == (len(qualities) == 2)
    for unit in units:
        assert set(key["mappings"][unit].values()) == {"lossless", *(f"quality-{q}" for q in qualities)}
    assert ("Partial candidate family" in page) == (len(qualities) == 1)
    assert f"repeat({len(qualities) + 2},1fr)" in page
    assert "Slowed to 10 fps" in page


@pytest.mark.parametrize("count", [0, 3])
def test_review_rejects_unsupported_candidate_count(tmp_path, count) -> None:
    from experiments.tennis import player_reference_review as review

    variants = [(tmp_path, "hash", tmp_path, tmp_path, tmp_path)] * count
    with pytest.raises(ValueError, match="one or two completed"):
        review.build(tmp_path, "hash", tmp_path, tmp_path, tmp_path,
                     variants, tmp_path, tmp_path / "output", 42)
