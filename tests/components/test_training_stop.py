"""Stop rule for ControlNet training: the task metric, not the diffusion loss.

A deliberately hopeless series must stop by epoch 3–4. A slow starter that is
still moving is not killed. Training loss is not an input, so it cannot gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.components.metrics.lpips import LpipsMetric
from src.shared.training.stop import (
    DEFAULT_FLOOR_LPIPS,
    DEFAULT_FLOOR_PSNR,
    DEFAULT_NULL_LPIPS,
    StopBounds,
    TaskStopRule,
    write_bounds,
)
from src.shared.training.task_eval import (
    bbox_from_mask,
    eval_snapshot_is_ephemeral,
    eval_snapshot_tag,
    mean_scores,
    score_item,
    static_copy_scores,
)


def _bounds() -> StopBounds:
    return StopBounds(
        floor_psnr=DEFAULT_FLOOR_PSNR,
        floor_lpips=DEFAULT_FLOOR_LPIPS,
        null_lpips=DEFAULT_NULL_LPIPS,
        source="test fixture; not a measured floor",
    )


def test_bounds_are_written_before_any_observe(tmp_path: Path) -> None:
    rule = TaskStopRule(_bounds(), tmp_path)
    payload = json.loads((tmp_path / "bounds.json").read_text())
    assert payload["written_before_training"] is True
    assert payload["do_not_stop_on_diffusion_loss"] is True
    assert payload["ranking_key"] == "lpips"
    assert payload["floor_lpips"] == DEFAULT_FLOOR_LPIPS
    assert rule.history == []


def test_write_bounds_refuses_a_floor_above_the_null() -> None:
    with pytest.raises(ValueError, match="below the unrelated-image null"):
        StopBounds(floor_psnr=13.51, floor_lpips=0.80, null_lpips=0.73)


def test_never_stops_before_epoch_3(tmp_path: Path) -> None:
    rule = TaskStopRule(_bounds(), tmp_path)
    for epoch in (1, 2):
        decision = rule.observe(epoch=epoch, lpips=0.70, psnr=11.0)
        assert decision.stop is False, f"stopped at epoch {epoch}: {decision.reason}"


def test_hopeless_flat_series_stops_by_epoch_3_or_4(tmp_path: Path) -> None:
    """The pose-ref shape: below the floor, flat, from epoch 1."""
    rule = TaskStopRule(_bounds(), tmp_path)
    stopped_at: int | None = None
    for epoch in range(1, 6):
        decision = rule.observe(epoch=epoch, lpips=0.70, psnr=11.18)
        if decision.stop:
            stopped_at = epoch
            break
    assert stopped_at in {3, 4}, f"stopped at {stopped_at}, reason={rule.stop_reason}"
    series = json.loads((tmp_path / "stop_series.json").read_text())
    assert series["stopped"] is True
    assert series["not_citable"] is True
    assert "floor" in series["stop_reason"] or "patience" in series["stop_reason"]
    assert all(row["lpips"] == 0.70 for row in series["series"])


def test_slow_starter_that_improves_is_not_killed_at_epoch_3(tmp_path: Path) -> None:
    rule = TaskStopRule(_bounds(), tmp_path)
    trajectory = {1: 0.70, 2: 0.62, 3: 0.54}
    for epoch, lpips in trajectory.items():
        decision = rule.observe(epoch=epoch, lpips=lpips, psnr=12.0)
        assert decision.stop is False, decision.reason
    assert rule.best_epoch == 3
    assert rule.best_lpips == pytest.approx(0.54)


def test_best_checkpoint_is_the_lowest_lpips_not_the_last(tmp_path: Path) -> None:
    rule = TaskStopRule(_bounds(), tmp_path)
    first = rule.observe(epoch=1, lpips=0.48, psnr=13.0)
    second = rule.observe(epoch=2, lpips=0.55, psnr=12.0)
    third = rule.observe(epoch=3, lpips=0.52, psnr=12.5)
    assert first.keep_as_best is True
    assert second.keep_as_best is False
    assert third.keep_as_best is False
    assert rule.best_epoch == 1
    assert rule.best_lpips == pytest.approx(0.48)


def test_patience_stops_when_improvement_stalls_after_min_epochs(tmp_path: Path) -> None:
    bounds = StopBounds(
        floor_psnr=13.51,
        floor_lpips=0.4505,
        null_lpips=0.7358,
        min_epochs=3,
        patience=3,
        source="patience fixture",
    )
    rule = TaskStopRule(bounds, tmp_path)
    scores = {1: 0.40, 2: 0.41, 3: 0.42, 4: 0.43}
    stopped_at: int | None = None
    for epoch, lpips in scores.items():
        decision = rule.observe(epoch=epoch, lpips=lpips, psnr=14.0)
        if decision.stop:
            stopped_at = epoch
            break
    assert stopped_at == 4
    assert "patience" in rule.stop_reason


def test_observe_does_not_accept_training_loss() -> None:
    """A falling diffusion loss is not in the API, so it cannot gate."""
    import inspect

    assert "loss" not in inspect.signature(TaskStopRule.observe).parameters


def test_observe_rejects_an_unknown_kind(tmp_path: Path) -> None:
    rule = TaskStopRule(_bounds(), tmp_path)
    with pytest.raises(ValueError, match="kind must be"):
        rule.observe(epoch=1, lpips=0.7, psnr=11.0, kind="loss")


def test_mid_epoch_eval_cannot_stop_before_min_epochs(tmp_path: Path) -> None:
    rule = TaskStopRule(_bounds(), tmp_path)
    for step in (100, 200, 300, 400):
        decision = rule.observe(epoch=1, lpips=0.70, psnr=11.0, step=step, kind="mid")
        assert decision.stop is False


def test_static_copy_is_the_floor_on_identical_crops() -> None:
    image = np.zeros((16, 20, 3), dtype=np.uint8)
    image[2:14, 3:17] = 80
    mask = np.zeros((16, 20), dtype=bool)
    mask[2:14, 3:17] = True

    def extractor(reference: np.ndarray, predicted: np.ndarray) -> float:
        return float(
            np.mean(np.abs(reference.astype(np.float32) - predicted.astype(np.float32)))
        )

    lpips = LpipsMetric(extractor=extractor)
    item = static_copy_scores(image, image, mask, key="identical", lpips=lpips)
    assert item.lpips == pytest.approx(0.0)
    assert item.psnr == pytest.approx(float("inf"))


def test_score_item_uses_mask_for_psnr_and_bbox_for_lpips() -> None:
    target = np.zeros((8, 8, 3), dtype=np.uint8)
    target[2:6, 2:6] = 100
    predicted = target.copy()
    predicted[0, 0] = 255
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    seen: list[tuple[int, ...]] = []

    def extractor(reference: np.ndarray, predicted: np.ndarray) -> float:
        seen.append(reference.shape)
        return 0.1

    lpips_value, psnr, n_pixels = score_item(
        target, predicted, mask, lpips=LpipsMetric(extractor=extractor)
    )
    assert n_pixels == 16
    assert psnr == pytest.approx(float("inf"))
    assert seen[0][1:3] == (4, 4)
    assert lpips_value == pytest.approx(0.1)
    assert bbox_from_mask(mask) == (2, 6, 2, 6)


def test_mean_scores_refuses_an_empty_list() -> None:
    with pytest.raises(ValueError, match="zero scored items"):
        mean_scores([])


def test_write_bounds_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "bounds.json"
    write_bounds(path, _bounds(), extra={"n_eval_clips": 4, "offset": 8})
    payload = json.loads(path.read_text())
    assert payload["n_eval_clips"] == 4
    assert payload["offset"] == 8
    assert payload["written_before_training"] is True


def test_mid_epoch_evals_in_one_epoch_get_distinct_snapshot_tags() -> None:
    first = eval_snapshot_tag(1, kind="mid", step=2000)
    second = eval_snapshot_tag(1, kind="mid", step=4000)
    epoch_end = eval_snapshot_tag(1, kind="epoch")
    assert first != second
    assert first != epoch_end
    assert second != epoch_end
    assert first.startswith("checkpoint-epoch-")
    assert second.startswith("checkpoint-epoch-")
    assert epoch_end == "checkpoint-epoch-1"


def test_mid_epoch_snapshot_is_ephemeral_epoch_snapshot_is_kept() -> None:
    assert eval_snapshot_is_ephemeral("mid") is True
    assert eval_snapshot_is_ephemeral("epoch") is False


def test_mid_epoch_snapshot_tag_requires_a_step() -> None:
    with pytest.raises(ValueError, match="requires step"):
        eval_snapshot_tag(1, kind="mid")


def test_eval_snapshot_tag_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind must be"):
        eval_snapshot_tag(1, kind="loss", step=1)


def test_train_controlnet_always_writes_eval_weights() -> None:
    """The stale-weight bug was skipping _save when checkpoint-epoch-N existed."""
    source = Path("scripts/train_controlnet.py").read_text()
    assert "if not Path(ckpt).is_dir()" not in source
    assert "eval_snapshot_tag(" in source
    assert "_save(tag)" in source
