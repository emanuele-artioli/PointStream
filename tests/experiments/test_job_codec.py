"""Synthetic evidence checks; these numbers are fixtures, not codec results."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.jobs import codec, monitor


@pytest.fixture
def policy(tmp_path: Path) -> dict[str, Any]:
    controls = tmp_path / "controls.json"
    monitor.write_json(controls, {"calibrated": True, "null_checked": True, "fixture": True})
    return {
        "codec": "av1",
        "tier": "balanced",
        "sweep": "payload",
        "scenes": [["video", "scene"]],
        "dataset_revision": "synthetic-only",
        "pilot_frames": 2,
        "confirmation_frames": 4,
        "final_frames": 8,
        "qps": [20, 30, 40],
        "qp_min": 10,
        "qp_max": 50,
        "qp_step": 10,
        "jpegs": [80, 60, 40],
        "jpeg_min": 20,
        "jpeg_max": 100,
        "jpeg_step": 20,
        "min_gap_db": 2,
        "max_adjustments": 1,
        "budget_seconds": 1000,
        "pair_timeout_seconds": 10,
        "bands": {"psnr_dB": [0, 60], "coded_bytes": [1, 10000], "seconds": [0, 100]},
        "bounds_basis": "hand-written synthetic fixtures, never citable",
        "controls_evidence": str(controls),
    }


def pair(policy: dict[str, Any], clustered: bool = False) -> dict[str, Any]:
    rows = [
        {
            "rate_value": qp,
            "coded_bytes": 300 - 100 * i,
            "psnr_dB": 40 - (0.5 if clustered else 5) * i,
            "seconds": 1,
        }
        for i, qp in enumerate(policy["qps"])
    ]
    candidate = copy.deepcopy(rows)
    if policy["sweep"] == "payload":
        for i, row in enumerate(candidate):
            row.update(
                rate_value=i,
                residual_rate=policy["qps"][i],
                background_jpeg_quality=policy["jpegs"][i],
            )
    return {
        "anchor_rungs": rows,
        "pointstream_rungs": candidate,
        "failures": [],
        "bound_alarms": [],
        "rungs_excluded_not_a_rate": [],
    }


@pytest.mark.parametrize("sweep", ["qp", "payload"])
def test_spacing_spreads_both_arms_within_bounds(policy: dict[str, Any], sweep: str) -> None:
    policy["sweep"] = sweep
    codec.validate(policy)
    assert codec.assess(pair(policy), policy) == "pass"
    assert codec.assess(pair(policy, clustered=True), policy) == "clustered"
    expanded = codec.widen(policy)
    assert expanded is not None and expanded["qps"] == [10, 30, 50]
    if sweep == "payload":
        assert expanded["jpegs"] == [100, 60, 20]
    assert codec.widen(expanded) is None
    assert policy["qps"] == [20, 30, 40]


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "nan",
        "null",
        "wrong_qp",
        "wrong_jpeg",
        "uncoded",
        "alarm",
        "flat_bytes",
        "nonmonotone",
    ],
)
def test_invalid_evidence_never_promotes(policy: dict[str, Any], damage: str) -> None:
    result = pair(policy)
    rows = result["pointstream_rungs"]
    if damage == "missing":
        rows.pop()
    elif damage in {"nan", "null"}:
        rows[0]["psnr_dB"] = float("nan") if damage == "nan" else None
    elif damage == "wrong_qp":
        rows[0]["residual_rate"] = 21
    elif damage == "wrong_jpeg":
        rows[0]["background_jpeg_quality"] = 81
    elif damage == "uncoded":
        result["rungs_excluded_not_a_rate"] = [rows[0]]
    elif damage == "alarm":
        result["bound_alarms"] = ["outside known range"]
    elif damage == "flat_bytes":
        for row in rows:
            row["coded_bytes"] = 100
    else:
        rows[1]["psnr_dB"] = 50
    assert codec.assess(result, policy) == "attention"


def install_worker(monkeypatch: pytest.MonkeyPatch, mode: str = "pass") -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class Worker:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            self.request = monitor.read_json(Path(argv[-2]))
            self.output = Path(argv[-1])
            calls.append(self.request)

        def wait(self, **kwargs: Any) -> int:
            if mode == "interrupt":
                raise KeyboardInterrupt
            if mode == "missing":
                return 0  # clean exit is insufficient evidence
            clustered = (mode == "widen" and len(calls) == 1) or (
                mode == "confirmation_cluster" and self.request["frames"] == 4
            )
            monitor.write_json(self.output, pair(self.request["policy"], clustered=clustered))
            return 0

    monkeypatch.setattr(
        codec,
        "subprocess",
        SimpleNamespace(
            Popen=Worker,
            check_output=lambda *a, **kw: "fixture-revision",
            STDOUT=-2,
            TimeoutExpired=TimeoutError,
        ),
    )
    monkeypatch.setattr(codec, "stop_child", lambda proc: None)
    return calls


def test_campaign_reprobes_after_widening_then_confirms_and_resumes_without_work(
    tmp_path: Path,
    policy: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_worker(monkeypatch, "widen")
    path, dest = tmp_path / "policy.json", tmp_path / "campaign"
    monitor.write_json(path, policy)
    assert codec.run(path, dest) == 0
    assert [r["frames"] for r in calls] == [2, 2, 4, 8]
    assert calls[1]["policy"]["qps"] == [10, 30, 50]
    state = monitor.read_json(dest / "state.json")
    assert state["citable"] is False
    assert state["decisions"][0]["action"] == "widen"
    assert codec.run(path, dest) == 0
    assert len(calls) == 4
    policy["budget_seconds"] += 1
    monitor.write_json(path, policy)
    with pytest.raises(ValueError, match="policy changed"):
        codec.run(path, dest)


@pytest.mark.parametrize(
    "mode,expected",
    [("missing", [2]), ("confirmation_cluster", [2, 4]), ("budget", []), ("no_adjustments", [2])],
)
def test_stage_failure_stops_before_expensive_work(
    tmp_path: Path,
    policy: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: list[int],
) -> None:
    if mode == "budget":
        policy["budget_seconds"] = 1
    if mode == "no_adjustments":
        policy["max_adjustments"] = 0
    calls = install_worker(monkeypatch, "widen" if mode == "no_adjustments" else mode)
    path, dest = tmp_path / "policy.json", tmp_path / "campaign"
    monitor.write_json(path, policy)
    assert codec.run(path, dest) == 2
    assert [r["frames"] for r in calls] == expected
    assert monitor.read_json(dest / "state.json")["status"] == "attention"


def test_interrupted_pair_is_not_silently_replayed(
    tmp_path: Path,
    policy: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_worker(monkeypatch, "interrupt")
    path, dest = tmp_path / "policy.json", tmp_path / "campaign"
    monitor.write_json(path, policy)
    with pytest.raises(KeyboardInterrupt):
        codec.run(path, dest)
    assert codec.run(path, dest) == 2
    assert len(calls) == 1
    assert "interrupted" in monitor.read_json(dest / "state.json")["reason"]


def test_payload_adapter_applies_custom_rungs_in_coarseness_order(
    policy: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from experiments.tier import ladder
    from experiments.tier.clip import TierClip
    from src.runner.config_io import load_tier

    seen = []

    def anchor(clip: Any, request: Any) -> Any:
        qp = request.rate
        return ladder.Rung(qp, 500 - 10 * qp, 50 - qp / 2, 1, {})

    def payload(clip: Any, config: Any, *, jpeg_quality: int, rate_value: int, rank: int) -> Any:
        seen.append((jpeg_quality, rate_value, rank))
        return ladder.Rung(
            rank,
            500 - 10 * rate_value,
            50 - rate_value / 2,
            1,
            {
                "is_rate": True,
                "residual_rate": rate_value,
                "background_jpeg_quality": jpeg_quality,
            },
        )

    monkeypatch.setattr(ladder, "anchor_rung", anchor)
    monkeypatch.setattr(ladder, "payload_rung", payload)
    result = ladder.pair_for_codec(
        TierClip(
            video="fixture",
            scene="fixture",
            frame_ids=(0, 1),
            frames=np.zeros((2, 32, 32, 3), dtype=np.uint8),
            objects=(),
            union_mask=np.zeros((2, 32, 32), dtype=bool),
            paste_back_mae=0,
            n_tracks=0,
        ),
        load_tier("balanced"),
        codec_name="av1",
        rungs=tuple(policy["qps"]),
        sweep="payload",
        payload_rungs=tuple(zip(policy["jpegs"], policy["qps"])),
    )
    assert seen == [(80, 20, 0), (60, 30, 1), (40, 40, 2)]
    assert codec.assess(result, policy) == "pass"


def test_spent_budget_blocks_final_stage(
    tmp_path: Path,
    policy: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = install_worker(monkeypatch)
    policy["budget_seconds"] = 20
    times = iter([0, 6, 6, 12])
    monkeypatch.setattr(codec.time, "monotonic", lambda: next(times))
    path, dest = tmp_path / "policy.json", tmp_path / "campaign"
    monitor.write_json(path, policy)
    assert codec.run(path, dest) == 2
    assert [r["frames"] for r in calls] == [2, 4]
    assert monitor.read_json(dest / "state.json")["spent_seconds"] == 12
