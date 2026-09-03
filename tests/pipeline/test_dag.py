"""The stage DAG is the enabled set, and a disabled stage costs nothing.

Behaviour the paper's ablations need: every lattice corner enumerated from
the contracts builds; a skip is measured, not read off the source. Plausible
misuse fails at build time, not mid-run with an empty draw.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.contracts.capabilities import CONDITION_POSE
from src.contracts.config import LatticeConfig
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import (
    FULL,
    NAMED_CORNERS,
    OPTIONAL_STAGES,
    REQUIRED_STAGES,
    STAGE_CODEC,
    STAGE_DETECTION,
    STAGE_GENERATION,
    STAGE_METRICS,
    STAGE_POSE,
    STAGE_TRANSPORT,
    STAGES,
    StageLattice,
)
from src.pipeline.dag import (
    as_lattice,
    build_dag,
    enabled_stages,
    iter_lattice_corners,
)
from tests.pipeline.clocks import ClockedStage, full_roster


def _corner_id(lattice: StageLattice) -> str:
    return lattice.label()


# --------------------------------------------------------------------------
# Enumeration from the contracts
# --------------------------------------------------------------------------


def test_the_corner_enumerator_covers_the_contract_vocabulary() -> None:
    """A hand-picked subset would let a new optional stage land untested."""
    corners = iter_lattice_corners()
    enabled_sets = {corner.enabled for corner in corners}
    assert StageLattice.all_on().enabled in enabled_sets
    assert StageLattice.all_off().enabled in enabled_sets
    for name, named in NAMED_CORNERS.items():
        assert named.enabled in enabled_sets, name
    for name in OPTIONAL_STAGES:
        assert any(name not in corner.enabled for corner in corners), name
    assert REQUIRED_STAGES <= set(STAGES)


@pytest.mark.parametrize("lattice", iter_lattice_corners(), ids=_corner_id)
def test_every_enumerated_corner_builds_a_runnable_dag(lattice: StageLattice) -> None:
    dag = build_dag(lattice, full_roster())
    assert dag.order == lattice.dag()
    assert set(dag.order) == set(lattice.enabled)
    bag = dag.run({"source": "chunk"})
    for name in lattice.enabled:
        assert name in bag


@pytest.mark.parametrize("lattice", iter_lattice_corners(), ids=_corner_id)
def test_a_disabled_stage_costs_nothing_on_every_corner(lattice: StageLattice) -> None:
    """If detection is off, the detector must not run. Proven by the clock,
    not by reading the code — a skip that still pays makes every ablation
    number meaningless."""
    roster = full_roster()
    dag = build_dag(lattice, roster)
    dag.run({"source": "chunk"})
    for name, clock in roster.items():
        if name in lattice.enabled:
            assert clock.calls == 1, name
            assert clock.cost == 1, name
        else:
            assert clock.calls == 0, name
            assert clock.cost == 0, name
            assert clock.elapsed_ns == 0, name


def test_the_all_off_corner_is_the_required_spine_not_a_special_path() -> None:
    """Graceful degradation to the baseline codec is the enabled set shrinking
    to codec, transport and metrics. Same constructor as every other corner."""
    lattice = StageLattice.all_off()
    dag = build_dag(lattice, full_roster())
    assert dag.order == (STAGE_CODEC, STAGE_TRANSPORT, STAGE_METRICS)
    assert set(dag.order) == set(REQUIRED_STAGES)
    assert dag.lattice == lattice


def test_pruning_detection_does_not_pay_the_detector() -> None:
    """A surcharge large enough that a nominal skip would still dominate."""
    detector = ClockedStage(value="subjects", surcharge=1_000_000)
    roster = full_roster(extra={STAGE_DETECTION: detector})
    pruned = FULL.prune(STAGE_DETECTION)
    assert STAGE_DETECTION not in pruned
    assert STAGE_POSE not in pruned
    assert STAGE_GENERATION not in pruned
    dag = build_dag(pruned, roster)
    dag.run({"source": "chunk"})
    assert detector.calls == 0
    assert detector.cost == 0
    assert roster[STAGE_POSE].calls == 0
    assert roster[STAGE_GENERATION].calls == 0
    assert roster[STAGE_CODEC].calls == 1


def test_on_stage_sees_every_enabled_stage_in_order() -> None:
    lattice = StageLattice.all_off()
    seen: list[str] = []

    def _note(name: str, elapsed: float) -> None:
        assert elapsed >= 0.0
        seen.append(name)

    dag = build_dag(lattice, full_roster())
    dag.run({"source": "chunk"}, on_stage=_note)
    assert seen == list(dag.order)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFERRED D5: src/pipeline/reconstruction/reconstruct.py:96 ships the "
        "branch this test forbids, so C1's 'all-off reduces to the source' "
        "claim passes by shortcut rather than by the architecture degrading. "
        "strict=True: when the generic path is fixed this XPASSes and fails, "
        "which is the signal to delete this marker. Do not delete the test."
    ),
)
def test_pipeline_source_has_no_baseline_routing_branch() -> None:
    """``if baseline:`` is the special case the architecture forbids."""
    root = Path(__file__).resolve().parents[2] / "src" / "pipeline"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            code = line.split("#", 1)[0]
            lowered = code.lower()
            if "if baseline" in lowered or "is_source_passthrough" in code:
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert offenders == []


# --------------------------------------------------------------------------
# DAG shape
# --------------------------------------------------------------------------


def test_the_dag_runs_producers_before_consumers() -> None:
    dag = build_dag(FULL, full_roster())
    position = {name: index for index, name in enumerate(dag.order)}
    assert position[STAGE_DETECTION] < position[STAGE_POSE]
    assert position[STAGE_POSE] < position[STAGE_GENERATION]
    pose = next(node for node in dag.nodes if node.name == STAGE_POSE)
    assert STAGE_DETECTION in pose.predecessors


def test_enabled_stages_reads_the_lattice_not_the_roster() -> None:
    lattice = FULL.prune(STAGE_DETECTION)
    assert enabled_stages(lattice) == lattice.enabled
    assert STAGE_DETECTION not in enabled_stages(lattice)
    assert enabled_stages(LatticeConfig()) == LatticeConfig().to_lattice().enabled


def test_as_lattice_accepts_a_lattice_config() -> None:
    config = LatticeConfig()
    assert as_lattice(config) == config.to_lattice()


# --------------------------------------------------------------------------
# Plausible misuse — fail at build, not at run
# --------------------------------------------------------------------------


def test_generation_without_detection_fails_at_build() -> None:
    """The generator would have nothing to draw. That is a config error."""
    broken = FULL.disable(STAGE_DETECTION)
    with pytest.raises(ConfigValueError, match="subjects") as excinfo:
        build_dag(broken, full_roster())
    assert STAGE_DETECTION in str(excinfo.value)


def test_a_missing_backend_for_an_enabled_stage_fails_at_build() -> None:
    roster = full_roster()
    del roster[STAGE_CODEC]
    with pytest.raises(ConfigValueError, match="pipeline.backends") as excinfo:
        build_dag(FULL, roster)
    message = str(excinfo.value)
    assert STAGE_CODEC in message
    assert "injected" in message


def test_a_non_callable_backend_fails_at_build() -> None:
    roster: dict[str, object] = dict(full_roster())
    roster[STAGE_CODEC] = "not-a-callable"
    with pytest.raises(ConfigValueError, match="not callable"):
        build_dag(StageLattice.all_off(), roster)  # type: ignore[arg-type]


def test_pose_conditioning_with_pose_off_fails_at_build() -> None:
    lattice = FULL.disable(STAGE_POSE)
    with pytest.raises(ConfigValueError, match=STAGE_POSE):
        build_dag(lattice, full_roster(), conditioning=(CONDITION_POSE,))


def test_as_lattice_rejects_an_unknown_source() -> None:
    with pytest.raises(TypeError, match="StageLattice"):
        as_lattice("full")  # type: ignore[arg-type]
