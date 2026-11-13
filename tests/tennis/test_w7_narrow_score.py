import json
from pathlib import Path

import pytest

from experiments.tennis import w7_narrow_score as narrow, w7_shared_score as shared


def test_exact_predeclared_roles_and_excludes_previous_point():
    root = Path(__file__).resolve().parents[2] / 'docs/experiments/w7-inputs'
    rows = shared.inventory(root / 'mkv-53-inventory-20261105.json', shared.INVENTORY_SHA)
    rows = shared.bound_rows(rows, json.loads((root / 'shared-score-bindings-20261105.json').read_text()))
    smoke = narrow.selected(rows, 'smoke')
    full = narrow.selected(rows, 'full')
    assert {(r['window'], r['name']) for r in smoke} == narrow.CONTROLS
    assert {(r['window'], r['name']) for r in full} == narrow.NEW
    assert ('med000', 's2_o20_crf70') not in narrow.NEW
    with pytest.raises(ValueError, match='cohort'):
        narrow.selected(rows[:-1], 'full')


def test_forecast_applies_worst_case_to_both_stages_and_budget():
    f = narrow.forecast(narrow.PRIOR_SETUP_LAUNCH, narrow.PRIOR_POINT, [20, 20], 950)
    assert f['smoke'] == pytest.approx(342.2228802710306)
    assert f['full'] == pytest.approx(474.6739895346109)
    assert f['gpu_inclusive'] == pytest.approx(926.8968698056415)
    assert f['cpu_inclusive'] == pytest.approx(1026.8968698056415)
    with pytest.raises(ValueError, match='exceeds'):
        narrow.forecast(narrow.PRIOR_SETUP_LAUNCH, 100, [20, 20], 950)
    with pytest.raises(ValueError, match='remains'):
        narrow.forecast(narrow.PRIOR_SETUP_LAUNCH, narrow.PRIOR_POINT, [20, 20], 490)


def test_actual_stage_dispatch_selects_only_its_fixed_roles(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2] / 'docs/experiments/w7-inputs'
    rows = shared.inventory(root / 'mkv-53-inventory-20261105.json', shared.INVENTORY_SHA)
    seen = []
    monkeypatch.setattr(narrow, 'prepare', lambda stage, scratch: ({}, rows))
    monkeypatch.setattr(shared, 'score', lambda args, selected_rows, inputs, pilot, selection: seen.append((args.action, pilot, selection)))
    narrow.run(tmp_path / 'smoke', tmp_path)
    narrow.run(tmp_path / 'full', tmp_path)
    assert seen == [('narrow5', False, narrow.CONTROLS), ('narrow5', False, narrow.NEW)]


def test_observed_staging_overhead_can_only_increase_forecast():
    baseline = narrow.forecast(narrow.PRIOR_SETUP_LAUNCH, narrow.PRIOR_POINT, [20, 20], 950)
    slower = narrow.forecast(narrow.PRIOR_SETUP_LAUNCH, narrow.PRIOR_POINT, [20, 20], 950, 30)
    assert slower['gpu_inclusive'] == pytest.approx(baseline['gpu_inclusive'] + 20)
    with pytest.raises(ValueError, match='exceeds'):
        narrow.forecast(narrow.PRIOR_SETUP_LAUNCH, narrow.PRIOR_POINT, [20, 20], 950, 60)
