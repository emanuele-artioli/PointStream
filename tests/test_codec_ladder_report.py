"""Paper output must authenticate and recompute accepted scalar evidence."""
import json
from pathlib import Path

import pytest

from experiments.codec_ladder.report import table_rows, verified_record, write_tables

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'docs/experiments/submission-ladder-tennis-standard-v2-analysis-20261115.json'
SCORE = ROOT / 'docs/experiments/submission-ladder-tennis-standard-v2-scores-20261115.json'


def test_paper_rows_reproduce_accepted_negative_and_standard_definition(tmp_path):
    record = write_tables([(RECORD, SCORE)], tmp_path / 'paper')
    assert len(record['rows']) == 3
    roi, patch, _ = record['rows']
    assert roi['weighted_dists_percent'] == pytest.approx(-8.528273190099035)
    assert patch['weighted_dists_percent'] == pytest.approx(17.096566317763975)
    assert roi['full_lpips_standard_percent'] == pytest.approx(1.1465711035690367)
    assert all(row['development_target_met'] is False for row in record['rows'])
    with pytest.raises(ValueError, match='duplicate source'):
        table_rows([(RECORD, SCORE), (RECORD, SCORE)])


def test_paper_cannot_trust_tampered_scores_or_analysis(tmp_path):
    record = json.loads(RECORD.read_text())
    destination = tmp_path / SCORE.name
    destination.write_bytes(SCORE.read_bytes())
    path = tmp_path / 'accepted.json'
    path.write_text(json.dumps(record))
    destination.write_bytes(destination.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='score hash'):
        verified_record(path, destination)
    destination.write_bytes(SCORE.read_bytes())
    record['decisions']['roi']['native']['development_target_met'] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='analysis differs'):
        verified_record(path, destination)
