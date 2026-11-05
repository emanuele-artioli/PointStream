from pathlib import Path
import json

import pytest

from experiments.tennis import k5d_full_study as study


def test_selection_preserves_all_frames_and_labels() -> None:
    clips = study.selection()
    assert len(clips) == 30
    assert sum(x['label_rows'] for x in clips.values()) == 5675
    assert clips['tracknet/game9/Clip5']['label_rows'] == 292
    assert clips['tracknet/game10/Clip12']['label_rows'] == 586


def test_forecast_uses_fixed_and_marginal_and_scales_slowdown() -> None:
    receipt = {'cost': {'score': {'seconds16': 100, 'seconds48': 164, 'margin': 1.5, 'reserve_seconds': 40}}}
    assert study.forecast(receipt, 'score', 586, 100) == 1.5 * (68 + 2 * 586) + 40
    assert study.forecast(receipt, 'score', 586, 120) == pytest.approx(1.5 * (68 + 2 * 586) * 1.2 + 40)
    receipt['cost']['score']['seconds48'] = 400
    with pytest.raises(ValueError, match='negative measured fixed'):
        study.forecast(receipt, 'score', 586, 100)


def test_admission_rejects_missing_or_invalid_receipts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='invalid frozen'):
        study.admission(tmp_path / 'gate', '0' * 64, tmp_path / 'evidence', '0' * 64,
                        'tracknet/game9/Clip5', {})
    with pytest.raises((ValueError, FileNotFoundError)):
        study.admission(tmp_path / 'gate', '0' * 64, tmp_path / 'evidence', '0' * 64,
                        'tracknet/game9/Clip5', {x: 'a' * 64 for x in
                        ('source_sha256', 'snapshot_sha256', 'environment_sha256')})


def test_confirmation_gate_only_admits_registered_confirmations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    identity = {x: 'a' * 64 for x in ('source_sha256', 'snapshot_sha256', 'environment_sha256')}
    evidence_sha = 'b' * 64
    allowed = ['tracknet/game9/Clip5']
    gate = {'schema': 1, 'approved': True, 'purpose': 'full-retrospective-benchmark',
            'selection_sha256': study.fleet.SOURCE_SHA, 'cost_receipt_sha256': evidence_sha,
            'execution_identity': identity, 'gpu_models': ['RTX A6000'],
            'fidelity_only_for_unqualified': True, 'historical_task_failures_preserved': True,
            'release': 'cost-confirmation', 'allowed_clips': allowed,
            'historical_source_evidence_sha256': study.proof.history.HISTORICAL_SHA,
            'historical_task_failures': sorted(study.proof.history.HISTORICAL_FAILURES)}
    evidence = {'schema': 1, 'passed': True, 'execution_identity': identity, 'gpu_model': 'RTX A6000',
                'selection_sha256': study.fleet.SOURCE_SHA, 'all_12_arms_full_crop_cost': True,
                'exact_parity_16_and_48': True}
    monkeypatch.setattr(study, 'checked_json', lambda path, sha: gate if path.name == 'gate' else evidence)
    assert study.admission(tmp_path / 'gate', 'c' * 64, tmp_path / 'evidence', evidence_sha,
                           'tracknet/game9/Clip5', identity)[2] == 292
    allowed.append('tracknet/game10/Clip3')
    with pytest.raises(ValueError, match='not admitted'):
        study.admission(tmp_path / 'gate', 'c' * 64, tmp_path / 'evidence', evidence_sha,
                        'tracknet/game9/Clip5', identity)
    allowed.pop()
    gate['historical_task_failures'] = []
    with pytest.raises(ValueError, match='historical task failure'):
        study.admission(tmp_path / 'gate', 'c' * 64, tmp_path / 'evidence', evidence_sha,
                        'tracknet/game9/Clip5', identity)


def test_checkpoint_accepts_an_arbitrary_registered_clip(tmp_path: Path) -> None:
    import tarfile

    clip = 'tracknet/game10/Clip1'
    frames = study.selection()[clip]['label_rows']
    identity = {name: 'a' * 64 for name in ('source_sha256', 'snapshot_sha256', 'environment_sha256')}
    root = tmp_path / 'publish'
    root.mkdir()
    manifest = root / 'checkpoint.json'
    manifest.write_text(json.dumps({'schema': 1, 'clip': clip, 'frames': frames, 'execution_identity': identity,
                                    'points': [list(p) for p in study.enc.POINTS]}))
    archive = tmp_path / 'checkpoint.tar'
    with tarfile.open(archive, 'w') as bundle:
        bundle.add(root, arcname='publish')
    entry = {'remote': '/home/itec/emanuele/Datasets/pointstream-data/jobs/example/published.tar',
             'archive_sha256': study.oracle.sha(archive), 'checkpoint_sha256': study.oracle.sha(manifest),
             'local': str(archive)}
    assert study.checkpoint(entry, 'prepare_full', clip, frames, identity)['name'] == 'prepare_full'
    with pytest.raises(ValueError, match='scope or identity'):
        study.checkpoint(entry, 'prepare_full', clip, frames - 1, identity)
