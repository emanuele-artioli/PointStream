from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from experiments.tennis import k5d_study_wave as wave


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = {key: 'a' * 64 for key in ('source_sha256', 'snapshot_sha256', 'environment_sha256')}
    rows = [{'clip': clip, 'frames': data['label_rows'], 'prepare': {'state': 'pending'},
             'score': {'state': 'pending'}} for clip, data in wave.study.selection().items()]
    ledger = {'schema': 1, 'selection_sha256': wave.study.fleet.SOURCE_SHA,
              'execution_identity': identity, 'gpu_model': 'RTX A6000', 'clips': rows}
    now = datetime.now(timezone.utc)
    inspection = {'schema': 1, 'gpu_models': ['RTX A6000'], 'inspected_hosts': [f'gpu{i}' for i in range(1, 7)],
                  'observed_at': (now - timedelta(seconds=1)).isoformat(),
                  'valid_until': (now + timedelta(minutes=2)).isoformat(), 'active_or_uncertain_jobs': []}
    evidence = {'cost': {phase: {'seconds16': 100, 'seconds48': 164, 'margin': 1.5, 'reserve_seconds': 20}
                         for phase in ('prepare', 'score')}}
    monkeypatch.setattr(wave, 'read', lambda path, sha: ledger if path.name == 'ledger' else inspection)
    monkeypatch.setattr(wave.study, 'admission', lambda gate, gs, ep, es, clip, ident:
                        ({}, evidence, wave.study.selection()[clip]['label_rows']))
    monkeypatch.setattr(wave, 'verify_completed', lambda *a: None)
    kwargs: dict[str, Any] = {name + '_path': tmp_path / name for name in ('ledger', 'gate', 'evidence', 'inspection')}
    kwargs.update({name + '_sha': 'b' * 64 for name in ('ledger', 'gate', 'evidence', 'inspection')})
    kwargs.update(identity=identity, gpu_budget_seconds=3000, cpu_budget_seconds=3000, now=now)
    return kwargs, ledger, inspection


def test_wave_requires_process_tree_cpu_proof(setup: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
    kwargs, _, _ = setup
    with pytest.raises(ValueError, match='process-tree CPU'):
        wave.propose(**kwargs)


def test_stop_on_uncertain_job_missing_host_or_budget(setup: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
    kwargs, ledger, inspection = setup
    ledger['clips'][0]['prepare'] = {'state': 'unknown', 'job_id': '20261105T000000Z-12345678'}
    with pytest.raises(ValueError, match='no replay'):
        wave.propose(**kwargs)
    ledger['clips'][0]['prepare'] = {'state': 'pending'}
    inspection['inspected_hosts'].pop()
    with pytest.raises(ValueError, match='all-host'):
        wave.propose(**kwargs)
    inspection['inspected_hosts'].append('gpu6')
    kwargs['cpu_budget_seconds'] = 100
    with pytest.raises(ValueError, match='process-tree CPU'):
        wave.propose(**kwargs)


def test_cannot_omit_historical_failures(setup: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> None:
    kwargs, ledger, _ = setup
    ledger['clips'] = [row for row in ledger['clips'] if row['clip'] != 'tracknet/game10/Clip11']
    with pytest.raises(ValueError, match='30-clip'):
        wave.propose(**kwargs)


def test_completed_receipt_binds_publications_and_fleet_records(tmp_path: Path) -> None:
    import json
    from experiments.jobs import inbox

    job = '20261105T000000Z-12345678'
    root = tmp_path / job
    root.mkdir()
    identity = {key: 'a' * 64 for key in ('source_sha256', 'snapshot_sha256', 'environment_sha256')}
    spec = {'gpu_models': ['RTX A6000'], 'entrypoint': ['-m', 'experiments.tennis.k5d_full_study'], 'inputs': []}

    def write(path: Path, value: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return wave.oracle.sha(path)

    records: dict[str, Any] = {'spec_sha256': write(root / 'spec.json', spec)}
    publications = {}
    for stage, frames in (('smoke', 16), ('full', 292)):
        artifact = tmp_path / (stage + '.tar')
        artifact.write_bytes(stage.encode())
        sha = wave.oracle.sha(artifact)
        publications[stage] = {'local_path': str(artifact), 'sha256': sha, 'checkpoint_sha256': 'b' * 64}
        dispatch = {'code': {'spec_sha256': inbox.digest(spec), 'source_sha256': identity['source_sha256'],
                             'snapshot_sha256': identity['snapshot_sha256']},
                    'staged_environment': {'sha256': identity['environment_sha256']},
                    'staged_inputs': [], 'input_identity': [],
                    'environment': {'gpu': {'name': 'NVIDIA RTX A6000'}},
                    'command': ['python', '-m', 'experiments.tennis.k5d_full_study', 'prepare',
                                '--clip', 'tracknet/game9/Clip5', '--frames', str(frames)]}
        staging = {'inputs': [], 'published': {'sha256': sha, 'path': f'/remote/{job}/{stage}/published.tar'}}
        records[stage] = {'dispatch_sha256': write(root / stage / 'dispatch.json', dispatch),
                          'staging_sha256': write(root / stage / 'staging.json', staging),
                          'execution_sha256': write(root / stage / 'execution.json', {'exit_code': 0})}
    receipt = {'job_id': job, 'state': 'succeeded', 'clip': 'tracknet/game9/Clip5', 'phase': 'prepare',
               'frames': 292, 'execution_identity': identity, 'gpu_model': 'RTX A6000',
               'job_directory': str(root), 'records': records, 'publications': publications}
    receipt_path = tmp_path / 'receipt.json'
    entry = {'job_id': job, 'receipt_path': str(receipt_path), 'receipt_sha256': write(receipt_path, receipt),
             'publications': publications}
    import tarfile
    with pytest.raises(tarfile.ReadError):
        wave.verify_completed('tracknet/game9/Clip5', 'prepare', entry, 292, identity)
    Path(publications['full']['local_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='publication SHA'):
        wave.verify_completed('tracknet/game9/Clip5', 'prepare', entry, 292, identity)


def test_wave_rejects_understated_margin(setup: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs, _, _ = setup
    evidence = {'cost': {'prepare': {'seconds16': 100, 'seconds48': 164, 'margin': 1.49, 'reserve_seconds': 20}}}
    monkeypatch.setattr(wave.study, 'admission', lambda gate, gs, ep, es, clip, ident:
                        ({}, evidence, wave.study.selection()[clip]['label_rows']))
    with pytest.raises(ValueError, match='margin'):
        wave.propose(**kwargs)


def test_score_rejects_wrong_producer_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager

    @contextmanager
    def unpack(*args: Any) -> Any:
        yield tmp_path

    monkeypatch.setattr(wave.cost, 'unpack', unpack)
    producer = {'publications': {'full': {'sha256': 'a' * 64, 'checkpoint_sha256': 'b' * 64}}}
    dispatch = {'command': ['experiments.tennis.k5d_control_study', '--checkpoint-sha256', 'c' * 64],
                'staged_inputs': [{'sha256': 'a' * 64}]}
    with pytest.raises(ValueError, match='not bound'):
        wave.validate_full('clip', 'score', 10, {}, {'local_path': 'unused', 'sha256': 'd' * 64}, producer, dispatch)
