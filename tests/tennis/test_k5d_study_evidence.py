import json
from pathlib import Path
from typing import Any

import pytest

from experiments.jobs import inbox
from experiments.tennis import k5d_study_evidence as ev


def write(path: Path, data: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return ev.oracle.sha(path)


def fixture_stage(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, Any]]:
    job = '20261105T000000Z-12345678'
    root = tmp_path / job
    identity = {'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'environment_sha256': 'c' * 64}
    pub, prep = 'd' * 64, 'e' * 64
    spec = {'gpu_models': ['RTX A6000'], 'cpu_threads': 1, 'single_cpu_campaign': True,
            'inputs': [], 'scale': {'frames': {'smoke': 16, 'full': 48}}}
    staged = [{'name': 'prepare16', 'sha256': prep}]
    env = {'sha256': identity['environment_sha256']}
    dispatch = {'code': {'source_sha256': identity['source_sha256'], 'snapshot_sha256': identity['snapshot_sha256'],
                         'spec_sha256': inbox.digest(spec)}, 'staged_environment': env, 'input_identity': [],
                'staged_inputs': staged, 'environment': {'gpu': {'name': 'NVIDIA RTX A6000'}},
                'command': ['python', '-m', 'experiments.tennis.k5d_coalesced_fleet', 'score', '--frames', '16']}
    staging = {'environment': env, 'inputs': staged,
               'published': {'sha256': pub, 'path': f'/dataset/{job}/smoke/published.tar'}}
    execution = {'exit_code': 0, 'seconds': 42, 'timing_contaminated': False}
    documents: dict[str, dict[str, Any]] = {'spec': spec, 'dispatch': dispatch, 'staging': staging, 'execution': execution}
    hashes = {k: write(root / ('spec.json' if k == 'spec' else f'smoke/{k}.json'), v)
              for k, v in documents.items()}
    entry = {'job': job, 'directory': str(root), 'sha256': hashes}
    expected = {'job': job, 'stage': 'smoke', 'published_sha256': pub, 'sha256': hashes.copy()}
    return entry, expected, identity, documents


def test_stage_binds_actual_job_and_source(tmp_path: Path) -> None:
    entry, expected, identity, _ = fixture_stage(tmp_path)
    assert ev.fleet_stage(entry, expected, identity, 'score', 16, 'd' * 64, 'e' * 64) == 42
    for change in ({'job': '20261105T000001Z-12345678'}, {'stage': 'full'}, {'published_sha256': 'f' * 64}):
        with pytest.raises(ValueError):
            ev.fleet_stage(entry, {**expected, **change}, identity, 'score', 16, 'd' * 64, 'e' * 64)
    with pytest.raises(ValueError):
        ev.fleet_stage(entry, expected, {**identity, 'source_sha256': 'f' * 64}, 'score', 16, 'd' * 64, 'e' * 64)
    with pytest.raises(ValueError):
        ev.fleet_stage(entry, expected, identity, 'score', 16, 'd' * 64, 'f' * 64)


@pytest.mark.parametrize('field', ['spec', 'dispatch', 'staging', 'execution'])
def test_rejects_replaced_fleet_record(tmp_path: Path, field: str) -> None:
    entry, expected, identity, _ = fixture_stage(tmp_path)
    path = Path(entry['directory']) / ('spec.json' if field == 'spec' else f'smoke/{field}.json')
    path.write_text('{}')
    with pytest.raises(ValueError):
        ev.fleet_stage(entry, expected, identity, 'score', 16, 'd' * 64, 'e' * 64)


def test_real_archives_reject_changed_point_guard_timing_and_parity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tarfile

    root = tmp_path / 'inputs'
    checkpoint = root / 'checkpoint' / 'publish'
    score = root / 'score' / 'publish'
    reference = root / 'reference' / 'publish'
    for path in (checkpoint, score, reference):
        path.mkdir(parents=True)
    identity = {'source_sha256': 'a' * 64, 'snapshot_sha256': 'b' * 64, 'environment_sha256': 'c' * 64}
    checkpoint_sha = write(checkpoint / 'checkpoint.json', {'execution_identity': identity})
    report = {'points': [{'streak': {'guard_off': False}} for _ in ev.enc.POINTS]}
    write(score / 'assembled' / 'pilot.json', report)
    write(score / 'assembled' / 'source-qualification.json', {'rows': [{'label_kind': 1}] * 16})
    write(reference / 'pilot.json', {'reference': True})
    point_records = []
    for codec, setting in ev.enc.POINTS:
        name = f'point-{codec}-{setting}.json'
        point_records.append({'name': name, 'sha256': write(score / name, {'codec': codec, 'point': setting})})
    write(score / 'coalesced-validation.json', {'checkpoint_sha256': checkpoint_sha, 'execution_identity': identity,
                                                'points': point_records, 'result': {'passed': True}})
    events = [{'name': f'{kind}.{codec}.{setting}.{arm}', 'completed': True}
              for codec, setting in ev.enc.POINTS for kind in ('metrics', 'judges') for arm in ('baseline', 'streak')]
    timing_sha = write(score / 'timing.json', {'events': events})
    write(score / 'comparison.json', {'passed': True, 'frames': 16})

    def archive(name: str) -> tuple[Path, str]:
        target = tmp_path / f'{name}.tar'
        with tarfile.open(target, 'w') as bundle:
            bundle.add(root / name / 'publish', arcname='publish')
        return target, ev.oracle.sha(target)

    reference_path, reference_sha = archive('reference')
    checkpoint_path, checkpoint_archive_sha = archive('checkpoint')
    monkeypatch.setattr(ev.cf, 'D8_SMOKE_SHA', reference_sha)
    monkeypatch.setattr(ev.cp, 'verify', lambda *a: None)

    def assemble(check: Path, sha: str, point_files: list[dict[str, str]], output: Path,
                 current_identity: dict[str, str]) -> dict[str, bool]:
        assert len(point_files) == 6
        write(output / 'pilot.json', report)
        return {'passed': True}

    monkeypatch.setattr(ev.cp, 'assemble', assemble)
    monkeypatch.setattr(ev.cp, 'compare_equivalence', lambda *a: {'passed': True, 'frames': 16})

    def check(expected_timing: str = timing_sha) -> None:
        score_path, score_sha = archive('score')
        ev.parity(score_path, score_sha, checkpoint_path, checkpoint_archive_sha, reference_path,
                  reference_sha, 16, identity, expected_timing)

    check()
    write(score / point_records[0]['name'], {'modified': True})
    with pytest.raises(ValueError, match='point proof hash'):
        check()
    write(score / point_records[0]['name'], {'codec': ev.enc.POINTS[0][0], 'point': ev.enc.POINTS[0][1]})
    report['points'][0]['streak']['guard_off'] = True
    write(score / 'assembled' / 'pilot.json', report)
    with pytest.raises(ValueError, match='all 12 arms'):
        check()
    report['points'][0]['streak']['guard_off'] = False
    write(score / 'assembled' / 'pilot.json', report)
    write(score / 'assembled' / 'source-qualification.json', {'rows': [{'label_kind': 0}] + [{'label_kind': 1}] * 15})
    with pytest.raises(ValueError, match='every frame visible'):
        check()
    write(score / 'assembled' / 'source-qualification.json', {'rows': [{'label_kind': 1}] * 16})
    events.pop()
    new_timing = write(score / 'timing.json', {'events': events})
    with pytest.raises(ValueError, match='timing identity'):
        check()
    with pytest.raises(ValueError, match='metric/judge'):
        check(new_timing)
    events.append({'name': f'judges.{ev.enc.POINTS[-1][0]}.{ev.enc.POINTS[-1][1]}.streak', 'completed': True})
    new_timing = write(score / 'timing.json', {'events': events})
    write(score / 'comparison.json', {'passed': False, 'frames': 16})
    with pytest.raises(ValueError, match='reference comparison'):
        check(new_timing)
