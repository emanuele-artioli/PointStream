import hashlib
import io
import json
import tarfile

import pytest

from experiments.jobs.inbox import validate_spec
from experiments.tennis import k5d_checkpoint_fleet as fleet


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archive(path, member, value):
    raw = json.dumps(value, sort_keys=True).encode()
    with tarfile.open(path, 'w') as tar:
        info = tarfile.TarInfo(member)
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    return sha(raw), fleet.oracle.sha(path)


def preparation(tmp_path, frames):
    path = tmp_path / f'prepare{frames}.tar'
    checkpoint_sha, archive_sha = archive(path, 'publish/checkpoint.json',
                                           {'schema': 1, 'clip': fleet.cost.CLIP, 'frames': frames})
    return {'local': str(path), 'remote': str(fleet.DATA_ROOT / f'prepare{frames}.tar'),
            'archive_sha256': archive_sha, 'checkpoint_sha256': checkpoint_sha}


def test_score_builder_requires_actual_16_and_48_checkpoints(tmp_path, monkeypatch):
    base = json.loads(fleet.BASE.read_text())
    small, full = preparation(tmp_path, 16), preparation(tmp_path, 48)
    ref = tmp_path / 'reference.tar'
    ref.write_bytes(b'reference')
    monkeypatch.setattr(fleet, 'D8_SMOKE_SHA', fleet.oracle.sha(ref))
    monkeypatch.setattr(fleet, 'prepare_baseline', lambda: (169.76982356747612, 235.41768866125494))
    spec = fleet.score_spec(base, point=fleet.enc.POINTS[0], small=small, full=full,
                            reference_local=ref, deadline='2030-01-01T00:00:00Z')
    validate_spec(spec)
    assert spec['budget_seconds'] == 400 and spec['smoke']['seconds'] == 120 and spec['full']['seconds'] == 240
    assert [a['name'] for a in spec['staged_inputs'][-3:]] == ['prepare16', 'prepare48', 'accepted16']
    assert all(a.get('extract') for a in spec['staged_inputs'][-3:-1])
    bad = {**small, 'checkpoint_sha256': 'a' * 64}
    with pytest.raises(ValueError, match='checkpoint manifest digest'):
        fleet.score_spec(base, point=fleet.enc.POINTS[0], small=bad, full=full,
                         reference_local=ref, deadline='2030-01-01T00:00:00Z')
    with pytest.raises((ValueError, FileNotFoundError)):
        fleet.score_spec(base, point=fleet.enc.POINTS[0], small={**small, 'local': '/not/published'}, full=full,
                         reference_local=ref, deadline='2030-01-01T00:00:00Z')


def test_score_parity_checks_both_judges_metrics_and_exact_pixels(tmp_path, monkeypatch):
    point = {'codec': 'dcvc', 'point': 0, 'native_sha256': 'n', 'wire_sha256': 'w',
             'encoder_decoded_sha256': 'd', 'encoder_rendered_sha256': 'r',
             'baseline': {'metrics': [1], 'judges': {'wasb': 2, 'balltrack': 3}},
             'streak': {'metrics': [4], 'judges': {'wasb': 5, 'balltrack': 6}, 'guard_off': False}}
    monkeypatch.setattr(fleet, 'reference16', lambda _: ({'points': [point]}, {}))
    record = {'checkpoint_sha256': 'h', 'execution_identity': {'source': 's'},
              'point': json.loads(json.dumps(point)),
              'receiver_replay': {'decoded_yuv_sha256': 'd', 'rendered_rgb_sha256': 'r'}}
    fleet.score_parity(record, tmp_path, ('dcvc', 0), 'h', {'source': 's'})
    record['point']['streak']['judges']['balltrack'] = 7
    with pytest.raises(ValueError, match='metrics or judges'):
        fleet.score_parity(record, tmp_path, ('dcvc', 0), 'h', {'source': 's'})
    record['point']['streak']['judges']['balltrack'] = 6
    record['receiver_replay']['rendered_rgb_sha256'] = 'changed'
    with pytest.raises(ValueError, match='pixels'):
        fleet.score_parity(record, tmp_path, ('dcvc', 0), 'h', {'source': 's'})


def test_point_archive_requires_matching_checkpoint_and_setting(tmp_path):
    path = tmp_path / 'point.tar'
    digest, archive_sha = archive(path, 'publish/point.json', {'checkpoint_sha256': 'x' * 64,
                                                               'point': {'codec': 'dcvc', 'point': 0}})
    entry = {'local': str(path), 'remote': str(fleet.DATA_ROOT / 'point.tar'),
             'archive_sha256': archive_sha, 'point_sha256': digest}
    assert fleet.point_archive('p', entry, ('dcvc', 0), 'x' * 64)[1] == digest
    with pytest.raises(ValueError, match='another checkpoint'):
        fleet.point_archive('p', entry, ('dcvc', 0), 'y' * 64)
    with pytest.raises(ValueError, match='receipt SHA256'):
        fleet.point_archive('p', {**entry, 'point_sha256': '0' * 64}, ('dcvc', 0), 'x' * 64)


def test_forecast_fails_closed_over_stage_or_campaign(tmp_path):
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 30.0}))
    (tmp_path / 'dispatch.json').write_text(json.dumps({'campaign_clock': {'started': 1., 'phase_started': 2., 'budget_end': 400.}}))
    assert fleet.forecast(tmp_path, 240, 3, 40.)['estimate_seconds'] == 132.5
    with pytest.raises(ValueError, match='forecast'):
        fleet.forecast(tmp_path, 100, 3, 40.)
    with pytest.raises(ValueError, match='forecast'):
        fleet.forecast(tmp_path, 240, 3, 290.)


def test_assemble_builder_uses_twelve_real_point_archives_and_no_gpu(tmp_path, monkeypatch):
    base = json.loads(fleet.BASE.read_text())
    small, full = preparation(tmp_path, 16), preparation(tmp_path, 48)
    ref = tmp_path / 'ref.tar'
    ref.write_bytes(b'reference')
    monkeypatch.setattr(fleet, 'D8_SMOKE_SHA', fleet.oracle.sha(ref))
    original_asset = fleet.asset
    def fixture_asset(name, remote, digest, local, *, extract):
        if name == 'reference48':
            return {'name': name, 'path': remote, 'sha256': digest, 'extract': extract}
        return original_asset(name, remote, digest, local, extract=extract)
    monkeypatch.setattr(fleet, 'asset', fixture_asset)
    all_points = []
    for n, prep in ((16, small), (48, full)):
        points = []
        for i, (codec, setting) in enumerate(fleet.enc.POINTS):
            path = tmp_path / f'{n}-{i}.tar'
            point_sha, archive_sha = archive(path, 'publish/point.json',
                {'checkpoint_sha256': prep['checkpoint_sha256'], 'point': {'codec': codec, 'point': setting}})
            points.append({'local': str(path), 'remote': str(fleet.DATA_ROOT / path.name),
                           'archive_sha256': archive_sha, 'point_sha256': point_sha})
        all_points.append(points)
    monkeypatch.setattr(fleet, 'prepare_baseline', lambda: (169.76982356747612, 235.41768866125494))
    spec = fleet.assemble_spec(base, small=small, full=full, points16=all_points[0], points48=all_points[1],
                               reference16_local=ref, reference48_local=ref, deadline='2030-01-01T00:00:00Z')
    validate_spec(spec)
    assert spec['device'] == 'cpu' and spec['gpu_models'] == [] and spec['budget_seconds'] == 120
    assert len(spec['staged_inputs']) == 16
    assert all(x['extract'] for x in spec['staged_inputs'] if x['name'].startswith('point'))
    with pytest.raises(ValueError, match='point archives required'):
        fleet.assemble_spec(base, small=small, full=full, points16=all_points[0][:-1], points48=all_points[1],
                            reference16_local=ref, reference48_local=ref, deadline='2030-01-01T00:00:00Z')


def test_score_dispatch_uses_actual_fleet_staged_environment_and_phase(tmp_path, monkeypatch):
    import argparse

    root = tmp_path / 'job'
    stage = root / 'smoke'
    scratch = tmp_path / 'scratch'
    stage.mkdir(parents=True)
    scratch.mkdir()
    def pin(name):
        path = tmp_path / name
        path.write_bytes(name.encode())
        return path, fleet.oracle.sha(path)
    source, source_sha = pin('manifest')
    exposure, exposure_sha = pin('exposure')
    audit, audit_sha = pin('audit')
    monkeypatch.setattr(fleet.fleet, 'SOURCE', source)
    monkeypatch.setattr(fleet.fleet, 'SOURCE_SHA', source_sha)
    monkeypatch.setattr(fleet.fleet, 'EXPOSURE', exposure)
    monkeypatch.setattr(fleet.fleet, 'EXPOSURE_SHA', exposure_sha)
    monkeypatch.setattr(fleet, 'AUDIT_PATH', audit)
    monkeypatch.setattr(fleet, 'AUDIT_SHA', audit_sha)
    entries = []
    paths = {}
    for name in ('wasb_code', 'balltrack_code'):
        parent = tmp_path / name
        parent.mkdir()
        tree = parent / 'tree'
        tree.mkdir()
        code_archive = parent / 'code.tar'
        code_archive.write_bytes(name.encode())
        entries.append({'name': name, 'source': '/data/code.tar', 'sha256': fleet.oracle.sha(code_archive)})
        paths[name] = str(tree)
    monkeypatch.setattr(fleet.fleet, 'archive_tree_sha', lambda _: 'treehash')
    monkeypatch.setattr(fleet.oracle, 'tree_sha', lambda _: 'treehash')
    (stage / 'staging.json').write_text(json.dumps({'inputs': entries}))
    code = {'git_head': 'a' * 40, 'source_sha256': 'b' * 64,
            'snapshot_sha256': 'c' * 64, 'spec_sha256': 'd' * 64}
    (stage / 'dispatch.json').write_text(json.dumps({'code': code,
        'staged_environment': {'sha256': 'e' * 64, 'prefix': '/runtime', 'python': '/runtime/bin/python'}}))
    monkeypatch.setenv('PS_STAGE_DIR', str(stage))
    monkeypatch.setenv('PS_SCRATCH_DIR', str(scratch))
    monkeypatch.setenv('PS_STAGE', 'smoke')
    captured = []
    monkeypatch.setattr(fleet.cp, 'score_point', lambda args, sha, identity: captured.append((args, sha, identity)))
    args = argparse.Namespace(action='score', frames=16, codec='dcvc', point=0,
                              prepare16=str(tmp_path / 'extract16'), prepare48=str(tmp_path / 'extract48'),
                              checkpoint16_sha256='1' * 64, checkpoint48_sha256='2' * 64,
                              **paths)
    fleet.run(args)
    actual, sha, identity = captured[0]
    assert actual.checkpoint == str(tmp_path / 'extract16' / 'publish')
    assert sha == '1' * 64 and actual.frames == 16
    assert identity == {'source_sha256': 'b' * 64, 'snapshot_sha256': 'c' * 64, 'environment_sha256': 'e' * 64}
    assert actual.cost_pilot and actual.manifest_sha256 == source_sha
    prepared = []
    monkeypatch.setattr(fleet, 'prepare_baseline', lambda: None)
    monkeypatch.setattr(fleet.enc, 'fidelity_gate', lambda *a, **kw: None)
    monkeypatch.setattr(fleet.cp, 'prepare', lambda data, ident: prepared.append((data, ident)))
    args.action = 'prepare'
    args.fidelity_gate = str(tmp_path / 'gate')
    args.fidelity_gate_sha256 = 'f' * 64
    fleet.run(args)
    prep_args, prep_identity = prepared[0]
    assert prep_args.frames == 16 and prep_args.cost_pilot
    assert prep_args.output == scratch / 'publish' and prep_identity == identity
    args.frames = 48
    with pytest.raises(ValueError, match='stage frame'):
        fleet.run(args)


def test_prepare_parity_rejects_changed_joint_fit_payload_or_guard(tmp_path, monkeypatch):
    path = tmp_path / 'prepared'
    path.mkdir()
    (path / 'checkpoint.json').write_text('{}')
    source = {'rows': [{'frame': 0}], 'scores': {'wasb': 'ok'}}
    point = {'codec': 'dcvc', 'point': 0, 'native_sha256': 'n', 'wire_sha256': 'w',
             'encoder_decoded_sha256': 'd', 'encoder_rendered_sha256': 'r',
             'baseline': {'zero_ball_wire_sha256': 'z'}, 'streak': {'guard_off': False}}
    report = {'frames': 16, 'conversion': {'yuv_sha256': 'src'}, 'points': [point]}
    (path / 'pilot.json').write_text(json.dumps(report))
    (path / 'source-qualification.json').write_text(json.dumps(source))
    monkeypatch.setattr(fleet.cp, 'verify', lambda *_: {})
    monkeypatch.setattr(fleet, 'reference16', lambda _: (json.loads(json.dumps(report)), source))
    monkeypatch.setattr(fleet.cost, 'compare_contract', lambda *_args, **_kwargs: None)
    fleet.prepare_parity(path, tmp_path, {})
    changed = json.loads(json.dumps(report))
    changed['points'][0]['wire_sha256'] = 'wrong'
    (path / 'pilot.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='fit payload'):
        fleet.prepare_parity(path, tmp_path, {})
    changed['points'][0]['wire_sha256'] = 'w'
    changed['points'][0]['streak']['guard_off'] = True
    (path / 'pilot.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='guard'):
        fleet.prepare_parity(path, tmp_path, {})


def test_prepare_builder_requires_real_strict_48_frame_gate(tmp_path, monkeypatch):
    gate = {'schema_version': 1, 'passed': True, 'scope': 'fidelity-only', 'purpose': '48-frame-cost-pilot',
            'archive_sha256': fleet.fleet.EXPECTED['archive'], 'calibration_job_id': fleet.cost.ACCEPTED_JOB,
            'evidence_sha256': 'a' * 64, 'checks': ['source_transform', 'native_and_wire', 'receiver_parity',
                                                    'perceptual_metrics', 'full_cost'],
            'clip': fleet.cost.CLIP, 'max_frames': 48, 'accepted_smoke_sha256': fleet.cost.ACCEPTED_SHA}
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(gate))
    ref = tmp_path / 'ref.tar'
    ref.write_bytes(b'reference')
    monkeypatch.setattr(fleet, 'D8_SMOKE_SHA', fleet.oracle.sha(ref))
    options = dict(gate_remote=str(fleet.DATA_ROOT / 'gate.json'), gate_sha=fleet.oracle.sha(path), gate_local=path,
                   reference_local=ref, deadline='2030-01-01T00:00:00Z')
    monkeypatch.setattr(fleet, 'prepare_baseline', lambda: (169.76982356747612, 235.41768866125494))
    spec = fleet.prepare_spec(json.loads(fleet.BASE.read_text()), **options)
    validate_spec(spec)
    assert spec['budget_seconds'] == 650 and spec['smoke']['seconds'] == 200 and spec['full']['seconds'] == 420
    gate['max_frames'] = 188
    path.write_text(json.dumps(gate))
    options['gate_sha'] = fleet.oracle.sha(path)
    with pytest.raises(ValueError, match='frame limit'):
        fleet.prepare_spec(json.loads(fleet.BASE.read_text()), **options)


def test_prepare_forecast_uses_pinned_outer_intervals_and_actual_slowdown(tmp_path, monkeypatch):
    baseline16, baseline48 = fleet.prepare_baseline()
    assert baseline16 == pytest.approx(169.76982356747612)
    assert baseline48 == pytest.approx(235.41768866125494)
    stage = tmp_path / 'smoke'
    stage.mkdir()
    (tmp_path / 'spec.json').write_text(json.dumps({'full': {'seconds': 420}}))
    (stage / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': baseline16}))
    (stage / 'dispatch.json').write_text(json.dumps({'campaign_clock': {'started': 1., 'phase_started': 2., 'budget_end': 650.}}))
    forecast = fleet.prepare_forecast(stage, 180.)
    assert forecast['estimate_seconds'] == pytest.approx(393.1265329918824)
    assert forecast['observed_slowdown'] == 1.0
    (stage / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': baseline16 * 1.10}))
    with pytest.raises(ValueError, match='forecast'):
        fleet.prepare_forecast(stage, 180.)
    (tmp_path / 'spec.json').write_text(json.dumps({'full': {'seconds': 500}}))
    assert fleet.prepare_forecast(stage, 180.)['estimate_seconds'] == pytest.approx(428.4391862910707)
    with pytest.raises(ValueError, match='forecast'):
        fleet.prepare_forecast(stage, 230.)
    for bad in (True, 0, 501):
        (tmp_path / 'spec.json').write_text(json.dumps({'full': {'seconds': bad}}))
        with pytest.raises(ValueError, match='stage cap'):
            fleet.prepare_forecast(stage, 180.)
    tampered = tmp_path / 'tampered.json'
    data = json.loads(fleet.PREPARE_RECEIPT.read_text())
    data['stages']['16']['scoring_events'][0]['wall_seconds'] += 1
    tampered.write_text(json.dumps(data))
    monkeypatch.setattr(fleet, 'PREPARE_RECEIPT', tampered)
    with pytest.raises(ValueError, match='evidence identity'):
        fleet.prepare_baseline()
