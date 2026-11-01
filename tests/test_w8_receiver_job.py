import io
import math
import tarfile

import pytest

from experiments.tennis import w8_receiver_job as job


def _cost(steps, sample):
    phase = {name: {'gpu_reserved_wall_seconds': value} for name, value in
             [('model_load', 10), ('original_probe', 2), ('context_probe', 3), ('publication', 1),
              ('sampling', sample)]}
    return {'scope': 'qualification_only', 'steps': steps, 'unit': 'unit', 'phase_costs': phase}


def test_two_point_projection_uses_paid_context_and_reserved_downstream():
    result = job.projection(_cost(1, 10), _cost(3, 14), 400, 40, 2)
    assert result['per_step_gpu_wall_seconds'] == 2
    assert result['two_unit_30_step_seconds_upper_projection'] == pytest.approx(1.35 * 2 * (68 + 16 + 2))
    assert result['fits']
    assert not job.projection(_cost(1, 10), _cost(3, 14), 200, 40, 2)['fits']
    with pytest.raises(ValueError, match='invalid cost'):
        job.projection(_cost(1, 10), _cost(3, 9), 400, 40, 2)
    with pytest.raises(ValueError, match='invalid cost'):
        job.projection(_cost(1, 10), _cost(3, 14), math.nan, 40, 2)
    with pytest.raises(ValueError, match='matching'):
        job.projection(_cost(1, 10), _cost(6, 14), 400, 40, 2)


def test_common_base_extract_checks_archive_and_exact_member(tmp_path, monkeypatch):
    archive = tmp_path / 'base.tar'
    payload = b'base-checkpoint'
    with tarfile.open(archive, 'w') as tar:
        entry = tarfile.TarInfo('publish/tennis-base@fit-100.pt')
        entry.size = len(payload)
        tar.addfile(entry, io.BytesIO(payload))
    monkeypatch.setattr(job, 'BASE_ARCHIVE_SHA', job.fixed.file_sha(archive))
    monkeypatch.setattr(job.receiver, 'BASE_SHA', job.fixed.digest(payload))
    result = job.extract_base(archive, tmp_path / 'base.pt')
    assert result.read_bytes() == payload
    with pytest.raises(FileExistsError):
        job.extract_base(archive, result)
    monkeypatch.setattr(job.receiver, 'BASE_SHA', '0' * 64)
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        job.extract_base(archive, tmp_path / 'other.pt')


def test_preflight_bundle_rejects_tampered_runtime_receipt(tmp_path):
    import hashlib
    import json
    from experiments.tennis.w8_receiver_preflight import EXPECTED_TORCH

    asset_names = {'main_archive': job.MAIN_ENV, 'jcv_archive': job.JCV_ENV, 'aa_archive': job.receiver.AA_ARCHIVE_SHA}
    for label, weight in [('vae_config', 'w0'), ('vae_weights', 'w1'), ('image_encoder_config', 'w2'),
                          ('image_encoder_weights', 'w3'), ('denoising_unet', 'w4'), ('reference_unet', 'w5'),
                          ('pose_guider', 'w6'), ('motion_module', 'w7')]:
        asset_names[f'weight_{label}'] = job.receiver.PINNED_WEIGHTS[job.WEIGHT_NAMES[weight]]
    identity = tmp_path / 'identity.json'
    identity.write_text(json.dumps({'files': {key: {'sha256': value} for key, value in asset_names.items()},
                                    'aa_installed': {'verified_files': 10},
                                    'sam_distribution': {'record_hashed_sources': 10}}))
    source = tmp_path / 'source'
    source.mkdir()
    program = tmp_path / 'python'
    program.write_text('python')
    receipts = {}
    runtimes = {}
    results = {}
    for name in ('main', 'jcv'):
        receipts[name] = {'torch': EXPECTED_TORCH[name], 'cuda_initialized': False, 'visible_devices': '',
                          'executable': str(program), 'isolation': {'landlock_abi': 4, 'pre_sandbox_threads': 1,
                          'extra_fds': False, 'network_interfaces': ['lo'], 'denials': {'/x': 13},
                          'cpu_bootstrap': {'before_landlock_sha256': 'x', 'after_landlock_sha256': 'x'}},
                          'source_modules': {'module': {'path': str(source / 'module.py'), 'sha256': 'code'}}}
        runtimes[name] = {'python': str(program)}
        results[name] = {'usage': {'status': 0}, 'receipt_sha256': hashlib.sha256(
            json.dumps(receipts[name], sort_keys=True).encode()).hexdigest()}
    bundle = {'passed': True, 'identity_sha256': job.fixed.file_sha(identity), 'source_tree': str(source),
              'runtimes': runtimes, 'receipts': receipts, 'frozen_source_files': {'module.py': 'code'},
              'result': {'assets': {'status': 0}, 'source_files': 1, 'runtimes': results}}
    job.check_preflight_bundle(bundle, identity)
    bundle['receipts']['main']['cuda_initialized'] = True
    with pytest.raises(ValueError, match='runtime receipt mismatch'):
        job.check_preflight_bundle(bundle, identity)


def test_stage_input_mapping_rejects_substituted_bytes(tmp_path, monkeypatch):
    import json

    stage = tmp_path / 'job' / 'smoke'
    stage.mkdir(parents=True)
    expected = {'packet': job.fixed.WINDOW_SHA, 'profile': job.fixed.PROFILE_SHA, 'oracle': job.ORACLE_SHA,
                'base': job.BASE_ARCHIVE_SHA, 'aa': job.receiver.AA_ARCHIVE_SHA,
                **{k: job.receiver.PINNED_WEIGHTS[v] for k, v in job.WEIGHT_NAMES.items()},
                'preflight': 'p' * 64, 'identity': 'i' * 64}
    declared = []
    staged = []
    for name, digest in expected.items():
        path = tmp_path / name
        path.write_text(name)
        declared.append({'name': name, 'path': str(path), 'sha256': digest, 'extract': False})
        staged.append({'name': name, 'path': str(path), 'sha256': digest, 'extract': False})
    spec = {'device': 'gpu', 'single_cpu_campaign': True, 'cpu_threads': 1,
            'environment': {'sha256': job.JCV_ENV}, 'budget_seconds': 500, 'staged_inputs': declared,
            'contention': {'policy': 'stop'}}
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    (stage / 'staging.json').write_text(json.dumps({'inputs': staged}))
    (stage / 'dispatch.json').write_text(json.dumps({'cpu_affinity': [0]}))
    monkeypatch.setattr(job.fixed, 'file_sha', lambda path: expected[path.name])
    result, _ = job.stage_inputs(stage)
    assert set(result) == set(expected)
    spec['gpu_sharing'] = {'mode': 'shared', 'basis': 'test', 'margin_mib': 1024}
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='one-core JCV'):
        job.stage_inputs(stage)
    del spec['gpu_sharing']
    (stage.parent / 'spec.json').write_text(json.dumps(spec))
    staged[0]['sha256'] = 'wrong'
    (stage / 'staging.json').write_text(json.dumps({'inputs': staged}))
    with pytest.raises(ValueError, match='staged identity mismatch'):
        job.stage_inputs(stage)


def test_published_cost_summary_requires_exact_sealed_archive(tmp_path):
    import json

    stage = tmp_path / 'smoke'
    stage.mkdir()
    archive = stage / 'published.tar'
    body = json.dumps({'scope': 'qualification_only'}).encode()
    with tarfile.open(archive, 'w') as tar:
        info = tarfile.TarInfo('publish/cost-summary.json')
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    (stage / 'staging.json').write_text(json.dumps({'published': {'sha256': job.fixed.file_sha(archive)}}))
    assert job.read_published_summary(stage)['scope'] == 'qualification_only'
    with archive.open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='archive mismatch'):
        job.read_published_summary(stage)


def test_fleet_template_passes_actual_spec_validation_when_frozen():
    import json
    from pathlib import Path
    from experiments.jobs.inbox import phase_command, validate_spec

    template = Path(__file__).resolve().parents[1] / 'docs/experiments/w8-inputs/receiver-20261101/fleet-cost-template.json'
    spec = json.loads(template.read_text())
    spec['deadline'] = '2099-01-01T00:00:00Z'
    for entry in spec['staged_inputs']:
        if entry['path'].startswith('<'):
            entry['path'] = '/home/itec/emanuele/Datasets/pointstream-data/reviewed/' + entry['name']
        if entry['sha256'].startswith('<'):
            entry['sha256'] = 'a' * 64
    spec['inputs'] = [{'path': entry['path'], 'sha256': entry['sha256']} for entry in spec['staged_inputs']]
    spec['arguments'][-1] = 'a' * 64
    normalized = validate_spec(spec, now=1)
    staged = [{'name': entry['name'], 'path': '/local/staged/' + entry['name']} for entry in spec['staged_inputs']]
    assert phase_command(normalized, 'smoke', staged, '/env/bin/python')[4:6] == ['--steps', '1']
    assert phase_command(normalized, 'full', staged, '/env/bin/python')[4:6] == ['--steps', '3']


def test_cost_summary_requires_bounded_completed_workers():
    worker = {'status': 0, 'timed_out': False, 'cpu_seconds': 4.0, 'wall_seconds': 8.0}
    summary = {'scope': 'qualification_only', 'quality_claim': False, 'actual_parent_cpu_seconds': 2,
               'parent_cpu_limit': 10, 'packet_usage': dict(worker), 'packet_cpu_limit': 12,
               'generation_usage': dict(worker), 'generator_cpu_limit': 50}
    job.check_summary_resources(summary)
    summary['packet_usage']['timed_out'] = True
    with pytest.raises(ValueError, match='worker accounting'):
        job.check_summary_resources(summary)
    summary['packet_usage']['timed_out'] = False
    summary['actual_parent_cpu_seconds'] = 11
    with pytest.raises(ValueError, match='parent CPU'):
        job.check_summary_resources(summary)


def test_retained_decoder_audit_checks_unit_and_manifest_binding(tmp_path, monkeypatch):
    import json

    checkpoint = tmp_path / 'decoded.pt'
    checkpoint.write_bytes(b'paid-state')
    unit = tmp_path / 'packet-unit.npz'
    unit.write_bytes(b'unit')
    fingerprints = {'tensor': {'sha256': 'fingerprint'}}
    unit_hashes = {job.fixed.UNITS[0]: job.fixed.file_sha(unit), job.fixed.UNITS[1]: 'other'}
    monkeypatch.setattr(job.receiver, 'ORACLE_UNIT_SHA', unit_hashes)
    monkeypatch.setattr(job.receiver, 'verify_serialized_match', lambda path, expected: job.fixed.file_sha(path))
    manifest = {'packet_sha256': job.fixed.WINDOW_SHA, 'profile_sha256': job.fixed.PROFILE_SHA,
                'checkpoint_sha256': job.fixed.file_sha(checkpoint), 'match_tensors': fingerprints,
                'unit_sha256': unit_hashes}
    manifest_path = tmp_path / 'decoded-manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    seal = {'checkpoint_sha256': job.fixed.file_sha(checkpoint),
            'decoder_manifest_sha256': job.fixed.file_sha(manifest_path)}
    job.audit_retained_inputs(tmp_path, seal, fingerprints)
    unit.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='do not bind'):
        job.audit_retained_inputs(tmp_path, seal, fingerprints)


def test_projection_reserves_complete_observed_stage_overhead(tmp_path):
    import json

    validation = _cost(3, 14)
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 45.0, 'timing_contaminated': False, 'gpu_sharing': {'mode': 'exclusive'}}))
    # 30 seconds were recorded in model/probe/sample/publication phases; the
    # remaining 15 seconds include packet setup and the enclosing process.
    assert job._measured_stage_overhead(tmp_path, validation) == pytest.approx(15.0)
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 45.0, 'timing_contaminated': True, 'gpu_sharing': {'mode': 'exclusive'}}))
    with pytest.raises(ValueError, match='execution receipt'):
        job._measured_stage_overhead(tmp_path, validation)
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': 29.0, 'timing_contaminated': False, 'gpu_sharing': {'mode': 'exclusive'}}))
    with pytest.raises(ValueError, match='phase costs exceed'):
        job._measured_stage_overhead(tmp_path, validation)
    (tmp_path / 'execution.json').write_text(json.dumps({'exit_code': 1, 'seconds': 45.0, 'timing_contaminated': False, 'gpu_sharing': {'mode': 'exclusive'}}))
    with pytest.raises(ValueError, match='execution receipt'):
        job._measured_stage_overhead(tmp_path, validation)


def test_independent_cost_audit_rejects_unreserved_observed_overhead(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    campaign = tmp_path / 'job-id'
    smoke, full = campaign / 'smoke', campaign / 'full'
    smoke.mkdir(parents=True)
    full.mkdir()
    retained = {smoke: tmp_path / 'retained-smoke', full: tmp_path / 'retained-full'}
    seal = {'files': {'unit': 'same'}, 'device': {'name': 'same'}}
    for path in retained.values():
        path.mkdir()
        (path / 'sealed.json').write_text(json.dumps(seal))
    for stage, seconds in ((smoke, 45.0), (full, 49.0)):
        (stage / 'execution.json').write_text(json.dumps({'exit_code': 0, 'seconds': seconds, 'timing_contaminated': False, 'gpu_sharing': {'mode': 'exclusive'}}))
    validation = {1: _cost(1, 10), 3: _cost(3, 14)}
    def summary(stage):
        steps = 1 if stage == smoke else 3
        return {'scope': 'qualification_only', 'quality_claim': False, 'steps': steps,
                'retained_output': str(retained[stage]), 'generation_seal': seal,
                'validation': validation[steps], 'inputs': {}, 'preflight_sha256': 'same'}
    monkeypatch.setattr(job, 'read_published_summary', summary)
    monkeypatch.setattr(job, 'check_summary_resources', lambda *_: None)
    monkeypatch.setattr(job, 'read_oracle', lambda *_: {})
    monkeypatch.setattr(job, 'retained_path', lambda stage: retained[stage])
    monkeypatch.setattr(job, 'audit_retained_inputs', lambda *_: None)
    monkeypatch.setattr(job.receiver, 'validate_cost_probe', lambda _out, _expected, steps: validation[steps])
    original_resolve = Path.resolve
    monkeypatch.setattr(Path, 'resolve', lambda self, strict=False: self if str(self) == '/local/users/emanuele'
                        else original_resolve(self, strict=strict))
    monkeypatch.setattr(Path, 'is_relative_to', lambda *_: True)
    with pytest.raises(ValueError, match='below measured stage overhead'):
        job.audit_pair(smoke, full, tmp_path / 'oracle', 500, 10, 18)
    result = job.audit_pair(smoke, full, tmp_path / 'oracle', 500, 10, 19)
    assert result['projection']['measured_stage_overhead_seconds_per_unit'] == pytest.approx(19)


def test_cost_audit_requires_and_records_cpu_cap(tmp_path, monkeypatch):
    import json
    import sys

    output = tmp_path / 'audit.json'
    captured = []
    monkeypatch.setattr(job.resource, 'setrlimit', lambda which, limits: captured.append((which, limits)))
    monkeypatch.setattr(job, 'audit_pair', lambda *_: {'passed': True})
    monkeypatch.setattr(sys, 'argv', ['w8_receiver_job', 'cost-audit', '--one', 'smoke', '--three', 'full',
                                     '--oracle', 'oracle', '--remaining-gpu', '100', '--downstream-reserve', '10',
                                     '--extra-generation-overhead', '15', '--cpu-seconds', '7', '--output', str(output)])
    assert job.main() == 0
    assert captured == [(job.resource.RLIMIT_CPU, (7, 7))]
    assert json.loads(output.read_text())['audit_cpu_limit'] == 7
    sys.argv[sys.argv.index('--cpu-seconds') + 1] = '21'
    with pytest.raises(ValueError, match='at most 20 seconds'):
        job.main()
