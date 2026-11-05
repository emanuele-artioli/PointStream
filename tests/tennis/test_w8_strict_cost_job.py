import copy

import pytest

from experiments.tennis import w8_strict_cost_job as job


def test_new_spec_limits_and_frozen_inputs():
    from experiments.jobs.inbox import validate_spec
    spec = job.make_spec('2026-11-05T04:00:00+00:00')
    validated = validate_spec(spec, now=1793830000)
    assert validated['budget_seconds'] == 600
    assert spec['scale']['steps'] == {'smoke': 1, 'full': 3}
    assert (spec['smoke']['seconds'], spec['full']['seconds'], spec['validator_seconds']) == (260, 300, 30)
    assert spec['receiver_cpu_budget_seconds'] <= 1000
    assert spec['gpu_sharing'] == {'mode': 'exclusive'}
    assert {entry['name'] for entry in spec['staged_inputs']} >= {'packet', 'profile', 'oracle', 'preflight', 'identity'}


def test_receipt_resource_check_requires_preserved_canaries():
    summary = {'scope': 'qualification_only', 'quality_claim': False, 'actual_parent_cpu_seconds': 1,
               'parent_cpu_limit': 30, 'packet_cpu_limit': 30, 'generator_cpu_limit': 210,
               'packet_usage': {'cpu_seconds': 1, 'wall_seconds': 1, 'status': 0, 'timed_out': False, 'canaries_preserved': True},
               'generation_usage': {'cpu_seconds': 1, 'wall_seconds': 1, 'status': 0, 'timed_out': False, 'canaries_preserved': True}}
    job.check_summary_resources(summary)
    bad = copy.deepcopy(summary)
    bad['generation_usage']['canaries_preserved'] = False
    with pytest.raises(ValueError, match='accounting'):
        job.check_summary_resources(bad)


def test_durable_archive_is_exclusive_and_detects_changed_member(tmp_path):
    source = tmp_path / 'input'
    source.write_bytes(b'fixed experiment output')
    retained = tmp_path / 'retained'
    retained.mkdir()
    names = job.durable_member_names()
    for name in names:
        job.copy_new(source, retained / name)
    record = job.seal_durable_archive(retained, names)
    job.verify_durable_archive(retained, record)
    with pytest.raises(FileExistsError):
        job.seal_durable_archive(retained, names)
    (retained / names[0]).chmod(0o644)
    (retained / names[0]).write_bytes(b'altered')
    with pytest.raises(ValueError, match='bytes differ'):
        job.verify_durable_archive(retained, record)


def test_exact_preflight_runtime_gets_only_qualified_device_grants(tmp_path):
    runtime = '/local/users/emanuele/pointstream/cache/abc/tree'
    policy = {'python': runtime + '/bin/python3.12', 'allow_paths': [runtime, '/usr/lib', '/lib', '/lib64']}
    files = {'packet': {'path': '/local/public-packet.psw8', 'sha256': 'a' * 64}}
    roots = job.strict_roots(tmp_path / 'source', tmp_path / 'output', policy, files)
    assert '/dev/nvidiactl' in roots and '/dev/nvidia-uvm' in roots
    assert '/proc/driver/nvidia/version' in roots
    assert '/local/public-packet.psw8' in roots
    altered = copy.deepcopy(policy)
    altered['allow_paths'].append('/home')
    with pytest.raises(ValueError, match='fixed profile'):
        job.strict_roots(tmp_path / 'source', tmp_path / 'output', altered, files)
