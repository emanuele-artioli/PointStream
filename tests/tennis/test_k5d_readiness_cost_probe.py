import json
from pathlib import Path

import pytest

from experiments.tennis import k5d_readiness_cost_probe as probe


def test_probe_requires_exact_checkpoint_before_spec_creation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='exact fe451'):
        probe.make_spec({'arguments': ['--frames', '{frames}']}, checkpoint_remote='/missing',
                        checkpoint_sha256='0' * 64, checkpoint_archive_sha256='1' * 64,
                        checkpoint_local=tmp_path / 'missing', deadline='2030-01-01T00:00:00Z')


def test_real_checkpoint_spec_is_cpu_only_and_bounded() -> None:
    checkpoint = Path('/private/tmp/pointstream-permission-prepare-audit-20261105/full/published.tar')
    if not checkpoint.exists():
        pytest.skip('saved actual calibration publication not present')
    base = json.loads(probe.cf.BASE.read_text())
    result = probe.make_spec(base, checkpoint_remote='/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/'
                             '20261105T025241Z-fe451f9e/full/published.tar',
                             checkpoint_sha256=probe.CHECKPOINT_SHA,
                             checkpoint_archive_sha256='4a5ba08205dee1b3a51ad4634dcf00b1f528534b91d772e30a316a4d860357a6',
                             checkpoint_local=checkpoint, deadline='2030-01-01T00:00:00Z')
    assert result['device'] == 'cpu' and result['gpu_models'] == [] and result['hosts'] == ['gpu2']
    assert result['cpu_affinity_index'] == 0 and result['budget_seconds'] == 240
    assert result['scale']['mode'] == {'smoke': 'assets', 'full': 'source'}
    assert result['environment']['sha256'] == probe.CHECKPOINT_IDENTITY['environment_sha256']
    assert result['arguments'][-2:] == ['--checkpoint', '{staged:checkpoint}']


def test_full_phase_reopens_and_binds_assets_smoke(tmp_path: Path) -> None:
    import io
    import tarfile

    job = tmp_path / 'job'
    smoke = job / 'smoke'
    full = job / 'full'
    smoke.mkdir(parents=True)
    full.mkdir()
    inputs = [{'name': 'archive', 'sha256': 'a' * 64}]
    record = {'scope': 'readiness-cost only; no hardware or production admission',
              'staged_inputs': {'archive': 'a' * 64}}
    payload = json.dumps(record).encode()
    published = smoke / 'published.tar'
    with tarfile.open(published, 'w') as archive:
        member = tarfile.TarInfo('publish/readiness-assets.json')
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    (smoke / 'staging.json').write_text(json.dumps({'inputs': inputs,
        'environment': {'sha256': probe.CHECKPOINT_IDENTITY['environment_sha256']},
        'published': {'sha256': probe.oracle.sha(published)}}))
    assert probe.verify_smoke(full) == record
    inputs[0]['sha256'] = 'b' * 64
    (smoke / 'staging.json').write_text(json.dumps({'inputs': inputs,
        'environment': {'sha256': probe.CHECKPOINT_IDENTITY['environment_sha256']},
        'published': {'sha256': probe.oracle.sha(published)}}))
    with pytest.raises(ValueError, match='differs'):
        probe.verify_smoke(full)
