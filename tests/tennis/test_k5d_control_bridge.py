import io
import tarfile
from pathlib import Path

import pytest

from experiments.tennis import k5d_control_bridge as bridge


def archive(path: Path, files: dict[str, bytes]) -> str:
    with tarfile.open(path, 'w') as tar:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bridge_checks_complete_manifest_and_rejects_scientific_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old = tmp_path / 'old.tar'
    new = tmp_path / 'new.tar'
    old_sha = archive(old, {'model.py': b'unchanged', 'codec.py': b'unchanged'})
    added = {name: b'new control' for name in bridge.CONTROL_ADDITIONS}
    new_sha = archive(new, {'model.py': b'unchanged', 'codec.py': b'unchanged', **added})
    old_id, _ = bridge.manifest(old, old_sha)
    new_id, _ = bridge.manifest(new, new_sha)
    monkeypatch.setattr(bridge, 'CALIBRATION_SOURCE', old_id)
    a = {'source_sha256': old_id, 'snapshot_sha256': old_sha, 'environment_sha256': 'e' * 64}
    b = {'source_sha256': new_id, 'snapshot_sha256': new_sha, 'environment_sha256': 'e' * 64}
    result = bridge.bridge(old, a, new, b)
    assert result['approved'] is False and result['changed_paths'] == sorted(bridge.CONTROL_ADDITIONS)
    admission = 'experiments/tennis/k5d_full_admission.py'
    assert admission in bridge.CONTROL_ADDITIONS
    missing = {name: data for name, data in added.items() if name != admission}
    missing_sha = archive(new, {'model.py': b'unchanged', 'codec.py': b'unchanged', **missing})
    missing_id, _ = bridge.manifest(new, missing_sha)
    with pytest.raises(ValueError, match='beyond approved'):
        bridge.bridge(old, a, new, {**b, 'source_sha256': missing_id, 'snapshot_sha256': missing_sha})
    changed_sha = archive(new, {'model.py': b'changed', 'codec.py': b'unchanged', **added})
    changed_id, _ = bridge.manifest(new, changed_sha)
    with pytest.raises(ValueError, match='beyond approved'):
        bridge.bridge(old, a, new, {**b, 'source_sha256': changed_id, 'snapshot_sha256': changed_sha})
    with pytest.raises(ValueError, match='environment'):
        bridge.bridge(old, a, new, {**b, 'environment_sha256': 'f' * 64})


def test_actual_7640_source_archive_matches_record() -> None:
    path = Path('/private/tmp/pointstream-7640-source-20261105.tar')
    if not path.exists():
        pytest.skip('calibration source archive not retained locally')
    actual, _ = bridge.manifest(path, 'c3ff5419167504afcd84d25931791eddfe8bae16b5e492ed124df31e43f66f67')
    assert actual == bridge.CALIBRATION_SOURCE


def test_recorded_calibration_uses_saved_complete_16_and_48_proofs() -> None:
    from experiments.tennis import k5d_study_evidence as evidence

    base = Path('/private/tmp/pointstream-full30-cost-inputs')
    reference = Path('/private/tmp/pointstream-checkpoint-repair1-audit-20261105')
    prep = base / '20261105T025241Z-fe451f9e'
    score = base / '20261105T032433Z-1423f92a'
    if not (score / 'full/published.tar').exists():
        pytest.skip('accepted calibration publications not retained locally')
    identity = {'source_sha256': '8a3b51c487cf9b456e304fdf24300f230ece3f3c7b2c8569ab6681fbd8166f13',
                'snapshot_sha256': 'c3ff5419167504afcd84d25931791eddfe8bae16b5e492ed124df31e43f66f67',
                'environment_sha256': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'}
    cases = [('smoke', 16, '84575156c209d324400c1c0bbc710698c9c44aceb0e1eb5f764d08a239405b14',
              'd6cb69e4c0f0cb7b9100283b8b5808d28146a2e0e31b269ab309249691ce20b7', 'd8b5-smoke.tar',
              '8758597ba524cafc017e25cbd20c15ecbeaeed534d5622dfe67d66eb16521377',
              '6aa2841e2e72682b5e8a2e78dc4395cad0b23dfa2703310c12ee60848b584451'),
             ('full', 48, '4a5ba08205dee1b3a51ad4634dcf00b1f528534b91d772e30a316a4d860357a6',
              '8dd50977e0fdd9ce15ec4d10529b7c1cd0f4cfaf9e140ee30a3de92d10c3d532', 'd8b5-full.tar',
              'f0dd118a5d64ca148be675a097606aa6dacb8ae49ffcac639ad30911ae74a727',
              'e30aff247dad65bbf3aaabb2ae5ee46d5bdcd9f1e45d00ea7b3a63a26e50e000')]
    for stage, frames, prep_sha, score_sha, ref, ref_sha, timing_sha in cases:
        evidence.parity(score / stage / 'published.tar', score_sha, prep / stage / 'published.tar', prep_sha,
                        reference / ref, ref_sha, frames, identity, timing_sha)
