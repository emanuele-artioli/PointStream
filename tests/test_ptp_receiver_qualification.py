import json
import struct

import numpy as np
import pytest

from experiments.jobs import inbox, staging
from experiments.tennis import paid_patch_profile as paid, ptp_receiver_qualification as bridge
from src.pointstream import tracked_patch


def test_factory_requires_four_concrete_paid_identities_and_validates():
    entries = {name: {'path': f'/data/{name}', 'sha256': str(index) * 64}
               for index, name in enumerate(('tennis_archive', 'tennis_stage', 'ego_archive', 'ego_stage'), 1)}
    spec = inbox.validate_spec(bridge.make_spec('2026-11-16T12:00:00+00:00', entries))
    assert spec['hosts'] == ['gpu6'] and spec['cpu_threads'] == 1 and spec['cpu_affinity_index'] == 0
    assert spec['scale']['cases'] == {'smoke': 1, 'full': 4}
    assert spec['budget_seconds'] == 1530 and spec['required_commands'] == ['ffmpeg']
    assert spec['staged_inputs'][0]['sha256'] == bridge.FIXED[spec['staged_inputs'][0]['name']]
    del entries['ego_stage']
    with pytest.raises(ValueError):
        bridge.make_spec('2026-11-16T12:00:00+00:00', entries)


def test_actual_protected_file_positive_identity_required(tmp_path):
    path = tmp_path / 'frame.jpg'
    path.write_bytes(b'actual source bytes')
    identity = bridge.row(path)
    assert identity['size'] == len(b'actual source bytes')
    assert bridge.row(path, identity['sha256']) == identity
    with pytest.raises(ValueError):
        bridge.row(path, '0' * 64)
    with pytest.raises(ValueError):
        bridge.row(tmp_path / 'not-present')


def test_bound_paid_full_archive_and_tampering(tmp_path):
    root = tmp_path / 'paid'
    root.mkdir()
    for name in paid.publication_members('ego'):
        value = {'domain': 'ego', 'frames': 64} if name == 'receipt.json' else {'name': name}
        (root / name).write_text(json.dumps(value))
    archive = tmp_path / 'paid.tar'
    binding = paid.publish(root, archive, 'ego')
    stage = tmp_path / 'stage.json'
    stage.write_text(json.dumps({'domain': 'ego', 'frames': 64, 'archive': binding,
                                 'cost': {'passed': True}, 'internal_checks': {'receiver': True}}))
    stage_sha = bridge.common.k5d_source_oracle.sha(stage)
    extracted, receipt = bridge.bind_input(archive, stage, binding['sha256'], stage_sha, 'ego', tmp_path / 'extracted')
    assert receipt['frames'] == 64 and (extracted / 'fallback.ptp').is_file()
    with pytest.raises(ValueError):
        bridge.bind_input(archive, stage, '0' * 64, stage_sha, 'ego', tmp_path / 'bad')


def test_packet_expected_pixels_are_parent_only_and_hash_bound(tmp_path):
    native = {'version': 1, 'profile': 'native-av1', 'width': 64, 'height': 64, 'fps': [25, 1],
              'frames': 48, 'full_range': False, 'decoder_sha256': '0' * 64}
    meta = {'version': 1, 'profile': 'tracked-pixel-v1', 'native': native, 'numpy_version': np.__version__,
            'positions': [[0, 0]] * 48, 'tile_size': 64, 'inner_box': [16, 16, 32, 32]}
    ivf = struct.pack('<4sHH4sHHIIII', b'DKIF', 0, 32, b'AV01', 64, 64, 25, 1, 48, 0)
    ivf += b''.join(struct.pack('<IQ', 2, i) + b'xx' for i in range(48))
    packet = tracked_patch.pack(meta, ivf, ivf)
    (tmp_path / '48.ptp').write_bytes(packet)
    decoded = {'decoded_sha256': 'd' * 64, 'rgb_frame_sha256': ['e' * 64] * 48}
    info = {'decode': decoded, 'encode': dict(decoded), 'inspect': tracked_patch.inspect(packet),
            'oracle_parity': {'passed': True}}
    (tmp_path / '48.json').write_text(json.dumps(info))
    case = bridge.expected_case(tmp_path, '48', 'tennis')
    assert case['expected'] == decoded and case['packet_sha256'] == bridge.common.digest(packet)
    info['encode']['decoded_sha256'] = '0' * 64
    (tmp_path / '48.json').write_text(json.dumps(info))
    with pytest.raises(ValueError):
        bridge.expected_case(tmp_path, '48', 'tennis')


def test_proof_archive_not_overwritten_when_scratch_released(tmp_path):
    scratch, stage = tmp_path / 'scratch', tmp_path / 'stage'
    pub = scratch / 'ptp-qualification-private' / 'publication'
    pub.mkdir(parents=True)
    stage.mkdir()
    for name in bridge.names(1):
        (pub / name).write_text('{}')
    archive = stage / 'published.tar'
    binding = bridge.publish(pub, archive, 1)
    assert staging.publish_scratch(scratch, stage) is None
    staging.release_scratch(scratch, stage)
    result = bridge.extract(archive, binding, 1, tmp_path / 'replayed')
    assert {path.name for path in result.iterdir()} == set(bridge.names(1))
    with pytest.raises(ValueError):
        bridge.extract(archive, dict(binding, sha256='0' * 64), 1, tmp_path / 'tampered')
    assert bridge.cost_record(140, 130, 1)['passed']
    assert not bridge.cost_record(146, 130, 1)['passed']


def bindings_fixture():
    from experiments.tennis import ptp_isolated_receiver as isolated
    profile = {'runtime_files': {'numpy.so': {'sha256': 'a' * 64, 'size': 100}},
               'source_files': {'decoder.py': {'sha256': 'b' * 64, 'size': 100}},
               'runtime_archive_sha256': isolated.RUNTIME_SHA}
    policy = {'profiles': {'gpu6': profile}}
    manifest = {'host': 'gpu6', 'ego_original_video_present': False}
    def record(path, sha, size=100):
        return {'path': path, 'sha256': sha, 'size': size}
    names = ('archive', 'producer', 'prepared')
    roles = {
        'source_archive': [record(bridge.ORIGINAL[n], bridge.FIXED[n]) for n in names],
        'staged_source_archive': [record('/scratch/' + n, bridge.FIXED[n]) for n in names],
        'private_detector': [record(bridge.ORIGINAL['model'], bridge.MODEL), record('/scratch/model', bridge.MODEL)],
        'source_yuv': [record('/scratch/tennis.yuv', bridge.common.SOURCE_YUV, 48 * bridge.common.FRAME_BYTES),
                       record('/scratch/ego.yuv', paid.EGO_SOURCE, 64 * 1920 * 1080 * 3 // 2)],
        'ground_truth': [record('/scratch/qualification.json', bridge.common.QUALIFICATION),
                         record('/scratch/masks.npz', bridge.EGO_MASKS)],
        'source_frame': [record('/scratch/frame.jpg', bridge.FIRST_JPEG),
                         record('/scratch/ego-frame.yuv', 'c' * 64, 1920 * 1080 * 3 // 2)],
        'encoder_state': [record('/scratch/actual-config-' + n + '.json', 'd' * 64)
                          for n in ('48', '55', '62', 'fallback')],
    }
    proof = {'filesystem_proof': {'profile_sha256': bridge.kernel._object_sha(profile),
             'runtime_inventory_sha256': bridge.kernel._object_sha(profile['runtime_files']),
             'source_inventory_sha256': bridge.kernel._object_sha(profile['source_files']),
             'protected_roles': roles}}
    return proof, manifest, policy


def test_archived_witness_binds_host_and_all_generic_inventories():
    from copy import deepcopy
    proof, manifest, policy = bindings_fixture()
    assert all(bridge.proof_bindings(proof, manifest, policy).values())
    for key in ('profile_sha256', 'runtime_inventory_sha256', 'source_inventory_sha256'):
        forged = deepcopy(proof)
        forged['filesystem_proof'][key] = 'f' * 64
        assert not all(bridge.proof_bindings(forged, manifest, policy).values())
    manifest['host'] = 'unreviewed-host'
    with pytest.raises(ValueError, match='host'):
        bridge.proof_bindings(proof, manifest, policy)


@pytest.mark.parametrize('role,index,key,value', [
    ('source_archive', 0, 'sha256', 'f' * 64),
    ('source_archive', 1, 'path', '/different-producer'),
    ('staged_source_archive', 2, 'sha256', 'f' * 64),
    ('private_detector', 1, 'sha256', 'f' * 64),
    ('source_yuv', 0, 'size', 1),
    ('source_yuv', 1, 'sha256', 'f' * 64),
    ('ground_truth', 0, 'sha256', 'f' * 64),
    ('ground_truth', 1, 'sha256', 'f' * 64),
    ('source_frame', 0, 'sha256', 'f' * 64),
    ('source_frame', 1, 'size', 1),
    ('encoder_state', 0, 'path', '/scratch/placeholder.json'),
])
def test_denied_role_labels_cannot_substitute_unrelated_files(role, index, key, value):
    proof, manifest, policy = bindings_fixture()
    # Even a self-consistent lower-level denial receipt must name the real inputs.
    proof['filesystem_proof']['protected_roles'][role][index][key] = value
    assert not all(bridge.proof_bindings(proof, manifest, policy).values())


def test_optional_original_video_is_checked_when_recorded():
    proof, manifest, policy = bindings_fixture()
    manifest['ego_original_video_present'] = True
    assert not all(bridge.proof_bindings(proof, manifest, policy).values())
    proof['filesystem_proof']['protected_roles']['source_archive'].append(
        {'path': bridge.ORIGINAL['ego_video'], 'sha256': bridge.VIDEO, 'size': 100})
    assert all(bridge.proof_bindings(proof, manifest, policy).values())
