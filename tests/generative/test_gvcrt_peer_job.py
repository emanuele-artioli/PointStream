import hashlib

import pytest

from experiments.generative import gvcrt_peer_job as job


def test_forecast_counts_one_i_thirteen_normal_and_two_reset_p():
    sender = {'reference_stack_seconds': 0.0, 'reference_save_seconds': 0.0, 'encode_seconds_per_frame': [1, 2, .5, 1],
              'reference_decode_seconds_per_frame': [1, 2, .5, 1],
              'encode_full_seconds_per_frame': [2, 3, 1, 1.5],
              'reference_decode_full_seconds_per_frame': [2, 3, 1, 1.5],
              'encode_full_timing_scope': 'source_hash_read_rgb_pad_transfer_compress_packet_per_frame',
              'decode_full_timing_scope': 'packet_parse_decode_crop_cpu_transfer_per_frame'}
    receiver = {'stack_seconds': 0.0, 'save_seconds': 0.0, 'decode_seconds_per_frame': [1, 2, .5, 1],
                'decode_full_seconds_per_frame': [2, 3, 1, 1.5],
                'decode_full_timing_scope': 'packet_parse_decode_crop_cpu_transfer_per_frame'}
    times = {'sender': 30, 'receiver': 20, 'staging': 2, 'prep': 1, 'probe': 1,
             'score': 1, 'output_copy': 0, 'validation_reserve': 3, 'release_reserve': 3}
    result = job.projection(sender, receiver, times, 500)
    assert result['variable_16_seconds'] == pytest.approx(3 * (2 + 13 * 1.5 + 2 * 3))
    assert result['forecast_gpu_seconds'] == pytest.approx(1.5 * (27.5 + 82.5 + 14))
    assert result['passes'] is True
    assert job.projection(sender, receiver, times, 100)['passes'] is False
    sender['encode_seconds_per_frame'][0] = 3
    with pytest.raises(ValueError, match='shorter'):
        job.projection(sender, receiver, times, 500)


def test_native_profile_checks_reset_as_p_frame_and_rejects_wrong_schedule(tmp_path):
    data = b'abcdefgh'
    stream = tmp_path / 'stream.bin'
    stream.write_bytes(data)
    manifest = {'schema': 'gvcrt-native-v1', 'source_revision': 'd0e32bfa3e8e282f9a77437c223c605858eea637',
                'frames': 4, 'fps': 30, 'original_width': 1280, 'original_height': 720,
                'canvas_width': 1920, 'canvas_height': 1088, 'base_qp': 3, 'reset_interval': 8,
                'padding_policy': 'released', 'ec_part': 1, 'model_sha256': {'I': job.I_SHA, 'P': job.P_SHA},
                'stream': {'filename': 'stream.bin', 'bytes': 8, 'sha256': hashlib.sha256(data).hexdigest()},
                'placements': [{'index': i, 'I': i == 0, 'reset': i == 1, 'qp': q,
                                'offset': 2 * i, 'bytes': 2} for i, q in enumerate((3, 5, 3, 4))]}
    job.check_manifest(manifest, 4, stream)
    manifest['placements'][1]['I'] = True
    with pytest.raises(ValueError, match='schedule'):
        job.check_manifest(manifest, 4, stream)


def test_comparison_keeps_four_frame_smoke_rates_unreported():
    labels = [[1, 10.0, 20.0]] * 4
    metrics = [{'frame': i, 'frame_psnr': 30.0, 'full_lpips': .2, 'full_dists': .1,
                'crop_lpips': .3, 'crop_dists': .2} for i in range(4)]
    rows = [{'detected': True, 'x': 10.0, 'y': 20.0} for _ in range(4)]
    scores = {'frames': 4, 'visible': 4, 'absent': 0, 'unknown': 0, 'hit5': 4, 'hit10': 4,
              'miss10': 0, 'false_on_absent': 0, 'recall5': 1.0, 'recall10': 1.0, 'false_per_negative': None}
    judge = {'rows': rows, 'scores': scores}
    peer = {'metrics': metrics, 'wasb': judge, 'balltrack': judge}
    base = [{'codec': 'svt', 'point': 62, 'native_bytes': 100, 'zero_ball_wire_bytes': 110,
             'metrics': metrics, 'judges': {'wasb': {'rows': rows}, 'balltrack': {'rows': rows}}}]
    result = job.paired_comparison(peer, base, labels, 4, 20, 30)
    assert result['peer']['wire_kbps'] is None
    assert result['baselines'][0]['native_kbps'] is None
    assert result['bd_rate'] is None
    metrics[0]['full_lpips'] = float('nan')
    with pytest.raises(ValueError, match='invalid metric'):
        job.paired_comparison(peer, base, labels, 4, 20, 30)


def test_isolated_command_is_direct_script_and_fresh_namespace(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{}')
    command = job.receiver_command(tmp_path / 'python', tmp_path / 'receiver.py', 'probe', config, isolated=True)
    assert command[:4] == ['/usr/bin/unshare', '--user', '--map-root-user', '--net']
    assert command[4:7] == [str(tmp_path / 'python'), str(tmp_path / 'receiver.py'), 'probe']


def test_isolation_receipt_requires_same_inode_alias_and_exact_allowlist():
    c = {'runtime': '/safe/runtime', 'author': '/safe/author', 'bridge': '/safe/bridge',
         'entropy': '/safe/entropy', 'i_model': '/safe/i', 'p_model': '/safe/p',
         'wire': '/safe/wire', 'output': '/safe/output', 'device_paths': [],
         'parent_user': 'user-parent', 'parent_net': 'net-parent',
         'denied_paths': {'source': ['/secret'], 'aliases': ['/alias']}}
    receipt = {'landlock_abi': 3, 'handled_fs_rights': (1 << 15) - 1,
               'mutation_denials': dict.fromkeys(('write', 'truncate', 'create', 'remove', 'mkdir', 'rename', 'link'), 13),
               'output_write_probe': True, 'parent_canary_post_sha256': dict.fromkeys(('write', 'truncate', 'remove', 'rename', 'link'), hashlib.sha256(b'canary').hexdigest()), 'denials': {'/secret': 13, '/alias': 13},
               'namespaces': {'user': 'user-child', 'net': 'net-child'}, 'network_interfaces': ['lo'],
               'pre_sandbox_threads': 1, 'extra_fds': False, 'cuda_preinitialized': False,
               'cpu_before_sha256': 'x', 'cpu_after_sha256': 'x',
               'allow_roots': ['/safe/runtime', '/safe/author', '/safe/bridge', '/safe/entropy',
                               '/safe/i', '/safe/p', '/safe/wire', '/safe/output'],
               'protected_identities': {'source': [{'path': '/secret', 'resolved': '/secret', 'device': 3, 'inode': 8}],
                                        'aliases': [{'path': '/alias', 'resolved': '/secret', 'device': 3, 'inode': 8}]}}
    job.check_isolation(receipt, c)
    receipt['protected_identities']['aliases'][0]['inode'] = 9
    with pytest.raises(ValueError, match='identity alias'):
        job.check_isolation(receipt, c)


def test_sixteen_frame_schedule_has_one_i_and_two_p_resets(tmp_path):
    stream = tmp_path / 'stream.bin'
    stream.write_bytes(b'x' * 16)
    shifts = (0, 2, 0, 1, 0, 1, 0, 1)
    placements = [{'index': i, 'I': i == 0, 'reset': i > 0 and i % 8 == 1,
                   'qp': 3 if i == 0 else 3 + shifts[i % 8], 'offset': i, 'bytes': 1} for i in range(16)]
    manifest = {'schema': 'gvcrt-native-v1', 'source_revision': 'd0e32bfa3e8e282f9a77437c223c605858eea637',
                'frames': 16, 'fps': 30, 'original_width': 1280, 'original_height': 720,
                'canvas_width': 1920, 'canvas_height': 1088, 'base_qp': 3, 'reset_interval': 8,
                'padding_policy': 'released', 'ec_part': 1, 'model_sha256': {'I': job.I_SHA, 'P': job.P_SHA},
                'stream': {'filename': 'stream.bin', 'bytes': 16, 'sha256': hashlib.sha256(b'x' * 16).hexdigest()},
                'placements': placements}
    job.check_manifest(manifest, 16, stream)
    assert [i for i, frame in enumerate(placements) if frame['I']] == [0]
    assert [i for i, frame in enumerate(placements) if frame['reset']] == [1, 9]
    placements[9]['reset'] = False
    with pytest.raises(ValueError, match='schedule'):
        job.check_manifest(manifest, 16, stream)


def test_ordered_source_members_and_labels_are_bound(tmp_path, monkeypatch):
    import zipfile

    rows = [f'{i}.jpg,1,10,20' for i in range(188)]
    label_bytes = ('file,visibility,x,y\n' + '\n'.join(rows) + '\n').encode()
    monkeypatch.setattr(job, 'LABEL_SHA', hashlib.sha256(label_bytes).hexdigest())
    archive = tmp_path / 'data.zip'
    members = []
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('Dataset/game10/Clip3/Label.csv', label_bytes)
        for i in range(16):
            name = f'Dataset/game10/Clip3/{i}.jpg'
            data = f'jpeg-{i}'.encode()
            z.writestr(name, data)
            members.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    receipt = {'tracknet_sha256': job.DATA_SHA, 'frames': 16, 'size': [1280, 720], 'fps': 30,
               'source_yuv_sha256': 'c82d2e12e18e19f0453cb4d1b497581841c423149b153a1fc4ba2de6ff4353f4',
               'conversion_floor_psnr': 41.45716184079476, 'source_members': members}
    labels = [[1, 10.0, 20.0] for _ in range(16)]
    job.verify_source_members(archive, receipt, labels)
    members[0], members[1] = members[1], members[0]
    with pytest.raises(ValueError, match='ordered source member'):
        job.verify_source_members(archive, receipt, labels)


def test_single_point_context_uses_anchor_hull_without_extrapolation():
    metric_names = ('frame_psnr', 'full_lpips', 'full_dists', 'crop_lpips', 'crop_dists')
    def point(codec, number, byte_count, value):
        metrics = {k: (30.0 + value if k == 'frame_psnr' else 1.0 - value / 10) for k in metric_names}
        return {'codec': codec, 'point': number, 'native_bytes': byte_count,
                'zero_ball_wire_bytes': byte_count, 'mean_metrics': metrics}
    anchors = [point('svt', 1, 100, 1), point('dcvc', 2, 200, 2), point('svt', 3, 400, 3)]
    peer_metrics = anchors[1]['mean_metrics']
    result = job.rate_context(peer_metrics, anchors, 200, 200, 16)
    assert result['native']['frame_psnr']['nearest']['point'] == 2
    assert result['native']['frame_psnr']['anchor_envelope_quality_at_peer_rate'] is not None
    outside = job.rate_context(peer_metrics, anchors, 1000, 1000, 16)
    assert outside['native']['frame_psnr']['anchor_envelope_quality_at_peer_rate'] is None
    assert job.rate_context(peer_metrics, anchors, 200, 200, 4) is None


def test_projection_rejects_impossible_enclosing_worker_timing():
    sender = {'reference_stack_seconds': 0.0, 'reference_save_seconds': 0.0, 'encode_seconds_per_frame': [1]*4, 'reference_decode_seconds_per_frame': [1]*4,
              'encode_full_seconds_per_frame': [2]*4, 'reference_decode_full_seconds_per_frame': [2]*4,
              'encode_full_timing_scope': 'source_hash_read_rgb_pad_transfer_compress_packet_per_frame',
              'decode_full_timing_scope': 'packet_parse_decode_crop_cpu_transfer_per_frame'}
    receiver = {'stack_seconds': 0.0, 'save_seconds': 0.0, 'decode_seconds_per_frame': [1]*4, 'decode_full_seconds_per_frame': [2]*4,
                'decode_full_timing_scope': 'packet_parse_decode_crop_cpu_transfer_per_frame'}
    times = {'sender': 3, 'receiver': 10, 'staging': 1, 'prep': 1, 'probe': 1, 'score': 1,
             'output_copy': 1, 'validation_reserve': 1, 'release_reserve': 1}
    with pytest.raises(ValueError, match='enclosing worker'):
        job.projection(sender, receiver, times, 500)
