import hashlib
from pathlib import Path

import pytest

from experiments.tennis import w7_mkv_profile as profile


def probes():
    frames = [{'best_effort_timestamp': i} for i in range(240)]
    source = {'streams': [{'codec_name': 'av1', 'width': 960, 'height': 544, 'pix_fmt': 'yuv420p',
                           'r_frame_rate': '50/1', 'time_base': '1/50', 'color_range': 'tv'}], 'frames': frames}
    dest = {'streams': [{'codec_name': 'av1', 'width': 960, 'height': 544, 'pix_fmt': 'yuv420p',
                         'r_frame_rate': '50/1', 'time_base': '1/1000', 'color_range': 'tv'}],
            'frames': [{'best_effort_timestamp': i * 20} for i in range(240)]}
    return source, dest


def test_explicit_profile_retains_unknown_original_and_requires_exact_time():
    original, dest = probes()
    meta = profile.probe_pair(original, dest)
    assert meta['original_color_metadata']['color_space'] is None
    assert 'not inferred' in meta['interpretation'] and meta['last'] == '239/50'
    dest['frames'][100]['best_effort_timestamp'] += 1
    with pytest.raises(ValueError, match='timestamps'):
        profile.probe_pair(original, dest)


def test_whole_wire_size_and_historical_native_hash_must_pass():
    original, dest = probes()
    wire = b'x' * profile.MAX_BYTES['med000']
    report = {'case': 'med000', 'status': 'passed', 'scope': 'new transport sensitivity only; no quality or old gate result',
              'metadata': profile.probe_pair(original, dest), 'wire_bytes': len(wire), 'byte_limit': len(wire),
              'wire_kbps': len(wire) * 8 / 4.8 / 1000,
              'wire_sha256': hashlib.sha256(wire).hexdigest(),
              'input_sha256': profile.old.CASES['med000']['stream'], 'ffmpeg_sha256': profile.old.FFMPEG_SHA,
              'ffprobe_sha256': profile.old.FFPROBE_SHA, 'dav1d_sha256': profile.DAV1D_SHA,
              'original_yuv_sha256': profile.old.CASES['med000']['decoded'],
              'delivery_yuv_sha256': profile.old.CASES['med000']['decoded']}
    profile.check_result(report, original, dest, wire, 'med000')
    report['delivery_yuv_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='native decode'):
        profile.check_result(report, original, dest, wire, 'med000')
    report['delivery_yuv_sha256'] = profile.old.CASES['med000']['decoded']
    wire += b'x'
    report['wire_bytes'] = len(wire)
    report['wire_kbps'] = len(wire) * 8 / 4.8 / 1000
    report['wire_sha256'] = hashlib.sha256(wire).hexdigest()
    with pytest.raises(ValueError, match='wire'):
        profile.check_result(report, original, dest, wire, 'med000')


def test_decoder_receives_only_new_container_via_internal_ivf():
    cmd = profile.commands('ffmpeg', 'ffprobe', 'dav1d', Path('/input.ivf'), Path('/scratch'))
    assert cmd['mux'][cmd['mux'].index('-c:v') + 1] == 'copy'
    assert cmd['extract_ivf'][cmd['extract_ivf'].index('-i') + 1] == '/scratch/delivered.mkv'
    assert cmd['decode_delivery'][cmd['decode_delivery'].index('-i') + 1] == '/scratch/decoder-scratch.ivf'
    assert '/input.ivf' not in cmd['decode_delivery']


def test_spec_binds_original_inputs_and_rejects_expired_deadline():
    with pytest.raises(ValueError, match='future'):
        profile.make_spec('2020-01-01T00:00:00+00:00')
    spec = profile.make_spec('2099-01-01T00:00:00+00:00')
    assert (spec['smoke']['seconds'], spec['full']['seconds'], spec['budget_seconds']) == (35, 35, 90)
    assert spec['device'] == 'cpu' and spec['single_cpu_campaign']
    assert {i['name']: i['sha256'] for i in spec['staged_inputs']} == {k: v['archive'] for k, v in profile.old.CASES.items()}


def test_delivered_color_cannot_contradict_declared_profile():
    original, dest = probes()
    dest['streams'][0]['color_space'] = 'bt2020nc'
    with pytest.raises(ValueError, match='contradicts'):
        profile.probe_pair(original, dest)
    dest['streams'][0]['color_space'] = 'bt709'
    dest['streams'][0]['color_range'] = 'pc'
    with pytest.raises(ValueError, match='limited range'):
        profile.probe_pair(original, dest)
