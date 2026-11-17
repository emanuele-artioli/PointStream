"""Synthetic contract tests
    these bytes are not research evidence."""
import copy
import struct

import numpy as np
import pytest

from experiments.tennis import common_player_packet as packet
from experiments.tennis import k6b_rate_streams as rates
from experiments.tennis import w8_payload as wire


def fixture():
    profile = {'mode': 'players', 'recipe': copy.deepcopy(packet.RECIPE), 'resources': {k: 'a' * 64 for k in packet.RESOURCES},
               'coded_size': [640, 480], 'canvas_size': [1920, 1080], 'cuts': [6], 'order': ['player'],
               'runs': [{'unit': 'player', 'start': 0, 'end': 4, 'reference': 'ref0'},
                        {'unit': 'player', 'start': 6, 'end': 8, 'reference': 'ref1'}]}
    base = struct.pack('<4sHH4sHHIIII', b'DKIF', 0, 32, b'AV01', 640, 480, 50, 1, 240, 0)
    base += b''.join(struct.pack('<IQ', 1, t) + b'x' for t in range(240))
    t = np.array([0, 1, 2, 3, 6, 7])
    kp = np.zeros((len(t), 17, 3))
    kp[:, :, :2] = 64
    kp[:, :, 2] = 1
    keypoints = rates.encode_keypoints_kps2({'native_frames': 240}, [{'name': 'player', 't': t, 'kp': kp}])
    image = np.zeros((784, 512, 3), dtype=np.uint8)
    refs = {r['reference']: rates.encode_reference(image, {'unit': 'player', 'reference_frame': r['start'],
            'reference_box': [0, 0, 100, 100], 'pixels': 'full_prepared_conditioning_crop'}) for r in profile['runs']}
    return profile, base, keypoints, refs


def test_roundtrip_accounting_and_runs():
    profile, base, keypoints, refs = fixture()
    packet.verify_resources(profile, profile['resources'].copy())
    with pytest.raises(ValueError):
        packet.verify_resources(profile, {})
    blob = packet.build(profile, base, keypoints, refs)
    sections, accounting, decoded = packet.unpack(blob, profile)
    assert sum(accounting.values()) == len(blob)
    assert accounting['frame'] == len(base) + len(keypoints)
    assert accounting['reference'] == sum(map(len, refs.values()))
    assert accounting['match'] == 0
    assert accounting['window'] == len(blob) - len(base) - len(keypoints) - sum(map(len, refs.values()))
    assert sections['keypoints'] == keypoints
    runs = decoded['player']
    assert [(r['start'], r['end']) for r in runs] == [(0, 4), (6, 8)]
    assert packet.pose_for_tick(runs[0], 0).shape == (784, 512, 3)
    assert runs[1]['seed'] == runs[0]['seed'] + 1
    active = packet.coverage(decoded, profile['order'])
    assert len(active) == 240 and active[0] == ['player'] and active[4] == []
    assert set(range(240)) - set(np.concatenate([r['t'] for r in runs])) >= {4, 5, 8, 239}


def test_corruption_and_closure():
    profile, base, keypoints, refs = fixture()
    blob = packet.build(profile, base, keypoints, refs)
    for bad in (blob[:-1], blob + b'x', blob[:-1] + bytes([blob[-1] ^ 1])):
        with pytest.raises(ValueError):
            packet.unpack(bad, profile)
    changed = copy.deepcopy(profile)
    changed['resources']['common_lora'] = 'b' * 64
    with pytest.raises(ValueError):
        packet.unpack(blob, changed)
    with pytest.raises(ValueError):
        packet.build(profile, base, keypoints, {**refs, 'extra': b'x'})
    with pytest.raises(ValueError):
        packet.build(profile, base, keypoints, {k: v for k, v in refs.items() if k != 'ref0'})
    wrong_roles = wire.pack(profile, [('base', 'match', base), ('keypoints', 'frame', keypoints)] +
                            [(n, 'reference', refs[n]) for n in sorted(refs)])
    with pytest.raises(ValueError):
        packet.unpack(wrong_roles, profile)


def test_cuts_gaps_and_ivf_checks():
    profile, base, keypoints, refs = fixture()
    altered = copy.deepcopy(profile)
    altered['cuts'] = [2]
    with pytest.raises(ValueError):
        packet.build(altered, base, keypoints, refs)
    altered = copy.deepcopy(profile)
    altered['runs'][1]['start'] = 5
    with pytest.raises(ValueError):
        packet.build(altered, base, keypoints, refs)
    bad = bytearray(base)
    bad[36] = 9
    with pytest.raises(ValueError):
        packet.build(profile, bytes(bad), keypoints, refs)
    with pytest.raises(ValueError):
        packet.build(profile, base + b'x', keypoints, refs)


def test_native_only_and_profile_rejection():
    _, base, keypoints, refs = fixture()
    profile = {'mode': 'native_only', 'recipe': copy.deepcopy(packet.NATIVE_RECIPE),
               'resources': {'native_decoder': 'a' * 64}, 'coded_size': [640, 480], 'canvas_size': [1920, 1080],
               'cuts': [2], 'runs': [], 'order': []}
    blob = packet.build(profile, base, None, {})
    _, accounting, decoded = packet.unpack(blob, profile)
    assert accounting['frame'] == len(base) and accounting['match'] == 0
    assert packet.coverage(decoded, []) == [[] for _ in range(240)]
    with pytest.raises(ValueError):
        packet.build(profile, base, keypoints, refs)
    profile['recipe']['native_frames'] = 239
    with pytest.raises(ValueError):
        packet.build(profile, base, None, {})


def test_cut_coverage_with_separate_units():
    profile, base, _, _ = fixture()
    profile['cuts'] = [4]
    profile['order'] = ['before', 'after']
    profile['runs'] = [{'unit': 'before', 'start': 0, 'end': 4, 'reference': 'before_ref'},
                       {'unit': 'after', 'start': 4, 'end': 8, 'reference': 'after_ref'}]
    kp = np.zeros((4, 17, 3))
    kp[:, :, :2] = 64
    kp[:, :, 2] = 1
    keypoints = rates.encode_keypoints_kps2({'native_frames': 240}, [
        {'name': 'before', 't': np.arange(4), 'kp': kp},
        {'name': 'after', 't': np.arange(4, 8), 'kp': kp},
    ])
    image = np.zeros((784, 512, 3), dtype=np.uint8)
    refs = {r['reference']: rates.encode_reference(image, {'unit': r['unit'],
            'reference_frame': r['start'], 'reference_box': [0, 0, 100, 100],
            'pixels': 'full_prepared_conditioning_crop'}) for r in profile['runs']}
    _, _, decoded = packet.unpack(packet.build(profile, base, keypoints, refs), profile)
    active = packet.coverage(decoded, profile['order'])
    assert active[3] == ['before'] and active[4] == ['after']


def test_empty_individual_track_rejected(monkeypatch):
    profile, base, keypoints, refs = fixture()
    _, units = rates.decode_keypoints(keypoints)
    units[0]['t'] = np.array([], dtype=np.int64)
    units[0]['kp'] = np.empty((0, 17, 3))
    monkeypatch.setattr(packet, '_decode_window', lambda *_: units)
    with pytest.raises(ValueError, match='invalid track'):
        packet.build(profile, base, keypoints, refs)


def test_coded_size_and_output_canvas_are_independent():
    profile, base, keypoints, refs = fixture()
    # IVF header only: synthetic frame payloads do not establish AV1 decodability.
    base = bytearray(base)
    struct.pack_into('<HH', base, 12, 480, 272)
    profile['coded_size'] = [480, 272]
    profile['canvas_size'] = [1920, 1080]
    blob = packet.build(profile, bytes(base), keypoints, refs)
    packet.unpack(blob, profile)
    mismatch = copy.deepcopy(profile)
    mismatch['coded_size'] = [1920, 1080]
    with pytest.raises(ValueError, match='IVF header'):
        packet.build(mismatch, bytes(base), keypoints, refs)
    mismatch = copy.deepcopy(profile)
    mismatch['canvas_size'] = [1920, 0]
    with pytest.raises(ValueError, match='dimensions'):
        packet.build(mismatch, bytes(base), keypoints, refs)
    mismatch = copy.deepcopy(profile)
    mismatch['recipe']['base_to_canvas'] = 'nearest'
    with pytest.raises(ValueError, match='recipe'):
        packet.build(mismatch, bytes(base), keypoints, refs)
    mismatch = copy.deepcopy(profile)
    mismatch['canvas_size'] = [1280, 720]
    with pytest.raises(ValueError, match='profile identity'):
        packet.unpack(blob, mismatch)


@pytest.mark.parametrize('name', ['GEN_W', 'GEN_H', 'CROP_MARGIN', 'SIZE_PERCENTILE', 'CENTRE_SMOOTH'])
def test_geometry_helper_drift_rejected(monkeypatch, name):
    profile, base, keypoints, refs = fixture()
    monkeypatch.setattr(packet.k6b, name, getattr(packet.k6b, name) + 1)
    with pytest.raises(ValueError, match='geometry helper'):
        packet.build(profile, base, keypoints, refs)


def test_confidence_threshold_drift_rejected(monkeypatch):
    profile, base, keypoints, refs = fixture()
    monkeypatch.setattr(packet.k6b.k4, 'KP_CONF', 0.6)
    with pytest.raises(ValueError, match='geometry helper'):
        packet.build(profile, base, keypoints, refs)


def test_embedded_profile_and_inspection():
    profile, base, keypoints, refs = fixture()
    blob = packet.build(profile, base, keypoints, refs)
    sections, accounting, decoded = packet.unpack(blob)
    assert sections['base'] == base and len(decoded['player']) == 2
    report = packet.inspect(blob, profile['resources'])
    assert report['profile'] == profile and report['required_resources'] == profile['resources']
    assert report['total_bytes'] == sum(accounting.values()) == len(blob)
    assert report['component_bytes']['keypoints'] == len(keypoints)
    assert report['role_bytes'] == accounting
    assert report['coverage']['ticks'] == 240 and report['coverage']['base_only_ticks'] == 234
    assert report['native_validity'] == 'container_only' and report['learned_receiver'] is False
    with pytest.raises(ValueError, match='installed resource'):
        packet.inspect(blob, {})
    changed = copy.deepcopy(profile)
    changed['canvas_size'] = [1280, 720]
    with pytest.raises(ValueError, match='profile identity'):
        packet.unpack(blob, changed)


def test_native_only_embedded_profile():
    _, base, _, _ = fixture()
    profile = {'mode': 'native_only', 'recipe': copy.deepcopy(packet.NATIVE_RECIPE),
               'resources': {'native_decoder': 'a' * 64}, 'coded_size': [640, 480],
               'canvas_size': [1920, 1080], 'cuts': [], 'runs': [], 'order': []}
    blob = packet.build(profile, base, None, {})
    assert packet.unpack(blob)[2] == {}
    assert packet.inspect(blob)['coverage']['base_only_ticks'] == 240


def test_embedded_header_rejection():
    profile, base, keypoints, refs = fixture()
    blob = packet.build(profile, base, keypoints, refs)
    size = struct.unpack_from('>I', blob, 5)[0]
    raw = blob[9:9 + size]
    # Whitespace makes JSON valid but noncanonical; oversized lengths are bounded.
    noncanonical = blob[:5] + struct.pack('>I', size + 1) + b' ' + raw + blob[9 + size:]
    for bad in (noncanonical, blob[:5] + struct.pack('>I', wire.MAX_HEADER + 1),
                b'bad!!' + blob[5:], blob[:9 + size - 1]):
        with pytest.raises(ValueError):
            packet.unpack(bad)


@pytest.mark.parametrize('index,value', [(1, 1), (1, 2), (1, 4), (2, 8), (3, 3), (4, 17),
                                         (5, 0), (5, 241), (6, 241), (7, 0), (7, 241)])
def test_inner_w2_rejected_before_decoder(monkeypatch, index, value):
    profile, base, keypoints, refs = fixture()
    meta, raw = rates.unseal(keypoints, b'KPS2')
    raw = bytearray(raw)
    raw[index] = value
    bad = rates.seal(b'KPS2', meta, bytes(raw))
    def forbidden(*_):
        raise AssertionError('unbounded decoder invoked')
    monkeypatch.setattr(packet, '_decode_window', forbidden)
    with pytest.raises(ValueError):
        packet.build(profile, base, bad, refs)


def test_transport_caps_before_decode(monkeypatch):
    profile, base, keypoints, refs = fixture()
    def forbidden(*_):
        raise AssertionError('decoder invoked')
    monkeypatch.setattr(packet, '_decode_window', forbidden)
    bad = bytearray(keypoints)
    struct.pack_into('>I', bad, 4, packet.MAX_KPS + 1)
    with pytest.raises(ValueError, match='size'):
        packet.build(profile, base, bytes(bad), refs)
    # Actual expansion exceeds a forged small declared length.
    import hashlib
    import zlib
    raw = b'x' * 10000
    coded = zlib.compress(raw)
    bad = rates.HEADER.pack(b'KPS2', 20, len(coded), hashlib.sha256(raw).digest()) + coded
    with pytest.raises(ValueError):
        packet.build(profile, base, bad, refs)


def test_reference_cap_and_png_dimensions():
    profile, base, keypoints, refs = fixture()
    bad = bytearray(refs['ref0'])
    struct.pack_into('>I', bad, 4, packet.MAX_REF + 1)
    with pytest.raises(ValueError, match='size'):
        packet.build(profile, base, keypoints, {**refs, 'ref0': bytes(bad)})
    meta, raw = rates.unseal(refs['ref0'], b'REF1')
    raw = bytearray(raw)
    struct.pack_into('>I', raw, 16, 8192)
    with pytest.raises(ValueError, match='PNG header'):
        packet.build(profile, base, keypoints, {**refs, 'ref0': rates.seal(b'REF1', meta, bytes(raw))})


def test_exp_golomb_prefix_is_bounded(monkeypatch):
    monkeypatch.setattr(packet.w2.Decoder, 'bit', lambda *_: 1)
    decoder = packet._BoundedDecoder(b'')
    with pytest.raises(ValueError, match='Exp-Golomb'):
        packet.w2._exp_golomb(decoder, None)


def test_zstd_forged_expansion_is_rejected():
    import hashlib
    import zstandard

    raw = b'x' * 10000
    coded = zstandard.ZstdCompressor().compress(raw)
    blob = rates.HEADER.pack(b'KPS2', 20, len(coded), hashlib.sha256(raw).digest()) + coded
    with pytest.raises(ValueError, match='content size'):
        packet._bounded_transport(blob, b'KPS2', packet.MAX_KPS)
