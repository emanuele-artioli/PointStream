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


def test_zstd_window_bounded_before_decompression(monkeypatch):
    import zstandard

    _, _, keypoints, _ = fixture()
    if not keypoints[rates.HEADER.size:].startswith(b'\x28\xb5\x2f\xfd'):
        pytest.skip('fixture compressor is not zstd')
    class Parameters:
        window_size = packet.MAX_KPS + 1
    monkeypatch.setattr(zstandard, 'get_frame_parameters', lambda _: Parameters())
    with pytest.raises(ValueError, match='window'):
        packet._bounded_transport(keypoints, b'KPS2', packet.MAX_KPS)


def test_adjacent_w2_runs_rejected_before_decode(monkeypatch):
    profile, base, keypoints, refs = fixture()
    meta, raw = rates.unseal(keypoints, b'KPS2')
    raw = bytearray(raw)
    # One unit: two runs (0, 4) and (6, 2); mutate second start to 4.
    assert list(raw[5:10]) == [2, 0, 4, 6, 2]
    raw[8] = 4
    bad = rates.seal(b'KPS2', meta, bytes(raw))
    def forbidden(*_):
        raise AssertionError('decoder invoked')
    monkeypatch.setattr(packet, '_decode_window', forbidden)
    with pytest.raises(ValueError, match='run geometry'):
        packet.build(profile, base, bad, refs)


@pytest.mark.parametrize('quality', [70, 40])
def test_jpeg_roundtrip_identity_and_accounting(quality):
    import cv2
    import hashlib

    profile, base, keypoints, _ = fixture()
    profile['recipe'] = copy.deepcopy(packet.JPEG_RECIPES[quality])
    profile['resources']['reference_decoder'] = hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()
    image = np.arange(784 * 512 * 3, dtype=np.uint8).reshape(784, 512, 3)
    refs = {r['reference']: rates.encode_reference(image, {'unit': 'player',
            'reference_frame': r['start'], 'reference_box': [0, 0, 100, 100],
            'pixels': 'full_prepared_conditioning_crop'}, quality) for r in profile['runs']}
    from experiments.tennis.player_requalification import array_sha
    profile['reference_rgb_sha256'] = {name: array_sha(rates.decode_reference(ref)[1])
                                       for name, ref in refs.items()}
    blob = packet.build(profile, base, keypoints, refs)
    sections, accounting, decoded = packet.unpack(blob)
    assert accounting['frame'] == len(base) + len(keypoints)
    assert accounting['reference'] == sum(map(len, refs.values()))
    assert sum(accounting.values()) == len(blob)
    assert sections['ref0'] == refs['ref0']
    np.testing.assert_array_equal(decoded['player'][0]['reference'], rates.decode_reference(refs['ref0'])[1])
    report = packet.inspect(blob, profile['resources'])
    assert report['native_validity'] == 'container_only' and not report['learned_receiver']
    assert report['required_resources']['reference_decoder'] == profile['resources']['reference_decoder']
    bad = copy.deepcopy(profile)
    bad['resources']['reference_decoder'] = '0' * 64
    with pytest.raises(ValueError, match='decoder build'):
        packet.build(bad, base, keypoints, refs)
    bad = copy.deepcopy(profile)
    bad['recipe']['nominal_quality'] = 90
    with pytest.raises(ValueError, match='recipe'):
        packet.build(bad, base, keypoints, refs)
    bad = copy.deepcopy(profile)
    bad['recipe'] = copy.deepcopy(packet.JPEG_RECIPES[40 if quality == 70 else 70])
    with pytest.raises(ValueError, match='paid reference'):
        packet.build(bad, base, keypoints, refs)


def test_jpeg_header_rejects_before_decoder(monkeypatch):
    import cv2
    import hashlib

    profile, base, keypoints, _ = fixture()
    profile['recipe'] = copy.deepcopy(packet.JPEG_RECIPES[70])
    profile['resources']['reference_decoder'] = hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()
    image = np.zeros((784, 512, 3), np.uint8)
    ref = rates.encode_reference(image, {'unit': 'player', 'reference_frame': 0,
          'reference_box': [0, 0, 100, 100], 'pixels': 'full_prepared_conditioning_crop'}, 70)
    meta, raw = rates.unseal(ref, b'REF1')
    sof = raw.index(b'\xff\xc0')
    wrong_size = bytearray(raw)
    wrong_size[sof + 5] ^= 1
    progressive = bytearray(raw)
    progressive[sof + 1] = 0xc2
    def forbidden(*_):
        raise AssertionError('image decoder invoked')
    monkeypatch.setattr(cv2, 'imdecode', forbidden)
    for bad in (bytes(wrong_size), bytes(progressive), raw[:-1], raw + b'x', raw[:sof + 3]):
        with pytest.raises(ValueError):
            packet._bounded_reference(rates.seal(b'REF1', meta, bad), True, profile['resources']['reference_decoder'])


def test_jpeg_reference_hash_bindings_and_salvaged_entropy():
    import cv2
    import hashlib

    profile, base, keypoints, _ = fixture()
    profile['recipe'] = copy.deepcopy(packet.JPEG_RECIPES[70])
    profile['resources']['reference_decoder'] = hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, (784, 512, 3), dtype=np.uint8)
    refs = {r['reference']: rates.encode_reference(image, {'unit': 'player',
            'reference_frame': r['start'], 'reference_box': [0, 0, 100, 100],
            'pixels': 'full_prepared_conditioning_crop'}, 70) for r in profile['runs']}
    from experiments.tennis.player_requalification import array_sha
    profile['reference_rgb_sha256'] = {name: array_sha(rates.decode_reference(ref)[1])
                                       for name, ref in refs.items()}
    packet.build(profile, base, keypoints, refs)
    missing = copy.deepcopy(profile)
    del missing['reference_rgb_sha256']
    with pytest.raises(ValueError, match='profile'):
        packet.build(missing, base, keypoints, refs)
    for hashes in ({'ref0': '0' * 64, 'ref1': profile['reference_rgb_sha256']['ref1']},
                   {'ref0': profile['reference_rgb_sha256']['ref0']},
                   {**profile['reference_rgb_sha256'], 'extra': '0' * 64},
                   {'wrong': profile['reference_rgb_sha256']['ref0'],
                    'ref1': profile['reference_rgb_sha256']['ref1']}):
        bad = copy.deepcopy(profile)
        bad['reference_rgb_sha256'] = hashes
        with pytest.raises(ValueError, match='reference RGB hash'):
            packet.build(bad, base, keypoints, refs)
    meta, raw = rates.unseal(refs['ref0'], b'REF1')
    sos = raw.index(b'\xff\xda')
    scan_start = sos + 2 + int.from_bytes(raw[sos + 2:sos + 4], 'big')
    truncated = raw[:scan_start] + b'\xff\xd9'
    # OpenCV may return a full-sized image even when the scan has no entropy.
    # A stricter installed decoder may reject it before the hash comparison.
    salvaged = cv2.imdecode(np.frombuffer(truncated, np.uint8), cv2.IMREAD_COLOR)
    assert salvaged is None or salvaged.shape == (784, 512, 3)
    with pytest.raises(ValueError, match='reference RGB hash mismatch|invalid reference image'):
        packet.build(profile, base, keypoints, {**refs, 'ref0': rates.seal(b'REF1', meta, truncated)})


def test_lossless_reference_packet_roundtrip_and_binding(monkeypatch):
    import cv2
    import hashlib

    from experiments.tennis.player_requalification import array_sha

    profile, base, keypoints, _ = fixture()
    profile['recipe'] = copy.deepcopy(packet.LOSSLESS_RECIPE)
    profile['resources']['reference_decoder'] = hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()
    image = np.random.default_rng(7).integers(0, 16, (784, 512, 3), dtype=np.uint8)
    refs = {r['reference']: rates.encode_reference(image, {'unit': 'player',
            'reference_frame': r['start'], 'reference_box': [0, 0, 100, 100],
            'pixels': 'full_prepared_conditioning_crop'}, lossless_auto=True) for r in profile['runs']}
    profile['reference_rgb_sha256'] = {name: array_sha(image) for name in refs}
    result = packet.build(profile, base, keypoints, refs)
    _, _, decoded = packet.unpack(result, profile)
    np.testing.assert_array_equal(decoded['player'][0]['reference'], image)
    for name, ref in refs.items():
        meta, raw = rates.unseal(ref, b'REF1')
        assert meta['codec'] in ('.png', '.webp')
        if meta['codec'] == '.webp':
            packet._webp_header(raw)
        png = rates.encode_reference(image, {k: v for k, v in meta.items() if k not in ('codec', 'quality', 'color')})
        assert len(ref) <= len(png)
    altered = copy.deepcopy(profile)
    altered['reference_rgb_sha256']['ref0'] = '0' * 64
    with pytest.raises(ValueError, match='reference RGB hash mismatch'):
        packet.build(altered, base, keypoints, refs)
    altered = copy.deepcopy(profile)
    altered['resources']['reference_decoder'] = '0' * 64
    with pytest.raises(ValueError, match='decoder build'):
        packet.build(altered, base, keypoints, refs)


def test_webp_header_rejects_invalid_payload_before_decoder(monkeypatch):
    import cv2

    image = np.zeros((784, 512, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode('.webp', image)
    assert ok
    raw = encoded.tobytes()
    packet._webp_header(raw)
    bad_size = bytearray(raw)
    bad_size[21] ^= 1
    bad_flags = bytearray(raw)
    bad_flags[24] |= 0x10
    def forbidden(*_):
        raise AssertionError('image decoder invoked')
    monkeypatch.setattr(cv2, 'imdecode', forbidden)
    for bad in (raw[:-1], raw + b'x', bytes(bad_size), bytes(bad_flags), raw[:20],
                raw[:12] + b'VP8 ' + raw[16:]):
        ref = rates.seal(b'REF1', {'codec': '.webp'}, bad)
        with pytest.raises(ValueError):
            packet._bounded_reference(ref, lossless=True)


def test_optimize_lossless_carrier_preserves_pixels_and_accounts_full_packet(monkeypatch):
    profile, base, keypoints, refs = fixture()
    image = np.random.default_rng(7).integers(0, 16, (784, 512, 3), dtype=np.uint8)
    refs = {r['reference']: rates.encode_reference(image, {'unit': 'player',
            'reference_frame': r['start'], 'reference_box': [0, 0, 100, 100],
            'pixels': 'full_prepared_conditioning_crop'}) for r in profile['runs']}
    original = packet.build(profile, base, keypoints, refs)
    optimized = packet.optimize_lossless_carrier(original)
    assert len(optimized) < len(original)
    sections, accounting, decoded = packet.unpack(optimized)
    assert sum(accounting.values()) == len(optimized)
    assert sections['base'] == base and sections['keypoints'] == keypoints
    for run in decoded['player']:
        np.testing.assert_array_equal(run['reference'], image)

    # No conversion is worthwhile when every candidate is larger than its
    # already paid original and the new profile adds header bytes.
    monkeypatch.setattr(rates, 'encode_reference', lambda *args, **kwargs: b'x' * (len(original) + 1))
    assert packet.optimize_lossless_carrier(original) == original


def test_separate_native_25fps_profile_and_mismatched_headers():
    profile, base50, keypoints, refs = fixture()
    profile25 = copy.deepcopy(profile)
    profile25['recipe'] = copy.deepcopy(packet.RECIPE_25)
    base25 = bytearray(base50)
    struct.pack_into('<II', base25, 16, 25, 1)
    encoded = packet.build(profile25, bytes(base25), keypoints, refs)
    sections, _, decoded = packet.unpack(encoded, profile25)
    assert sections['base'] == bytes(base25)
    assert len(decoded['player'][0]['t']) == 4
    with pytest.raises(ValueError, match='invalid native IVF header'):
        packet.build(profile25, base50, keypoints, refs)
    with pytest.raises(ValueError, match='invalid native IVF header'):
        packet.build(profile, bytes(base25), keypoints, refs)
    with pytest.raises(ValueError, match='unsupported recipe'):
        invalid = copy.deepcopy(profile25)
        invalid['recipe']['fps'] = [30, 1]
        packet.build(invalid, bytes(base25), keypoints, refs)


def test_fractional_native_profile_rejects_rounded_cadence():
    profile, base, keys, refs = fixture()
    profile['recipe'] = copy.deepcopy(packet.RECIPE_2997)
    n = bytearray(base)
    struct.pack_into('<II', n, 16, 30000, 1001)
    blob = packet.build(profile, bytes(n), keys, refs)
    assert packet.unpack(blob)[0]['base'] == n
    with pytest.raises(ValueError):
        packet.build(profile, base, keys, refs)
    struct.pack_into('<II', n, 16, 30, 1)
    with pytest.raises(ValueError):
        packet.build(profile, bytes(n), keys, refs)

def test_sixty_fps_packet_demands_a_matching_timebase():
    profile, base, keys, refs = fixture()
    profile['recipe'] = copy.deepcopy(packet.RECIPE_60)
    calibrated = bytearray(base)
    struct.pack_into('<II', calibrated, 16, 60, 1)
    stream = packet.build(profile, bytes(calibrated), keys, refs)
    assert packet.unpack(stream)[0]['base'] == calibrated
    for opposite in ((25, 1), (50, 1), (60000, 1001)):
        struct.pack_into('<II', calibrated, 16, *opposite)
        with pytest.raises(ValueError):
            packet.build(profile, bytes(calibrated), keys, refs)
