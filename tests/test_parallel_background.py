from pathlib import Path
import struct
import json

import numpy as np
import pytest

from experiments.tennis.parallel_background import compare, header_decode, inspect_stream, packet_header, pack_residual, reconstruct, reconstruct_packet


def test_residual_roundtrip_covers_extreme_signed_range():
    source = np.arange(256, dtype=np.uint8)[:, None].repeat(256, axis=1)
    reference = source.T.copy()
    decoded = reconstruct(reference, pack_residual(source, reference))
    assert np.max(np.abs(decoded.astype(int) - source.astype(int))) <= 1
    assert int(decoded[0, 255]) == 0
    assert int(decoded[255, 0]) == 254
    with pytest.raises(ValueError):
        reconstruct(reference.astype(float), source)


def test_header_pays_reference_and_residual_identity(tmp_path: Path):
    ref, residual = tmp_path / 'ref', tmp_path / 'residual'
    ref.write_bytes(b'reference')
    residual.write_bytes(b'residual')
    header = packet_header(ref, residual, 240)
    assert b'identity' in header and b'"frames":240' in header
    residual.write_bytes(b'changed')
    assert packet_header(ref, residual, 240) != header


def test_one_point_must_win_both_metrics_without_extrapolation():
    curves = {m: [(10., -.3), (30., -.1)] for m in ('lpips', 'dists')}
    assert compare({'lpips': .19, 'dists': .19}, curves, 20)['verdict'] == 'PASS'
    assert compare({'lpips': .19, 'dists': .21}, curves, 20)['verdict'] == 'FAIL'
    assert compare({'lpips': .19, 'dists': .1995}, curves, 20)['verdict'] == 'UNRESOLVED'
    assert compare({'lpips': .19, 'dists': .19}, curves, 31)['verdict'] == 'UNRESOLVED/ANCHOR-COVERAGE'


def test_native_stream_rejects_reordered_timestamps_and_wrong_geometry(tmp_path: Path):
    path = tmp_path / 'stream.ivf'
    header = b'DKIF' + struct.pack('<HH4sHHIIII', 0, 32, b'AV01', 960, 544, 25, 1, 2, 0)
    path.write_bytes(header + struct.pack('<IQ', 1, 0) + b'x' + struct.pack('<IQ', 1, 1) + b'y')
    assert inspect_stream(path, 2)['timestamps'] == [0, 1]
    path.write_bytes(header + struct.pack('<IQ', 1, 0) + b'x' + struct.pack('<IQ', 1, 0) + b'y')
    with pytest.raises(ValueError, match='reordered'):
        inspect_stream(path, 2)
    with pytest.raises(ValueError, match='geometry'):
        inspect_stream(path, 1)


def test_nonfinite_scores_cannot_be_reported_as_unresolved():
    with pytest.raises(ValueError, match='finite'):
        compare({'lpips': float('nan'), 'dists': .2}, {}, 20)


def test_paid_decoder_rejects_changed_contract_and_uses_serialized_parameters(tmp_path: Path):
    ref, residual = tmp_path / 'ref', tmp_path / 'residual'
    ref.write_bytes(b'reference')
    residual.write_bytes(b'residual')
    raw = packet_header(ref, residual, 240)
    header = header_decode(raw, ref, residual, 240)
    pixels = np.full((816, 960), 100, np.uint8)
    delta = np.full_like(pixels, 150)
    assert np.all(reconstruct_packet(pixels, delta, header) == 144)
    for key, changed in [('width', 480), ('transform', 'free-homography'),
                         ('residual_decode', {'scale': 1, 'offset': 0, 'clip': [0, 255]}),
                         ('reference_sha256', 'wrong')]:
        bad = dict(header, **{key: changed})
        with pytest.raises(ValueError, match='contract'):
            header_decode(json.dumps(bad).encode(), ref, residual, 240)
    residual.write_bytes(b'mutated')
    with pytest.raises(ValueError, match='identity'):
        header_decode(raw, ref, residual, 240)
    with pytest.raises(ValueError, match='planes'):
        reconstruct_packet(pixels[:100], delta[:100], header)
