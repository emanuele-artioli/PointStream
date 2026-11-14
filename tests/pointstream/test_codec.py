import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from src.pointstream import codec


def sample():
    meta = dict(version=1, profile='native-av1', width=64, height=64, fps=[25, 1],
                frames=1, full_range=False, decoder_sha256='0' * 64)
    native = struct.pack('<4sHH4sHHIIII', b'DKIF', 0, 32, b'AV01', 64, 64, 25, 1, 1, 0)
    native += struct.pack('<IQ', 2, 0) + b'xx'
    return meta, native


def test_packet_accounting_and_corruption():
    meta, native = sample()
    wire = codec.pack(meta, native)
    assert codec.unpack(wire) == (meta, native)
    record = codec.inspect(wire)
    assert record['packet_bytes'] == len(wire)
    assert record['native_bytes'] + record['framing_bytes'] == len(wire)
    for bad in [wire[:-1], wire + b'x', wire[:20] + bytes([wire[20] ^ 1]) + wire[21:]]:
        with pytest.raises(ValueError):
            codec.unpack(bad)


@pytest.mark.parametrize('field,value', [('width', 63), ('frames', True), ('fps', [50, 2]),
                                        ('full_range', 0), ('decoder_sha256', 'missing')])
def test_invalid_contract(field, value):
    meta, native = sample()
    meta[field] = value
    with pytest.raises(ValueError):
        codec.pack(meta, native)


def test_native_contract_and_noncanonical_json():
    meta, native = sample()
    with pytest.raises(ValueError):
        codec.pack({**meta, 'width': 128}, native)
    with pytest.raises(ValueError):
        codec.pack(meta, native[:-1])
    encoded = json.dumps(meta).encode()
    body = codec.HEADER.pack(codec.MAGIC, len(encoded)) + encoded + native
    with pytest.raises(ValueError, match='noncanonical'):
        codec.unpack(body + hashlib.sha256(body).digest())


def test_preserve_saved_results(tmp_path):
    path = tmp_path / 'saved'
    path.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        codec.write_new(path, b'replacement')
    assert path.read_bytes() == b'original'


@pytest.mark.skipif(not all(shutil.which(t) for t in ['SvtAv1EncApp', 'dav1d']), reason='native tools absent')
def test_native_encode_fresh_decode_without_source(tmp_path: Path):
    # Infrastructure correctness only, never scientific compression evidence.
    probe = subprocess.run(['ffprobe', '-version'], capture_output=True) if shutil.which('ffprobe') else None
    if probe is None or probe.returncode:
        pytest.skip('ffprobe native runtime unavailable; full endpoint not qualified')
    source = tmp_path / 'input.yuv'
    source.write_bytes((bytes([80]) * 4096 + bytes([128]) * 2048) * 2)
    config = dict(width=64, height=64, fps=[25, 1], frames=2, full_range=False,
                  crf=40, preset=12, input_format='yuv420p')
    packet, output = tmp_path / 'sample.psc', tmp_path / 'output.yuv'
    result = codec.encode(source, packet, config, scratch=tmp_path)
    source.unlink()
    decoded = codec.decode(packet, output, scratch=tmp_path)
    assert output.stat().st_size == 64 * 64 * 3
    assert result['packet_bytes'] == packet.stat().st_size == decoded['packet_bytes']
    meta, native = codec.unpack(packet.read_bytes())
    wrong = tmp_path / 'wrong.psc'
    wrong.write_bytes(codec.pack({**meta, 'decoder_sha256': '0' * 64}, native))
    with pytest.raises(ValueError, match='identity'):
        codec.decode(wrong, tmp_path / 'bad.yuv', scratch=tmp_path)


def test_expansion_limit_and_timestamp_validation():
    meta, native = sample()
    with pytest.raises(ValueError, match='512 MiB'):
        codec.pack({**meta, 'width': 8192, 'height': 8192, 'frames': 65535}, native)
    changed = native[:36] + struct.pack('<Q', 7) + native[44:]
    with pytest.raises(ValueError, match='timestamp'):
        codec.pack(meta, changed)


def test_probe_rejects_vfr_and_coded_geometry(monkeypatch, tmp_path):
    meta, _ = sample()
    meta['frames'] = 3
    stream = dict(width=64, height=64, pix_fmt='yuv420p', color_range='tv',
                  color_space='bt709', color_transfer='bt709', color_primaries='bt709')
    response = {'streams': [stream], 'frames': [{'best_effort_timestamp_time': str(t)} for t in (0, .04, .09)]}
    monkeypatch.setattr(codec.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess([], 0, json.dumps(response), ''))
    with pytest.raises(ValueError, match='cadence'):
        codec.probe_video(tmp_path / 'source', meta, selected=True)
    stream.update(width=32, height=128)
    with pytest.raises(ValueError, match='geometry'):
        codec.probe_video(tmp_path / 'source', meta, selected=False)


def test_cli_rejects_record_output_collision(monkeypatch, tmp_path):
    from src.pointstream.__main__ import main
    source = tmp_path / 'source'
    source.write_bytes(b'saved')
    output = tmp_path / 'new'
    monkeypatch.setattr('sys.argv', ['pointstream', 'decode', str(source), str(output),
                                    '--scratch', str(tmp_path), '--record', str(output)])
    with pytest.raises(SystemExit):
        main()
    assert not output.exists()
    assert source.read_bytes() == b'saved'
