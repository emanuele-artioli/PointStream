import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.tennis import windows
from src.pointstream import codec, tracked_patch as ptp


def ivf(width=64, height=64, frames=2):
    result = struct.pack('<4sHH4sHHIIII', b'DKIF', 0, 32, b'AV01', width, height, 25, 1, frames, 0)
    return result + b''.join(struct.pack('<IQ', 2, i) + b'xx' for i in range(frames))


def sample(positions=None):
    if positions is None:
        positions = [[0, 0], None]
    native = {'version': 1, 'profile': 'native-av1', 'width': 64, 'height': 64,
              'fps': [25, 1], 'frames': 2, 'full_range': False, 'decoder_sha256': '0' * 64}
    meta = {'version': 1, 'profile': 'tracked-pixel-v1', 'native': native, 'numpy_version': np.__version__,
            'positions': positions, 'tile_size': 64, 'inner_box': [16, 16, 32, 32]}
    return meta, ivf(), ivf() if any(x is not None for x in positions) else b''


def test_pack_accounting_checksum_and_canonical_json():
    meta, base, patch = sample()
    data = ptp.pack(meta, base, patch)
    assert ptp.unpack(data) == (meta, base, patch)
    info = ptp.inspect(data)
    assert info['output_format'] == 'rgb24'
    assert info['native_base_bytes'] + info['patch_bytes'] + info['placement_and_framing_bytes'] == len(data)
    for broken in (data[:-1], data + b'x', data[:20] + bytes([data[20] ^ 1]) + data[21:]):
        with pytest.raises(ValueError):
            ptp.unpack(broken)
    encoded = json.dumps(meta).encode()
    body = ptp.HEADER.pack(ptp.MAGIC, len(encoded), len(base), len(patch)) + encoded + base + patch
    with pytest.raises(ValueError, match='noncanonical'):
        ptp.unpack(body + hashlib.sha256(body).digest())


@pytest.mark.parametrize('positions', [[[1, 0], None], [[34, 0], None], [[True, 0], None], [[0, 0]], [None, [0, -2]]])
def test_rejects_bad_placement(positions):
    meta, base, patch = sample()
    meta['positions'] = positions
    with pytest.raises(ValueError):
        ptp.pack(meta, base, patch)


def test_patch_fallback_and_extra_frames():
    meta, base, _ = sample([None, None])
    assert ptp.unpack(ptp.pack(meta, base))[2] == b''
    with pytest.raises(ValueError):
        ptp.pack(meta, base, ivf())
    meta, base, patch = sample()
    with pytest.raises(ValueError):
        ptp.pack(meta, base, patch + b'extra')
    with pytest.raises(ValueError):
        ptp.pack(meta, base, ivf(frames=3))
    with pytest.raises(ValueError):
        ptp.pack({**meta, 'native': {**meta['native'], 'width': 8192, 'height': 8192, 'frames': 65535}}, base, patch)


def test_limited_rgb_matches_historical_exactly_and_full_black():
    rng = np.random.default_rng(4)
    for _ in range(5):
        frame = rng.integers(0, 256, (96, 64), dtype=np.uint8)
        assert np.array_equal(ptp.yuv_to_rgb(frame, 64, 64, False), windows.yuv_to_rgb(frame, 64, 64))
    full_black = np.concatenate((np.zeros(4096, np.uint8), np.full(2048, 128, np.uint8))).reshape(96, 64)
    assert np.all(ptp.yuv_to_rgb(full_black, 64, 64, True) == 0)


def test_even_centres_and_patch_source(tmp_path):
    assert ptp.origin([0, 0], 64, 64) == [0, 0]
    assert ptp.origin([63, 63], 64, 64) == [32, 32]
    source = tmp_path / 'source.yuv'
    source.write_bytes((bytes([70]) * 4096 + bytes([90]) * 1024 + bytes([160]) * 1024) * 2)
    out = tmp_path / 'patch.yuv'
    ptp._patch_video(source, out, {'width': 64, 'height': 64, 'full_range': True}, [[0, 0], None])
    data = np.frombuffer(out.read_bytes(), np.uint8).reshape(2, 6144)
    assert data[0, 0] == 0 and np.all(data[0, :4096].reshape(64, 64)[16:48, 16:48] == 70)
    assert np.all(data[1, :4096] == 0) and np.all(data[1, 4096:] == 128)


@pytest.mark.skipif(not all(shutil.which(t) for t in ('SvtAv1EncApp', 'dav1d')), reason='native tools unavailable')
@pytest.mark.parametrize('positions,full_range', [([[25, 25], None], False), ([None, None], True)])
def test_real_native_roundtrip_no_source(tmp_path: Path, positions, full_range):
    source = tmp_path / 'source.yuv'
    source.write_bytes((bytes([80]) * 4096 + bytes([128]) * 2048) * 2)
    cfg = {'profile': 'tracked-pixel-v1', 'width': 64, 'height': 64, 'fps': [25, 1], 'frames': 2,
           'full_range': full_range, 'crf': 40, 'preset': 12, 'input_format': 'yuv420p', 'positions': positions}
    packet, output = tmp_path / 'packet.ptp', tmp_path / 'render.rgb'
    encoded = ptp.encode(source, packet, cfg, scratch=tmp_path)
    source.unlink()
    decoded = ptp.decode(packet, output, scratch=tmp_path)
    assert output.stat().st_size == 2 * 64 * 64 * 3
    assert decoded['decoded_sha256'] == encoded['decoded_sha256']
    assert decoded['rgb_frame_sha256'] == encoded['rgb_frame_sha256']
    assert (decoded['patch_bytes'] == 0) == all(x is None for x in positions)
    with pytest.raises(FileExistsError):
        ptp.decode(packet, output, scratch=tmp_path)
    assert codec.read_packet(packet).startswith(ptp.MAGIC)


def test_decode_native_rejects_extra_actual_frames(tmp_path, monkeypatch):
    meta, base, _ = sample()
    def fake_run(command, timeout):
        del timeout
        y4m = Path(command[command.index('-o') + 1])
        y4m.write_bytes(b'YUV4MPEG2 W64 H64 F25:1 Ip C420jpeg\n' + (b'FRAME\n' + bytes(6144)) * 3)
        return {'wall': 0.1, 'cpu': 0.1}
    monkeypatch.setattr(ptp.svtav1, 'run', fake_run)
    with pytest.raises(ValueError, match='extra decoded frames'):
        ptp.decode_native(base, tmp_path / 'decoded.yuv', meta['native'], 'dav1d')


def test_cli_inspect_dispatches_by_magic(tmp_path, monkeypatch, capsys):
    from src.pointstream.__main__ import main

    meta, base, patch = sample()
    path = tmp_path / 'tracked.ptp'
    path.write_bytes(ptp.pack(meta, base, patch))
    monkeypatch.setattr('sys.argv', ['pointstream', 'inspect', str(path)])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)['output_format'] == 'rgb24'
    native = tmp_path / 'native.psc'
    native.write_bytes(codec.pack(meta['native'], base))
    monkeypatch.setattr('sys.argv', ['pointstream', 'inspect', str(native)])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)['metadata']['profile'] == 'native-av1'


def test_patch_input_matches_accepted_oracle_construction(tmp_path):
    from experiments.tennis import ball_pixel_oracle as reference

    source = tmp_path / 'source.yuv'
    with source.open('wb') as handle:
        handle.truncate(48 * reference.common.FRAME_BYTES)
    frame = np.memmap(source, mode='r+', dtype=np.uint8, shape=(48, reference.common.FRAME_BYTES))
    frame[0] = np.random.default_rng(9).integers(0, 256, reference.common.FRAME_BYTES, dtype=np.uint8)
    frame.flush()
    del frame
    track = [(101, 103), (0, 0), (1279, 719)] + [None] * 45
    reference_frames, _ = reference.patch_frames(source, track, 48)
    placements = [ptp.origin(list(p), 1280, 720) if p is not None else None for p in track]
    output = tmp_path / 'patches.yuv'
    ptp._patch_video(source, output, {'width': 1280, 'height': 720, 'full_range': False}, placements)
    assert output.read_bytes() == reference_frames.tobytes()
