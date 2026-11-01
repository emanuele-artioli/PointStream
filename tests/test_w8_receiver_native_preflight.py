import hashlib
import json
from pathlib import Path

import pytest

from experiments.tennis import w8_receiver_finish as finish
from experiments.tennis import w8_receiver_native_preflight as preflight


def test_real_saved_packet_extracts_exact_paid_stream():
    path = Path('/private/tmp/w8-bundle-oracle-20261101-v2/window.psw8')
    if not path.exists():
        pytest.skip('optional locally recovered packet is absent')
    blob = path.read_bytes()
    assert hashlib.sha256(preflight.extract_ivf(blob)).hexdigest() == preflight.IVF_SHA
    with pytest.raises(ValueError, match='packet mismatch'):
        preflight.extract_ivf(blob[:-1] + bytes([blob[-1] ^ 1]))


def test_maps_qualifies_exact_roots_and_fails_on_deleted_files(tmp_path):
    so = tmp_path / 'libdav1d.so.7'
    so.write_bytes(b'library')
    row = f'7f00-7f01 r-xp 00000000 00:01 12 {so}\n'
    assert preflight.mapped_files(row, (tmp_path,)) == [so]
    with pytest.raises(ValueError, match='deleted'):
        preflight.mapped_files(row.rstrip() + ' (deleted)\n', (tmp_path,))
    with pytest.raises(ValueError, match='unapproved'):
        preflight.mapped_files(row, (tmp_path / 'other',))


def test_qualification_only_generation_is_never_a_candidate(tmp_path):
    unit = finish.fixed.UNITS[0]
    crop = tmp_path / 'crop.npz'
    crop.write_bytes(b'crop')
    receipt = tmp_path / 'sealed.json'
    receipt.write_text(json.dumps({'version': 1, 'unit': unit, 'packet_sha256': finish.fixed.WINDOW_SHA,
        'profile_sha256': finish.fixed.PROFILE_SHA, 'scope': 'qualification_only',
        'sidecar': {'scope': 'qualification_only', 'npz_sha256': finish.fixed.file_sha(crop)},
        'recipe': {'steps': 30, 'cfg': 2, 'context': [24, 4], 'size': [512, 784]}}))
    with pytest.raises(ValueError, match='generation seal'):
        finish._generation_seals({f'generation_seal:{unit}': receipt, f'crop:{unit}': crop})


def test_maps_accept_only_exact_loader_cache_and_no_other_etc_file(tmp_path, monkeypatch):
    cache = tmp_path / 'ld.so.cache'
    other = tmp_path / 'sensitive'
    cache.write_bytes(b'actual cache')
    other.write_bytes(b'other')
    monkeypatch.setattr(preflight, 'LD_CACHE', cache)
    row = f'7f00-7f01 r--p 00000000 00:01 12 {cache}\n'
    assert preflight.mapped_files(row, ()) == [cache]
    with pytest.raises(ValueError, match='unapproved'):
        preflight.mapped_files(row.replace(str(cache), str(other)), ())
    alias = tmp_path / 'alias'
    alias.symlink_to(cache)
    with pytest.raises(ValueError, match='unapproved'):
        preflight.mapped_files(row.replace(str(cache), str(alias)), ())


def test_native_exec_grants_must_match_independent_inventory(tmp_path, monkeypatch):
    from experiments.tennis import w8_receiver_native_bootstrap as native

    paths = {name: tmp_path / name for name in ('ffmpeg', 'ld-linux', 'libdav1d.so.7', 'ld.so.cache')}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    hashes = {path: native.file_sha(path) for path in paths.values()}
    monkeypatch.setattr(native, 'FFMPEG_SHA', hashes[paths['ffmpeg']])
    receipt = {'passed': True, 'ffmpeg_sha256': native.FFMPEG_SHA, 'ivf_sha256': native.IVF_SHA,
               'yuv_sha256': native.EXPECTED_YUV, 'libdav1d_version': '1.5.4',
               'ffmpeg_path': str(paths['ffmpeg']), 'elf_interpreter': str(paths['ld-linux']),
               'libdav1d_path': str(paths['libdav1d.so.7']),
               'mapped_files': {str(path): {'sha256': value} for path, value in hashes.items()}}
    file = tmp_path / 'inventory.json'
    file.write_text(json.dumps(receipt))
    receipt_sha = native.file_sha(file)
    libraries = {paths['libdav1d.so.7']: hashes[paths['libdav1d.so.7']],
                 paths['ld.so.cache']: hashes[paths['ld.so.cache']]}
    native.validate_inventory(file, receipt_sha, paths['ffmpeg'], paths['ld-linux'], hashes[paths['ld-linux']], libraries)
    with pytest.raises(ValueError, match='differs from inventory'):
        native.validate_inventory(file, receipt_sha, paths['ffmpeg'], paths['ld-linux'], hashes[paths['ld-linux']],
                                  {paths['libdav1d.so.7']: hashes[paths['libdav1d.so.7']]})
    libraries[paths['ld.so.cache']] = '0' * 64
    with pytest.raises(ValueError, match='differs from inventory'):
        native.validate_inventory(file, receipt_sha, paths['ffmpeg'], paths['ld-linux'], hashes[paths['ld-linux']], libraries)
    with pytest.raises(ValueError, match='receipt identity'):
        native.validate_inventory(file, '0' * 64, paths['ffmpeg'], paths['ld-linux'], hashes[paths['ld-linux']], libraries)
