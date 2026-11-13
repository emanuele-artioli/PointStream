import pytest

from experiments.tennis import w8_strict_source as source


def test_source_only_snapshot_rejects_changed_and_extra_files(tmp_path):
    root = tmp_path / 'snapshot'
    tennis = root / 'experiments/tennis'
    tennis.mkdir(parents=True)
    (root / 'src').mkdir()
    (root / 'tools').mkdir()
    (tennis / 'w8_strict_receiver_worker.py').write_text('value = 1\n')
    (tennis / 'protected.json').write_text('never included')
    (root / 'src' / '__init__.py').write_text('')
    out = tmp_path / 'isolated'
    manifest = source.stage_source(root, out)
    source.verify_source(out, manifest)
    assert not (out / 'experiments/tennis/protected.json').exists()
    with pytest.raises(ValueError, match='must be new'):
        source.stage_source(root, out)
    (out / 'extra.txt').write_text('unexpected')
    with pytest.raises(ValueError, match='membership'):
        source.verify_source(out, manifest)


def test_source_snapshot_rejects_symlink(tmp_path):
    root = tmp_path / 'snapshot'
    tennis = root / 'experiments/tennis'
    tennis.mkdir(parents=True)
    (root / 'src').mkdir()
    (root / 'tools').mkdir()
    target = tmp_path / 'external.py'
    target.write_text('value = 1\n')
    (tennis / 'w8_strict_receiver_worker.py').symlink_to(target)
    with pytest.raises(ValueError, match='non-regular'):
        source.stage_source(root, tmp_path / 'isolated')


def test_real_source_only_tree_imports_packet_and_generator_helpers(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    staged = tmp_path / 'source'
    manifest = source.stage_source(root, staged)
    source.verify_source(staged, manifest)
    env = {**os.environ, 'PYTHONPATH': str(staged), 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    code = ('from experiments.tennis.w8_strict_receiver_worker import preload; '
            'preload(); from experiments.tennis import k6b, w2_pose_rate, w8_payload; '
            'assert k6b.__file__.startswith(__import__("os").environ["PYTHONPATH"])')
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, check=True, timeout=20)


def test_fresh_process_rejects_checkout_shadowing_and_accepts_pinned_source(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    original = Path(__file__).resolve().parents[2]
    staged = tmp_path / 'source'
    source.stage_source(original, staged)
    env = {**os.environ, 'PYTHONPATH': str(staged), 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    code = ('from pathlib import Path; from experiments.tennis.w8_strict_receiver_worker import '
            'preload, validate_source_origins; preload(); validate_source_origins(Path(__import__("os").environ["PYTHONPATH"]))')
    good = subprocess.run([sys.executable, '-c', code], cwd=staged, env=env, capture_output=True, text=True, timeout=20)
    assert good.returncode == 0, good.stderr
    shadowed = subprocess.run([sys.executable, '-c', code], cwd=original, env=env, capture_output=True, text=True, timeout=20)
    assert shadowed.returncode != 0
    assert 'worker is not executing from authenticated source root' in shadowed.stderr
