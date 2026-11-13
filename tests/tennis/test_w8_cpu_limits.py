"""Exercise real inherited OS limits, including proof import under the run CLI."""
import subprocess
import sys


def test_proof_import_and_child_cpu_limit() -> None:
    code = '''
import resource, subprocess, sys
resource.setrlimit(resource.RLIMIT_CPU, (650, 650))
sys.argv = ['w8_witness_cpu_proof_job', 'run']
from experiments.tennis import w8_witness_cpu_proof_job as proof
assert proof._CPU_LIMIT_BEFORE_FIXED_IMPORT == (650, 650)
assert proof._CPU_LIMIT_AFTER_FIXED_IMPORT == (650, 650)
proof.fixed.set_cpu_soft_limit(100, required_child=160)
child = "import resource; resource.setrlimit(resource.RLIMIT_CPU,(160,resource.getrlimit(resource.RLIMIT_CPU)[1])); assert resource.getrlimit(resource.RLIMIT_CPU)==(160,650)"
subprocess.run([sys.executable, '-c', child], check=True)
assert resource.getrlimit(resource.RLIMIT_CPU) == (100, 650)
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


def test_inadequate_hard_limit_rejected_before_child() -> None:
    code = '''
import resource, subprocess, sys
resource.setrlimit(resource.RLIMIT_CPU, (150, 150))
from experiments.tennis import w8_input_job as fixed
try:
    fixed.set_cpu_soft_limit(100, required_child=160)
except ValueError:
    pass
else:
    subprocess.run([sys.executable, '-c', 'raise SystemExit(7)'], check=True)
    raise AssertionError('insufficient hard limit accepted')
assert resource.getrlimit(resource.RLIMIT_CPU) == (150,150)
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


def test_repaired_input_module_is_bound_to_archive(tmp_path) -> None:
    import tarfile
    from pathlib import Path

    import pytest
    from experiments.tennis import w8_input_job as fixed, w8_witness_job as job

    stage = tmp_path / 'smoke'
    stage.mkdir()
    source = Path(fixed.__file__).resolve().parent
    bootstrap = source / 'w8_receiver_bootstrap.py'
    input_job = Path(fixed.__file__).resolve()
    policy = source / 'w8_cuda_system_leaves.py'
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(bootstrap, arcname='experiments/tennis/w8_receiver_bootstrap.py')
        tar.add(input_job, arcname='experiments/tennis/w8_input_job.py')
        tar.add(policy, arcname='experiments/tennis/w8_cuda_system_leaves.py')
    dispatch = {'code': {'snapshot_sha256': fixed.file_sha(archive)}}
    job.check_bootstrap_archive(stage, dispatch, fixed.file_sha(bootstrap), fixed.file_sha(input_job), fixed.file_sha(policy))
    with pytest.raises(ValueError):
        job.check_bootstrap_archive(stage, dispatch, fixed.file_sha(bootstrap), '0' * 64, fixed.file_sha(policy))
    with pytest.raises(ValueError):
        job.check_bootstrap_archive(stage, dispatch, fixed.file_sha(bootstrap), '', fixed.file_sha(policy))
    with pytest.raises(ValueError, match='system leaves policy differs'):
        job.check_bootstrap_archive(stage, dispatch, fixed.file_sha(bootstrap), fixed.file_sha(input_job), '0' * 64)
    wrong = tmp_path / 'wrong-input.py'
    wrong.write_text('changed input module')
    with tarfile.open(archive, 'w') as tar:
        tar.add(bootstrap, arcname='experiments/tennis/w8_receiver_bootstrap.py')
        tar.add(wrong, arcname='experiments/tennis/w8_input_job.py')
        tar.add(policy, arcname='experiments/tennis/w8_cuda_system_leaves.py')
    dispatch['code']['snapshot_sha256'] = fixed.file_sha(archive)
    with pytest.raises(ValueError, match='input module differs'):
        job.check_bootstrap_archive(stage, dispatch, fixed.file_sha(bootstrap), fixed.file_sha(input_job), fixed.file_sha(policy))
