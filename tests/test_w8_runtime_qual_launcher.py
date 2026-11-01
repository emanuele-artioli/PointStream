import hashlib
import json
from pathlib import Path
import subprocess
import sys


LAUNCHER = (Path(__file__).resolve().parents[1] / 'docs/experiments/w8-inputs/receiver-20261101/'
            'runtime-qual-b581-scoped-launcher.py.txt')


def test_runtime_launcher_rejects_unapproved_evidence_before_any_execution(tmp_path):
    gate = tmp_path / 'gate.json'
    gate.write_text(json.dumps({'cheap_v3': {'passed': False}, 'source_audit': {'passed': False}}))
    missing = tmp_path / 'not-created'
    result = subprocess.run([sys.executable, '-B', str(LAUNCHER), '--gate', str(gate), '--gate-sha256',
                             hashlib.sha256(gate.read_bytes()).hexdigest(), '--cheap-result', str(missing),
                             '--cheap-checkpoint', str(missing), '--cheap-audit', str(missing), '--source-audit', str(missing)],
                            capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode != 0
    assert 'passing reviewed cheap check and independent source audit required' in result.stderr
    assert not missing.exists()
