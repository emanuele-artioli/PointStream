import json
import sys
from pathlib import Path

import pytest

from experiments.tennis import k5d_cpu_accounting as accounting


def receipt() -> dict:
    return {'schema': 1, 'method': 'linux-subreaper-waited-rusage-v1', 'phase': 'prepare', 'stage': 'full',
            'clip': 'clip', 'frames': 48, 'execution_identity': {'source': 'hash'}, 'child_exit_code': 0,
            'wrapper_pid': 1, 'command_pid': 2, 'adopted_children_waited': 0, 'children_cpu_seconds': 3.0, 'wrapper_cpu_seconds': 0.5,
            'complete_descendant_wait': True,
            'scope': 'stage command only; excludes fleet staging, publication and validation'}


def parse(row: dict) -> float:
    return accounting.read_receipt('K5D_CPU_ACCOUNTING ' + json.dumps(row), phase='prepare', stage='full',
                                   clip='clip', frames=48, identity={'source': 'hash'})


def test_accounting_rejects_forged_scope_bad_values_and_duplicate() -> None:
    data = receipt()
    assert parse(data) == 3.5
    for key, value in [('complete_descendant_wait', False), ('frames', 16), ('child_exit_code', 1),
                       ('children_cpu_seconds', float('nan')), ('wrapper_cpu_seconds', True)]:
        bad = dict(data, **{key: value})
        with pytest.raises(ValueError):
            parse(bad)
    line = 'K5D_CPU_ACCOUNTING ' + json.dumps(data)
    with pytest.raises(ValueError, match='exactly one'):
        accounting.read_receipt(line + '\n' + line, phase='prepare', stage='full', clip='clip', frames=48,
                                identity={'source': 'hash'})


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux subreaper required')
def test_subreaper_waits_for_orphan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The direct child exits before its forked child; the wrapper must adopt and reap it.
    marker = tmp_path / 'finished'
    code = ('import os,time; p=os.fork(); '
            f"p or (time.sleep(.1),open({str(marker)!r},'w').write('done'),os._exit(0))")
    row = accounting.measure([sys.executable, '-c', code], phase='prepare', stage='full', clip='clip', frames=48,
                             identity={'source': 'hash'}, timeout_seconds=5)
    assert marker.read_text() == 'done'
    assert row['adopted_children_waited'] == 1
    assert parse(json.loads(capsys.readouterr().out.split('K5D_CPU_ACCOUNTING ', 1)[1])) > 0
