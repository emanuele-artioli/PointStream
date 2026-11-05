"""One self-pidfd capability check; helper bytes arrive on stdin."""
import hashlib
import json
import os
import sys
import time
import traceback

start = time.monotonic()
cpu_start = time.process_time()
record = {'schema': 1, 'scope': 'self pidfd signal-zero capability only',
          'pid': os.getpid(), 'pgid': os.getpgrp(), 'interpreter': sys.executable,
          'environment_sha256': '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661',
          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
          'torch_before': 'torch' in sys.modules, 'status': 'failed'}
exit_code = 1
try:
    raw = sys.stdin.buffer.read()
    record['helper_sha256'] = hashlib.sha256(raw).hexdigest()
    if record['helper_sha256'] != '21ed2775eaafd2f55c19cc65c91c2a4f73c042b4bfc7a616bbcde36b99e6bd8e':
        raise ValueError('helper SHA differs')
    if record['cuda_visible_devices'] != '' or record['torch_before']:
        raise ValueError('CUDA must be hidden and Torch absent')
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(allowed)})
    namespace = {'__name__': 'pidfd_capability_probe'}
    exec(compile(raw, '<sha-bound-helper>', 'exec'), namespace)
    record['capability'] = namespace['Pidfd']().self_check()
    if 'torch' in sys.modules:
        raise ValueError('Torch was imported')
    record['status'] = 'passed'
    exit_code = 0
except Exception:
    record['traceback'] = traceback.format_exc()
finally:
    record['torch_after'] = 'torch' in sys.modules
    try:
        record['affinity'] = sorted(os.sched_getaffinity(0))
    except Exception:
        record['affinity'] = None
    record['cpu_seconds'] = time.process_time() - cpu_start
    record['wall_seconds'] = time.monotonic() - start
    record['exit_code'] = exit_code
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
sys.exit(exit_code)
