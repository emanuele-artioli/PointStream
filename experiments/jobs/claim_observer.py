"""Bounded, read-only claim observer. No claim is released or process signalled.

Success bounds a claim interval, conditional on independently verified NFS visibility.
It does not measure GPU utilization or replace the supervisor's CPU accounting.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
import multiprocessing
import os
import resource
import socket
import stat
import time
from pathlib import Path
from typing import Any

from experiments.jobs import inbox
from experiments.jobs.claims import get_process_start_time, sanitize_uuid

TERMINAL = {'complete', 'failed', 'cancelled', 'contended', 'budget_exhausted', 'interrupted', 'expired'}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def bounded_bytes(path: Path, limit: int = 8 * 1024 * 1024) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('input is not a bounded regular file')
        data = f.read(limit + 1)
        if len(data) > limit:
            raise ValueError('input exceeds limit')
        return data


def read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(bounded_bytes(path))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'not a JSON object: {path}')
    return value


def append(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('ab', buffering=0) as f:
        f.write(data)
        os.fsync(f.fileno())


def observe(config: dict[str, Any], binding: dict[str, Any] | None, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate recorded samples; missing, ambiguous or interrupted evidence is unknown."""
    unknown: dict[str, Any] = {'status': 'unknown', 'reason': 'incomplete or ambiguous observation'}
    try:
        if not binding or binding['nonce'] != config['nonce'] or binding['spec_sha256'] != config['spec_sha256'] or binding['source_sha256'] != config['source_sha256'] or binding.get('inputs_sha256') != config['inputs_sha256'] or binding.get('management_host') != config['management_host'] or binding.get('management_boot') != config['boot']:
            return unknown
        job = binding['job_id']
        expected_dir = str(Path(config['inbox']) / job / 'run')
        if not job or '/' in job or job in ('.', '..') or not samples:
            return unknown
        if any(s.get('error') or s['boot'] != config['boot'] for s in samples):
            return unknown
        times = [s['start_ns'] for s in samples]
        if any(b <= a or b-a > config['max_gap_ns'] for a, b in zip(times, times[1:])):
            return unknown
        if any(s['end_ns'] < s['start_ns'] or s['end_ns']-s['start_ns'] > config['max_gap_ns'] for s in samples):
            return unknown
        seen = [i for i,s in enumerate(samples) if s.get('gpu_job_dir') == expected_dir]
        if not seen or seen[0] == 0:
            return unknown
        first = seen[0]
        prior = samples[first-1]
        if prior.get('gpu_state') != 'absent':
            return unknown
        token = samples[first]['gpu_token_sha256']
        if not token or any(samples[i].get('gpu_token_sha256') != token for i in seen):
            return unknown
        # No gap in the period in which our claim is observed; a replacement means
        # release, and subsequent reappearance makes the whole measurement ambiguous.
        last = seen[-1]
        if seen != list(range(first,last+1)) or last + 1 >= len(samples):
            return unknown
        tail = samples[last+1:]
        if any(s.get('gpu_state') not in ('absent', 'other') or s.get('gpu_token_sha256') == token for s in tail):
            return unknown
        end = tail[-1]
        joined = next((s for s in samples[first:last+1] if s.get('job_join') is True), None)
        if (not joined or joined.get('cpu_token_sha256') is None or not joined.get('leader_live')
                or not joined.get('supervisor_live') or joined.get('cpu_token_present') is not True):
            return unknown
        if (end.get('cpu_token_sha256') != joined['cpu_token_sha256']
                or any(end.get(k) != joined.get(k) for k in ('leader_pid', 'leader_birth', 'supervisor_pid', 'supervisor_birth', 'pgid'))
                or end.get('cpu_token_present') is not False or end.get('group_absent') is not True
                or end.get('supervisor_absent') is not True or end.get('terminal') not in TERMINAL
                or end.get('spec_sha256') != config['spec_sha256'] or end.get('source_sha256') != config['source_sha256']
                or not isinstance(end.get('published_ns'), int)
                or end['published_ns'] <= config['ready_wall_ns']):
            return unknown
        return {'status':'bounded', 'job_id':job, 'gpu_token_sha256':token,
                'start_lower_ns':prior['start_ns'], 'release_upper_ns':end['end_ns'],
                'claim_upper_seconds':(end['end_ns']-prior['start_ns'])/1e9,
                'scope':'reservation interval only; conditional on independently verified NFS visibility',
                'cpu_accounting':'not established by this observer'}
    except (KeyError, TypeError, ValueError):
        return unknown



def local_mount(path: Path) -> bool:
    """Accept only a separately mounted local Linux filesystem."""
    target = str(path.resolve(strict=True))
    records = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, _, right = line.partition(' - ')
        parts, tail = left.split(), right.split()
        if len(parts) < 5 or not tail:
            continue
        mount = parts[4].replace('\\040', ' ')
        if target == mount or target.startswith(mount.rstrip('/') + '/'):
            records.append((len(mount), mount, tail[0]))
    if not records:
        return False
    _, mount, kind = max(records)
    return mount != '/' and kind in ('ext4', 'xfs', 'btrfs')


def sample_once(cfg: dict[str, Any], binding: dict[str, Any] | None) -> dict[str, Any]:
    s: dict[str, Any] = {}
    claim = read(Path(cfg['gpu_claim']))
    if claim is None:
        # A lock directory without its record may be a partial acquisition.
        if Path(cfg['gpu_claim']).parent.exists():
            raise ValueError('GPU claim lock without record')
        s['gpu_state'] = 'absent'
    else:
        if claim['host'] != cfg['host_fqdn'] or claim['device_uuid'] != cfg['gpu_uuid'] or claim['sharing'] != 'exclusive':
            raise ValueError('claim device identity mismatch')
        s.update(gpu_state='other', gpu_job_dir=claim['job_dir'], gpu_token_sha256=digest(claim['token']),
                 gpu_owner_pid=claim['pid'], gpu_owner_birth=claim['proc_start_time'])
    if binding is None:
        return s
    job = binding['job_id']
    if not inbox.fleet.JOB_ID_RE.fullmatch(job):
        raise ValueError('invalid job ID')
    run = Path(cfg['inbox']) / job / 'run'
    record, status, supervisor = read(run/'claim.json'), read(run/'status.json'), read(run/'supervisor.json')
    ready, spec, request = read(run.parent/'ready.json'), read(run.parent/'spec.json'), read(run/'request.json')
    cpu = read(Path(cfg['cpu_claim']))
    if any(x is None for x in (record, status, supervisor, ready, spec, request)):
        return s
    assert record is not None and status is not None and supervisor is not None and ready is not None and spec is not None and request is not None
    if (spec.get('hosts') != ['gpu6'] or spec.get('single_cpu_campaign') is not True
            or spec.get('cpu_threads') != 1 or spec.get('gpu_sharing', {}).get('mode') != 'exclusive'
            or inbox.digest(spec) != cfg['spec_sha256'] or ready['spec_sha256'] != cfg['spec_sha256']
            or ready['source_sha256'] != cfg['source_sha256'] or inbox.digest(spec['inputs']) != binding['inputs_sha256']):
        raise ValueError('canonical spec, source or inputs differ')
    control = request['fleet_control']
    claims = request['claims']
    if (control['root'] != cfg['control_root'] or control['files'] != cfg['control_files']
            or claims['claims_dir'] != cfg['claims_dir'] or claims['gpu'] != cfg['gpu_uuid']
            or claims['cpu_threads'] != 1 or request['gpu_sharing'] != {'mode': 'exclusive'}):
        raise ValueError('actual supervisor request differs from pinned control and claims')
    expected_gpu = Path(cfg['claims_dir']) / 'devices' / (cfg['host_fqdn'] + '_' + sanitize_uuid(cfg['gpu_uuid'])) / 'claim.json'
    expected_cpu = Path(cfg['claims_dir']) / 'cpu' / (cfg['host_fqdn'] + '.json')
    if Path(cfg['gpu_claim']) != expected_gpu or Path(cfg['cpu_claim']) != expected_cpu:
        raise ValueError('claim paths differ from actual request')
    dc, cc = record['device_claim'], record['cpu_claim']
    if (dc['job_dir'] != str(run) or dc['device_uuid'] != cfg['gpu_uuid'] or dc['host'] != cfg['host_fqdn']
            or cc['job_dir'] != str(run) or cc['host'] != cfg['host_fqdn'] or cc['threads'] != 1
            or any(x.get('pid') != supervisor['pid'] or x.get('proc_start_time') != supervisor['proc_start_time']
                   for x in (dc, cc))):
        raise ValueError('run claim identity mismatch')
    s['job_join'] = (s.get('gpu_token_sha256') == digest(dc['token'])
                     and s.get('gpu_owner_pid') == supervisor['pid']
                     and s.get('gpu_owner_birth') == supervisor['proc_start_time'])
    s['cpu_token_sha256'] = digest(cc['token'])
    if cpu is None:
        raise ValueError('CPU allocation file absent')
    allocation = cpu['allocations'].get(cc['token'])
    if allocation is not None and any(allocation.get(k) != cc.get(k)
                                      for k in ('token', 'host', 'job_dir', 'threads', 'pid', 'proc_start_time')):
        raise ValueError('CPU allocation owner differs')
    s['cpu_token_present'] = allocation is not None
    s['terminal'] = status['status']
    s['spec_sha256'], s['source_sha256'] = ready['spec_sha256'], ready['source_sha256']
    s['published_ns'] = int(ready['published'] * 1_000_000_000)
    leader, sup, pgid = status['pid'], supervisor['pid'], status['process_group_id']
    if (pgid != leader or record['child_pid'] != leader or record['process_group_id'] != pgid
            or record['child_proc_start_time'] != status['proc_start_time']):
        raise ValueError('process group record differs')
    def birth(pid: int) -> float | None:
        value = get_process_start_time(pid)
        if value is None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return None
            raise ValueError('live PID has unreadable birth')
        return value
    leader_birth, sup_birth = birth(leader), birth(sup)
    if leader_birth is not None and os.getpgid(leader) != pgid:
        raise ValueError('live leader group differs')
    if leader_birth is not None and leader_birth != status['proc_start_time']:
        raise ValueError('leader PID recycled')
    if sup_birth is not None and sup_birth != supervisor['proc_start_time']:
        raise ValueError('supervisor PID recycled')
    s.update(leader_pid=leader, leader_birth=status['proc_start_time'], supervisor_pid=sup,
             supervisor_birth=supervisor['proc_start_time'], pgid=pgid,
             leader_live=leader_birth is not None, supervisor_live=sup_birth is not None)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        s['group_absent'] = True
    else:
        s['group_absent'] = False
    s['supervisor_absent'] = sup_birth is None
    return s


def _capped_cpu_limit(seconds: float, hard: int) -> int:
    limit = math.floor(seconds)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    if limit < 1:
        raise ValueError('insufficient remaining CPU budget')
    return limit


def _control_child(controls: dict[str, str], seconds: int, pipe: Any) -> None:
    try:
        limit = _capped_cpu_limit(seconds, resource.getrlimit(resource.RLIMIT_CPU)[1])
        resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
        for name, expected in controls.items():
            if hashlib.sha256(bounded_bytes(Path(name))).hexdigest() != expected:
                raise ValueError('installed control identity differs')
        pipe.send(True)
    except BaseException as exc:
        pipe.send(repr(exc))
    finally:
        pipe.close()


def bounded_controls(cfg: dict[str, Any]) -> None:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    remaining = _capped_cpu_limit(cfg['cpu_seconds'] - cfg['parent_cpu_reserve'] - time.process_time() - usage.ru_utime - usage.ru_stime, resource.getrlimit(resource.RLIMIT_CPU)[1])
    parent, child = multiprocessing.Pipe(duplex=False)
    proc = multiprocessing.Process(target=_control_child,
                                   args=(cfg['control_sha256'], remaining, child))
    proc.start()
    child.close()
    try:
        if not parent.poll(cfg['read_timeout_seconds']):
            proc.kill()
            raise TimeoutError('control reader timed out')
        if parent.recv() is not True:
            raise ValueError('installed control identity differs')
    finally:
        parent.close()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        if proc.is_alive():
            raise RuntimeError('owned control reader could not be reaped')


def _read_child(cfg: dict[str, Any], binding: dict[str, Any] | None, pipe: Any) -> None:
    try:
        limit = _capped_cpu_limit(cfg['reader_cpu_limit'], resource.getrlimit(resource.RLIMIT_CPU)[1])
        resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
        current = read(Path(cfg['binding_file']))
        if binding is not None and current != binding:
            raise ValueError('binding changed')
        pipe.send({'sample': sample_once(cfg, current), 'binding': current})
    except BaseException as exc:
        pipe.send({'error': repr(exc)})
    finally:
        pipe.close()


def bounded_sample(cfg: dict[str, Any], binding: dict[str, Any] | None) -> dict[str, Any]:
    _capped_cpu_limit(cfg['reader_cpu_limit'], resource.getrlimit(resource.RLIMIT_CPU)[1])
    parent, child = multiprocessing.Pipe(duplex=False)
    proc = multiprocessing.Process(target=_read_child, args=(cfg, binding, child))
    proc.start()
    child.close()
    try:
        if not parent.poll(cfg['read_timeout_seconds']):
            proc.kill()
            raise TimeoutError('claim reader timed out')
        reply = parent.recv()
        if 'error' in reply:
            raise ValueError(reply['error'])
        return dict(reply)
    finally:
        parent.close()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)
        if proc.is_alive():
            raise RuntimeError('owned claim reader could not be reaped')


def _validate_journal(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in bounded_bytes(path, 64 * 1024 * 1024).splitlines()]
    if len(rows) < 3 or rows[0].get('event') != 'ready' or rows[-1].get('event') != 'final':
        return {'status': 'unknown', 'reason': 'incomplete journal'}
    if (rows[1].get('event') != 'ready_complete'
            or any(r.get('event') not in ('sample', 'binding', 'terminal_control') for r in rows[2:-1])):
        return {'status': 'unknown', 'reason': 'journal order differs'}
    cfg = rows[0]['config']
    previous = rows[0]['monotonic_ns']
    if type(previous) is not int or rows[0]['boot'] != cfg['boot']:
        return {'status': 'unknown', 'reason': 'ready time or boot differs'}
    for row in rows[1:]:
        if row['event'] == 'sample':
            start, end = row['start_ns'], row['end_ns']
            if type(start) is not int or type(end) is not int or start < previous or end < start:
                return {'status': 'unknown', 'reason': 'sample intervals overlap'}
            previous = end
        else:
            stamp = row['monotonic_ns']
            if type(stamp) is not int or stamp < previous:
                return {'status': 'unknown', 'reason': 'journal time order differs'}
            previous = stamp
        if row.get('boot') != cfg['boot']:
            return {'status': 'unknown', 'reason': 'journal boot differs'}
    readiness = [r for r in rows if r.get('event') == 'ready_complete']
    if len(readiness) != 1:
        return {'status': 'unknown', 'reason': 'readiness incomplete'}
    cfg['ready_wall_ns'] = readiness[0]['wall_ns']
    bindings = [r['binding'] for r in rows if r.get('event') == 'binding']
    if len(bindings) != 1:
        return {'status': 'unknown', 'reason': 'binding incomplete'}
    samples = [{k:v for k,v in r.items() if k != 'event'} for r in rows if r.get('event') == 'sample']
    result = observe(cfg, bindings[0], samples)
    final = rows[-1]['result']
    if result.get('status') == 'bounded':
        controls = [r for r in rows if r.get('event') == 'terminal_control']
        if (len(controls) != 1 or rows[-2] != controls[0]
                or controls[0].get('control_sha256') != cfg['control_sha256']):
            return {'status': 'unknown', 'reason': 'terminal control check missing'}
        parent, children = rows[-1]['observer_cpu_seconds'], rows[-1]['reader_cpu_seconds']
        if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in (parent, children))
                or parent > cfg['parent_cpu_reserve'] or parent + children > cfg['cpu_seconds']):
            return {'status': 'unknown', 'reason': 'CPU usage invalid or exhausted'}
    if final != result:
        return {'status': 'unknown', 'reason': 'final receipt differs from replay'}
    return result

def validate_journal(path: Path) -> dict[str, Any]:
    try:
        return _validate_journal(path)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {'status': 'unknown', 'reason': 'invalid journal'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, nargs='?')
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--validate', type=Path)
    parser.add_argument('--config-sha256')
    args = parser.parse_args()
    if args.validate is not None:
        print(json.dumps(validate_journal(args.validate), sort_keys=True))
        return
    if args.config is None or args.output is None:
        parser.error('config and output are required')
    if not local_mount(args.config.parent):
        raise ValueError('config must be on a separate local filesystem')
    raw_config = bounded_bytes(args.config)
    if len(raw_config) > 8 * 1024 * 1024 or hashlib.sha256(raw_config).hexdigest() != args.config_sha256:
        raise ValueError('pinned configuration identity differs')
    cfg = json.loads(raw_config)
    # Configuration and control files must be pinned before READY.
    if (cfg['host'] != 'gpu6' or cfg['host_fqdn'].split('.')[0] != 'gpu6'
            or not cfg['gpu_uuid'].startswith('GPU-') or not local_mount(args.output.parent)
            or not local_mount(Path(cfg['binding_file']).parent)
            or socket.gethostname().split('.')[0] != cfg['host'] or cfg['max_samples'] < 1
            or cfg['max_samples'] * (cfg['poll_seconds'] + cfg['read_timeout_seconds'] + 4) > 2700 or cfg['poll_seconds'] < 2
            or not 0 < cfg['read_timeout_seconds'] <= 10 or cfg['max_gap_ns'] <= 0
            or cfg['cpu_seconds'] <= cfg['parent_cpu_reserve'] + 1 or cfg['parent_cpu_reserve'] < 1
            or len(os.sched_getaffinity(0)) != 1
            or not cfg['control_sha256'] or not cfg['nonce']
            or any(len(cfg[k]) != 64 or any(c not in '0123456789abcdef' for c in cfg[k])
                   for k in ('spec_sha256', 'source_sha256', 'inputs_sha256'))):
        raise ValueError('host, one-core affinity or bounded observer configuration differs')
    if (not Path(cfg['control_root']).is_absolute() or any(Path(k).is_absolute() or '..' in Path(k).parts
            for k in cfg['control_files'])):
        raise ValueError('invalid control root or relative file path')
    expected_control = {str(Path(cfg['control_root']) / k): v for k, v in cfg['control_files'].items()}
    if (expected_control != cfg['control_sha256'] or cfg.get('management_host') != 'gpu6'
            or cfg.get('management_boot') != Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
        raise ValueError('control or same-boot management attestation differs')
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    parent_limit = math.floor(cfg['parent_cpu_reserve'])
    if hard != resource.RLIM_INFINITY and parent_limit > hard:
        raise ValueError('parent CPU limit exceeds existing hard limit')
    resource.setrlimit(resource.RLIMIT_CPU, (parent_limit, parent_limit))
    cfg['boot'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    bounded_controls(cfg)
    control_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if time.process_time() + control_usage.ru_utime + control_usage.ru_stime > cfg['cpu_seconds']:
        raise ValueError('observer CPU guard exceeded before ready')
    # A fresh directory and local journal avoid overwriting evidence.
    cfg['config_sha256'] = hashlib.sha256(raw_config).hexdigest()
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    journal = args.output / 'journal.jsonl'
    append(journal, {'event':'ready', 'host':cfg['host'], 'boot':cfg['boot'], 'nonce':cfg['nonce'],
                     'spec_sha256':cfg['spec_sha256'], 'source_sha256':cfg['source_sha256'],
                     'monotonic_ns':time.monotonic_ns(), 'wall_ns':time.time_ns(),
                     'control_sha256':cfg['control_sha256'], 'config':cfg})
    directory_fd = os.open(args.output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    cfg['ready_wall_ns'] = time.time_ns()
    append(journal, {'event':'ready_complete', 'wall_ns':cfg['ready_wall_ns'],
                     'monotonic_ns':time.monotonic_ns(), 'boot':cfg['boot']})
    samples: list[dict[str, Any]] = []
    binding: dict[str, Any] | None = None
    reason = 'deadline'
    try:
        for _ in range(cfg['max_samples']):
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            if time.process_time() + usage.ru_utime + usage.ru_stime >= cfg['cpu_seconds']:
                reason = 'CPU limit'
                break
            cfg['reader_cpu_limit'] = cfg['cpu_seconds'] - cfg['parent_cpu_reserve'] - time.process_time() - usage.ru_utime - usage.ru_stime
            if cfg['reader_cpu_limit'] < 1:
                reason = 'CPU reserve'
                break
            start = time.monotonic_ns()
            s: dict[str, Any] = {'start_ns':start, 'wall_ns':time.time_ns(), 'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
            try:
                reply = bounded_sample(cfg, binding)
                s.update(reply['sample'])
                b = reply['binding']
                child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                if time.process_time() + child_usage.ru_utime + child_usage.ru_stime > cfg['cpu_seconds']:
                    raise ValueError('observer CPU guard exceeded')
            except (OSError, ValueError, KeyError, TypeError, EOFError, TimeoutError) as exc:
                s['error'] = str(exc)
            s['end_ns'] = time.monotonic_ns()
            samples.append(s)
            append(journal, {'event':'sample', **s})
            if not s.get('error') and b is not None and binding is None:
                binding = b
                append(journal, {'event':'binding', 'binding':b, 'monotonic_ns':time.monotonic_ns(), 'boot':s['boot']})
            result = observe(cfg, binding, samples)
            if result['status'] == 'bounded':
                bounded_controls(cfg)
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                if (time.process_time() > cfg['parent_cpu_reserve']
                        or time.process_time() + after.ru_utime + after.ru_stime > cfg['cpu_seconds']):
                    raise ValueError('observer CPU guard exceeded during terminal control check')
                append(journal, {'event':'terminal_control', 'control_sha256':cfg['control_sha256'],
                                 'monotonic_ns':time.monotonic_ns(), 'boot':cfg['boot']})
                reason = 'complete'
                break
            if s.get('error'):
                reason = 'read error'
                break
            time.sleep(cfg['poll_seconds'])
    except BaseException as exc:
        reason = type(exc).__name__
    result = observe(cfg, binding, samples)
    if reason != 'complete':
        result = {'status':'unknown', 'reason':reason}
    final_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    append(journal, {'event':'final', 'result':result, 'observer_cpu_seconds':time.process_time(),
                     'reader_cpu_seconds':final_usage.ru_utime + final_usage.ru_stime, 'max_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     'monotonic_ns':time.monotonic_ns(), 'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})


if __name__ == '__main__':
    main()
