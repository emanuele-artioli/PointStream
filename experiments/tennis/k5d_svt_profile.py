"""Prospective SVT-only oracle selection and shared-Ada resource contract."""
from __future__ import annotations
import json
import math
import os
import socket
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

PROFILE = 'k5d-svt-only16-shared-ada-v1'
STUDY_PROFILE = 'k5d-svt-study-shared-ada-v1'
STUDY_STAGES = ('qualification16', 'cost48', 'confirmation292', 'confirmation586', 'study')
POINTS = (('svt', 48), ('svt', 55), ('svt', 62))
ALLOCATOR_MIB = 8192
OTHER_MIB = 2048
CLAIM_MIB = ALLOCATOR_MIB + OTHER_MIB
BASIS = 'SVT-only quality/cost oracle; shared timings are not codec speed evidence.'


def select(default: tuple, profile: str | None) -> tuple:
    if profile is None:
        return default
    if profile not in (PROFILE, STUDY_PROFILE):
        raise ValueError('unknown separately named point profile')
    return POINTS


def scope(stage: object, clip: object, frames: object) -> dict:
    """Validate a prospective span against the immutable registered selection."""
    from experiments.tennis.k5d_full_study import selection

    if not isinstance(stage, str) or not isinstance(clip, str) or stage not in STUDY_STAGES or type(frames) is not int:
        raise ValueError('unknown study stage or invalid frame count')
    clips = selection()
    if clip not in clips:
        raise ValueError('clip absent from frozen selection')
    label_rows = clips[clip]['label_rows']
    fixed = {
        'qualification16': ('tracknet/game10/Clip3', 16),
        'cost48': ('tracknet/game10/Clip3', 48),
        'confirmation292': ('tracknet/game9/Clip5', 292),
        'confirmation586': ('tracknet/game10/Clip12', 586),
    }
    if stage == 'study':
        valid = frames == label_rows
    else:
        valid = (clip, frames) == fixed[stage] and frames <= label_rows
        if stage.startswith('confirmation'):
            valid = valid and frames == label_rows
    if not valid:
        raise ValueError('study stage clip or span differs from registered scope')
    purpose = ('all-visible calibration; not evidence of absence behavior' if stage == 'cost48'
               else 'qualification prefix; no absence claim' if stage == 'qualification16'
               else 'complete registered clip')
    return {'stage': stage, 'clip': clip, 'frames': frames, 'label_rows': label_rows,
            'full_context': frames == label_rows, 'purpose': purpose}


def admit(spec: dict, dispatch: dict, env: Mapping[str, str], torch: Any) -> dict:
    sharing = {'mode':'shared', 'basis':BASIS, 'margin_mib':8192}
    contention = {'policy':'continue', 'basis':BASIS, 'resume_attempts':0}
    uuid = env.get('CUDA_VISIBLE_DEVICES')
    selected = spec.get('k5d_evaluation_profile')
    if (socket.gethostname().split('.')[0] != 'gpu6' or spec.get('cpu_threads') != 1
            or spec.get('single_cpu_campaign') is not True or selected not in (PROFILE, STUDY_PROFILE) or spec.get('hosts') != ['gpu6']
            or spec.get('gpu_models') != ['RTX 6000 Ada'] or spec.get('gpu_memory_mib') != CLAIM_MIB
            or spec.get('gpu_sharing') != sharing or spec.get('contention') != contention
            or dispatch.get('gpu_sharing') != sharing or dispatch.get('contention_policy') != contention
            or env.get('PS_GPU_SHARING') != 'shared' or env.get('PS_GPU_MEMORY_LIMIT_MIB') != str(CLAIM_MIB)
            or not uuid or not uuid.startswith('GPU-') or env.get('PS_CLAIMED_GPU_UUID') != uuid
            or env.get('PS_CPU_ALLOWANCE') != '1' or len(os.sched_getaffinity(0)) != 1
            or env.get('PS_STAGE') not in (('smoke', 'full') if selected == STUDY_PROFILE else ('smoke',))
            or torch.cuda.device_count() != 1
            or dispatch.get('environment', {}).get('gpu', {}).get('uuid') != uuid
            or torch.cuda.get_device_name(0) != 'NVIDIA RTX 6000 Ada Generation'
            or tuple(torch.cuda.get_device_capability(0)) != (8, 9)):
        raise ValueError('explicit shared Ada oracle admission differs')
    if torch.cuda.memory_allocated(0) or torch.cuda.memory_reserved(0):
        raise ValueError('CUDA allocator was used before the cap')
    total = torch.cuda.get_device_properties(0).total_memory
    if total < CLAIM_MIB * 2**20:
        raise ValueError('GPU capacity below declared claim')
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_MIB * 2**20 / total, 0)
    # Materialize the capped CUDA context before requiring its actual driver row.
    probe = torch.empty(1, dtype=torch.uint8, device='cuda:0')
    torch.cuda.synchronize(0)
    del probe
    torch.cuda.reset_peak_memory_stats(0)
    from experiments.tennis.k5d_scoped_namespace import host_group_identity
    host = host_group_identity(Path(env['PS_STAGE_DIR']).parent/'run'/'status.json')
    return {'profile':selected, 'gpu_uuid':uuid, 'allocator_mib':ALLOCATOR_MIB,
            'context_nonallocator_allowance_mib':OTHER_MIB, 'claim_mib':CLAIM_MIB,
            'allocator_before':0, 'total_bytes':total, 'fraction':ALLOCATOR_MIB*2**20/total,
            **host, 'scope':'prospective cap plus observed memory snapshots; not an absolute hardware peak bound'}


def check_snapshot(value: dict) -> None:
    for key in ('driver_used_bytes','reserved_bytes','peak_reserved_bytes','allocated_bytes'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError('missing or invalid actual GPU memory receipt')
    if (value['driver_used_bytes'] > CLAIM_MIB*2**20 or value['peak_reserved_bytes'] > ALLOCATOR_MIB*2**20
            or max(0,value['driver_used_bytes']-value['reserved_bytes']) > OTHER_MIB*2**20):
        raise ValueError('observed shared memory allowance exceeded')


def snapshot(torch: Any, receipt: dict, phase: str) -> dict:
    result = subprocess.run(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                             '--format=csv,noheader,nounits'], check=True,capture_output=True,text=True,timeout=10)
    own = None
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 3:
            raise ValueError('malformed GPU process memory inventory')
        if fields[0] != receipt['gpu_uuid']:
            continue
        pid = int(fields[1])
        if pid == receipt['host_pid']:
            if own is not None:
                raise ValueError('duplicate model process memory row')
            own = int(fields[2])*2**20
        else:
            try:
                if os.getpgid(pid) == receipt['campaign_pgid']:
                    raise ValueError('a second campaign process acquired GPU memory')
            except ProcessLookupError:
                continue
    if own is None:
        raise ValueError('model process actual GPU memory receipt missing')
    value = {'phase':phase, 'monotonic_ns':time.monotonic_ns(), 'host_pid':receipt['host_pid'],
             'gpu_uuid':receipt['gpu_uuid'], 'driver_used_bytes':own,
             'reserved_bytes':int(torch.cuda.memory_reserved(0)),
             'peak_reserved_bytes':int(torch.cuda.max_memory_reserved(0)),
             'allocated_bytes':int(torch.cuda.memory_allocated(0))}
    check_snapshot(value)
    return value


def validate_memory(report: dict, expected_profile: str = PROFILE) -> None:
    setup, samples = report['setup'], report['snapshots']
    total = setup['total_bytes']
    if (expected_profile not in (PROFILE, STUDY_PROFILE) or setup['profile'] != expected_profile
            or setup['allocator_mib'] != ALLOCATOR_MIB
            or setup['context_nonallocator_allowance_mib'] != OTHER_MIB or setup['claim_mib'] != CLAIM_MIB
            or setup['allocator_before'] != 0 or type(total) is not int or total < CLAIM_MIB*2**20
            or setup['fraction'] != ALLOCATOR_MIB*2**20/total or not samples):
        raise ValueError('resource profile or actual snapshots missing')
    if (not isinstance(setup.get('gpu_uuid'),str) or not setup['gpu_uuid'].startswith('GPU-')
            or any(type(setup.get(k)) is not int or setup[k] <= 0 for k in ('host_pid','host_pgid','campaign_pgid','start_ticks'))
            or setup['host_pgid'] != setup['campaign_pgid']):
        raise ValueError('actual model process lineage missing')
    if not {'before_models','after_complete'}.issubset({x['phase'] for x in samples}):
        raise ValueError('cold setup or complete pipeline memory evidence missing')
    previous = -1
    for value in samples:
        check_snapshot(value)
        if (value['host_pid'] != setup['host_pid'] or value['gpu_uuid'] != setup['gpu_uuid']
                or type(value['monotonic_ns']) is not int or value['monotonic_ns'] <= previous):
            raise ValueError('memory receipt process or timing differs')
        previous = value['monotonic_ns']
    for key in ('self_cpu_seconds','waited_children_cpu_seconds','wall_seconds'):
        value = report[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            raise ValueError('incomplete oracle cost counters')


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
