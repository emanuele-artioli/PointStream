"""W7 fleet job specifications for the Tier 1 windows (python -m experiments.tennis.w7_specs WINDOWS.json OUT_DIR).

Nothing is submitted here. Only stage groups whose entrypoint already reads the window manifest get a spec
(`k6b_native_base`, `w1_masks`, `w1_encode`, `w1_score`); the others are listed in `STAGES` with their status, so the
GPU estimate of the whole chain is in one place (docs/experiments/w7-untouched.md, 2026-10-30 estimate entry).
Staged files must live under the fleet data root (a 2026-10-24 attempt failed on sources outside it): `W7_DIR` is
where `w7_select stage` copies the six clips; windows.json, the window state and prior-stage archives come as (path, sha256).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from experiments.tennis import window_manifest as wm
from experiments.tennis.w1_specs import DATA, ENVIRONMENT

W7_DIR = DATA + '/tennis/w7-tier1-20261030'
DEADLINE = '2026-11-02T12:00:00+00:00'
HOSTS_GPU = ['gpu3', 'gpu5', 'gpu6']
GPU_MODELS = ['RTX A6000', 'RTX 6000 Ada']
FULL_SECONDS = 2600
SMOKE_SECONDS = 600

#: Per-window GPU seconds from the recorded jobs (A6000 numbers; Ada in brackets where recorded). Sources: k6b-players.md
#: 2026-10-24 budget table (features 0.29 h per 6 windows, anchors 0.37 h per 3, render 0.35-0.39 h per window), the
#: receiver-frame pilot (1639 s for 2 windows), jobs 20261027T074823Z-135421ee (167 s / 2), 20261030T072024Z-10c9606e (816 s / 2),
#: 20261030T091552Z-fe376abe (681 s / 1), W3 (489 s / 2). Render on an A6000 is 863 s per player-unit (lead), Ada about 680.
STAGES: list[dict[str, Any]] = [
    {'stage': 'features', 'module': 'k6b_confirm features', 'gpu_s': 175, 'status': 'manifest cohort accepted (cohort field); sources/--windows generic'},
    {'stage': 'anchors/keypoints (K6 codes)', 'module': 'k6b_confirm_run --kind anchors', 'gpu_s': 444, 'status': 'cohort accepted; k6 run inputs not re-checked'},
    {'stage': 'prepare-match', 'module': 'k6b_confirm_run --kind prepare', 'gpu_s': 60, 'status': 'cohort accepted'},
    {'stage': 'generation (2 players, FULL reference)', 'module': 'k6b_generate generate (k6b_confirm_run --kind generate)', 'gpu_s': 1735, 'status': 'cohort accepted; adapter per match; reference = window frame 0'},
    {'stage': 'rates (KPS, adapter, bank)', 'module': 'k6b_rate_transport', 'gpu_s': 0, 'status': 'NOT generalised (smoke list pins Al031/Med014/Al016); CPU 473 s'},
    {'stage': 'paid base (scale 8/4/2)', 'module': 'k6b_native_base', 'gpu_s': 0, 'status': 'generalised (--windows-file); CPU 84 s/window'},
    {'stage': 'GPU source control', 'module': 'k6b_rgb_diagnostic', 'gpu_s': 84, 'status': 'NOT generalised (prior rows, wid by name)'},
    {'stage': 'receiver masks (every-frame SAM)', 'module': 'k6b_receiver_frame', 'gpu_s': 820, 'status': 'NOT generalised'},
    {'stage': 'compose + hybrid score', 'module': 'k6b_real_base_judge', 'gpu_s': 410, 'status': 'NOT generalised (PILOTS, gen_al/gen_med, prior controls)'},
    {'stage': 'W1/W5 SVT anchors R/S masks + encodes', 'module': 'w1_masks, w1_encode', 'gpu_s': 0, 'status': 'generalised (--windows-file --window-state); CPU about 150 s/window'},
    {'stage': 'W1/W5 scoring of anchors', 'module': 'w1_score', 'gpu_s': 681, 'status': 'generalised (--windows-file --window-state)'},
    {'stage': 'DCVC-UF encodes + scoring', 'module': 'w3_neural_baseline, w5_dcvc', 'gpu_s': 245, 'status': 'NOT generalised (SOURCES al/med, two_windows)'},
]


def per_window_gpu_seconds() -> int:
    return sum(s['gpu_s'] for s in STAGES)


def staged(items: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    return [{'name': n, 'path': p, 'sha256': h, 'extract': False} for n, (p, h) in items.items()]


def common(module: str, arguments: list[str], files: dict[str, tuple[str, str]]) -> dict[str, Any]:
    items = staged(files)
    return {'schema': 1, 'entrypoint': ['-m', module], 'arguments': arguments, 'staged_inputs': items,
            'inputs': [{'path': x['path'], 'sha256': x['sha256']} for x in items], 'environment': ENVIRONMENT,
            'required_commands': ['ffmpeg'], 'validator': ['{python}', '-m', module, '--validate'],
            'deadline': DEADLINE, 'verify_staged_inputs': True, 'stall_seconds': 900.0,
            'contention': {'policy': 'stop', 'resume_attempts': 0}}


def source_files(windows: list[wm.WindowSpec]) -> dict[str, tuple[str, str]]:
    return {w.name: (f'{W7_DIR}/sources/{w.source_file}', w.source_sha256) for w in windows}


def native_base(windows: list[wm.WindowSpec], manifest: tuple[str, str], frames: int) -> dict[str, Any]:
    """Paid bases (scale 8/4/2 plain CRF 63) of the selected windows in one single-CPU job; smoke = 24 frames (the code's
    grid is 24 or 240, not 16) of the first window only. The job is not checkpointed per window (one bundle), so a
    full job over six windows needs about 6 x 84 s, far inside the 1300 s alarm."""
    chosen = windows[:1] if frames == 24 else windows
    arguments = ['--windows-file', '{staged:windows}', *sum((['--window', w.name] for w in chosen), []),
                 *sum((['--source', '{staged:' + w.name + '}'] for w in chosen), []), '--frames', '{frames}']
    spec = common('experiments.tennis.k6b_native_base', arguments, {'windows': manifest, **{w.name: source_files(windows)[w.name] for w in chosen}})
    spec.update({'device': 'cpu', 'hosts': HOSTS_GPU, 'gpu_models': [], 'gpu_memory_mib': 0, 'cpu_threads': 1, 'local_storage_gib': 40,
                 'scale': {'frames': {'smoke': 24, 'full': 240}},
                 'smoke': {'seconds': 300, 'representative_basis': 'First window, 24 frames, scale 2 only: the same conversion, SVT encode, fresh decode and bundle as the full stage.'},
                 'full': {'seconds': min(FULL_SECONDS, 1300)}, 'validator_seconds': 120, 'budget_seconds': 1800})
    return spec


def w1_window_stage(module: str, window: wm.WindowSpec, manifest: tuple[str, str], state: tuple[str, str], extra_arguments: list[str],
                    extra_files: dict[str, tuple[str, str]], *, gpu: bool, full_seconds: int) -> dict[str, Any]:
    """One w1_* job for one window (those modules take a single `--window`; there is no six-window job)."""
    arguments = ['--windows-file', '{staged:windows}', '--window-state', '{staged:state}', '--window', window.name, *extra_arguments]
    files = {'windows': manifest, 'state': state, window.name: source_files([window])[window.name], **extra_files}
    spec = common(module, arguments, files)
    spec.update({'device': 'gpu' if gpu else 'cpu', 'hosts': HOSTS_GPU, 'gpu_models': GPU_MODELS if gpu else [],
                 'gpu_memory_mib': 8000 if gpu else 0, 'cpu_threads': 4 if gpu else 16, 'local_storage_gib': 30,
                 'smoke': {'seconds': SMOKE_SECONDS, 'representative_basis': 'First saved ticks of the window, as W1/W5.'},
                 'full': {'seconds': min(full_seconds, FULL_SECONDS)}, 'validator_seconds': 120, 'budget_seconds': min(full_seconds, FULL_SECONDS) + 800})
    return spec


def main(argv: list[str]) -> None:
    manifest_path, out = Path(argv[0]), Path(argv[1])
    windows = wm.load(manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    sha = wm.file_sha256(manifest_path)
    manifest = (W7_DIR + '/windows.json', sha)
    for frames in (24, 240):
        spec = native_base(windows, manifest, frames)
        (out / f'native-base-{"smoke" if frames == 24 else "full"}.json').write_text(json.dumps(spec, indent=1) + '\n')
    (out / 'stages.json').write_text(json.dumps({'stages': STAGES, 'gpu_seconds_per_window': per_window_gpu_seconds(),
                                                 'windows': [w.name for w in windows]}, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
