"""Read-only packet structure and frozen CPU scheduler counts; no GPU or quality claim."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time

UNIT_SHA = ('387544a3fc344685f7c07632879affbdafcfd54fd6e1c00e0d9d6c06ca98995d',
            '74be8c9e2f8eb65ca67e6339ef9a2a2f8238df36c60a4b5a07a45fc68d3712a7')
SOURCE_SHA = ('bb30f9e4292a8f1d687f70f1af8b25325ea4df504789c7154eb84f923d68d996',
              '00248aa4fdd47e634f82b37a7a33c99f1794440d65cbfe9395255a4225735542',
              'aa019bb1dac4e7e736231c85d48131e21ea51012347b1810cb03c8b37cade71a',
              '480971c58c845c8636a9b0e616708f07bd7d7a5c19ae3c20b9439c0c9ac90a7f')


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (45, 45))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('CUDA must be hidden and process affinity must be one CPU')
    if len(sys.argv) != 7:
        raise ValueError('expected two unit NPZ paths and four frozen source paths')
    paths = [Path(x).resolve(strict=True) for x in sys.argv[1:]]
    for path, expected in zip(paths, UNIT_SHA + SOURCE_SHA):
        if not path.is_file() or sha(path) != expected:
            raise ValueError('input SHA mismatch: ' + str(path))
    import numpy as np
    import torch
    from diffusers import DDIMScheduler
    if torch.cuda.is_initialized():
        raise ValueError('CUDA initialized')
    scheduler_module = runpy.run_path(str(paths[4]))
    uniform = scheduler_module['get_context_scheduler']('uniform')
    units = []
    for path in paths[:2]:
        with np.load(path, allow_pickle=False) as obj:
            t, judged, run, pose, reference = (obj[x] for x in ('t', 'judged', 'run', 'pose', 'reference'))
            if (not np.array_equal(t, np.arange(240)) or judged.shape != (240,)
                    or not np.array_equal(np.flatnonzero(judged), np.arange(0, 240, 3))
                    or run.shape != (240,) or pose.shape[0] != 240):
                raise ValueError('unit timeline or pose coverage mismatch')
            runs = [{'id': int(k), 'length': int(np.count_nonzero(run == k)),
                     'contiguous': bool(np.all(np.diff(np.flatnonzero(run == k)) == 1))}
                    for k in np.unique(run)]
            for steps in (1, 3, 30):
                scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='linear',
                    clip_sample=False, steps_offset=1, prediction_type='v_prediction',
                    rescale_betas_zero_snr=True, timestep_spacing='trailing')
                scheduler.set_timesteps(steps, device='cpu')
                if len(scheduler.timesteps) != steps or scheduler.order != 1:
                    raise ValueError('actual DDIM timestep count differs')
                for run_record in runs:
                    n = run_record['length']
                    probe = list(uniform(0, steps, n, 24, 1, 0))
                    used = list(uniform(0, steps, n, 24, 1, 4))
                    if not probe or not used or any(not 0 <= int(i) < n for w in used for i in w):
                        raise ValueError('invalid scheduler windows')
                    run_record[str(steps)] = {'discarded_windows_per_step': len(probe),
                        'used_windows_per_step': len(used), 'denoiser_calls': steps * len(used),
                        'image_encoder_calls': 1, 'reference_unet_calls': 1, 'pose_guider_calls': 1,
                        'vae_encode_calls': 1, 'vae_decode_calls': n,
                        'scheduler_invocations': 2 * steps,
                        'wrapped_windows_per_step': sum(any(b <= a for a, b in zip(w, w[1:])) for w in used),
                        'window_lengths': sorted(set(map(len, used)))}
            units.append({'sha256': sha(path), 'runs': runs, 'frames': len(t),
                          'ticks': int(np.count_nonzero(judged)), 'pose_shape': list(pose.shape),
                          'pose_dtype': str(pose.dtype), 'reference_shape': list(reference.shape),
                          'reference_dtype': str(reference.dtype)})
    syntax = []
    for path in paths[4:]:
        tree = ast.parse(path.read_text())
        syntax.append({'sha256': sha(path), 'functions': [n.name for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if torch.cuda.is_initialized():
        raise ValueError('CUDA initialized')
    print(json.dumps({'status': 'complete', 'units': units,
                      'source_sha256': [sha(p) for p in paths[2:]], 'scheduler_syntax': syntax,
                      'scheduler': {'name': 'uniform', 'step_argument': 0, 'context_frames': 24,
                                    'stride': 1, 'overlap': 4, 'closed_loop': True,
                                    'context_batch_size': 1, 'guidance': 2, 'render_size': [512, 784],
                                    'latent_frames': 'run length; no temporal padding',
                                    'discarded_overlap': 0, 'guidance_batch_multiplier': 2},
                      'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                      'affinity': sorted(os.sched_getaffinity(0)), 'python': sys.executable,
                      'pid': os.getpid(), 'pgid': os.getpgrp(), 'wall_seconds': time.monotonic() - started,
                      'cpu_seconds': usage.ru_utime + usage.ru_stime,
                      'scope': 'CPU structure and scheduler counts only; no GPU or quality claim'}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
