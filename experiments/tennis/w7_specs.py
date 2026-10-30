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


#: Tier 1 GPU budget (user, 2026-10-30): the first window, which is also the smoke, up to 1.3 GPU-h; all six windows up to 8 GPU-h.
BUDGET_FIRST_S = 1.3 * 3600
BUDGET_TOTAL_S = 8 * 3600


def budget_check(n_windows: int = 6, per_window: int | None = None) -> dict[str, Any]:
    """Estimated GPU seconds against the Tier 1 budget; every stage kept, keypoint codes included."""
    each = per_window_gpu_seconds() if per_window is None else per_window
    total = each * n_windows
    return {'per_window_s': each, 'first_window_s': each, 'first_window_budget_s': BUDGET_FIRST_S, 'total_s': total,
            'total_budget_s': BUDGET_TOTAL_S, 'first_ok': each <= BUDGET_FIRST_S, 'total_ok': total <= BUDGET_TOTAL_S,
            'margin_total_s': BUDGET_TOTAL_S - total}


#: staged inputs that are directories packed as a tar (the build-inputs output, read as `--inputs DIR`); every other tar stays packed.
EXTRACTED = frozenset({'inputs'})


def staged(items: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    return [{'name': n, 'path': p, 'sha256': h, 'extract': n in EXTRACTED} for n, (p, h) in items.items()]


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


PENDING = '0' * 64  # sha256 of an upstream archive that does not exist yet; the lead fills it in when that job has completed
JCV_ENVIRONMENT = {'path': DATA + '/environments/pointstream-jcv-20261022T124459Z.tar.gz',
                   'sha256': 'bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e'}
ASSETS: dict[str, tuple[str, str]] = {  # path under the data root, sha256: copied from the 2026-10-24 specs
    'sam': ('/background/g1-2026-10-08/inputs/sam3.1_multiplex.pt', '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'),
    'yolo11x_pose': ('/tennis/k4-2026-10-13/inputs/yolo/yolo11x-pose.pt', '013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8'),
    'inspect': ('/visor/h1-2026-10-08/inputs/aom/aom-inspect-v3.12.1-ps1', '9b2e75a288309a6d91c9dcd2976f30425923a8f6733e74dd65475f7f4e443959'),
    'lpips': ('/visor/h3-2026-10-10/inputs/lpips/alexnet-owt-7be5be79.pth', '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'),
    'dists': ('/visor/h3b-2026-10-11/inputs/dists/vgg16-397923af.pth', '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'),
    'dcvc_image': ('/visor/b2-2026-10-07/inputs/dcvc/cvpr2026_image.pth.tar', 'b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302'),
    'dcvc_video': ('/visor/b2-2026-10-07/inputs/dcvc/cvpr2026_video_htl.pth.tar', '934bde4a12fc0b6b0c679ec6ae21d41cb7722debf8b631ee0dd750d030b1a6b3'),
    'w0': ('/tennis/k6b-2026-10-22/inputs/aa/vae/config.json', '92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e'),
    'w1': ('/tennis/k6b-2026-10-22/inputs/aa/vae/diffusion_pytorch_model.safetensors', 'a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815'),
    'w2': ('/tennis/k6b-2026-10-22/inputs/aa/image_encoder/config.json', 'ba41139cc9dad159163a53e2aa847356b627f53fef988ad461659bd2d587d886'),
    'w3': ('/tennis/k6b-2026-10-22/inputs/aa/image_encoder/pytorch_model.bin', '89d2aa29b5fdf64f3ad4f45fb4227ea98bc45156bbae673b85be1af7783dbabb'),
    'w4': ('/tennis/k6b-2026-10-22/inputs/aa/original/denoising_unet.pth', 'b9e5a2c34fac369e8a922972ca2210916c6af175a0dad907deccf6235816ad52'),
    'w5': ('/tennis/k6b-2026-10-22/inputs/aa/original/reference_unet.pth', 'beddccb08d49a8b29b0f4d6d456c6521d4382a8d8d48884fa60ba8802509c214'),
    'w6': ('/tennis/k6b-2026-10-22/inputs/aa/original/pose_guider.pth', '1a8b7c1b4db92980fd977b4fd003c1396bbae9a9cdea00c35d452136d5e4f488'),
    'w7': ('/tennis/k6b-2026-10-22/inputs/aa/original/motion_module.pth', '0d11e01a281b39880da2efeea892215c1313e5713fca3d100a7fbb72ee312ef9'),
}


def assets(*names: str) -> dict[str, tuple[str, str]]:
    return {n: (DATA + ASSETS[n][0], ASSETS[n][1]) for n in names}


def gpu_stage(module: str, arguments: list[str], files: dict[str, tuple[str, str]], *, models: list[str], mem_mib: int, threads: int,
              smoke_seconds: int, full_seconds: int, scale: dict[str, Any], basis: str, environment: dict[str, str] | None = None,
              device: str = 'gpu', validator_module: str | None = None, validator: list[str] | None = None) -> dict[str, Any]:
    spec = common(module, arguments, files)
    if environment is not None:
        spec['environment'] = environment
    if validator_module is not None:
        spec['validator'] = ['{python}', '-m', validator_module, '--validate-only']
    if validator is not None:
        spec['validator'] = validator
    spec.update({'device': device, 'hosts': HOSTS_GPU, 'gpu_models': models if device == 'gpu' else [], 'gpu_memory_mib': mem_mib if device == 'gpu' else 0,
                 'cpu_threads': threads, 'local_storage_gib': 60, 'scale': scale,
                 'smoke': {'seconds': min(smoke_seconds, SMOKE_SECONDS), 'representative_basis': basis},
                 'full': {'seconds': min(full_seconds, FULL_SECONDS)}, 'validator_seconds': 120, 'budget_seconds': min(full_seconds, FULL_SECONDS) + 800})
    return spec


def features(first: wm.WindowSpec, manifest: tuple[str, str], source: tuple[str, str]) -> dict[str, Any]:
    """SAM 3.1 + YOLO11x-pose features of the window (k6b_confirm features); smoke = the same window, one-window grid."""
    args = ['features', '--manifest', '{staged:manifest}', '--sam-checkpoint', '{staged:sam}', '--pose', '{staged:yolo11x_pose}',
            '--max-windows', '{windows}', '--windows', first.id, '--sources', '{staged:c0}']
    return gpu_stage('experiments.tennis.k6b_confirm', args, {'manifest': manifest, **assets('sam', 'yolo11x_pose'), 'c0': source},
                     models=['RTX 6000 Ada'], mem_mib=32000, threads=8, smoke_seconds=600, full_seconds=1800,
                     scale={'windows': {'smoke': 1, 'full': 0}}, basis='The first window with the full 240-frame decode; the smallest grid the stage supports is one whole window.',
                     validator=['{python}', '-m', 'experiments.tennis.k6b_confirm', 'validate-features', '--manifest', manifest[0]])  # the validator gets no placeholder expansion


def build_inputs(first: wm.WindowSpec, confirm: tuple[str, str], features_archive: tuple[str, str]) -> dict[str, Any]:
    """Geometry-only masks, player selection and indices from the feature archive (k6b_confirm build-inputs), CPU. The 2026-10-24 job
    `20261024T200247Z-b63794f9` ran it the same way; the validator rebuilds and compares byte for byte, so it names its inputs literally."""
    spec = common('experiments.tennis.k6b_confirm', ['build-inputs', '--manifest', '{staged:manifest}', '--feature-archive', '{staged:f0}',
                                                      '--max-windows', '{windows}'], {'manifest': confirm, 'f0': features_archive})
    spec.update({'device': 'cpu', 'hosts': HOSTS_GPU, 'gpu_models': [], 'gpu_memory_mib': 0, 'cpu_threads': 4, 'local_storage_gib': 40,
                 'scale': {'windows': {'smoke': 1, 'full': 0}},
                 'smoke': {'seconds': 300, 'representative_basis': 'The one feature window of the first window: geometry-only masks, player selection, indices and archive; full is the same.'},
                 'full': {'seconds': 600}, 'validator_seconds': 180, 'budget_seconds': 1200,
                 'validator': ['{python}', '-m', 'experiments.tennis.k6b_confirm', 'validate-inputs', '--manifest', confirm[0], '--feature-archive', features_archive[0]]})
    return spec


def anchors(first: wm.WindowSpec, manifest: tuple[str, str], inputs: tuple[str, str], source: tuple[str, str]) -> dict[str, Any]:
    """K6 run: SVT/DCVC rough anchors, keypoint codes and window rows (k6b_confirm_run --kind anchors); `inputs` = `k6b_confirm build-inputs` output."""
    args = ['--kind', 'anchors', '--manifest', '{staged:manifest}', '--inputs', '{staged:inputs}', '--', 'run', '--rough-anchors-first',
            '--inputs', '{staged:inputs}', '--pose', '{staged:yolo11x_pose}', '--inspect', '{staged:inspect}',
            '--image-ckpt', '{staged:dcvc_image}', '--image-sha256', ASSETS['dcvc_image'][1], '--video-ckpt', '{staged:dcvc_video}',
            '--video-sha256', ASSETS['dcvc_video'][1], '--lpips-backbone', '{staged:lpips}', '--dists-backbone', '{staged:dists}',
            '--frames', '{frames}', '--max-windows', '{windows}', '--windows', first.id, '--sources', '{staged:c0}']
    files = {'manifest': manifest, 'inputs': inputs, **assets('yolo11x_pose', 'inspect', 'dcvc_image', 'dcvc_video', 'lpips', 'dists'), 'c0': source}
    return gpu_stage('experiments.tennis.k6b_confirm_run', args, files, models=['RTX 6000 Ada'], mem_mib=24000, threads=16,
                     smoke_seconds=600, full_seconds=1800, scale={'frames': {'smoke': 48, 'full': 240}, 'windows': {'smoke': 1, 'full': 0}},
                     basis='First window, 48 frames (the 2026-10-24 smoke grid).', validator_module='experiments.tennis.k6b_confirm_run')


def prepare(first: wm.WindowSpec, manifest: tuple[str, str], inputs: tuple[str, str], anchor: tuple[str, str], source: tuple[str, str]) -> dict[str, Any]:
    args = ['--kind', 'prepare', '--manifest', '{staged:manifest}', '--inputs', '{staged:inputs}', '--k6-stage', '{staged:a0}', '--', 'prepare',
            '--inputs', '{staged:inputs}', '--k6-stage', '{staged:a0}', '--windows', first.id, '--sources', '{staged:c0}', '--max-windows', '{windows}']
    return gpu_stage('experiments.tennis.k6b_confirm_run', args, {'manifest': manifest, 'inputs': inputs, 'a0': anchor, 'c0': source},
                     models=GPU_MODELS, mem_mib=0, threads=8, smoke_seconds=300, full_seconds=900, device='cpu',
                     scale={'windows': {'smoke': 1, 'full': 0}}, basis='The first window: prepare-match is per window, so smoke equals full.',
                     validator_module='experiments.tennis.k6b_confirm_run')


def generate(first: wm.WindowSpec, manifest: tuple[str, str], prepared: tuple[str, str], units: list[str]) -> dict[str, Any]:
    """FULL-reference generation of the window's player units with the match adapter (C tennis base, 4-bit, guidance 2.0, 30 steps).
    `units` are the player-unit names of the prepare output (two per window)."""
    adapter = first.adapter
    base = (DATA + f"/jobs/fleet/inbox/{wm.BASE['job']}/full/published.tar", wm.BASE['published_sha256'])
    fits = (DATA + f"/jobs/fleet/inbox/{adapter['job']}/full/published.tar", adapter['published_sha256'])
    weights = [x for k in range(8) for x in ('--weight', ['vae/config.json', 'vae/diffusion_pytorch_model.safetensors', 'image_encoder/config.json',
              'image_encoder/pytorch_model.bin', 'original/denoising_unet.pth', 'original/reference_unet.pth', 'original/pose_guider.pth',
              'original/motion_module.pth'][k], '{staged:w%d}' % k)]
    args = ['--kind', 'generate', '--manifest', '{staged:manifest}', '--prepared', '{staged:prepared}', '--windows', first.id, '--', 'generate',
            '--prepared', '{staged:prepared}', *weights, '--units', ','.join(units), '--max-units', '0', '--max-frames', '{frames}', '--profiles', '',
            '--fit-archive', '{staged:base}', '--fit-archive', '{staged:fits}', '--from-t', '0', '--steps', '30', '--gpu-class', 'RTX 6000 Ada',
            '--fit', 'match5', f"{adapter['name']}@fit-100-fp16.pt", '--fit-base', 'match5', 'tennis-base@fit-100.pt', '--fit-bits', 'match5', '4', '--cfg', 'match5', '2.0']
    files = {'manifest': manifest, 'prepared': prepared, **assets(*[f'w{k}' for k in range(8)]), 'base': base, 'fits': fits}
    spec = gpu_stage('experiments.tennis.k6b_confirm_run', args, files, models=['RTX 6000 Ada'], mem_mib=24000, threads=8, smoke_seconds=180,
                     full_seconds=1700, scale={'frames': {'smoke': 8, 'full': 0}}, environment=JCV_ENVIRONMENT,
                     basis='First window, 8 frames of every unit (the 2026-10-24 smoke grid); the full stage renders both players of one window (about 1,370 s Ada).',
                     validator_module='experiments.tennis.k6b_confirm_run')
    spec['contention'] = {'policy': 'continue', 'basis': 'the sampler is seeded per unit and run; outputs do not depend on wall time; recorded seconds are not evidence', 'resume_attempts': 0}
    return spec


def receiver(windows: list[wm.WindowSpec], manifest: tuple[str, str], transport: tuple[str, str], anchor: tuple[str, str],
             generated: list[tuple[str, str]], judged: int) -> dict[str, Any]:
    """Receiver masks (every-frame fresh-prompt SAM, k6b_receiver_frame, manifest mode: no frozen diagnostic or old controls)."""
    chosen = windows[:1] if judged else windows
    args = ['--windows-file', '{staged:windows}', *sum((['--window', w.name] for w in chosen), []), '--transport', '{staged:transport}',
            '--anchor', '{staged:a0}', *sum((['--source', '{staged:' + w.name + '}'] for w in chosen), []), '--backbone', '{staged:lpips}',
            '--backbone', '{staged:dists}', *sum((['--generated', '{staged:g%d}' % i] for i in range(len(chosen))), []),
            '--max-judged', '{judged}', '--weights', '{staged:sam}', '--text', '{staged:sam}']
    files = {'windows': manifest, 'transport': transport, 'a0': anchor, **assets('lpips', 'dists', 'sam'),
             **{w.name: source_files(windows)[w.name] for w in chosen}, **{f'g{i}': g for i, g in enumerate(generated[:len(chosen)])}}
    return gpu_stage('experiments.tennis.k6b_receiver_frame', args, files, models=['RTX 6000 Ada'], mem_mib=12000, threads=4, smoke_seconds=300,
                     full_seconds=FULL_SECONDS, scale={'judged': {'smoke': 2, 'full': 0}}, basis='First window, two judged ticks: SAM prompting twice, masks, composites and metrics.')


def dcvc(windows: list[wm.WindowSpec], manifest: tuple[str, str], anchor: tuple[str, str], transport: tuple[str, str], prior_frame: tuple[str, str],
         frames: int) -> dict[str, Any]:
    """DCVC-UF points and SVT bases with decode checks (w3_neural_baseline, manifest mode: per-window checkpoints under PS_CHECKPOINT_DIR)."""
    chosen = windows[:1] if frames == 25 else windows
    args = ['--windows-file', '{staged:windows}', *sum((['--window', w.name] for w in chosen), []),
            *sum((['--source', '{staged:' + w.name + '}'] for w in chosen), []), '--anchor', '{staged:anchor}', '--transport', '{staged:transport}',
            '--prior-frame', '{staged:prior_frame}', '--image-ckpt', '{staged:dcvc_image}', '--video-ckpt', '{staged:dcvc_video}',
            '--lpips', '{staged:lpips}', '--dists', '{staged:dists}', '--frames', '{frames}', '--points', '{points}', '--svt-scales', '{svt_scales}']
    files = {'windows': manifest, 'anchor': anchor, 'transport': transport, 'prior_frame': prior_frame,
             **assets('dcvc_image', 'dcvc_video', 'lpips', 'dists'), **{w.name: source_files(windows)[w.name] for w in chosen}}
    return gpu_stage('experiments.tennis.w3_neural_baseline', args, files, models=['RTX 6000 Ada', 'RTX A6000'], mem_mib=12000, threads=4,
                     smoke_seconds=600, full_seconds=FULL_SECONDS,
                     scale={'frames': {'smoke': 25, 'full': 240}, 'points': {'smoke': 1, 'full': 3}, 'svt_scales': {'smoke': '8', 'full': '8,4,2'}},
                     basis='First window, 25 frames, one DCVC point and the scale 8 SVT base (the W3 smoke grid).')


def _first(windows: list[wm.WindowSpec], frames_or_smoke: bool) -> list[wm.WindowSpec]:
    return windows[:1] if frames_or_smoke else windows


def rgb_diagnostic(windows: list[wm.WindowSpec], manifest: tuple[str, str], base: tuple[str, str], frames: int) -> dict[str, Any]:
    """GPU decode and resize of the paid bases (their `-gpu.npy` pixels are what the judge scores). A6000 only (the code asserts it).
    Smoke = 24 frames of the first window; it reads the 240-frame native-base bundle (3 scales), so that job must have run first.
    No prior run: the control is two identical fresh conversions and the IVF-to-recon decode check, hashes recorded."""
    chosen = _first(windows, frames == 24)
    arguments = ['--base', '{staged:base}', '--windows-file', '{staged:windows}', *sum((['--window', w.name] for w in chosen), []),
                 *sum((['--source', '{staged:' + w.name + '}'] for w in chosen), []), '--frames', '{frames}']
    spec = common('experiments.tennis.k6b_rgb_diagnostic', arguments,
                  {'windows': manifest, 'base': base, **{w.name: source_files(windows)[w.name] for w in chosen}})
    spec.update({'device': 'gpu', 'hosts': ['gpu3', 'gpu2'], 'gpu_models': ['RTX A6000'], 'gpu_memory_mib': 12000, 'cpu_threads': 4, 'local_storage_gib': 40,
                 'scale': {'frames': {'smoke': 24, 'full': 240}},
                 'smoke': {'seconds': 360, 'representative_basis': 'First window, 24 frames (8 ticks): same GPU conversion, bicubic resize and base decode checks as the full stage.'},
                 'full': {'seconds': 600}, 'validator_seconds': 120, 'budget_seconds': 1000})
    return spec


def rate_transport(windows: list[wm.WindowSpec], manifest: tuple[str, str], anchors: list[tuple[str, str]],
                   prepared: list[tuple[str, str]], max_windows: int) -> dict[str, Any]:
    """CPU transport packets (KPS2 keypoints, exact mattes, reference crops) for the windows; `--max-windows` is not used in manifest
    mode, so the smoke is the first window by `--window`. Per-window rows are written as they finish."""
    chosen = windows[:1] if max_windows else windows
    arguments = ['--windows-file', '{staged:windows}', *sum((['--window', w.name] for w in chosen), [])]
    files: dict[str, tuple[str, str]] = {'windows': manifest}
    for i, a in enumerate(anchors):
        arguments += ['--anchor', '{staged:a%d}' % i]
        files[f'a{i}'] = a
    for i, q in enumerate(prepared):
        arguments += ['--prepared', '{staged:p%d}' % i]
        files[f'p{i}'] = q
    spec = common('experiments.tennis.k6b_rate_transport', arguments, files)
    spec.update({'device': 'cpu', 'hosts': HOSTS_GPU, 'gpu_models': [], 'gpu_memory_mib': 0, 'cpu_threads': 4, 'local_storage_gib': 30,
                 'scale': {}, 'smoke': {'seconds': 400, 'representative_basis': 'First window, every packet kind, validator included.'},
                 'full': {'seconds': 900}, 'validator_seconds': 120, 'budget_seconds': 1500})
    return spec


def real_base_judge(windows: list[wm.WindowSpec], files: dict[str, tuple[str, str]], gens: list[tuple[str, str]], rates: tuple[str, str],
                    judged: int) -> dict[str, Any]:
    """Hybrid compose and score on manifest windows (in-job controls replace the prior-run gate). `files` needs anchor, transport, masks,
    gpu_base, base, prior (the receiver stage's result), lpips, dists, windows. Smoke = 3 judged ticks of the first window."""
    chosen = _first(windows, judged != 0)
    arguments = ['--windows-file', '{staged:windows}', *sum((['--window', w.name] for w in chosen), []),
                 *sum((['--source', '{staged:' + w.name + '}'] for w in chosen), []),
                 *sum((['--gen', '{staged:g%d}' % i] for i in range(len(chosen))), []), '--rates', '{staged:rates}',
                 '--anchor', '{staged:anchor}', '--transport', '{staged:transport}', '--masks', '{staged:masks}',
                 '--gpu-base', '{staged:gpu_base}', '--base', '{staged:base}', '--prior', '{staged:prior}',
                 '--lpips', '{staged:lpips}', '--dists', '{staged:dists}', '--judged', '{judged}', '--cost-model', 'separated-v1', '--no-early-stop']
    staged_files = {**files, 'rates': rates, **{w.name: source_files(windows)[w.name] for w in chosen}}
    staged_files.update({f'g{i}': g for i, g in enumerate(gens[:len(chosen)])})
    spec = common('experiments.tennis.k6b_real_base_judge', arguments, staged_files)
    spec.update({'device': 'gpu', 'hosts': HOSTS_GPU, 'gpu_models': GPU_MODELS, 'gpu_memory_mib': 16500, 'cpu_threads': 4, 'local_storage_gib': 60,
                 'scale': {'judged': {'smoke': 3, 'full': 0}},
                 'gpu_sharing': {'mode': 'shared', 'basis': 'Saved-pixel scoring; scores do not depend on wall time', 'margin_mib': 4096},
                 'smoke': {'seconds': SMOKE_SECONDS, 'representative_basis': 'Three saved ticks (0/3/6) of the first window: all three base scales, hybrid, the in-job controls and all four metrics.'},
                 'full': {'seconds': FULL_SECONDS}, 'validator_seconds': 120, 'budget_seconds': FULL_SECONDS + 800})
    return spec


def first_window(argv: list[str]) -> None:
    """`first-window STAGE WINDOWS.json OUT_DIR [NAME=PATH:SHA256 ...]`: the spec of one stage of the first window (al028), written to OUT_DIR/STAGE.json.
    Upstream archives come as NAME=PATH:SHA256 (path under the data root; sha256sum of the completed job's full/published.tar)."""
    stage, manifest_path, out = argv[0], Path(argv[1]), Path(argv[2])
    ups = {k: (v.rsplit(':', 1)[0], v.rsplit(':', 1)[1]) for k, v in (a.split('=', 1) for a in argv[3:])}
    windows = wm.load(manifest_path)
    first = windows[0]
    windows_ref = (W7_DIR + '/windows.json', wm.file_sha256(manifest_path))
    confirm = ups.get('manifest', (W7_DIR + '/manifest.json', ''))
    source = source_files(windows)[first.name]
    if stage == 'features':
        spec = features(first, confirm, source)
    elif stage == 'native-base':
        spec = native_base([first], windows_ref, 24)
    elif stage == 'anchors':
        spec = anchors(first, confirm, ups['inputs'], source)
    elif stage == 'prepare':
        spec = prepare(first, confirm, ups['inputs'], ups['anchor'], source)
    elif stage == 'build-inputs':
        spec = build_inputs(first, confirm, ups['features'])
    else:
        raise SystemExit(f'unknown stage {stage}')
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{stage}.json').write_text(json.dumps(spec, indent=1) + '\n')


def main(argv: list[str]) -> None:
    if argv and argv[0] == 'first-window':
        return first_window(argv[1:])
    manifest_path, out = Path(argv[0]), Path(argv[1])
    windows = wm.load(manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    sha = wm.file_sha256(manifest_path)
    manifest = (W7_DIR + '/windows.json', sha)
    for frames in (24, 240):
        spec = native_base(windows, manifest, frames)
        (out / f'native-base-{"smoke" if frames == 24 else "full"}.json').write_text(json.dumps(spec, indent=1) + '\n')
    base = (W7_DIR + '/native-base-published.tar', '0' * 64)  # placeholder until the native-base job's archive hash is known
    spec = rgb_diagnostic(windows, manifest, base, 24)
    (out / 'rgb-diagnostic-smoke.json').write_text(json.dumps(spec, indent=1) + '\n')
    (out / 'rgb-diagnostic-full.json').write_text(json.dumps(rgb_diagnostic(windows, manifest, base, 240), indent=1) + '\n')
    (out / 'stages.json').write_text(json.dumps({'stages': STAGES, 'gpu_seconds_per_window': per_window_gpu_seconds(), 'budget': budget_check(len(windows)),
                                                 'windows': [w.name for w in windows]}, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
