"""Post-seal GPU scoring of W8's 80 receiver frames against original evidence.

Source RGB, target regions, old scores and judge weights are offline-only. The
caller must stage a source RGB NPY whose pixels reproduce the saved original
receiver hashes. No model or source data is visible to receiver stages.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis import w8_input_job as fixed
from experiments.tennis import w8_receiver_finish as finish

REGION_ARCHIVE = '4eb91e79c287a49accc17fcd115819e7345bc078d49e1477ce43ccdf87cd549b'
JUDGE_ARCHIVE = 'bf05e2630d898a2ec68452f81407e063b0f0ad0490365c41d53a3988f5adb299'
RECEIVER_SHA = '46ae0e636f9efe44741cae1dd4731c39d54aa845b3ed3ddb7cff3611ca582751'
LPIPS_SHA = '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'
DISTS_SHA = '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'
CODE_SHA = {
    'experiments/visor/h3b.py': 'a5b51f9586e17e1e86b33e34fd6bc296e9e0ede8203e0cbf3829f1af88acc6e2',
    'experiments/tennis/k6.py': 'a8da4febac96249983713e74144e1deec1953a644c25a3ce111a2decec985dd7',
    'src/codecs/quality.py': 'bb330efdcf49694e86edca8311fcfdb4e08eb1db3316d81a112f6b808e9b8b46',
}


def _member(path: Path, name: str, maximum: int) -> bytes:
    found = []
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            if member.name == name:
                if not member.isfile() or not 0 < member.size <= maximum:
                    raise ValueError('invalid judge archive member')
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError('missing archive member data')
                data = stream.read(maximum + 1)
                if len(data) != member.size:
                    raise ValueError('truncated archive member')
                found.append(data)
    if len(found) != 1:
        raise ValueError(f'missing or duplicate archive member {name}')
    return found[0]


def _qualify_regions(region_archive: Path) -> tuple[np.ndarray, dict[str, Any]]:
    meta = json.loads(_member(region_archive, 'publish/masks.json', 2_000_000))
    packed_bytes = _member(region_archive, 'publish/masks.npz', 100_000_000)
    if (not meta.get('matches_k6_record') or meta.get('ticks') != finish.TICKS.tolist()
            or meta.get('masks_sha256') != fixed.digest(packed_bytes)):
        raise ValueError('original region metadata mismatch')
    with np.load(io.BytesIO(packed_bytes), allow_pickle=False) as archive:
        ticks, r, s = archive['ticks'], archive['r'], archive['s']
        if (not np.array_equal(ticks, finish.TICKS) or r.shape != (80, 259200)
                or s.shape != (80, 259200) or r.dtype != np.uint8 or s.dtype != np.uint8):
            raise ValueError('scoring region coverage')
        for i, tick in enumerate(finish.TICKS):
            expected = meta['hashes'][str(tick)]
            if fixed.digest(r[i].tobytes()) != expected['r'] or fixed.digest(s[i].tobytes()) != expected['s']:
                raise ValueError(f'original region hash at tick {tick}')
        return s.copy(), meta


def _source_hashes(receiver: Path) -> dict[str, str]:
    data = json.loads(receiver.read_text())
    rows = [row for row in data['rows'] if row['id'] == fixed.WINDOW]
    if len(rows) != 1 or set(rows[0]['source_rgb_frame_sha256']) != {str(t) for t in finish.TICKS}:
        raise ValueError('saved source RGB identities incomplete')
    return rows[0]['source_rgb_frame_sha256']


def run(config: dict[str, Any]) -> None:
    """Run after every receiver output has been sealed and checked."""
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('Linux and one-CPU affinity required')
    cap = config['cpu_seconds']
    if type(cap) is not int or not 1 <= cap <= 3600:
        raise ValueError('bounded CPU seconds required')
    if resource.getrlimit(resource.RLIMIT_CPU) != (cap, cap):
        raise ValueError('scoring bootstrap CPU limit missing')
    started = time.monotonic()
    files = config['files']
    required = set(finish.CANDIDATE_KEYS) | {'source_rgb', 'receiver', 'region_archive', 'judge_archive', 'lpips', 'dists'}
    if set(files) != required:
        raise ValueError('unexpected scoring inputs')
    # Open and bind the candidate seal before touching any validator-only data.
    candidate_names = set(finish.CANDIDATE_KEYS)
    candidate = finish._inputs({'files': {k: files[k] for k in candidate_names}}, candidate_names)
    finish.validate_candidate_seals(candidate)
    remaining = required - set(candidate)
    paths = candidate | finish._inputs({'files': {k: files[k] for k in remaining}}, remaining)
    for name, sha in {'receiver': RECEIVER_SHA, 'region_archive': REGION_ARCHIVE,
                      'judge_archive': JUDGE_ARCHIVE, 'lpips': LPIPS_SHA, 'dists': DISTS_SHA}.items():
        if fixed.file_sha(paths[name]) != sha:
            raise ValueError(f'original scoring input mismatch: {name}')
    repo = Path(__file__).resolve().parents[2]
    if any(fixed.file_sha(repo / path) != sha for path, sha in CODE_SHA.items()):
        raise ValueError('historical scoring code identity mismatch')
    output = finish._new_output(config)
    source_hashes = _source_hashes(paths['receiver'])
    masks, region_meta = _qualify_regions(paths['region_archive'])
    source = np.load(paths['source_rgb'], mmap_mode='r', allow_pickle=False)
    composite = np.load(paths['composite'], mmap_mode='r', allow_pickle=False)
    if (source.shape != (80, 1080, 1920, 3) or source.dtype != np.uint8
            or composite.shape != source.shape or composite.dtype != np.uint8):
        raise ValueError('source or composite coverage')
    import torch
    import torchmetrics
    import torchvision

    from experiments.tennis.k6 import DISTS_PAD
    from experiments.visor.h3b import dists_crops, load_dists, lpips_reduce
    from src.codecs.quality import load_lpips

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU metric networks required')
    if (torch.__version__ != '2.10.0+cu128' or torchmetrics.__version__ != '1.9.0'
            or torchvision.__version__ != '0.25.0+cu128' or np.__version__ != '1.26.4'):
        raise ValueError('historical metric runtime versions mismatch')
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) not in ((8, 6), (8, 9)):
        raise ValueError('unqualified metric GPU class')
    torch.cuda.set_per_process_memory_fraction(min(1.0, 20000 * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    lpips = load_lpips(paths['lpips'], 'cuda')
    dists = load_dists(paths['dists'], 'cuda')
    if not all(next(net.parameters()).is_cuda for net in (lpips, dists)):
        raise ValueError('metric networks are not on CUDA')
    scores = {name: np.full(80, np.nan, dtype=np.float64) for name in finish.METRICS}
    whole = np.ones((1080, 1920), dtype=bool)
    for i, tick in enumerate(finish.TICKS):
        truth = np.asarray(source[i])
        image = np.asarray(composite[i])
        if finish._array_sha(truth) != source_hashes[str(tick)]:
            raise ValueError(f'source RGB hash mismatch at tick {tick}')
        s = np.unpackbits(masks[i]).reshape(1080, 1920).astype(bool)
        ys, xs = np.nonzero(s)
        if not len(ys):
            raise ValueError('empty foreground mask')
        crop = (slice(max(0, ys.min()-DISTS_PAD), min(1080, ys.max()+1+DISTS_PAD)),
                slice(max(0, xs.min()-DISTS_PAD), min(1920, xs.max()+1+DISTS_PAD)))
        scores['full_lpips'][i], scores['foreground_lpips'][i] = lpips_reduce(lpips, [image], truth, [[whole, s]], 'cuda')[0]
        scores['full_dists'][i] = dists_crops(dists, [image], truth, whole, 'cuda')[0]
        scores['foreground_dists'][i] = dists_crops(dists, [image[crop]], truth[crop], s[crop], 'cuda')[0]
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError('nonfinite scores')
    score_file = output / 'scores.npz'
    np.savez_compressed(score_file, **scores, ticks=finish.TICKS)
    old = _member(paths['judge_archive'], 'publish/nadal_kyrgios__000@120/scores.npz', 2_000_000)
    try:
        with np.load(io.BytesIO(old), allow_pickle=False) as archive:
            if not np.array_equal(archive['ticks'], finish.TICKS):
                raise ValueError('original score ticks mismatch')
            comparison = finish._compare_scores(scores, {name: archive[name][3] for name in finish.METRICS})
        passed = True
    except ValueError as exc:
        comparison = {'error': str(exc)}
        passed = False
    scorer_identity = {'source_rgb_sha256': fixed.file_sha(paths['source_rgb']),
                       'receiver_sha256': RECEIVER_SHA, 'regions_archive_sha256': REGION_ARCHIVE,
                       'region_npz_sha256': region_meta['masks_sha256'], 'judge_archive_sha256': JUDGE_ARCHIVE,
                       'lpips_sha256': LPIPS_SHA, 'dists_sha256': DISTS_SHA, 'code_sha256': CODE_SHA}
    receipt = {'version': 1, 'passed': passed, 'composite_sha256': fixed.file_sha(paths['composite']),
               'scores_sha256': fixed.file_sha(score_file), 'ticks': finish.TICKS.tolist(),
               'scorer_identity': scorer_identity, 'comparison': comparison,
               'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'torchmetrics': torchmetrics.__version__,
               'torchvision': torchvision.__version__, 'numpy': np.__version__, 'cpu_seconds': sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
               'wall_seconds': time.monotonic()-started}
    (output / 'sealed.json').write_text(json.dumps(receipt, sort_keys=True))
    if not passed:
        raise ValueError('original per-tick metric gate failed')


def prepare_source(config: dict[str, Any]) -> None:
    """Post-seal, reproduce the original 80 source RGB ticks from the original clip."""
    if sys.platform != 'linux' or len(os.sched_getaffinity(0)) != 1:
        raise ValueError('Linux and one CPU required')
    parent, child, wall = (config[k] for k in ('cpu_seconds', 'child_cpu_seconds', 'wall_seconds'))
    if any(type(n) is not int or not 1 <= n <= 3600 for n in (parent, child, wall)):
        raise ValueError('bounded source preparation required')
    if resource.getrlimit(resource.RLIMIT_CPU) != (parent, max(parent, child)):
        raise ValueError('source bootstrap CPU limits missing')
    files = config['files']
    required = set(finish.CANDIDATE_KEYS) | {'source_clip', 'receiver', 'ffmpeg'}
    if set(files) != required:
        raise ValueError('unexpected source preparation inputs')
    candidate_names = set(finish.CANDIDATE_KEYS)
    paths = finish._inputs({'files': {k: files[k] for k in candidate_names}}, candidate_names)
    finish.validate_candidate_seals(paths)
    remaining = required - candidate_names
    paths.update(finish._inputs({'files': {k: files[k] for k in remaining}}, remaining))
    expected = {'source_clip': 'b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d',
                'ffmpeg': finish.FFMPEG_SHA, 'receiver': RECEIVER_SHA}
    for name, sha in expected.items():
        if fixed.file_sha(paths[name]) != sha:
            raise ValueError(f'source preparation identity: {name}')
    output = finish._new_output(config)
    from experiments.tennis.k4b_features import convert_command
    argv = convert_command(str(paths['ffmpeg']), paths['source_clip'], output / 'source.yuv', 0, 240)
    argv[argv.index('-y')] = '-n'
    from experiments.tennis.w8_receiver_native_bootstrap import parent_death_signal
    parent_pid = os.getpid()
    def child_limit() -> None:
        parent_death_signal(parent_pid)
        resource.setrlimit(resource.RLIMIT_CPU, (child, child))
    started = time.monotonic()
    with (output / 'source.stdout').open('xb') as stdout, (output / 'source.stderr').open('xb') as stderr:
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                    close_fds=True, start_new_session=True, preexec_fn=child_limit)
        except OSError as exc:
            (output / 'source-execution.json').write_text(json.dumps(
                {'status': 'launch_failed', 'pid': None, 'pgid': None,
                 'started_monotonic': started, 'error': str(exc)}, sort_keys=True))
            raise
        if proc.stdin is not None:
            proc.stdin.close()
        handle = {'pid': proc.pid, 'pgid': proc.pid, 'started_monotonic': started,
                  'argv': argv, 'wall_limit_seconds': wall}
        (output / 'source-handle.json').write_text(json.dumps(handle, sort_keys=True))
        status = 'completed'
        try:
            returncode = proc.wait(timeout=wall)
        except subprocess.TimeoutExpired:
            status = 'timeout'
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = proc.wait(timeout=10)
        (output / 'source-execution.json').write_text(json.dumps(
            {**handle, 'status': status, 'returncode': returncode, 'wall_seconds': time.monotonic()-started},
            sort_keys=True))
    yuv = output / 'source.yuv'
    if (status != 'completed' or returncode or not yuv.is_file() or yuv.stat().st_size != 240 * 1920 * 1080 * 3 // 2
            or fixed.file_sha(yuv) != '5a26ceb4c08fdfac2726805bb37c5cefe29ff18b9a3c733690fd4e263d3d972d'):
        raise ValueError('original source conversion failed')
    import torch

    from src.codecs.quality import read_frames, yuv420_to_rgb
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('source color conversion requires CUDA')
    hashes = _source_hashes(paths['receiver'])
    target = np.lib.format.open_memmap(output / 'source-rgb.npy', mode='w+', dtype=np.uint8,
                                       shape=(80, 1080, 1920, 3))
    saved = 0
    for first in range(0, 240, 8):
        planes = torch.from_numpy(read_frames(yuv, 1920, 1080, first, min(8, 240-first))).to('cuda')
        rgb = yuv420_to_rgb(planes, 1920, 1080, full_range=False)
        for tick in range(first, min(240, first+8)):
            if tick % 3:
                continue
            frame = rgb[tick-first].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            if finish._array_sha(frame) != hashes[str(tick)]:
                raise ValueError(f'original source RGB differs at tick {tick}')
            target[saved] = frame
            saved += 1
    if saved != 80:
        raise ValueError('source tick coverage')
    target.flush()
    del target
    record = {'version': 1, 'source_rgb_sha256': fixed.file_sha(output / 'source-rgb.npy'),
              'source_yuv_sha256': fixed.file_sha(yuv), 'source_clip_sha256': expected['source_clip'],
              'ffmpeg_sha256': expected['ffmpeg'], 'receiver_sha256': RECEIVER_SHA,
              'ticks': finish.TICKS.tolist(), 'argv': argv, 'device': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'parent_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
              'child_cpu_seconds': sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2]),
              'wall_seconds': time.monotonic() - started}
    (output / 'sealed.json').write_text(json.dumps(record, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('source', 'score'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    data = args.config.read_bytes()
    if fixed.digest(data) != args.config_sha256:
        raise ValueError('scoring config mismatch')
    config = json.loads(data)
    if args.phase == 'source':
        prepare_source(config)
    else:
        run(config)


if __name__ == '__main__':
    main()
