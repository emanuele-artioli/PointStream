"""Prospective shared-model W7 scorer; no job or budget admission.

A point uses the existing W1 scorer, including its pinned native decoder,
colour conversion, 80 ticks, cubic upscale, masks, and control tolerances.
The Matroska byte receipt must already exist; no encoding or network access.
"""
from __future__ import annotations

import json
import math
import os
import re
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import w1_roi as roi, w1_score as old, w7_mkv_profile as profile, w7_analysis
from experiments.tennis.k5d_timing import Timer

WINDOWS = ('al028', 'al034', 'al043', 'med000', 'med024', 'med019')
INVENTORY_SHA = '867d9fff8974f4ec5320c9b830138d4aac2ee22c285d7de45aebb04dff3bab11'
BRIDGE_SHA = 'a32a91b0e52d80cd34e48168e6bc40dcd2335dd83eccda803d9f3c41c0407642'
BRIDGE_PATH = Path(__file__).resolve().parents[2] / 'docs/experiments/w7-inputs/w1-scorer-source-compatibility-20261105.json'


def source_bridge() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if old.digest(BRIDGE_PATH) != BRIDGE_SHA:
        raise ValueError('reviewed W1 source bridge changed')
    bridge = json.loads(BRIDGE_PATH.read_text())
    if (bridge.get('schema') != 1 or old.digest(Path(old.__file__)) != bridge['repair_w1_score_sha256']
            or not {'score_point', 'control_errors'}.issubset(set(bridge['same_function_ast']))
            or any(old.digest(root / name) != sha for name, sha in bridge['common_helpers_sha256'].items())):
        raise ValueError('current scorer or reviewed helper identity differs')
    return bridge


def check_hybrid_state(window: roi.Window, actual_sha: str, spec_path: Path, ready_path: Path) -> None:
    if window.hybrid_archive_sha is not None:
        if window.hybrid_archive_sha != actual_sha:
            raise ValueError('hybrid control archive differs from frozen window')
        return
    from experiments.jobs.inbox import digest as spec_digest
    legacy = {
        'med000': ('9b3d405e22326c26a88ce6f6a15c3aa8b63fb59f8eccdb6714221de7c82339fc',
                   '8a7575bf77e28c573a8c9a35e5d294d258d1413c98f4244a12056302d4b94b98',
                   '2f01e4df611c1b6c3c08890ec88cc1c6881517c288cbec722eb20699e741cf03'),
        'med019': ('ef489d1009f57e47b23480ee94388461e19b64bc9a87d0d64b61df1fadecfb07',
                   'c1a5274c365be81083eb03d3c98751b43e290370f5b465d01073220bc9572e34',
                   '812c18a2aa7943c1d9ce28a4685529a5fcfebefabf8d30b441a238f4b9d92335')}
    expected = legacy.get(window.name)
    if expected is None or old.digest(spec_path) != expected[0] or old.digest(ready_path) != expected[1]:
        raise ValueError('null hybrid state requires authenticated historical spec and ready')
    spec, ready = json.loads(spec_path.read_text()), json.loads(ready_path.read_text())
    records = [x for x in spec['staged_inputs'] if x['name'] == 'hybrid']
    if (spec_digest(spec) != ready.get('spec_sha256') or len(records) != 1
            or records[0]['sha256'] != actual_sha
            or actual_sha != expected[2]):
        raise ValueError('historical spec hybrid identity differs')


def check_point(row: dict[str, Any], proof: dict[str, Any], validation: dict[str, Any], mkv: Path, window: roi.Window,
                source_sha: str, masks_sha: str, hybrid_sha: str, scorer_sha: str, bridge: dict[str, Any]) -> None:
    """Bind the actual native-YUV receipt and reviewed historical scoring inputs."""
    for key in ('window', 'name', 'scale', 'offset', 'crf', 'kind', 'score_archive_sha256',
                'encode_archive_sha256', 'ivf_member', 'original_ivf_sha256', 'historical_native_yuv_sha256'):
        if row[key] != proof[key]:
            raise ValueError('byte proof point identity mismatch: ' + key)
    if (row['window'] != window.name or proof['fps'] != str(window.fps) or proof['frames'] != 240
            or proof['profile'] != profile.PROFILE or proof.get('native_yuv_proven') is not True
            or proof.get('native_decoder_sha256') != profile.DAV1D_SHA
            or proof.get('native_yuv_sha256') != row['historical_native_yuv_sha256']
            or proof.get('ordered_payloads_equal') is not True or proof.get('timestamps_equal') is not True
            or validation.get('checks') != {'reextract': True, 'payloads': True, 'timestamps': True, 'native_yuv': True}
            or validation.get('source_ivf_sha256') != row['original_ivf_sha256']
            or validation.get('returned_ivf_sha256') != proof.get('reconstructed_ivf_sha256')
            or validation.get('delivered_mkv_sha256') != proof.get('delivered_mkv_sha256')
            or validation.get('native_yuv_sha256') != row['historical_native_yuv_sha256']
            or validation.get('native_decoder_sha256') != profile.DAV1D_SHA
            or old.digest(mkv) != proof['delivered_mkv_sha256']
            or mkv.stat().st_size != proof['delivered_mkv_bytes']
            or proof.get('wire_kbps') != mkv.stat().st_size * 8 * window.fps / 240 / 1000):
        raise ValueError('delivered profile or native input identity mismatch')
    binds = proof['source_and_scorer']
    original = row['source_and_scorer']
    for name in ('source_archive_sha256', 'masks_archive_sha256', 'hybrid_archive_sha256',
                 'lpips_sha256', 'dists_sha256', 'ticks', 'score_source_identity'):
        if binds[name] != original[name]:
            raise ValueError('byte proof does not bind the frozen scorer inventory: ' + name)
    if (binds['source_archive_sha256'] != source_sha or source_sha != window.source_sha
            or binds['masks_archive_sha256'] != masks_sha or binds['hybrid_archive_sha256'] != hybrid_sha
            or binds['lpips_sha256'] != roi.WEIGHTS['lpips'] or binds['dists_sha256'] != roi.WEIGHTS['dists']
            or binds['ticks'] != roi.TICKS or binds['score_source_identity'] is not None
            or scorer_sha != bridge['repair_w1_score_sha256']):
        raise ValueError('source, masks, models, ticks or scorer identity mismatch')
    expected_job = w7_analysis.W7_INPUTS[row['window']]['score']['job']
    if row['kind'] == 'control' and row.get('score_job') != expected_job:
        raise ValueError('historical control score job differs')
    if row['kind'] == 'new' and row.get('score_job') is not None:
        raise ValueError('new point cannot claim a historical score')
    source_records = [item for item in bridge['original_score_jobs'] if item['job'] == expected_job]
    if (len(source_records) != 1 or source_records[0]['files']['experiments/tennis/w1_score.py'] != bridge['historical_w1_score_sha256']):
        raise ValueError('historical W1 producer not bound to reviewed bridge')



def check_proof_publication(archive: Path, archive_sha: str, rows: list[dict[str, Any]], inputs: dict[str, Path],
                            *, require_complete: bool = False) -> None:
    """Bind all consumed files to a SHA-identified actual byte-proof publication."""
    import hashlib
    import tarfile
    if old.digest(archive) != archive_sha:
        raise ValueError('byte-proof publication hash differs')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        def read(name: str) -> bytes:
            matches = [m for m in members if m.name == name]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 100_000_000:
                raise ValueError('missing or ambiguous byte-proof member')
            stream = tar.extractfile(matches[0])
            if stream is None:
                raise ValueError('byte-proof member unavailable')
            return stream.read()
        coverage = json.loads(read('publish/coverage.json'))['completed']
        if require_complete and (len(coverage) != 75
                or len({(item['window'], item['name']) for item in coverage}) != 75):
            raise ValueError('full byte-proof publication requires 75 distinct roles')
        for row in rows:
            matches = [x for x in coverage if x['window'] == row['window'] and x['name'] == row['name']]
            if len(matches) != 1:
                raise ValueError('byte-proof publication role missing or ambiguous')
            entry = matches[0]
            prefix = f"publish/role-{entry['index']:03d}/"
            proof = read(prefix + 'proof.json')
            validation = read(prefix + 'independent-validation.json')
            mkv = read(prefix + 'delivered.mkv')
            if (entry['proof_sha256'] != hashlib.sha256(proof).hexdigest()
                    or entry['independent_validation_sha256'] != hashlib.sha256(validation).hexdigest()
                    or proof != inputs[row['proof_asset']].read_bytes()
                    or validation != inputs[row['validation_asset']].read_bytes()
                    or mkv != inputs[row['mkv_asset']].read_bytes()):
                raise ValueError('point assets do not match byte-proof publication')


def materialize_proofs(archive: Path, archive_sha: str, rows: list[dict[str, Any]], destination: Path) -> dict[str, Path]:
    """Extract only explicitly selected, coverage-bound proof assets from the audited publication."""
    import tarfile
    if old.digest(archive) != archive_sha:
        raise ValueError('byte-proof publication hash differs')
    destination.mkdir(parents=True, exist_ok=False)
    output: dict[str, Path] = {}
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        def read(name: str) -> bytes:
            matches = [m for m in members if m.name == name]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 100_000_000:
                raise ValueError('missing or ambiguous proof member')
            stream = tar.extractfile(matches[0])
            if stream is None:
                raise ValueError('unreadable proof member')
            return stream.read()
        coverage = json.loads(read('publish/coverage.json'))['completed']
        if len(coverage) != 75 or len({(x['window'], x['name']) for x in coverage}) != 75:
            raise ValueError('75-role coverage required')
        for row in rows:
            matches = [x for x in coverage if (x['window'], x['name']) == (row['window'], row['name'])]
            if len(matches) != 1:
                raise ValueError('missing selected role')
            prefix = f"publish/role-{matches[0]['index']:03d}/"
            for key, filename in (('proof_asset', 'proof.json'), ('validation_asset', 'independent-validation.json'), ('mkv_asset', 'delivered.mkv')):
                name = row[key]
                if name in output or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
                    raise ValueError('duplicate or unsafe selected asset name')
                path = destination / name
                path.write_bytes(read(prefix + filename))
                output[name] = path
    check_proof_publication(archive, archive_sha, rows, output, require_complete=True)
    return output

def cohort(rows: list[dict[str, Any]]) -> None:
    if (len(rows) != 22 or any(not re.fullmatch(r'[A-Za-z0-9_-]+', r['name']) for r in rows)
            or len({(x['window'], x['name']) for x in rows}) != 22):
        raise ValueError('exactly 22 distinct required scores')
    controls = [r for r in rows if r['kind'] == 'control']
    new = [r for r in rows if r['kind'] == 'new']
    if ({(r['window'], r['scale']) for r in controls} != {(w, s) for w in WINDOWS for s in (8, 4, 2)}
            or len(controls) != 18 or any(r['crf'] != 63 or r['offset'] != 0 for r in controls)
            or {(r['window'], r['crf']) for r in new} != {(w, c) for w in ('med000', 'med019') for c in (47, 70)}
            or len(new) != 4 or any(r['scale'] != 2 or r['offset'] != -20 for r in new)):
        raise ValueError('control or new-point cohort differs')


def forecast(pilot: dict[str, Any], *, source_bounds: dict[str, dict[str, float]],
             gpu_remaining: float, cpu_remaining: float, gpu_staging: float, cpu_staging: float,
             gpu_reserve: float, cpu_reserve: float) -> dict[str, float]:
    """Bound 22 scores and six source setups from two production-path timings."""
    if pilot.get('window') != 'med000' or pilot.get('point_count') != 2 or pilot.get('passed') is not True:
        raise ValueError('complete med000 two-point pilot required')
    if set(source_bounds) != set(WINDOWS):
        raise ValueError('six separately supported source-context bounds required')
    out = {}
    for kind, budget, reserve, staging in (('wall', gpu_remaining, gpu_reserve, gpu_staging),
                                           ('cpu', cpu_remaining, cpu_reserve, cpu_staging)):
        measured = pilot['cost'][kind]
        setup, source, points, spent = (measured[k] for k in ('setup', 'source', 'points', 'pilot_total'))
        sources = [source_bounds[w][kind] for w in WINDOWS]
        values = [setup, source, *points, spent, budget, reserve, staging, *sources]
        if (len(points) != 2 or any(isinstance(x, bool) or not isinstance(x, (float, int))
                or not math.isfinite(x) or x < 0 for x in values) or min(points) <= 0 or reserve <= 0
                or staging <= 0 or source_bounds['med000'][kind] < source):
            raise ValueError('invalid pilot timing or reserve')
        estimate = 1.5 * (setup + sum(sources) + 22 * max(points)) + staging + reserve + spent
        if estimate > budget:
            raise ValueError('complete scoring forecast exceeds remaining budget')
        out[kind + '_seconds'] = estimate
    return out


def score(args: Any, rows: list[dict[str, Any]], inputs: dict[str, Path], pilot: bool,
          selection: set[tuple[str, str]] | None = None) -> None:
    """Run all points in one process; `inputs` are already SHA-verified staged assets."""
    import torch
    from experiments.tennis.k4b_features import convert_command
    from experiments.visor.h3b import load_dists
    from src.codecs import svtav1
    from src.codecs.quality import load_lpips, yuv420_to_rgb

    started = time.monotonic()
    own = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    roi.preload_windows(['--windows-file', str(inputs['windows']), '--window-state', str(inputs['state'])])
    selected = [r for r in rows if r['window'] == 'med000' and
                ((r['kind'] == 'control' and r['scale'] == 8) or (r['kind'] == 'new' and r['crf'] == 70))] if pilot else rows
    cohort(rows)
    if selection is not None:
        selected = [r for r in rows if (r['window'], r['name']) in selection]
        if len(selected) != len(selection):
            raise ValueError('selected scoring roles missing')
    if pilot and len(selected) != 2:
        raise ValueError('required med000 control and CRF70 absent')
    if not torch.cuda.is_available() or 'A6000' not in torch.cuda.get_device_name(0) or 'Ada' in torch.cuda.get_device_name(0):
        raise ValueError('single A6000 GPU required')
    if len(os.sched_getaffinity(0)) != 1 or torch.cuda.device_count() != 1:
        raise ValueError('one CPU and one GPU required')
    torch.set_num_threads(4)  # Match the historical W1 scorer; affinity remains one CPU.
    tools = {name: svtav1.tool(name) for name in roi.TOOLS}
    if any(tools[name]['sha256'] != sha for name, sha in roi.TOOLS.items()):
        raise ValueError('native encoder/decoder tool identity changed')
    for name in ('lpips', 'dists'):
        if old.digest(inputs[name]) != roi.WEIGHTS[name]:
            raise ValueError('perceptual backbone identity changed')
    scorer_sha = old.digest(Path(old.__file__))
    bridge = source_bridge()
    check_proof_publication(inputs['byte_publication'], args.byte_publication_sha256, selected, inputs,
                            require_complete=args.action == 'full-study')
    for row in selected:
        w = row['window']
        proof = json.loads(inputs[row['proof_asset']].read_text())
        check_hybrid_state(roi.WINDOWS[w], old.digest(inputs[w + '_hybrid']),
                           inputs[w + '_historical_spec'] if w + '_historical_spec' in inputs else inputs['historical_spec'],
                           inputs[w + '_historical_ready'] if w + '_historical_ready' in inputs else inputs['historical_ready'])
        validation = json.loads(inputs[row['validation_asset']].read_text())
        check_point(row, proof, validation, inputs[row['mkv_asset']], roi.WINDOWS[w], old.digest(inputs[w + '_source']),
                    old.digest(inputs[w + '_masks']), old.digest(inputs[w + '_hybrid']), scorer_sha, bridge)
    scratch = Path(os.environ['PS_SCRATCH_DIR'])
    publish = scratch / 'publish'
    publish.mkdir(exist_ok=False)
    timer = Timer(publish / 'timing.json')
    def load_sync(loader: Any, path: Path) -> Any:
        net = loader(path, 'cuda')
        torch.cuda.synchronize()
        return net
    nets = {'lpips': timer.measure('model.lpips', lambda: load_sync(load_lpips, inputs['lpips'])),
            'dists': timer.measure('model.dists', lambda: load_sync(load_dists, inputs['dists']))}
    if not all(next(net.parameters()).is_cuda for net in nets.values()):
        raise ValueError('perceptual models must be on GPU')
    records = []
    for window_name in WINDOWS:
        subset = sorted((r for r in selected if r['window'] == window_name),
                        key=lambda r: (r['kind'] != 'control', r['scale'], r['name']))
        if not subset:
            continue
        window = roi.WINDOWS[window_name]
        work = scratch / ('w7-' + window_name)
        work.mkdir(exist_ok=False)
        def source_context() -> tuple[list[np.ndarray], Any, Any]:
            masks_root = old.extract(inputs[window_name + '_masks'], work / 'masks')
            hybrid_root = old.extract(inputs[window_name + '_hybrid'], work / 'hybrid')
            with np.load(masks_root / 'masks.npz') as z:
                masks = {k: z[k] for k in ('ticks', 'r', 's')}
            mask_summary = json.loads((masks_root / 'masks.json').read_text())
            if old.digest(masks_root / 'masks.npz') != mask_summary['masks_sha256']:
                raise ValueError('mask hash mismatch')
            with np.load(hybrid_root / window.hybrid_dir / 'scores.npz') as z:
                hybrid = {k: z[k] for k in old.SERIES}
                hybrid_ticks = z['ticks'].tolist()
            if masks['ticks'].tolist() != roi.TICKS or hybrid_ticks != roi.TICKS:
                raise ValueError('80-tick cohort differs')
            yuv = work / 'source.yuv'
            cmd = convert_command('ffmpeg', inputs[window_name + '_source'], yuv, window.source_start, roi.FRAMES)
            subprocess.run(cmd, check=True, capture_output=True, timeout=900)
            if old.digest(yuv) != window.source_yuv_sha:
                raise ValueError('source conversion differs from frozen window')
            raw = np.memmap(yuv, np.uint8, 'r', shape=(roi.FRAMES, 1620, 1920))
            truth = [yuv420_to_rgb(torch.from_numpy(np.array(raw[t], copy=True)[None]).cuda(), 1920, 1080,
                                   full_range=False)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy() for t in roi.TICKS]
            return truth, masks, hybrid
        truth, masks, hybrid = timer.measure('source.' + window_name, source_context)
        for row in subset:
            proof = json.loads(inputs[row['proof_asset']].read_text())
            mkv = inputs[row['mkv_asset']]
            ivf = work / (row['name'] + '.ivf')
            cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(mkv), '-map', '0:v:0',
                   '-an', '-c:v', 'copy', '-fps_mode', 'passthrough', '-f', 'ivf', '-n', str(ivf)]
            out = publish / f"{window_name}-{row['name']}.npz"
            def evaluate() -> dict[str, Any]:
                subprocess.run(cmd, check=True, capture_output=True, timeout=900)
                if old.digest(ivf) != proof['reconstructed_ivf_sha256']:
                    raise ValueError('Matroska reconstructed IVF differs from validated receiver input')
                point = {**row, 'size': roi.CODED[row['scale']], 'path': ivf,
                         'decoded_sha256': proof['historical_native_yuv_sha256']}
                result = old.score_point(point, work, roi.TICKS, truth, masks, nets, torch)
                if row['kind'] == 'control':
                    errors = old.control_errors(result['values'], hybrid, row['scale'], len(roi.TICKS))
                    if max(errors.values()) > roi.control_tolerance(window_name):
                        raise ValueError('historical control failed: ' + str(errors))
                else:
                    errors = None
                torch.cuda.synchronize()
                np.savez_compressed(out, ticks=np.asarray(roi.TICKS), values=result['values'])
                return {'errors': errors}
            measured = timer.measure(f"point.{window_name}.{row['name']}", evaluate)
            errors = measured['errors']
            records.append({'window': window_name, 'name': row['name'], 'kind': row['kind'],
                            'scale': row['scale'], 'proof_sha256': old.digest(inputs[row['proof_asset']]),
                            'mkv_sha256': proof['delivered_mkv_sha256'], 'mkv_bytes': proof['delivered_mkv_bytes'],
                            'fps': window.fps, 'wire_kbps': proof['wire_kbps'],
                            'native_yuv_sha256': proof['historical_native_yuv_sha256'],
                            'original_ivf_sha256': proof['original_ivf_sha256'],
                            'source_yuv_sha256': window.source_yuv_sha,
                            'source_and_scorer': proof['source_and_scorer'], 'control_errors': errors,
                            'score_sha256': old.digest(out)})
    torch.cuda.synchronize()
    end_own = resource.getrusage(resource.RUSAGE_SELF)
    end_child = resource.getrusage(resource.RUSAGE_CHILDREN)
    elapsed = time.monotonic() - started
    cpu = (end_own.ru_utime + end_own.ru_stime - own.ru_utime - own.ru_stime
           + end_child.ru_utime + end_child.ru_stime - child.ru_utime - child.ru_stime)
    timing_cost = {}
    if pilot:
        for kind, total, fields in (('wall', elapsed, ('wall_seconds',)),
                                    ('cpu', cpu, ('self_cpu_seconds', 'waited_children_cpu_seconds'))):
            def duration(event: dict[str, Any]) -> float:
                return float(sum(event[field] for field in fields))
            source_events = [e for e in timer.events if e['name'] == 'source.med000' and e['completed']]
            point_events = [e for e in timer.events if e['name'].startswith('point.med000.') and e['completed']]
            if len(source_events) != 1 or len(point_events) != 2:
                raise ValueError('pilot timing coverage differs')
            source = duration(source_events[0])
            points = [duration(event) for event in point_events]
            setup = total - source - sum(points)
            if setup < 0:
                raise ValueError('pilot timing intervals overlap unexpectedly')
            timing_cost[kind] = {'setup': setup, 'source': source, 'points': points, 'pilot_total': total}
    (publish / 'score.json').write_text(json.dumps({'schema': 1, 'pilot': pilot, 'points': records,
        'ticks': roi.TICKS, 'series': old.SERIES, 'tools': tools, 'scorer_sha256': scorer_sha,
        'source_identity': json.loads((Path(os.environ['PS_STAGE_DIR']) / 'dispatch.json').read_text())['code'],
        'cost': timing_cost, 'passed': True, 'window': 'med000' if pilot else None,
        'point_count': len(records), 'scope': 'saved per-tick W1 scores; no envelope or study admission'}, indent=2, allow_nan=False) + '\n')


def inventory(path: Path, sha: str) -> list[dict[str, Any]]:
    if sha != INVENTORY_SHA or old.digest(path) != sha:
        raise ValueError('inventory identity mismatch')
    data = json.loads(path.read_text())
    if (data.get('schema') != 1 or len(data.get('points', [])) != 53
            or len(data.get('controls', [])) != 18 or len(data.get('new_points', [])) != 4):
        raise ValueError('incomplete frozen 53/18/4 inventory')
    historical = data['points']
    if len({(row['window'], row['name']) for row in historical}) != 53:
        raise ValueError('historical inventory contains duplicates')
    rows = data['controls'] + data['new_points']
    cohort(rows)
    return rows


def bound_rows(rows: list[dict[str, Any]], bindings: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        key = row['window'] + '/' + row['name']
        value = bindings.get(key)
        if value is None:
            result.append(row)
            continue
        if (not isinstance(value, dict) or set(value) != {'proof_asset', 'validation_asset', 'mkv_asset'}
                or any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in value.values())):
            raise ValueError('invalid point asset binding')
        result.append({**row, **value})
    if set(bindings) - {r['window'] + '/' + r['name'] for r in rows}:
        raise ValueError('unregistered point binding')
    return result
