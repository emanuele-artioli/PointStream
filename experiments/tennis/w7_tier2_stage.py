"""Extract and verify selected W7 Tier 2 windows, then stage new fleet inputs.

Each lossless H.264 clip is compared byte-for-byte after K4's YUV conversion
with the corresponding native broadcast frames before it is published.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from experiments.tennis import k4b_features, k6b_confirm, w7_select, window_manifest
from tools.datasets.k6b_recovery import SCOPE, sha

FRAME_BYTES = 1920 * 1080 * 3 // 2
SOURCE_SHA256 = "676331e0f4843f81a250b24c8a2f7ac0bb119ead4ffad0f145651ffb91a43e66"


def selected_rows(record: dict[str, Any], chunks: Path) -> list[dict[str, Any]]:
    """Read and rebase the exact 240 broadcast registration rows for one clip."""
    start = int(record['broadcast_start'])
    first = start // 200 * 200
    rows = [r for s in range(first, start + 240, 200)
            for r in json.loads((chunks / f'chunk-{s:06d}.json').read_text())
            if start <= r['frame'] < start + 240]
    if [r['frame'] for r in rows] != list(range(start, start + 240)):
        raise ValueError('registration coverage is incomplete')
    rebased = [{**r, 'frame': i} for i, r in enumerate(rows)]
    if len(k6b_confirm.gate(rebased)) < 228:
        raise ValueError('selected clip no longer passes registration')
    return rebased


def converted_hash(ffmpeg: str, video: Path, start: int, out: Path) -> str:
    subprocess.run(k4b_features.convert_command(ffmpeg, video, out, start, 240), check=True, capture_output=True, timeout=1800)
    if out.stat().st_size != 240 * FRAME_BYTES:
        raise ValueError(f'{video}: incorrect converted length')
    return w7_select.file_sha256(out)


def run(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text())
    half = selection['half']
    if half not in ('fit', 'eval') or len(selection['selected']) != 3:
        raise ValueError('selection must contain exactly three selected shots')
    if w7_select.file_sha256(Path(args.source)) != SOURCE_SHA256:
        raise ValueError('broadcast source hash changed')
    work, dest = Path(args.work), Path(args.dest)
    work.mkdir(parents=True, exist_ok=False)
    (work / 'sources' / 'nadal_kyrgios').mkdir(parents=True)
    records = selection['selected']
    windows: list[window_manifest.WindowSpec] = []
    inventories: dict[str, dict[str, Any]] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    lineage = []
    for i, rec in enumerate(records):
        start = int(rec['broadcast_start'])
        if not int(rec['shot_start']) <= start or start + 240 > int(rec['shot_end']):
            raise ValueError('selection crosses a scene cut')
        if (half == 'fit' and start + 240 > 57753) or (half == 'eval' and start < 57753):
            raise ValueError('selection crosses the frozen time split')
        number = f'{i + (100 if half == "fit" else 0):03d}'
        name, cid = 'nad' + number, f'nadal_kyrgios/{number}'
        raw = work / f'{name}-source.yuv'
        original = converted_hash(args.ffmpeg, Path(args.source), start, raw)
        clip = work / 'sources' / f'{cid}.mp4'
        subprocess.run([args.ffmpeg, '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
                        '-s:v', '1920x1080', '-r', '50', '-i', str(raw), '-frames:v', '240', '-an',
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0', '-pix_fmt', 'yuv420p', str(clip)],
                       check=True, capture_output=True, timeout=1800)
        check_raw = work / f'{name}-decoded.yuv'
        decoded = converted_hash(args.ffmpeg, clip, 0, check_raw)
        if decoded != original:
            raise ValueError(f'{name}: lossless extraction differs from broadcast conversion')
        digest = w7_select.file_sha256(clip)
        windows.append(window_manifest.WindowSpec(name, 'nadal_kyrgios', number, 0, 240, 50, digest, original, tier='w7-tier2'))
        inventories[cid] = {'frames': 240, 'size': '1920x1080'}
        all_rows[cid] = selected_rows(rec, selection_path.parent)
        lineage.append({'window': name, 'broadcast_start': start, 'broadcast_end': start + 240,
                        'source_yuv_sha256': original, 'clip_sha256': digest})
        raw.unlink()
        check_raw.unlink()
        (work / 'lineage.json').write_text(json.dumps(lineage, indent=2))
        print(name, 'verified', start, start + 240, flush=True)
    scope = Path(args.scope)
    if sha(scope) != SCOPE:
        raise ValueError('changed old input inventory')
    old = sorted({v['clip'] for v in json.loads(scope.read_text())['inventory'].values()})
    manifest = w7_select.confirm_manifest(windows, inventories, all_rows, old, str(dest), str(work / 'sources'))
    manifest['broadcast_lineage'] = lineage
    k6b_confirm.check_manifest(manifest)
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'sources').mkdir()
    for w in windows:
        source = work / 'sources' / w.clip_id
        target = dest / 'sources' / w.source_file
        shutil.copyfile(source.with_suffix('.mp4'), target)
        if w7_select.file_sha256(target) != w.source_sha256:
            raise ValueError(f'{w.name}: staged source differs')
    window_manifest.dump(windows, dest / 'windows.json', selection_sha256=w7_select.file_sha256(selection_path), seed=20261030)
    (dest / 'manifest.json').write_text(json.dumps(manifest))
    (work / 'staged.json').write_text(json.dumps({'dest': str(dest), 'windows_sha256': w7_select.file_sha256(dest / 'windows.json'),
                                                  'manifest_sha256': w7_select.file_sha256(dest / 'manifest.json')}, indent=2))
    print('staged', dest, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--scope', required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--dest', required=True)
    p.add_argument('--ffmpeg', default='/opt/local/bin/ffmpeg')
    run(p.parse_args())


if __name__ == '__main__':
    main()
