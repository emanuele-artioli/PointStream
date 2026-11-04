"""Bounded CPU fleet adapter for the two existing W7 CRF-70 streams.

The native oracle checks full decoded bytes before scratch is released. Only
small transport files, probes and the report are published. No quality claim.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from typing import Any

from experiments.tennis import w7_remux_oracle as oracle

ENV_SHA = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
FFMPEG_SHA = '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2'
FFPROBE_SHA = 'f4e72e0094397518602881be95f16a1d60ec41b03ecfb0aaa6f8e45ffa14583a'
MEMBER = 'publish/streams/s2_o20_crf70.ivf'
CASES = {
    'med000': {'archive': '172fea604e4ea1464dc22917b0737e315515ed38629723a7d5dcde7c2c2e84db',
               'stream': '29266563b82ff50bef0fbc11644b9d1040829c5385d949be547ca32794112111',
               'decoded': '4331b42da6c3aeb8445f035f3ac0fd41b42874db3db908cead44b1b5498c6426'},
    'med019': {'archive': '3fdb14f3c3ca3c5c39477d85735a58ddebe65eee0ccf01e7e5424df1170a5ee5',
               'stream': 'a7a5678ca18ec89c8b28e00198b441d1c780325ef706129a8d52ed3e5c82b80b',
               'decoded': '651e5a67f4d383cb535cdb1d85e2552546fde3ddbed93921ee1d1876e0226dce'},
}


def context(stage: Path, case: str) -> tuple[Path, Path, dict[str, Any]]:
    spec = json.loads((stage.parent / 'spec.json').read_text())
    staged = json.loads((stage / 'staging.json').read_text())
    dispatch = json.loads((stage / 'dispatch.json').read_text())
    if (case != {'smoke': 'med000', 'full': 'med019'}.get(stage.name)
            or spec.get('device') != 'cpu' or spec.get('single_cpu_campaign') is not True
            or spec.get('cpu_threads') != 1 or len(dispatch.get('cpu_affinity', [])) != 1
            or not 0 < spec.get('budget_seconds', 0) <= 250
            or not 0 < spec.get('validator_seconds', 0) <= 15
            or any(not 0 < spec.get(phase, {}).get('seconds', 0) <= 100
                   for phase in ('smoke', 'full'))
            or spec.get('environment', {}).get('sha256') != ENV_SHA):
        raise ValueError('wrong case, one-CPU campaign, environment or budget')
    declarations = {item['name']: item for item in spec['staged_inputs']}
    records = {item['name']: item for item in staged['inputs']}
    if (len(declarations) != len(spec['staged_inputs']) or len(records) != len(staged['inputs'])
            or set(declarations) != set(CASES) or set(records) != set(CASES)):
        raise ValueError('exactly the two distinct archived inputs required')
    for name, expected in CASES.items():
        if any(item.get('sha256') != expected['archive'] or item.get('extract') is not True
               for item in (declarations[name], records[name])):
            raise ValueError(f'archive staging identity mismatch: {name}')
    root = Path(staged['local_root']).resolve(strict=True)
    tree = Path(records[case]['path']).resolve(strict=True)
    source = (tree / MEMBER).resolve(strict=True)
    if not tree.is_relative_to(root) or not source.is_relative_to(tree):
        raise ValueError('source must be inside staged host-local tree')
    oracle.check_hash(source, CASES[case]['stream'])
    bin_dir = Path(staged['environment']['bin']).resolve(strict=True)
    if not bin_dir.is_relative_to(root) or staged['environment']['sha256'] != ENV_SHA:
        raise ValueError('packed tools must be host local')
    return source, bin_dir, spec


def audit_report(report: dict[str, Any], original: dict[str, Any], remux: dict[str, Any],
                 remux_bytes: bytes, case: str) -> None:
    import hashlib
    result = oracle.validate_probes(original, remux, width=960, height=544, frames=240, fps=Fraction(50))
    identities = report.get('identities', {})
    if (report.get('status') != 'passed' or report.get('scope') != 'transport only; no quality score'
            or identities != {'input': CASES[case]['stream'], 'ffmpeg': FFMPEG_SHA, 'ffprobe': FFPROBE_SHA}
            or report.get('historical_decoded_sha256') != CASES[case]['decoded']
            or report.get('decoded_yuv_sha256') != CASES[case]['decoded']
            or report.get('historical_decoded_match') is not True
            or report.get('metadata') != result
            or report.get('remux_mkv_bytes') != len(remux_bytes)
            or report.get('remux_sha256') != hashlib.sha256(remux_bytes).hexdigest()
            or not isinstance(report.get('source_ivf_bytes'), int)
            or report.get('bytes_saved') != report['source_ivf_bytes'] - len(remux_bytes)):
        raise ValueError('published report, decoded reference or transport bytes disagree')


def run(stage: Path, scratch: Path, case: str) -> None:
    source, bin_dir, _ = context(stage, case)
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError('one CPU affinity required')
    scratch = scratch.resolve(strict=True)
    if not scratch.is_relative_to(Path(json.loads((stage / 'staging.json').read_text())['local_root']).resolve(strict=True)):
        raise ValueError('scratch must be host local')
    args = argparse.Namespace(input=str(source), input_sha256=CASES[case]['stream'],
                              ffmpeg=str(bin_dir / 'ffmpeg'), ffmpeg_sha256=FFMPEG_SHA,
                              ffprobe=str(bin_dir / 'ffprobe'), ffprobe_sha256=FFPROBE_SHA,
                              workdir=str(scratch / 'oracle'), width=960, height=544,
                              frames=240, fps='50', original_decoded_sha256=CASES[case]['decoded'])
    report = oracle.run(args)
    work = scratch / 'oracle'
    original = json.loads((work / 'probe_original.json').read_text())
    remux = json.loads((work / 'probe_remux.json').read_text())
    audit_report(report, original, remux, (work / 'remux.mkv').read_bytes(), case)
    if report['source_ivf_bytes'] != source.stat().st_size:
        raise ValueError('source byte count disagrees')
    publish = scratch / 'publish'
    publish.mkdir()
    for name in ('report.json', 'probe_original.json', 'probe_remux.json', 'remux.mkv', 'commands.json'):
        shutil.copyfile(work / name, publish / name)


def validate(stage: Path, output: Path) -> int:
    try:
        case = 'med000'
        source, _, spec = context(stage, case)
        with tarfile.open(stage / 'published.tar') as archive:
            def read(name: str) -> bytes:
                member = archive.getmember('publish/' + name)
                if not member.isfile() or member.size > 2_000_000:
                    raise ValueError('unexpected publication member')
                file = archive.extractfile(member)
                if file is None:
                    raise ValueError('unreadable publication member')
                return file.read()
            report = json.loads(read('report.json'))
            original = json.loads(read('probe_original.json'))
            remux = json.loads(read('probe_remux.json'))
            audit_report(report, original, remux, read('remux.mkv'), case)
        if report['source_ivf_bytes'] != source.stat().st_size:
            raise ValueError('published source byte count differs')
        dispatch = json.loads((stage / 'dispatch.json').read_text())
        clock = dispatch['campaign_clock']
        now = time.monotonic()
        measured = now - clock['phase_started']
        projected = max(5.0, 1.5 * measured)
        remaining = clock['budget_end'] - now
        reserve = max(10.0, 1.25 * (clock['phase_started'] - clock['started']))
        if (not clock['started'] <= clock['phase_started'] <= now < clock['budget_end']
                or projected > spec['full']['seconds'] or remaining < projected + reserve):
            raise ValueError('full stage does not fit conservative remaining budget projection')
        result = {'passed': True, 'checks': {'staged_member': True, 'remux_sha': True,
                  'probe_cadence_color': True, 'historical_yuv_sha': True, 'full_budget': True},
                  'projection': {'smoke_seconds': measured, 'full_seconds': projected,
                                 'reserve_seconds': reserve, 'remaining_seconds': remaining}}
        output.write_text(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        output.write_text(json.dumps({'passed': False, 'checks': {'remux_audit': False},
                                      'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=sorted(CASES))
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    stage = Path(os.environ['PS_STAGE_DIR'])
    if args.validate:
        return validate(stage, Path(os.environ['PS_VALIDATION_PATH']))
    if args.case is None:
        parser.error('--case required')
    run(stage, Path(os.environ['PS_SCRATCH_DIR']), args.case)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
