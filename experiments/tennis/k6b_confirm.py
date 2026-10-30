"""Frozen clip-local K6b diagnostic: input gate and feature-stage validation.

No quality selection, fitting or source recovery. Later stages must also pass
native K6/generator/judge validators; this gate cannot certify independence.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

FROZEN = (('alcaraz_hurkacz/031', 9, 25, '054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76'), ('alcaraz_hurkacz/013', 93, 25, '1af169200b28659c503499979863b6219b0d331f532962ac69fb505d13c4976c'), ('alcaraz_hurkacz/023', 108, 25, '88345a9deac75a4b6180daa56a2ec7cda2afea25babb5b549b937105bca45d1a'), ('alcaraz_hurkacz/016', 153, 25, '305a369d74882a628ff7efd6b49cf3f75c56a985d78ffd798863542df20551bd'), ('alcaraz_hurkacz/008', 141, 25, '355e4b0a3408e4c8f049baadf749c937e52930f57df5018867e5baa0cdd49e54'), ('alcaraz_hurkacz/047', 3, 25, '7bb3a53e3b416d3b84c0344c99b29d1374f8c49b54f970034f4f29a95609e99b'), ('medvedev_struff/014', 74, 50, 'a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be'), ('medvedev_struff/037', 229, 50, '97372d45ba53068078fdc85300a550221b2c4bd31a3074958b52081326e3feeb'), ('medvedev_struff/002', 218, 50, '49cc09d072d740bd4d5ed02fdd24f2b3edbdc5fb16a660f66f1e8148ff8be14a'), ('medvedev_struff/017', 86, 50, '64cc4bc0a0712b5daa6bc00c5ed592b31a0f8333cb36a4caf5b05d24f9e39873'), ('medvedev_struff/029', 18, 50, '013979d403ec9ae3e48ce3dc49bbf261b37fb6a3dd78af71d5b8d81419612e8f'), ('medvedev_struff/021', 126, 50, 'fa2da44fdc64d6d95c9a111953dd6d16a088cb534dc7336a98d3cb2557d30b4a'))
PROVENANCE = {"clip_audit": "complete", "original_source_gate": "unresolved",
              "claim_scope": "clip_local_diagnostic"}


def gate(rows: list[Any]) -> list[int]:
    """Relative frame indices passing the frozen K8b fits/finite-H rule."""
    if len(rows) != 240:
        raise ValueError("Registration must have exactly 240 rows, including missing rows")
    valid = []
    for t, r in enumerate(rows):
        if not r or r.get("fits") is not True:
            continue
        try:
            h = np.asarray(r.get("homography"), dtype=float)
            if h.shape == (3, 3) and np.isfinite(h).all() and np.linalg.matrix_rank(h) == 3:
                valid.append(t)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            continue
    return valid


def cohort_of(manifest: dict[str, Any]) -> tuple[tuple[str, int, int, str], ...]:
    """The cohort a manifest freezes: its `cohort` list (W7 manifests, experiments/tennis/window_manifest.py) or `FROZEN`."""
    if "cohort" not in manifest:
        return FROZEN
    return tuple((str(c), int(s), int(f), str(h)) for c, s, f, h in manifest["cohort"])


def check_manifest(manifest: dict[str, Any], frozen: tuple[tuple[str, int, int, str], ...] | None = None,
                   centered: bool | None = None) -> dict[str, list[int]]:
    """Reject changed cohort, overlap claims and mismatched per-clip placement.

    `frozen`/`centered` default to the manifest's own `cohort` and `centered` fields, and for a manifest without them to the
    2026-10-25 cohort, centered; those manifests are checked exactly as before.
    """
    frozen = cohort_of(manifest) if frozen is None else frozen
    centered = manifest.get("centered", True) if centered is None else centered
    if manifest.get("provenance") != PROVENANCE or manifest.get("budget_gpu_hours") != 8:
        raise ValueError("Wrong diagnostic scope or budget")
    clips = manifest["clips"]
    if [c["id"] for c in clips] != [r[0] for r in frozen]:
        raise ValueError("Frozen cohort changed or duplicated")
    old = manifest["old_clips"]
    if len(old) != 49 or set(old) & {r[0] for r in frozen}:
        raise ValueError("Old-manifest clip intersection or incomplete audit")
    # Every frozen clip is >=240 frames and the review frame is start+120.
    # Keep this gate independent of the main environment's mask dependencies:
    # it also runs in the isolated AnimateAnyone environment.
    plan = {f"{c['id']}@{f}": {"id": f"{c['id']}@{f}", "clip": c['id'],
            "frame": f, "start": f-120, "frames": 240, "target": 120,
            "fps": c['fps'], "sha256": c['sha256']} for c in clips for f in c['review_frames']}
    if manifest.get('contains_human_marks') is not False or any(c['frames'] < 240 for c in clips):
        raise ValueError('Wrong source-only window manifest')
    gates = {}
    for c, (cid, start, fps, digest) in zip(clips, frozen):
        wid = f"{cid}@{start + 120}"
        w = plan.get(wid, {})
        if (w.get("start"), w.get("frames"), w.get("target"), w.get("fps"), w.get("sha256")) != (start, 240, 120, fps, digest):
            raise ValueError(f"Changed source/span/fps: {cid}")
        if c["review_frames"] != [start + 120] or (centered and (c["frames"] - 240) // 2 != start):
            raise ValueError(f"Wrong centered frame placement: {cid}")
        if c["staged"]["sha256"] != digest:
            raise ValueError(f"Staged source mismatch: {cid}")
        if any(r is not None and r.get("frame") != start+t for t, r in enumerate(c["registration"])):
            raise ValueError(f"Misaligned native registration frame: {cid}")
        gates[wid] = gate(c["registration"])
        # All 12 frozen windows passed in the original pre-quality audit.
        if len(gates[wid]) < 228:
            raise ValueError(f"Changed registration eligibility: {cid}")
    if len(plan) != len(frozen):
        raise ValueError("Extra windows")
    return gates


def check_feature_rows(manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    from experiments.tennis.k4b_features import plan_windows

    check_manifest(manifest)
    plan = plan_windows(manifest)
    requested = summary["requested"][:summary["configured_windows"] or None]
    if not requested or len(set(requested)) != len(requested) or any(w not in plan for w in requested):
        raise ValueError("Invalid requested windows")
    if [r["id"] for r in summary["windows"]] != requested:
        raise ValueError("Missing, duplicated or additional feature output")
    for row in summary["windows"]:
        if any(row.get(k) != v for k, v in plan[row["id"]].items()):
            raise ValueError("Feature/source span mismatch")
        if row["conversion"]["frames"] != 240:
            raise ValueError("Incomplete source decode")



def build_inputs(manifest: dict[str, Any], archives: list[Path], destination: Path, max_windows: int = 0) -> None:
    """Freeze geometry-only comparison masks/indices before quality is opened."""
    from experiments.tennis.k6 import FRAME, frame_regions, window_players
    from experiments.tennis.k4 import safe
    from experiments.visor.b1 import file_sha256
    from src.segmentation.masks import ClipMasks

    gates = check_manifest(manifest)
    clips = {c['id']: c for c in manifest['clips']}
    record: dict[str, Any] = {'schema': 1, 'provenance': PROVENANCE, 'windows': {}}
    with tempfile.TemporaryDirectory(prefix='k6b-confirm-inputs-', dir='/tmp') as temp:
        root = Path(temp)
        for i, archive in enumerate(archives):
            with tarfile.open(archive) as tar:
                tar.extractall(root / str(i), filter='data')
        windows = sorted(root.glob('*/publish/*/window.json'))[:max_windows or None]
        if not windows:
            raise ValueError('No complete feature windows')
        for window in windows:
            w = json.loads(window.read_text())
            wid, cid = w['id'], w['clip']
            if wid in record['windows'] or wid not in gates:
                raise ValueError('Repeated/unknown window')
            for name, digest in w['files'].items():
                if file_sha256(window.parent / name) != digest:
                    raise ValueError('Changed feature bytes')
            start = w['start']
            c = clips[cid]
            fits = {str(start+t): c['registration'][t]['homography'] for t in gates[wid]}
            features = json.loads((window.parent / 'features.json').read_text())
            clip = ClipMasks.load(window.parent / 'sam-masks.rle')
            if len(clip.frames) != 240:
                raise ValueError('Incomplete masks')
            players = window_players(features, clip, start, {int(t): np.array(h) for t,h in fits.items()})['players']
            if not players:
                raise ValueError('No players: keep window unresolved, never replace it')
            scored, hashes = [], {}
            for t in gates[wid]:
                regions = frame_regions(clip.frames[t], features['aliases'], players)['regions']
                r = np.logical_or.reduce([*regions.values(), np.zeros((FRAME[1], FRAME[0]), bool)])
                if r.any():
                    scored.append(t)
                    hashes[str(t)] = hashlib.sha256(np.packbits(r).tobytes()).hexdigest()
            if not scored:
                raise ValueError('No scored frames')
            record['windows'][wid] = {'window': {k:w[k] for k in ('id','clip','frame','start','frames','target','fps','sha256')},
                                     'fits': fits, 'files': {n:w['files'][n] for n in ('features.json','sam-masks.rle','pose.json.zst')},
                                     'source': c['staged'], 'comparison': {'players': players, 'scored': scored,
                                     'judged': [t for t in scored if t % 3 == 0], 'r_masks': hashes}}
        with tarfile.open(destination, 'w') as tar:
            import io
            data = (json.dumps(record, sort_keys=True) + '\n').encode()
            info = tarfile.TarInfo('inputs.json')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            for window in windows:
                wid = json.loads(window.read_text())['id']
                for name in ('features.json','sam-masks.rle','pose.json.zst'):
                    tar.add(window.parent / name, arcname=f'windows/{safe(wid)}/{name}')
    print(json.dumps({'windows': list(record['windows']), 'sha256': file_sha256(destination)}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("features", "validate-features", "preflight", "build-inputs", "validate-inputs"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--feature-archive", action="append", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--pose")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--windows", nargs="+")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    gates = check_manifest(manifest)
    if args.command in ("build-inputs", "validate-inputs"):
        if not args.feature_archive:
            raise ValueError("Feature archives required")
        from experiments.visor.b1 import stage_dir, file_sha256, write_json
        target = args.out or stage_dir() / "inputs.tar"
        if args.command == "build-inputs":
            build_inputs(manifest, args.feature_archive, target, args.max_windows)
            write_json(stage_dir() / "confirm-inputs.json", {"windows": args.max_windows, "sha256": file_sha256(target)})
            return 0
        metadata = json.loads((stage_dir() / "confirm-inputs.json").read_text())
        with tempfile.TemporaryDirectory(prefix="k6b-confirm-recheck-", dir="/tmp") as temp:
            rebuilt = Path(temp) / "inputs.tar"
            build_inputs(manifest, args.feature_archive, rebuilt, metadata["windows"])
            with tarfile.open(target) as actual, tarfile.open(rebuilt) as expected:
                names = actual.getnames()
                checks = {"unique_files": len(names) == len(set(names)), "files_exact": names == expected.getnames(),
                          "recorded_hash": file_sha256(target) == metadata["sha256"]}
                for name in expected.getnames():
                    a, b = actual.extractfile(name), expected.extractfile(name)
                    checks[name] = a is not None and b is not None and a.read() == b.read()
        report: dict[str, Any] = {"passed": all(checks.values()), "checks": checks, "provenance": PROVENANCE}
        if path := os.environ.get("PS_VALIDATION_PATH"):
            write_json(Path(path), report)
        print(json.dumps(report))
        return 0 if report["passed"] else 1
    if args.command == "preflight":
        print(json.dumps({"windows": len(gates), "fits": {k: len(v) for k, v in gates.items()}, **PROVENANCE}))
        return 0
    from experiments.tennis import k4b_features
    from experiments.visor.b1 import stage_dir, write_json

    if args.command == "features":
        if not args.windows or not args.sources or not args.sam_checkpoint or not args.pose:
            raise ValueError("Missing feature arguments")
        if len(set(args.windows)) != len(args.windows) or any(w not in gates for w in args.windows):
            raise ValueError("Unknown/duplicate window before GPU loading")
        code = k4b_features.run(args)
        if code:
            return code
        summary = json.loads((stage_dir() / "k4b-features.json").read_text())
        check_feature_rows(manifest, summary)
        checked = k4b_features.validate(Path(os.environ["PS_SCRATCH_DIR"]) / "publish", summary, args.manifest)
        write_json(stage_dir() / "scientific-validation.json", checked)
        return 0 if checked["passed"] else 1
    stage = stage_dir()
    summary = json.loads((stage / "k4b-features.json").read_text())
    check_feature_rows(manifest, summary)
    with tempfile.TemporaryDirectory(prefix="k6b-confirm-validate-", dir="/tmp") as temp:
        with tarfile.open(stage / "published.tar") as archive:
            archive.extractall(temp, filter="data")
        report = k4b_features.validate(Path(temp) / "publish", summary, args.manifest)
    report["checks"]["frozen_cohort_and_geometry"] = True
    report["provenance"] = PROVENANCE
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
