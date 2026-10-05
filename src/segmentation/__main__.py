"""Command line for the segmentation module.

    python -m src.segmentation run     --backend yoloe-26s --domain tennis --source clip.mp4 --out DIR
    python -m src.segmentation dataset --backend sam31 --domain egocentric --out ROOT [--source ...]
    python -m src.segmentation bench   --reference ROOT/sam31 --candidates ROOT/yoloe-26* --out report.json
    python -m src.segmentation suite   --domain tennis --backends sam31,yoloe-26n,... --out ROOT
    python -m src.segmentation export  --masks DIR --out PNG_DIR [--kind foreground|labels]
    python -m src.segmentation preview --source clip.mp4 --masks DIR --out overlay.mp4

A run directory holds ``masks.rle`` (lossless instance masks) and
``provenance.json``. A dataset root holds one run directory per clip. Without
``--source``, clips come from the domain's ``clips`` in domains.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from src.segmentation import BACKENDS, REFERENCE_BACKEND, ClipMasks, build, domain_names, load_domain
from src.segmentation.masks import mask_file


def clip_id(path: Path) -> str:
    """``<parent>_<stem>``, so scene files like ``000.mp4`` stay distinct across matches."""
    name = path.stem if path.suffix else path.name
    return f"{path.parent.name}_{name}" if name.isdigit() else name


def resolve_sources(domain: str, sources: list[Path] | None) -> list[tuple[str, Path]]:
    paths = sources or load_domain(domain).clip_paths()
    return [(clip_id(path), path) for path in paths]


def run_one(backend: Any, domain_name: str, clip_id: str, source: Path, out: Path, max_frames: int | None) -> ClipMasks:
    from src.segmentation.sources import code_identity, runtime_identity, source_identity, write_provenance

    domain = load_domain(domain_name)
    started = time.time()
    masks = backend.segment(source, domain, max_frames=max_frames)
    masks.meta.update({
        "clip": clip_id,
        "domain": domain_name,
        "source": source_identity(source, max_frames if source.is_dir() else None),
        "runtime": runtime_identity(),
    })
    out.mkdir(parents=True, exist_ok=True)
    masks.save(out)
    write_provenance(out, {
        "schema": "pointstream.segmentation.provenance.v1",
        "command": sys.argv,
        "started_unix": started,
        "code": code_identity(),
        **{key: masks.meta[key] for key in masks.meta},
        "frames": len(masks),
        "max_frames": max_frames,
    })
    return masks


def cmd_run(args: argparse.Namespace) -> int:
    run_one(build(args.backend), args.domain, clip_id(args.source), args.source, args.out, args.max_frames)
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    backend = build(args.backend)
    for clip_id, source in resolve_sources(args.domain, args.source):
        target = args.out / clip_id
        if (target / "provenance.json").is_file() and not args.force:
            print(f"skip {clip_id}: {target} exists", flush=True)
            continue
        print(f"{args.backend} {clip_id} <- {source}", flush=True)
        run_one(backend, args.domain, clip_id, source, target, args.max_frames)
    return 0


def bench(reference: Path, candidates: list[Path]) -> dict[str, Any]:
    from src.segmentation.evaluate import report_row, summarize

    rows = []
    for ref_dir in sorted(p for p in reference.iterdir() if p.is_dir()):
        try:
            ref = ClipMasks.load(ref_dir)
        except FileNotFoundError:
            continue
        for root in candidates:
            try:
                cand = ClipMasks.load(root / ref_dir.name)
            except FileNotFoundError:
                continue
            rows.append(report_row(ref_dir.name, str(cand.meta.get("backend") or root.name), cand, ref))
    reference_timing = []
    for ref_dir in sorted(p for p in reference.iterdir() if p.is_dir()):
        try:
            meta = ClipMasks.load(ref_dir).meta
        except FileNotFoundError:
            continue
        reference_timing.append({"clip": ref_dir.name, **(meta.get("timing") or {})})
    return {
        "schema": "pointstream.segmentation.bench.v1",
        "reference": str(reference),
        "reference_note": "SAM 3.1 is the assumed ground truth; scores are agreement with it.",
        "reference_timing": reference_timing,
        "summary": summarize(rows),
        "rows": rows,
    }


def cmd_bench(args: argparse.Namespace) -> int:
    report = bench(args.reference, args.candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print_table(report["summary"])
    return 0


def print_table(summary: list[dict[str, Any]]) -> None:
    keys = ("backend", "clips", "frames", "J", "F", "J&F", "recall", "precision", "flicker", "ms_per_frame", "fps")
    print("\t".join(keys))
    for row in summary:
        print("\t".join(str(row.get(key)) for key in keys))


def cmd_suite(args: argparse.Namespace) -> int:
    """Every backend over every clip of each domain, then the benchmark per domain."""
    out = args.out or Path(os.environ.get("PS_STAGE_DIR", ""))
    if not str(out):
        raise SystemExit("--out is required outside a fleet stage")
    backends = [name.strip() for name in args.backends.split(",") if name.strip()]
    failures: list[dict[str, str]] = []
    for domain in args.domain:
        sources = resolve_sources(domain, args.source)
        for name in backends:
            try:
                backend = build(name)
            except Exception as exc:  # a missing weight is reported, not fatal
                failures.append({"domain": domain, "backend": name, "error": repr(exc)})
                continue
            for clip_id, source in sources:
                target = out / domain / name / clip_id
                if (target / "provenance.json").is_file():
                    continue
                print(f"[{domain}] {name} {clip_id}", flush=True)
                try:
                    run_one(backend, domain, clip_id, source, target, args.max_frames)
                except Exception as exc:
                    failures.append({"domain": domain, "backend": name, "clip": clip_id, "error": repr(exc)})
                    if name == REFERENCE_BACKEND and not args.keep_going:
                        raise
                try:
                    from experiments.jobs.monitor import publish_progress

                    publish_progress(os.environ.get("PS_STAGE", "local"), 1)
                except Exception:
                    pass
            del backend
        reference = out / domain / REFERENCE_BACKEND
        if reference.is_dir():
            candidates = [out / domain / name for name in backends if name != REFERENCE_BACKEND]
            report = bench(reference, [c for c in candidates if c.is_dir()])
            (out / domain / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"== {domain}")
            print_table(report["summary"])
    (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    return 1 if failures and not args.keep_going else 0


def cmd_export(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    masks = ClipMasks.load(args.masks)
    args.out.mkdir(parents=True, exist_ok=True)
    for index in range(len(masks)):
        if args.kind == "labels":
            plane = masks.labels(index)
        else:
            plane = masks.foreground(index).astype(np.uint8) * 255
        cv2.imwrite(str(args.out / f"{index:06d}.png"), plane)
    return 0


PALETTE_BGR = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0))


def cmd_preview(args: argparse.Namespace) -> int:
    import cv2

    from src.segmentation.sources import iter_frames

    masks = ClipMasks.load(args.masks)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), masks.fps, (masks.width, masks.height))
    try:
        for index, frame in enumerate(iter_frames(args.source, len(masks))):
            labels = masks.labels(index)
            overlay = frame.copy()
            for code, _name in enumerate(masks.classes, start=1):
                overlay[labels == code] = PALETTE_BGR[(code - 1) % len(PALETTE_BGR)]
            writer.write(cv2.addWeighted(frame, 0.5, overlay, 0.5, 0))
    finally:
        writer.release()
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m src.segmentation", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)
    domains = list(domain_names())

    run = sub.add_parser("run", help="segment one clip")
    run.add_argument("--backend", choices=BACKENDS, required=True)
    run.add_argument("--domain", choices=domains, required=True)
    run.add_argument("--source", type=Path, required=True, help="video file or image directory")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--max-frames", type=int)
    run.set_defaults(func=cmd_run)

    dataset = sub.add_parser("dataset", help="segment a set of clips into ROOT/<clip>/")
    dataset.add_argument("--backend", choices=BACKENDS, required=True)
    dataset.add_argument("--domain", choices=domains, required=True)
    dataset.add_argument("--source", type=Path, nargs="*", help="default: the domain's clips in domains.yaml")
    dataset.add_argument("--out", type=Path, required=True)
    dataset.add_argument("--max-frames", type=int)
    dataset.add_argument("--force", action="store_true")
    dataset.set_defaults(func=cmd_dataset)

    bench_p = sub.add_parser("bench", help="score candidate roots against the reference root")
    bench_p.add_argument("--reference", type=Path, required=True)
    bench_p.add_argument("--candidates", type=Path, nargs="+", required=True)
    bench_p.add_argument("--out", type=Path, required=True)
    bench_p.set_defaults(func=cmd_bench)

    suite = sub.add_parser("suite", help="all backends over all clips, then bench")
    suite.add_argument("--domain", choices=domains, nargs="+", required=True)
    suite.add_argument("--backends", default=",".join(BACKENDS))
    suite.add_argument("--source", type=Path, nargs="*")
    suite.add_argument("--out", type=Path)
    suite.add_argument("--max-frames", type=int)
    suite.add_argument("--keep-going", action="store_true")
    suite.set_defaults(func=cmd_suite)

    export = sub.add_parser("export", help="write per-frame PNG masks")
    export.add_argument("--masks", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--kind", choices=("foreground", "labels"), default="foreground")
    export.set_defaults(func=cmd_export)

    preview = sub.add_parser("preview", help="overlay video for the demo")
    preview.add_argument("--source", type=Path, required=True)
    preview.add_argument("--masks", type=Path, required=True)
    preview.add_argument("--out", type=Path, required=True)
    preview.set_defaults(func=cmd_preview)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
