"""Clip-mode probe: coding-task LPIPS and PSNR for every wired generator.

Appearance from a keyframe, conditioning from frame N, score against frame N.
Two baselines always run first — ``static-copy`` (the floor) and
``unrelated-image`` (the null control) — and the run refuses to rank anything
if those two do not separate on the metric.

Ranking is on LPIPS, lower better, PSNR reported beside it. Not citable. No
CLAIM lines. ``python -m experiments.probe``.

Does not bind CUDA_VISIBLE_DEVICES. Stream A owns cuda:0; default here is cuda:1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.probe.clips import CLIP_MODE_OFFSETS, DEFAULT_KEYFRAME, DEFAULT_OFFSETS
from experiments.probe.engines import DEVICE, PLANS, SEED
from experiments.probe.run import drive_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path("outputs") / "bp12-clip-roster")
    parser.add_argument("--probe-root", type=Path, default=None)
    parser.add_argument("--keyframe", type=int, default=DEFAULT_KEYFRAME)
    parser.add_argument(
        "--mode",
        choices=("clip", "sparse"),
        default="clip",
        help=(
            "clip: contiguous offsets 1-8, the only shape a temporal model can "
            "be driven in. sparse: the old 8/16/24/32 spread, kept for "
            "comparison with pre-BP12 runs."
        ),
    )
    parser.add_argument(
        "--offset",
        action="append",
        type=int,
        dest="offsets",
        help="Coding-task offset from the keyframe (repeatable). Overrides --mode.",
    )
    parser.add_argument(
        "--engine",
        action="append",
        dest="engines",
        help="Drive only this engine (repeatable). Both baselines still always run.",
    )
    parser.add_argument(
        "--self-recon",
        action="store_true",
        help=(
            "Also measure self-reconstruction. Off by default: it is not the "
            "coding task, it is never ranked on, and for a temporal engine it "
            "would re-open the single-frame path."
        ),
    )
    args = parser.parse_args(argv)
    if args.offsets:
        offsets = tuple(args.offsets)
    else:
        offsets = CLIP_MODE_OFFSETS if args.mode == "clip" else DEFAULT_OFFSETS
    print("[probe] bounds were written in experiments/probe/bounds.py before this run")
    print(
        f"[probe] seed={args.seed} device={args.device} keyframe={args.keyframe} "
        f"mode={args.mode} offsets={offsets} out={args.out}"
    )
    print("[probe] engines:", ", ".join(args.engines or [plan.name for plan in PLANS]))
    summary = drive_all(
        device=args.device,
        seed=args.seed,
        out_dir=args.out,
        probe_root=args.probe_root,
        engines=tuple(args.engines) if args.engines else None,
        keyframe_index=args.keyframe,
        offsets=offsets,
        self_recon=args.self_recon,
    )
    control = summary.get("control", {})
    print(f"[probe] control: {control.get('note', 'not run')}")
    print(f"[probe] rank (LPIPS, lower first): {summary.get('rank') or 'none — see control'}")
    print(f"[probe] done. summary at {args.out / 'summary.json'} (not citable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
