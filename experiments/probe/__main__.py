<<<<<<< HEAD
"""Triage probe: coding-task region-scoped PSNR for every wired generator.

Appearance from a keyframe, conditioning from frame N, score against frame N.
Static copy is a permanent arm. Not citable. No CLAIM lines.
``python -m experiments.probe``.

Does not bind CUDA_VISIBLE_DEVICES. Stream A owns cuda:0; default here is cuda:1.
=======
"""Triage probe: region-scoped PSNR for every wired generator.

Not citable. No CLAIM lines. ``python -m experiments.probe``.
>>>>>>> phase-bp/bp5
"""

from __future__ import annotations

import argparse
<<<<<<< HEAD
from pathlib import Path

from experiments.probe.clips import DEFAULT_KEYFRAME, DEFAULT_OFFSETS
from experiments.probe.engines import DEVICE, PLANS, SEED
from experiments.probe.run import drive_all
=======
import os
from pathlib import Path

# Bind the process to physical cuda:0 before torch is imported anywhere else
# in this process. Leaves cuda:1 free on the two-card host.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from experiments.probe.engines import DEVICE, SEED, PLANS  # noqa: E402
from experiments.probe.run import drive_all  # noqa: E402
>>>>>>> phase-bp/bp5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
<<<<<<< HEAD
    parser.add_argument("--out", type=Path, default=Path("outputs") / "bp9-probe")
    parser.add_argument("--probe-root", type=Path, default=None)
    parser.add_argument("--keyframe", type=int, default=DEFAULT_KEYFRAME)
    parser.add_argument(
        "--offset",
        action="append",
        type=int,
        dest="offsets",
        help="Coding-task offset from the keyframe (repeatable). Default: 8 16 24 32.",
    )
=======
    parser.add_argument("--out", type=Path, default=Path("outputs") / "bp5-probe")
    parser.add_argument("--probe-root", type=Path, default=None)
>>>>>>> phase-bp/bp5
    parser.add_argument(
        "--engine",
        action="append",
        dest="engines",
<<<<<<< HEAD
        help="Drive only this engine (repeatable). Static copy still always runs.",
    )
    args = parser.parse_args(argv)
    offsets = tuple(args.offsets) if args.offsets else DEFAULT_OFFSETS
    print("[probe] bounds were written in experiments/probe/bounds.py before this run")
    print(
        f"[probe] seed={args.seed} device={args.device} keyframe={args.keyframe} "
        f"offsets={offsets} out={args.out}"
    )
=======
        help="Drive only this engine (repeatable). Default: the full BP5 list.",
    )
    args = parser.parse_args(argv)
    print("[probe] bounds were written in experiments/probe/bounds.py before this run")
    print(f"[probe] seed={args.seed} device={args.device} out={args.out}")
>>>>>>> phase-bp/bp5
    print("[probe] engines:", ", ".join(args.engines or [plan.name for plan in PLANS]))
    drive_all(
        device=args.device,
        seed=args.seed,
        out_dir=args.out,
        probe_root=args.probe_root,
        engines=tuple(args.engines) if args.engines else None,
<<<<<<< HEAD
        keyframe_index=args.keyframe,
        offsets=offsets,
=======
>>>>>>> phase-bp/bp5
    )
    print(f"[probe] done. summary at {args.out / 'summary.json'} (not citable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
