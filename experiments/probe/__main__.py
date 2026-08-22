"""Triage probe: region-scoped PSNR for every wired generator.

Not citable. No CLAIM lines. ``python -m experiments.probe``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Bind the process to physical cuda:0 before torch is imported anywhere else
# in this process. Leaves cuda:1 free on the two-card host.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from experiments.probe.engines import DEVICE, SEED, PLANS  # noqa: E402
from experiments.probe.run import drive_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path("outputs") / "bp5-probe")
    parser.add_argument("--probe-root", type=Path, default=None)
    parser.add_argument(
        "--engine",
        action="append",
        dest="engines",
        help="Drive only this engine (repeatable). Default: the full BP5 list.",
    )
    args = parser.parse_args(argv)
    print("[probe] bounds were written in experiments/probe/bounds.py before this run")
    print(f"[probe] seed={args.seed} device={args.device} out={args.out}")
    print("[probe] engines:", ", ".join(args.engines or [plan.name for plan in PLANS]))
    drive_all(
        device=args.device,
        seed=args.seed,
        out_dir=args.out,
        probe_root=args.probe_root,
        engines=tuple(args.engines) if args.engines else None,
    )
    print(f"[probe] done. summary at {args.out / 'summary.json'} (not citable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
