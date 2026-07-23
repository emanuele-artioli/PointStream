from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.experiment_evaluation import evaluate_run_summary  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a PointStream experiment folder.")
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to an experiment folder containing run_summary.json.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames used for PSNR computation.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path for the evaluation JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    summary_path = experiment_dir / "run_summary.json"
    if not summary_path.exists() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing run summary at {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evaluation = evaluate_run_summary(summary=summary, experiment_dir=experiment_dir, max_frames=args.max_frames)

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else (experiment_dir / "evaluation_summary.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(evaluation, indent=2))
    print(f"Evaluation written to: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())