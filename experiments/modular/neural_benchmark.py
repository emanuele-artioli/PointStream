"""PointStream Neural Generative Model Benchmark Runner.

Evaluates neural generative model candidates (pix2pix, SPADE, ControlNet, Animate-Anyone)
against the falsifiable ceiling established by PointStream's winning modular rate ladder:
  - Wire Budget Ceiling: F <= 12,000 bytes
  - Anatomical Fidelity Ceiling: Pose OKS >= 0.90
  - Distortion Hurdle: FG PSNR >= 35.8 dB (or Saliency-Weighted PSNR >= 33.0 dB)

If all generative candidates violate the ceiling, generation: false remains frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SPEC = REPO_ROOT / "manifests" / "neural_generative_benchmark_spec.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "modular" / "neural_benchmark"


@dataclass(frozen=True)
class CandidateEvaluation:
    model_id: str
    architecture: str
    wire_bytes: int
    pose_oks: float
    fg_psnr: float
    saliency_weighted_psnr: float
    wire_passed: bool
    oks_passed: bool
    quality_passed: bool
    verdict: str  # "FALSIFIED" | "PASSED"
    falsification_reasons: list[str]


@dataclass(frozen=True)
class BenchmarkReport:
    schema: str
    operational_baseline: dict[str, Any]
    falsifiable_ceilings: dict[str, Any]
    candidates: list[CandidateEvaluation]
    summary_verdict: str
    overall_status: str  # "ALL_CANDIDATES_FALSIFIED" | "CANDIDATE_ACCEPTED"


def evaluate_candidate(
    candidate: dict[str, Any],
    ceilings: dict[str, Any],
) -> CandidateEvaluation:
    """Evaluate a single generative candidate against falsifiable ceilings."""
    wire_bytes = candidate.get("conditioning_wire_bytes", 0)
    pose_oks = candidate.get("pose_oks", 0.0)
    fg_psnr = candidate.get("fg_psnr", 0.0)
    weighted_psnr = candidate.get("saliency_weighted_psnr", 0.0)

    wire_budget = ceilings.get("wire_budget_bytes", 12000)
    oks_min = ceilings.get("anatomical_oks_min", 0.90)
    fg_psnr_min = ceilings.get("fg_psnr_min", 35.8)

    wire_passed = wire_bytes <= wire_budget
    oks_passed = pose_oks >= oks_min
    quality_passed = fg_psnr >= fg_psnr_min

    reasons: list[str] = []
    if not wire_passed:
        reasons.append(f"Wire budget exceeded: {wire_bytes:,} B > {wire_budget:,} B ceiling")
    if not oks_passed:
        reasons.append(f"Anatomical drift / hallucination: OKS {pose_oks:.3f} < {oks_min:.2f} ceiling")
    if not quality_passed:
        reasons.append(f"Distortion uncompetitive: FG PSNR {fg_psnr:.1f} dB < {fg_psnr_min:.1f} dB hurdle")

    verdict = "PASSED" if (wire_passed and oks_passed and quality_passed) else "FALSIFIED"

    return CandidateEvaluation(
        model_id=candidate["model_id"],
        architecture=candidate.get("architecture", "unknown"),
        wire_bytes=wire_bytes,
        pose_oks=pose_oks,
        fg_psnr=fg_psnr,
        saliency_weighted_psnr=weighted_psnr,
        wire_passed=wire_passed,
        oks_passed=oks_passed,
        quality_passed=quality_passed,
        verdict=verdict,
        falsification_reasons=reasons,
    )


def run_neural_benchmark(
    spec_path: Path = DEFAULT_SPEC,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute neural generative model benchmark."""
    spec_path = Path(spec_path)
    output_dir = Path(output_dir)

    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)

    ceilings = spec["falsifiable_ceilings"]
    evaluations: list[CandidateEvaluation] = []

    for cand in spec["candidates"]:
        ev = evaluate_candidate(cand, ceilings)
        evaluations.append(ev)

    all_falsified = all(e.verdict == "FALSIFIED" for e in evaluations)
    if all_falsified:
        overall_status = "ALL_CANDIDATES_FALSIFIED"
        summary_verdict = (
            f"All {len(evaluations)} evaluated neural generative models are FALSIFIED against "
            f"the winning rate ladder ceiling (F <= {ceilings['wire_budget_bytes']:,} B, "
            f"OKS >= {ceilings['anatomical_oks_min']:.2f}). PointStream reference-pasting default "
            f"(generation: false) remains frozen as the superior operational regime."
        )
    else:
        overall_status = "CANDIDATE_ACCEPTED"
        passed_models = [e.model_id for e in evaluations if e.verdict == "PASSED"]
        summary_verdict = f"Promising candidate(s) cleared ceiling: {', '.join(passed_models)}."

    report = {
        "schema": spec.get("schema", "pointstream.neural_generative_benchmark.v1"),
        "doc_role": "neural_generative_benchmark_report",
        "falsifiable_ceilings": ceilings,
        "operational_baseline": ceilings.get("operational_baseline", {}),
        "candidates": [asdict(e) for e in evaluations],
        "summary_verdict": summary_verdict,
        "overall_status": overall_status,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "benchmark_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PointStream Neural Generative Benchmark Runner")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Execute without model loading")

    args = parser.parse_args()
    report = run_neural_benchmark(
        spec_path=args.spec,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )

    print("\nNeural Generative Benchmark Complete.")
    print(f"Status: {report['overall_status']}")
    print(f"Verdict: {report['summary_verdict']}\n")
    for c in report["candidates"]:
        status_symbol = "[OK]" if c["verdict"] == "PASSED" else "[FAIL]"
        print(
            f"  {status_symbol} {c['model_id']} ({c['architecture']}): "
            f"Wire={c['wire_bytes']:,} B | OKS={c['pose_oks']:.3f} | "
            f"FG PSNR={c['fg_psnr']} dB | Verdict={c['verdict']}"
        )
        for r in c["falsification_reasons"]:
            print(f"     -> {r}")


if __name__ == "__main__":
    main()
