from __future__ import annotations

from pathlib import Path


DEFAULT_REQUIRED_WEIGHTS = [
    "yolo26n.pt",
    "yolo26n-seg.pt",
    "yolo26n-pose.pt",
]

# Optional weights for open-vocabulary and high-precision segmentation ablations.
OPTIONAL_ABLATION_WEIGHTS = [
    "yoloe-26n-seg.pt",
    "mobileclip2_b.ts",
    "sam3.pt",
    "sam2_b.pt",
]


def ensure_weights(weights_dir: Path, required_weights: list[str]) -> None:
    missing = [name for name in required_weights if not (weights_dir / name).exists()]
    if missing:
        missing_list = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            "Missing required weights in assets/weights/.\n"
            f"{missing_list}\n"
            "Place the files manually or update scripts/download_weights.py with valid sources."
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    weights_dir = project_root / "assets" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Standard placeholder behavior for scaffold phase: verify local presence only.
    ensure_weights(weights_dir, DEFAULT_REQUIRED_WEIGHTS)
    print(f"All required weights are available in: {weights_dir}")

    missing_optional = [name for name in OPTIONAL_ABLATION_WEIGHTS if not (weights_dir / name).exists()]
    if missing_optional:
        missing_list = "\n".join(f"- {name}" for name in missing_optional)
        print(
            "Optional ablation weights are missing (needed for --detector yoloe / --segmenter yoloe|sam3|sam):\n"
            f"{missing_list}"
        )


if __name__ == "__main__":
    main()
