from __future__ import annotations

from pathlib import Path


DEFAULT_REQUIRED_WEIGHTS = [
    "yolo26n.pt",
    "yolo26n-seg.pt",
    "yolo26n-pose.pt",
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


if __name__ == "__main__":
    main()
