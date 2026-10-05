from __future__ import annotations

from pathlib import Path
import urllib.request
from src.contracts import paths


DEFAULT_REQUIRED_WEIGHTS = [
    "yolo26n.pt",
    "yoloe-26n-seg.pt",
    "mobileclip2_b.ts",
    "yolo26n-pose.pt",
]

# Larger YOLOE-26 segmenters benchmarked against the SAM 3.1 reference
# (src.segmentation). SAM 3.1 itself lives in the pointstream-sam31 env.
OPTIONAL_ABLATION_WEIGHTS = [
    "yoloe-26s-seg.pt",
    "yoloe-26m-seg.pt",
    "yoloe-26l-seg.pt",
    "yoloe-26x-seg.pt",
]

# Optional weight for the FVD perceptual-quality metric (src/shared/fvd.py):
# I3D R50, pretrained on Kinetics-400 via the pytorchvideo model zoo ("8x8"
# checkpoint). Unlike the weights above, this one is auto-fetchable from a
# public URL, so it is downloaded rather than merely presence-checked.
FVD_I3D_WEIGHT_NAME = "i3d_r50_kinetics.pyth"
FVD_I3D_WEIGHT_URL = (
    "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D_8x8_R50.pyth"
)


def weight_path(weights_dir: Path, name: str) -> Path:
    if weights_dir == paths.models_root():
        return paths.model_asset(name)
    return weights_dir / name


def ensure_weights(weights_dir: Path, required_weights: list[str]) -> None:
    missing = [name for name in required_weights if not weight_path(weights_dir, name).exists()]
    if missing:
        missing_list = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            "Missing required weights in the configured Models root.\n"
            f"{missing_list}\n"
            "Place the files manually or update scripts/download_weights.py with valid sources."
        )


def ensure_fvd_i3d_weight(weights_dir: Path) -> None:
    """Fetch the I3D R50 (Kinetics-400) checkpoint used by the FVD metric, if missing."""
    destination = weight_path(weights_dir, FVD_I3D_WEIGHT_NAME)
    if destination.exists():
        return
    print(f"Downloading FVD I3D weight from {FVD_I3D_WEIGHT_URL} ...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FVD_I3D_WEIGHT_URL, str(destination))
    print(f"Saved FVD I3D weight to: {destination}")


def main() -> None:
    weights_dir = paths.models_root()
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Standard placeholder behavior for scaffold phase: verify local presence only.
    ensure_weights(weights_dir, DEFAULT_REQUIRED_WEIGHTS)
    print(f"All required weights are available in: {weights_dir}")

    missing_optional = [
        name for name in OPTIONAL_ABLATION_WEIGHTS if not weight_path(weights_dir, name).exists()
    ]
    if missing_optional:
        missing_list = "\n".join(f"- {name}" for name in missing_optional)
        print(
            "Optional ablation weights are missing (needed for --detector yoloe / --segmenter yoloe|sam3|sam):\n"
            f"{missing_list}"
        )

    try:
        ensure_fvd_i3d_weight(weights_dir)
    except Exception as exc:  # noqa: BLE001 - weight fetch failures should not crash the whole script
        print(
            f"Could not fetch FVD I3D weight automatically ({exc}); FVD metric will be unavailable until it exists."
        )


if __name__ == "__main__":
    main()
