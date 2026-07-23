"""Weight resolution for the generative backends.

Weights resolve from assets/weights/ before a backend loads, so a missing
file fails here rather than triggering a silent download mid-run."""

from __future__ import annotations
import logging
from pathlib import Path
_LOGGER = logging.getLogger(__name__)


def _resolve_local_weight_path(model_name: str) -> Path | None:
    candidate = Path(model_name)
    if candidate.exists():
        return candidate

    project_root = Path(__file__).resolve().parents[2]
    assets_candidate = project_root / "assets" / "weights" / model_name
    if assets_candidate.exists():
        return assets_candidate

    return None
def _require_local_or_optin_weight(model_name: str, allow_download: bool = False) -> str:
    local_path = _resolve_local_weight_path(model_name)
    if local_path is not None:
        return str(local_path)

    if allow_download:
        return model_name

    raise FileNotFoundError(
        f"Required model weights not found for '{model_name}'. "
        "Place weights in assets/weights/ or set allow-auto-model-download to true."
    )
