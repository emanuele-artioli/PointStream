"""Weight resolution for the generative backends.

Weights resolve from assets/weights/ before a backend loads, so a missing
file fails here rather than triggering a silent download mid-run."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
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


def _resolve_strategy_weight(config: Any, model_name: str, allow_download: bool = False) -> str:
    """Resolve a strategy's primary weight, honouring `genai-checkpoint-override`.

    The override exists so evaluation can score an arbitrary campaign checkpoint
    (e.g. `outputs/campaign/<name>/checkpoints/<variant>/...`) through the *same*
    strategy classes the decoder runs, instead of the fixed `assets/weights/`
    name. When unset, behaviour is identical to `_require_local_or_optin_weight`.
    """
    override = getattr(config, "genai_checkpoint_override", None) if config is not None else None
    if override:
        candidate = Path(str(override))
        if not candidate.exists():
            raise FileNotFoundError(
                f"genai-checkpoint-override points at a missing path: '{override}'"
            )
        return str(candidate)
    return _require_local_or_optin_weight(model_name, allow_download=allow_download)
