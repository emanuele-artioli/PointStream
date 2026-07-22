"""Generative compositing: putting the actors back on the reconstructed frame.

Split out of the single 1545-line `src.decoder.genai_compositor` module:

    weights      weight resolution for the generative backends
    pose_render  pose tensors -> the conditioning images every engine consumes
    strategies   the generative strategies a run can select, by config name
    compositor   BaseCompositor and DiffusersCompositor

`compositor.py` is still large because `DiffusersCompositor` is one class;
splitting it further would mean changing behaviour, not moving it, so that is
left for a change that has a reason beyond file size.

Names are re-exported here, and `src.decoder.genai_compositor` remains as a
shim, so existing imports keep working.
"""

from src.decoder.compositing.compositor import (
    BaseCompositor,
    DiffusersCompositor,
    GenAICompositor,
)
from src.decoder.compositing.strategies import (
    AnimateAnyoneStrategy,
    BaselineControlNetStrategy,
    BaseGenAIStrategy,
)

# Private helpers re-exported for the engine modules that import them from the
# old `genai_compositor` path. Kept out of __all__ because they are not API.
from src.decoder.compositing.pose_render import (  # noqa: F401
    _render_pose_condition,
    _render_pose_with_racket,
    _to_numpy_bgr,
)
from src.decoder.compositing.weights import (  # noqa: F401
    _require_local_or_optin_weight,
    _resolve_local_weight_path,
)

__all__ = [
    "AnimateAnyoneStrategy",
    "BaseCompositor",
    "BaseGenAIStrategy",
    "BaselineControlNetStrategy",
    "DiffusersCompositor",
    "GenAICompositor",
]
