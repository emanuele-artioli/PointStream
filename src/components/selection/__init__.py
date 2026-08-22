"""Subject selectors.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.capabilities import CAP_OPEN_VOCABULARY, domains
from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("selection")

REGISTRY.register(
    BackendSpec(
        name="heuristic",
        target="src.components.selection.heuristic:HeuristicSelector",
        capabilities=domains("tennis"),
        summary="Ad-hoc tennis heuristic: two players, not ball kids or crowd.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="open-vocabulary",
        target="src.components.selection.prompt:PromptSelector",
        capabilities=frozenset({CAP_OPEN_VOCABULARY}),
        summary="Keep detections matching the domain / config class prompt.",
    )
)
