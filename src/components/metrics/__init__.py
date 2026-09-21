"""Quality-metric backends.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.components.metrics.evaluator import (
    EvaluationRecord,
    Evaluator,
    ScopedScore,
    triage,
)
from src.components.metrics.region import MIN_REGION_PIXELS, Region, RegionKind, RegionRole
from src.contracts.metrics import (
    FVMD,
    LPIPS,
    PALETTE,
    POSE_OKS,
    PSNR,
    REID,
    SAM_IOU,
    SSIM,
    VMAF,
)
from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("metric")

REGISTRY.register(
    BackendSpec(
        name=PSNR.name,
        target="src.components.metrics.psnr:PsnrMetric",
        capabilities=frozenset({"frame", "reference"}),
        summary=PSNR.summary,
    )
)
REGISTRY.register(
    BackendSpec(
        name=SSIM.name,
        target="src.components.metrics.ssim:SsimMetric",
        capabilities=frozenset({"frame", "reference"}),
        summary=SSIM.summary,
    )
)
REGISTRY.register(
    BackendSpec(
        name=VMAF.name,
        target="src.components.metrics.vmaf:VmafMetric",
        capabilities=frozenset({"frame", "reference"}),
        summary=VMAF.summary,
    )
)
REGISTRY.register(
    BackendSpec(
        name=LPIPS.name,
        target="src.components.metrics.lpips:LpipsMetric",
        capabilities=frozenset({"frame", "reference"}),
        summary=LPIPS.summary,
    )
)
REGISTRY.register(
    BackendSpec(
        name=REID.name,
        target="src.components.metrics.reid:ReidMetric",
        capabilities=frozenset({"frame", "reference", "identity"}),
        summary=REID.summary,
    )
)
REGISTRY.register(
    BackendSpec(
        name=PALETTE.name,
        target="src.components.metrics.palette:PaletteMetric",
        capabilities=frozenset({"frame", "reference", "identity"}),
        summary=PALETTE.summary,
    )
)
REGISTRY.register(
    BackendSpec(
        name=FVMD.name,
        target="src.components.metrics.fvmd:FvmdMetric",
        capabilities=frozenset({"sequence", "reference", "temporal-sequence"}),
        summary=FVMD.summary,
    )
)
REGISTRY.register(
    BackendSpec(
        name=POSE_OKS.name,
        target="src.components.metrics.pose:PoseMetric",
        capabilities=frozenset({"frame", "reference", "pose"}),
        summary=POSE_OKS.summary,
    )
)
REGISTRY.register(
    BackendSpec(
        name=SAM_IOU.name,
        target="src.components.metrics.semantic_mask:SamIouMetric",
        capabilities=frozenset({"frame", "reference", "mask"}),
        summary=SAM_IOU.summary,
    )
)

__all__ = [
    "MIN_REGION_PIXELS",
    "REGISTRY",
    "EvaluationRecord",
    "Evaluator",
    "Region",
    "RegionKind",
    "RegionRole",
    "ScopedScore",
    "triage",
]
