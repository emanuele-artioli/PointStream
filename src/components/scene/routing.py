"""Route scene classes to the semantic pipeline or the fallback codec.

This is the only load-bearing decision scene classification makes. Point
spans go through the object-centric path; interludes (and anything else) go
to the fallback codec. The classifier that produces the labels is optional
and cheap; this mapping is what a reviewer is asking about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

POINT: Final = "point"
INTERLUDE: Final = "interlude"

ROUTE_SEMANTIC: Final = "semantic"
ROUTE_FALLBACK: Final = "fallback"


@dataclass(frozen=True)
class SceneSpan:
    """One contiguous span and where it should be encoded."""

    start_frame: int
    end_frame: int
    scene_class: str
    route: str

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"SceneSpan end_frame {self.end_frame} is before start_frame {self.start_frame}."
            )
        if self.start_frame < 0:
            raise ValueError(f"SceneSpan start_frame must be >= 0, got {self.start_frame}.")


def route_for(scene_class: str) -> str:
    """Map a domain scene class onto a pipeline route.

    Tennis declares ``("point", "interlude")``. Only ``point`` is eligible for
    the semantic pipeline; everything else, including an unknown label, falls
    back. That is deliberate: a misclassified interlude in the semantic path
    is a quietly wrong reconstruction, while sending a point through the
    fallback codec is just a missed saving.
    """
    if scene_class == POINT:
        return ROUTE_SEMANTIC
    return ROUTE_FALLBACK


def span(start_frame: int, end_frame: int, scene_class: str) -> SceneSpan:
    return SceneSpan(
        start_frame=start_frame,
        end_frame=end_frame,
        scene_class=scene_class,
        route=route_for(scene_class),
    )
