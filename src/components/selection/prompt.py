"""Open-vocabulary prompt selector.

Keeps detections whose class matches the configured prompt (or the domain
profile's detection prompts). No tennis-specific near/far or crowd filter —
that is the point of having this as a selectable alternative to the heuristic.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.components.detection.types import Detection, is_ball, is_person, is_racket
from src.contracts.domain import DomainProfile, SalientClass, profile as domain_profile


class PromptSelector:
    """Keep detections that match a class prompt.

    ``prompt`` comes from ``BackendConfig.prompt``. When it is unset, the
    domain profile's per-class ``detection_prompt`` values are used instead.
    """

    def __init__(
        self,
        prompt: str | None = None,
        domain: str | DomainProfile | None = "tennis",
    ) -> None:
        self._prompt = prompt.strip() if prompt and prompt.strip() else None
        if isinstance(domain, DomainProfile):
            self._profile: DomainProfile | None = domain
        elif domain:
            self._profile = domain_profile(domain)
        else:
            self._profile = None

    def select(
        self,
        detections: Sequence[Detection],
        frame_shape: tuple[int, int] | None = None,
    ) -> list[Detection]:
        _ = frame_shape
        wanted = self._wanted_classes()
        selected: list[Detection] = []
        for item in detections:
            name = self._name_for(item, wanted)
            if name is None:
                continue
            selected.append(item.with_class_name(name))
        return selected

    def _wanted_classes(self) -> tuple[SalientClass, ...] | None:
        if self._profile is None:
            return None
        if self._prompt is None:
            return self._profile.salient_classes
        return tuple(
            salient
            for salient in self._profile.salient_classes
            if _matches_prompt(salient.name, (self._prompt,))
            or _matches_prompt(salient.detection_prompt, (self._prompt,))
        )

    def _name_for(
        self, item: Detection, wanted: tuple[SalientClass, ...] | None
    ) -> str | None:
        if wanted is not None:
            for salient in wanted:
                if self._matches_salient(item, salient):
                    return salient.name
            return None
        if self._prompt and _matches_prompt(item.class_name, (self._prompt,)):
            return self._prompt
        if self._prompt:
            return None
        return item.class_name

    def _matches_salient(self, item: Detection, salient: SalientClass) -> bool:
        if _matches_prompt(item.class_name, (salient.name, salient.detection_prompt)):
            return True
        if salient.name == "player" and is_person(item.class_name):
            return True
        if salient.name == "racket" and is_racket(item.class_name):
            return True
        if salient.name == "ball" and is_ball(item.class_name):
            return True
        return False


def _matches_prompt(class_name: str, prompts: Sequence[str]) -> bool:
    lowered = class_name.strip().lower()
    for prompt in prompts:
        target = prompt.strip().lower()
        if not target:
            continue
        if lowered == target or target in lowered or lowered in target:
            return True
    return False
