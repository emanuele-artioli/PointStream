"""Component wrappers around the contract domain profiles.

The contract already owns `TENNIS` and `GENERAL`. This module does not fork
those definitions: it names the selector the profile wants and surfaces
background-method checks so a panorama under parallax fails with the contract's
message rather than being swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.contracts.domain import GENERAL, TENNIS, DomainProfile


@dataclass(frozen=True)
class DomainBackend:
    """One registered domain, ready for a runner to consume.

    Args:
        profile: The contract profile this backend wraps. Identity, not a copy.
        selector: Selection backend the profile names. The rule itself lives on
            the selection axis; this is only the name.
    """

    profile: DomainProfile
    selector: str

    def __post_init__(self) -> None:
        if not self.selector:
            raise ValueError(
                f"Domain {self.profile.name!r} must name a selector; an empty "
                f"string would look like selection is off rather than delegated."
            )

    @property
    def name(self) -> str:
        """Config key this backend was registered under."""
        return self.profile.name

    def assert_background_valid(self, method: str, *, path: str = "background.method") -> None:
        """Reject a background method the camera assumption cannot support.

        Delegates to the contract check. Callers must not catch and ignore this:
        a panorama under a free-moving camera is not a slightly worse plate, it
        is a quietly incoherent one.
        """
        self.profile.assert_background_valid(method, path=path)


def build_tennis(*, selector: str = "heuristic") -> DomainBackend:
    """Broadcast tennis. Selector name only — the heuristic itself is B2's."""
    return DomainBackend(profile=TENNIS, selector=selector)


def build_general(*, selector: str = "identity") -> DomainBackend:
    """DAVIS-human video. Every detected person is salient; no tennis rules."""
    return DomainBackend(profile=GENERAL, selector=selector)
