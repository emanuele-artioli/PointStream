"""Backend registration and lookup, shared by every pluggable axis.

One mechanism serves detection, pose, segmentation, generation, background,
codec, transport and metrics, so learning it once is enough to add a backend
anywhere. Adding one is a `BackendSpec` literal plus the class it points at;
nothing else, anywhere, has to change.

Three properties are deliberate:

**Names are matched exactly, never by substring.** The arrangement this replaces
selected backends with `"yolo" in name`, which is ordering-dependent and
misroutes anything whose name merely contains a registered one.

**Construction targets are import strings, not classes.** Registry tables are
imported by config validation, and validation must not drag in torch or
diffusers to discover that a backend exists. The module is imported only when a
backend is actually built.

**Capabilities are namespaced strings.** A plain entry like `temporal-sequence`
says what a backend can do; a namespaced entry like `motion:keypoints` says what
it accepts. One vocabulary covers both, so the pairing checks in
`objectstream` and the capability checks in the pipeline are the same lookup.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import importlib
from typing import Any, Generic, TypeVar

from src.contracts.errors import DuplicateBackendError, UnknownBackendError

T = TypeVar("T")

#: Separates a capability's namespace from its value, as in ``motion:keypoints``.
NAMESPACE_SEPARATOR = ":"


@dataclass(frozen=True)
class BackendSpec(Generic[T]):
    """One registered backend on one axis.

    Args:
        name: Canonical key. Config values are matched against this exactly.
        target: Import path as ``"package.module:AttributeName"``. Resolved
            lazily, on first construction.
        aliases: Alternative spellings accepted for `name`. Use for genuine
            synonyms (``animate_anyone`` for ``animate-anyone``), not for
            near-miss typos, which the error messages already handle.
        capabilities: What this backend can do and what it accepts. Plain
            entries are abilities; ``namespace:value`` entries are acceptances.
        requires: Conditioning kinds this backend needs the encoder to produce.
            This is what makes cross-axis effects derived rather than
            string-matched: a generator asking for ``mask`` is why segmentation
            runs, instead of some unrelated module inspecting the generator's
            name.
        defaults: Constructor keyword defaults, overridable at build time.
        summary: One line, shown when listing an axis.
    """

    name: str
    target: str
    aliases: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    defaults: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""

    def __post_init__(self) -> None:
        if NAMESPACE_SEPARATOR not in self.target and "." not in self.target:
            raise ValueError(
                f"BackendSpec {self.name!r} has target {self.target!r}, which is not "
                f"an import path. Expected 'package.module:AttributeName'."
            )

    def supports(self, capability: str) -> bool:
        """Whether this backend declares `capability`."""
        return capability in self.capabilities

    def accepted(self, namespace: str) -> frozenset[str]:
        """Every value this backend accepts within `namespace`.

        ``spec.accepted("motion")`` on a pose-driven generator returns
        ``{"keypoints"}``. An empty result means the backend declares nothing in
        that namespace, which callers should read as "does not participate in
        this axis" rather than "accepts nothing".
        """
        prefix = f"{namespace}{NAMESPACE_SEPARATOR}"
        return frozenset(
            capability[len(prefix) :]
            for capability in self.capabilities
            if capability.startswith(prefix)
        )

    def accepts(self, namespace: str, value: str) -> bool:
        """Whether this backend accepts `value` within `namespace`."""
        return f"{namespace}{NAMESPACE_SEPARATOR}{value}" in self.capabilities

    def resolve(self) -> Any:
        """Import and return the target attribute.

        Separate from `Registry.build` so a caller can inspect a class —
        checking it satisfies a protocol, say — without constructing it.
        """
        module_path, _, attribute = self.target.partition(NAMESPACE_SEPARATOR)
        if not attribute:
            module_path, _, attribute = self.target.rpartition(".")
        module = importlib.import_module(module_path)
        try:
            return getattr(module, attribute)
        except AttributeError as exc:
            raise ImportError(
                f"BackendSpec {self.name!r} targets {self.target!r}, but "
                f"{module_path!r} has no attribute {attribute!r}."
            ) from exc


class Registry(Generic[T]):
    """The registered backends for one axis.

    Args:
        axis: Human-readable axis name, used in error messages ("detector",
            "generator", "codec").
    """

    def __init__(self, axis: str) -> None:
        self.axis = axis
        self._specs: dict[str, BackendSpec[T]] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: BackendSpec[T]) -> BackendSpec[T]:
        """Add `spec` to this axis.

        Raises:
            DuplicateBackendError: If the name or any alias is already claimed.
                Raised at import time, so a collision stops the process rather
                than silently shadowing one of the two backends.
        """
        for key in (spec.name, *spec.aliases):
            existing = self._specs.get(key) or self._specs.get(self._aliases.get(key, ""))
            if existing is not None:
                raise DuplicateBackendError(self.axis, key, existing.name)
        self._specs[spec.name] = spec
        for alias in spec.aliases:
            self._aliases[alias] = spec.name
        return spec

    def names(self) -> list[str]:
        """Canonical names, sorted. Aliases are excluded."""
        return sorted(self._specs)

    def spec(self, name: str) -> BackendSpec[T]:
        """Look up `name`, following aliases.

        Raises:
            UnknownBackendError: With the registered names and a close-match
                suggestion.
        """
        canonical = self._aliases.get(name, name)
        try:
            return self._specs[canonical]
        except KeyError:
            raise UnknownBackendError(self.axis, name, self.names()) from None

    def has(self, name: str) -> bool:
        """Whether `name` resolves, without raising if it does not."""
        return (self._aliases.get(name, name)) in self._specs

    def build(self, name: str, **kwargs: Any) -> T:
        """Construct the backend registered as `name`.

        Keyword arguments override the spec's `defaults`. The target module is
        imported here and not before.
        """
        spec = self.spec(name)
        factory = spec.resolve()
        built: T = factory(**{**spec.defaults, **kwargs})
        return built

    def supporting(self, capability: str) -> list[BackendSpec[T]]:
        """Every registered backend declaring `capability`, by canonical name."""
        return [spec for spec in self if spec.supports(capability)]

    def __iter__(self) -> Iterator[BackendSpec[T]]:
        for name in self.names():
            yield self._specs[name]

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.has(name)

    def describe(self) -> str:
        """A readable table of this axis, for `list-backends`-style output."""
        if not self._specs:
            return f"{self.axis}: (nothing registered)"
        width = max(len(spec.name) for spec in self)
        lines = [f"{self.axis}:"]
        for spec in self:
            alias_note = f"  (aka {', '.join(spec.aliases)})" if spec.aliases else ""
            lines.append(f"  {spec.name.ljust(width)}  {spec.summary}{alias_note}")
        return "\n".join(lines)
