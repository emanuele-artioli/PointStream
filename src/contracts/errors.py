"""Every way a configuration can be rejected before a run starts.

These exist so that a bad configuration fails at validation time with a message
naming the offending knob, rather than succeeding and producing something other
than what was asked for. That failure mode is not hypothetical: `libsvtav1`
accepts `-pix_fmt yuv444p` on the command line and emits yuv420p regardless, so
every residual encode that asked for 4:4:4 silently got 4:2:0 instead, and the
ablation built on it measured nothing.

All of them carry the *legal* alternatives, not just the rejected value. A
message that says only "unsupported" sends the reader to the source; a message
that lists what would have worked usually does not.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence


class ContractError(Exception):
    """Base for every contract violation, so callers can catch the family."""


def _suggest(value: str, candidates: Iterable[str], *, cutoff: float = 0.6) -> str:
    """Render a ' Did you mean X?' clause, or '' when nothing is close.

    Typos are the common case for these errors — `background-layerr` for
    `background-layer` — and a close-match hint turns a hunt through the schema
    into a one-line fix.
    """
    matches = difflib.get_close_matches(value, list(candidates), n=3, cutoff=cutoff)
    if not matches:
        return ""
    if len(matches) == 1:
        return f" Did you mean {matches[0]!r}?"
    rendered = ", ".join(repr(match) for match in matches)
    return f" Did you mean one of {rendered}?"


def _render_options(options: Iterable[str], *, limit: int = 24) -> str:
    """List legal values, truncating a very long axis rather than flooding."""
    ordered = sorted(options)
    if not ordered:
        return "(none registered)"
    if len(ordered) <= limit:
        return ", ".join(ordered)
    head = ", ".join(ordered[:limit])
    return f"{head}, … (+{len(ordered) - limit} more)"


class UnknownBackendError(ContractError):
    """A config named a backend that is not registered on that axis."""

    def __init__(self, axis: str, name: str, known: Iterable[str]) -> None:
        known_list = list(known)
        self.axis = axis
        self.name = name
        self.known = known_list
        super().__init__(
            f"Unknown {axis} backend {name!r}."
            f"{_suggest(name, known_list)}"
            f" Registered {axis} backends: {_render_options(known_list)}."
        )


class DuplicateBackendError(ContractError):
    """Two backends claimed the same name or alias on one axis.

    Raised at import time rather than lookup time, so a collision is a failure
    to start rather than a silently shadowed backend.
    """

    def __init__(self, axis: str, name: str, existing: str) -> None:
        self.axis = axis
        self.name = name
        self.existing = existing
        super().__init__(
            f"Cannot register {name!r} on the {axis} axis: that name or alias is "
            f"already taken by {existing!r}."
        )


class ConfigKeyError(ContractError):
    """A config key does not correspond to any field in the schema.

    Unknown keys are rejected rather than ignored. Silently dropping them is how
    `canny-lower-threshold` sat in the shipped config for months, documented and
    commented, with no backing field and therefore no effect whatsoever.
    """

    def __init__(self, path: str, key: str, known: Iterable[str]) -> None:
        known_list = list(known)
        self.path = path
        self.key = key
        self.known = known_list
        where = f" under {path!r}" if path else " at the top level"
        super().__init__(
            f"Unknown config key {key!r}{where}."
            f"{_suggest(key, known_list)}"
            f" Valid keys there: {_render_options(known_list)}."
        )


class ConfigValueError(ContractError):
    """A config value is the wrong type, or outside its legal range."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        super().__init__(f"Invalid config at {path!r}: {message}")


class ConfigError(ContractError):
    """Several config problems at once.

    Validation collects every failure before raising, so a config with five
    typos reports five. Fixing them one run at a time is the kind of friction
    that makes people stop running validation.
    """

    def __init__(self, problems: Sequence[ContractError]) -> None:
        self.problems = list(problems)
        count = len(self.problems)
        heading = "1 configuration problem:" if count == 1 else f"{count} configuration problems:"
        body = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"{heading}\n{body}")


class CodecConstraintError(ContractError):
    """A codec was asked for something it does not actually honour.

    The motivating case: `libsvtav1` ignores any `-pix_fmt` other than yuv420p
    without complaint, so the request appeared to succeed while producing a
    different pixel format than the one every downstream byte measurement
    assumed.
    """

    def __init__(self, codec: str, knob: str, value: object, allowed: Iterable[str]) -> None:
        self.codec = codec
        self.knob = knob
        self.value = value
        super().__init__(
            f"Codec {codec!r} does not support {knob}={value!r}. "
            f"Supported: {_render_options(str(item) for item in allowed)}."
        )


class UnsupportedCapabilityError(ContractError):
    """A component was asked for a capability it does not declare."""

    def __init__(self, capability: str, component: str, declared: Iterable[str]) -> None:
        self.capability = capability
        self.component = component
        super().__init__(
            f"{component} does not support the {capability!r} capability. "
            f"It declares: {_render_options(declared)}."
        )


class MissingConditioningError(ContractError):
    """A generator required conditioning that the bundle did not carry.

    Named fields, so the failure surfaces at the call site as "this generator
    needs a pose and got none" rather than forty lines deeper as a shape
    mismatch on a tensor nobody expected to be absent.
    """

    def __init__(self, missing: Sequence[str], present: Iterable[str]) -> None:
        self.missing = list(missing)
        needed = ", ".join(self.missing)
        super().__init__(
            f"Conditioning is missing required field(s): {needed}. "
            f"Present: {_render_options(present)}."
        )


class UndecodableStreamError(ContractError):
    """No registered generator can decode this appearance/motion pairing.

    The axes are independent where they genuinely are, but not every
    combination has a decoder — an appearance carried as edges cannot be turned
    back into colour by anything. Rejecting the pair at validation is what keeps
    the design from sprawling into combinations nothing implements.
    """

    def __init__(
        self,
        appearance: str,
        motion: str,
        workable: Iterable[tuple[str, str]] = (),
    ) -> None:
        self.appearance = appearance
        self.motion = motion
        pairs = [f"{app}+{mot}" for app, mot in workable]
        tail = f" Decodable pairings: {_render_options(pairs)}." if pairs else ""
        super().__init__(
            f"No registered generator can decode appearance {appearance!r} "
            f"together with motion {motion!r}.{tail}"
        )


class LayerViolationError(ContractError):
    """A module imported outward across an architectural layer boundary.

    Layers are only real if something checks them. The prior arrangement had
    experiment scripts shelling out to the CLI and scraping stdout, which is
    what an unchecked boundary decays into.
    """

    def __init__(self, importer: str, imported: str, importer_layer: str, imported_layer: str) -> None:
        self.importer = importer
        self.imported = imported
        super().__init__(
            f"{importer} ({importer_layer}) imports {imported} ({imported_layer}), "
            f"which is an outward import. {importer_layer} may only depend on layers "
            f"beneath it."
        )
