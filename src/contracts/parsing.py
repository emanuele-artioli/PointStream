"""Building nested frozen dataclasses from config mappings, strictly.

The mechanism, separated from the schema that uses it. Two properties matter,
and both exist because of specific failures:

**Unknown keys are errors.** The arrangement being retired filtered incoming
mappings against the dataclass fields and silently dropped anything unmatched.
That is how `canny-lower-threshold`, `canny-upper-threshold` and
`pose-heuristic-mask-dilation` sat in the shipped config for months — documented,
commented, and read by code through `getattr(config, ..., default)` that always
returned the default, because the attribute never existed. A knob that does
nothing is worse than a missing one: it produces a clean, plausible, entirely
fictional ablation.

**Every problem is reported at once.** Validation collects failures and raises
them together, so a config with five typos reports five. Fixing them one run at
a time is the kind of friction that stops people running validation at all.

Keys are accepted in either spelling: `residual_pix_fmt` and `residual-pix-fmt`
both reach the same field, because YAML convention and Python convention
disagree and neither is worth fighting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
import enum
from pathlib import Path
import types
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from src.contracts.errors import ConfigError, ConfigKeyError, ConfigValueError, ContractError

T = TypeVar("T")


def normalise_key(key: str) -> str:
    """Config spelling to Python attribute spelling."""
    return key.strip().replace("-", "_")


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return type(None) in get_args(annotation)
    return False


def _unwrap_optional(annotation: Any) -> Any:
    """`int | None` -> `int`. Leaves non-optional annotations alone."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        remaining = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(remaining) == 1:
            return remaining[0]
    return annotation


def _coerce_scalar(value: Any, annotation: Any, path: str) -> Any:
    """Convert one YAML scalar to the annotated type.

    Deliberately narrow. YAML already produces real ints, floats, bools and
    strings, so the only conversions performed are the ones YAML cannot express:
    enum members from their values, and `Path` from a string. Anything broader
    would let a genuinely wrong value through by coercing it into shape.
    """
    target = _unwrap_optional(annotation)

    if target is Any or target is None:
        return value

    if isinstance(target, type) and issubclass(target, enum.Enum):
        try:
            return target(value)
        except ValueError:
            legal = ", ".join(repr(member.value) for member in target)
            raise ConfigValueError(path, f"{value!r} is not one of {legal}") from None

    if target is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigValueError(path, f"expected a path, got {type(value).__name__}")
        return Path(value)

    if target is bool:
        if not isinstance(value, bool):
            raise ConfigValueError(path, f"expected true or false, got {value!r}")
        return value

    if target is int:
        # bool is a subclass of int; accepting it here would let `true` become 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigValueError(path, f"expected a whole number, got {value!r}")
        return value

    if target is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigValueError(path, f"expected a number, got {value!r}")
        return float(value)

    if target is str:
        if not isinstance(value, str):
            raise ConfigValueError(path, f"expected text, got {value!r}")
        return value

    return value


def _coerce_sequence(value: Any, annotation: Any, path: str) -> Any:
    """Convert a YAML list to a tuple or frozenset of the annotated element type."""
    target = _unwrap_optional(annotation)
    origin = get_origin(target)
    args = get_args(target)

    if not isinstance(value, (list, tuple)):
        raise ConfigValueError(path, f"expected a list, got {type(value).__name__}")

    element = args[0] if args else Any
    items = [
        _coerce(item, element, f"{path}[{index}]")
        for index, item in enumerate(value)
    ]

    if origin in (frozenset, set):
        return frozenset(items)
    return tuple(items)


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    target = _unwrap_optional(annotation)

    if value is None:
        if _is_optional(annotation):
            return None
        raise ConfigValueError(path, "may not be null")

    if is_dataclass(target) and isinstance(value, Mapping):
        return build(target, value, path=path)  # type: ignore[arg-type]

    origin = get_origin(target)
    if origin in (tuple, list, frozenset, set):
        return _coerce_sequence(value, annotation, path)

    if origin is dict or target is dict:
        if not isinstance(value, Mapping):
            raise ConfigValueError(path, f"expected a mapping, got {type(value).__name__}")
        return dict(value)

    return _coerce_scalar(value, annotation, path)


def build(dataclass_type: type[T], data: Mapping[str, Any], *, path: str = "") -> T:
    """Construct `dataclass_type` from `data`, rejecting anything unrecognised.

    Args:
        dataclass_type: A dataclass. Nested dataclass fields are built
            recursively from nested mappings.
        data: The mapping, with keys in either `snake_case` or `kebab-case`.
        path: Dotted position in the enclosing config, used in error messages.

    Raises:
        ConfigError: Collecting every unknown key, missing required field and
            bad value found anywhere beneath `path`.
    """
    # Kept as a separate name so narrowing `is_dataclass` does not erase T from
    # `dataclass_type`, which is what the return annotation depends on.
    schema: Any = dataclass_type
    if not is_dataclass(schema):
        raise TypeError(f"{dataclass_type!r} is not a dataclass")

    hints = get_type_hints(schema)
    field_names = {item.name for item in fields(schema)}
    problems: list[ContractError] = []

    normalised: dict[str, Any] = {}
    for raw_key, value in data.items():
        key = normalise_key(str(raw_key))
        if key not in field_names:
            problems.append(ConfigKeyError(path, str(raw_key), sorted(field_names)))
            continue
        normalised[key] = value

    kwargs: dict[str, Any] = {}
    for item in fields(schema):
        item_path = _join(path, item.name)
        if item.name in normalised:
            try:
                kwargs[item.name] = _coerce(normalised[item.name], hints[item.name], item_path)
            except ConfigError as exc:
                problems.extend(exc.problems)
            except ContractError as exc:
                problems.append(exc)
            continue

        has_default = item.default is not MISSING or item.default_factory is not MISSING
        if not has_default:
            problems.append(ConfigValueError(item_path, "is required and was not given"))

    if problems:
        raise ConfigError(problems)

    try:
        built: T = dataclass_type(**kwargs)
    except (ValueError, TypeError) as exc:
        raise ConfigError([ConfigValueError(path or "<root>", str(exc))]) from exc
    return built


def to_mapping(instance: Any, *, kebab: bool = True) -> dict[str, Any]:
    """Render a config dataclass back to a plain mapping.

    Used to write the shipped default config *from* the schema rather than
    maintaining it by hand. That is what permanently closes the
    documented-but-unreachable-knob bug: a key can only appear in the file if a
    field exists to produce it.
    """

    def render(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                (item.name.replace("_", "-") if kebab else item.name): render(
                    getattr(value, item.name)
                )
                for item in fields(value)
            }
        if isinstance(value, enum.Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, frozenset):
            return sorted(render(item) for item in value)
        if isinstance(value, (tuple, list)):
            return [render(item) for item in value]
        if isinstance(value, Mapping):
            return {key: render(item) for key, item in value.items()}
        return value

    rendered = render(instance)
    if not isinstance(rendered, dict):
        raise TypeError(f"{instance!r} did not render to a mapping")
    return rendered


def flat_keys(dataclass_type: type[Any], *, prefix: str = "") -> list[str]:
    """Every dotted key path the schema accepts.

    Powers the "did you mean" suggestions across sections, and lets a test
    assert that the shipped config file names only keys that exist.
    """
    hints = get_type_hints(dataclass_type)
    keys: list[str] = []
    for item in fields(dataclass_type):
        here = _join(prefix, item.name)
        target = _unwrap_optional(hints[item.name])
        if isinstance(target, type) and is_dataclass(target):
            keys.extend(flat_keys(target, prefix=here))
        else:
            keys.append(here)
    return keys


def describe_problems(error: ConfigError, *, limit: int = 20) -> str:
    """Render collected problems for a terminal, truncating a very long list."""
    lines = [str(problem) for problem in error.problems[:limit]]
    if len(error.problems) > limit:
        lines.append(f"… and {len(error.problems) - limit} more")
    return "\n".join(f"  - {line}" for line in lines)


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge `override` onto `base`, for layering config files.

    Nested mappings merge; everything else replaces. Keys are normalised on the
    way in, so a base written in kebab-case and an override written in
    snake_case still combine rather than producing two separate entries that
    later validation would report as one unknown key.
    """
    result: dict[str, Any] = {normalise_key(str(key)): value for key, value in base.items()}
    for raw_key, value in override.items():
        key = normalise_key(str(raw_key))
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = merge(existing, value)
        else:
            result[key] = value
    return result


def require_known_keys(dataclass_type: type[Any], data: Mapping[str, Any]) -> None:
    """Check keys without constructing anything.

    Lets a tool report every unknown key in a config file — including ones in
    sections that would have failed earlier for an unrelated reason — without
    needing the values to be valid.
    """
    problems: list[ContractError] = []

    def walk(current_type: type[Any], mapping: Mapping[str, Any], path: str) -> None:
        hints = get_type_hints(current_type)
        names = {item.name for item in fields(current_type)}
        for raw_key, value in mapping.items():
            key = normalise_key(str(raw_key))
            if key not in names:
                problems.append(ConfigKeyError(path, str(raw_key), sorted(names)))
                continue
            target = _unwrap_optional(hints[key])
            if is_dataclass(target) and isinstance(value, Mapping):
                walk(target, value, _join(path, key))  # type: ignore[arg-type]

    walk(dataclass_type, data, "")
    if problems:
        raise ConfigError(problems)


__all__ = [
    "build",
    "describe_problems",
    "flat_keys",
    "merge",
    "normalise_key",
    "require_known_keys",
    "to_mapping",
]
