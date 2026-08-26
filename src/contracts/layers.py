"""The architectural layers, and the check that keeps them real.

Layers are only real if something enforces them, so the layer list lives here as
data and CI walks the import graph against it. Nothing is written down twice:
there is no prose description of the architecture that could drift from this.

Innermost first, each layer may import only layers *beneath* it:

``contracts``
    Protocols, schemas, capability declarations, config validation. Imports
    nothing else from the project, and no heavy third-party dependency either —
    validating a config must not require torch to be installed.

``components``
    One package per pluggable axis, each implementing a contract and registering
    its backends.

``pipeline``
    The encoder, decoder, reconstruction and stage DAG. Depends on `contracts`
    **but deliberately not on `components`**: the pipeline must never know which
    backend was chosen. Components reach it by injection.

``runner``
    The single run path — chunk loop, scene routing, accounting, evaluation,
    invariants. This is the layer allowed to compose concrete components into a
    pipeline, and the only one.

``experiments``
    Ablation matrices, sweeps, training campaigns, dataset tooling. Consumes
    `runner` as a library. The arrangement this replaces had experiment scripts
    shell out to the CLI and scrape results from stdout, which is what an
    unchecked boundary decays into.

``legacy``
    The pre-rebuild tree, exempt while it is being replaced. Every module here
    is one the rebuild has not reached yet, so the set shrinks to nothing and
    then the layer goes away.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.contracts.errors import LayerViolationError

#: Layers innermost-first. A layer may import itself and anything earlier.
LAYER_ORDER: tuple[str, ...] = (
    "contracts",
    "components",
    "pipeline",
    "runner",
    "experiments",
)

#: Package under `src/` that defines each layer.
LAYER_PACKAGES: dict[str, str] = {
    "contracts": "src.contracts",
    "components": "src.components",
    "pipeline": "src.pipeline",
    "runner": "src.runner",
    "experiments": "src.experiments",
}

#: Packages not yet migrated. Exempt from the direction rule in both directions:
#: legacy code may import anything, and anything may import legacy code, because
#: forbidding either would just stall the rebuild. Delete entries as they are
#: replaced; when this is empty, remove it.
#:
#: BP22: ``src.shared`` stays condemned. It is not promoted to a layer. New
#: code does not go here. The set shrinks as callers move or die.
LEGACY_PACKAGES: frozenset[str] = frozenset(
    {
        "src.shared",
        "src.transport",
    }
)

LEGACY_LAYER = "legacy"


@dataclass(frozen=True)
class ImportEdge:
    """One import, located precisely enough to fix."""

    importer: str
    imported: str
    file: Path
    line: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.importer} -> {self.imported}"


def layer_of(module: str) -> str | None:
    """Which layer `module` belongs to, or None if it is outside the scheme.

    Legacy packages return `LEGACY_LAYER`. Modules outside `src.` entirely —
    third-party imports — return None and are ignored by the check.
    """
    for package in LEGACY_PACKAGES:
        if module == package or module.startswith(f"{package}."):
            return LEGACY_LAYER
    for layer, package in LAYER_PACKAGES.items():
        if module == package or module.startswith(f"{package}."):
            return layer
    return None


def may_import(importer_layer: str, imported_layer: str) -> bool:
    """Whether a module in `importer_layer` may import one in `imported_layer`."""
    if LEGACY_LAYER in (importer_layer, imported_layer):
        return True
    if importer_layer == imported_layer:
        return True
    return LAYER_ORDER.index(imported_layer) < LAYER_ORDER.index(importer_layer)


def module_name_for(path: Path, root: Path) -> str:
    """Dotted module name for a source file, relative to the repository root."""
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def iter_imports(path: Path, module: str) -> Iterator[tuple[str, int]]:
    """Every module `path` imports, with line numbers.

    Parses rather than imports, so the check runs without the project's runtime
    dependencies installed and without executing any module-level code.
    Relative imports are resolved against `module`.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return

    package_parts = module.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package_parts[: len(package_parts) - node.level + 1]
                target = ".".join([*base, node.module]) if node.module else ".".join(base)
            else:
                target = node.module or ""
            if target:
                yield target, node.lineno


def find_violations(root: Path, *, packages: Iterable[str] | None = None) -> list[ImportEdge]:
    """Every outward import under `root`.

    Args:
        root: Repository root — the directory containing `src/`.
        packages: Restrict the scan to these dotted package prefixes. Defaults
            to every layer package.

    Returns:
        Offending edges, sorted by file then line.
    """
    prefixes = tuple(packages) if packages is not None else tuple(LAYER_PACKAGES.values())
    violations: list[ImportEdge] = []

    for path in sorted((root / "src").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = module_name_for(path, root)
        if not any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes):
            continue
        importer_layer = layer_of(module)
        if importer_layer is None or importer_layer == LEGACY_LAYER:
            continue

        for imported, line in iter_imports(path, module):
            imported_layer = layer_of(imported)
            if imported_layer is None:
                continue
            if not may_import(importer_layer, imported_layer):
                violations.append(ImportEdge(module, imported, path, line))

    return violations


def assert_no_violations(root: Path, *, packages: Iterable[str] | None = None) -> None:
    """Raise on the first outward import found.

    Raises:
        LayerViolationError: Naming the importer, the imported module, and both
            layers.
    """
    violations = find_violations(root, packages=packages)
    if not violations:
        return
    first = violations[0]
    importer_layer = layer_of(first.importer) or "?"
    imported_layer = layer_of(first.imported) or "?"
    error = LayerViolationError(first.importer, first.imported, importer_layer, imported_layer)
    if len(violations) > 1:
        listing = "\n".join(f"  {edge}" for edge in violations)
        raise LayerViolationError(
            first.importer, first.imported, importer_layer, imported_layer
        ) from AssertionError(f"{len(violations)} outward imports:\n{listing}")
    raise error


def describe(root: Path, *, packages: Iterable[str] | None = None) -> str:
    """A report suitable for CI output."""
    violations = find_violations(root, packages=packages)
    if not violations:
        return "Import direction: OK — no outward imports."
    lines = [f"Import direction: {len(violations)} violation(s)."]
    for edge in violations:
        importer_layer = layer_of(edge.importer) or "?"
        imported_layer = layer_of(edge.imported) or "?"
        lines.append(f"  {edge}  [{importer_layer} -> {imported_layer}]")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: `python -m src.contracts.layers`."""
    import argparse

    parser = argparse.ArgumentParser(description="Check architectural import direction.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (the directory containing src/).",
    )
    args = parser.parse_args(argv)

    report = describe(args.root)
    print(report)
    return 0 if report.endswith("no outward imports.") else 1


if __name__ == "__main__":
    raise SystemExit(main())
