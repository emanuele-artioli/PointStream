"""Dataset manifests, clip resolution, and a runner-facing iterator.

A future Phase C runner consumes `iter_dataset`: each item carries a domain
name plus either a video path or a sequence of frame paths. This module does
not encode, decode, or download anything.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from src.contracts.errors import UnknownBackendError

MissingPolicy = Literal["skip", "error"]

_MANIFEST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
_REGISTERED_DATASETS = ("tennis", "general")


class DatasetMissingError(FileNotFoundError):
    """A manifest clip is not on disk at any of the expected locations."""

    def __init__(
        self,
        domain: str,
        clip_id: str,
        expected: Sequence[Path],
    ) -> None:
        self.domain = domain
        self.clip_id = clip_id
        self.expected = tuple(expected)
        looked = ", ".join(str(path) for path in expected) or "(no search roots)"
        super().__init__(
            f"Domain {domain!r} clip {clip_id!r} is not on disk. "
            f"Looked in: {looked}. The manifest lists expected paths; this "
            f"component does not download datasets."
        )


@dataclass(frozen=True)
class ClipSpec:
    """One clip in a domain's minimal set."""

    id: str
    kind: str
    path: str
    pattern: str = ""
    summary: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"video", "frames"}:
            raise ValueError(
                f"Clip {self.id!r} has kind {self.kind!r}; expected 'video' or 'frames'."
            )


@dataclass(frozen=True)
class DatasetManifest:
    """The on-disk index for one domain's minimal set."""

    domain: str
    selector: str
    search_roots: tuple[str, ...]
    clips: tuple[ClipSpec, ...]
    summary: str = ""


@dataclass(frozen=True)
class DatasetItem:
    """One clip a runner can consume, tagged with the domain it belongs to."""

    domain: str
    clip_id: str
    kind: str
    source: Path
    frames: tuple[Path, ...] = ()
    summary: str = ""

    @property
    def sample_path(self) -> Path:
        """A single path a smoke test can open: first frame, or the video file."""
        if self.frames:
            return self.frames[0]
        return self.source


def manifest_path(domain: str) -> Path:
    """YAML index for `domain`.

    Raises:
        UnknownBackendError: If this stream does not ship a manifest for it.
    """
    path = _MANIFEST_DIR / f"{domain}.yaml"
    if not path.is_file():
        raise UnknownBackendError("domain dataset", domain, _REGISTERED_DATASETS)
    return path


def parse_manifest(data: Mapping[str, Any]) -> DatasetManifest:
    """Build a manifest from a mapping, rejecting a document that cannot be used."""
    domain = str(data.get("domain") or "")
    if not domain:
        raise ValueError("Dataset manifest is missing 'domain'.")
    clips_raw = data.get("clips")
    if not clips_raw:
        raise ValueError(
            f"Dataset manifest for {domain!r} lists no clips, so a runner would "
            f"have nothing to iterate."
        )
    clips = tuple(
        ClipSpec(
            id=str(item["id"]),
            kind=str(item["kind"]),
            path=str(item["path"]),
            pattern=str(item.get("pattern") or ""),
            summary=str(item.get("summary") or ""),
        )
        for item in clips_raw
    )
    roots_raw = data.get("search_roots") or ()
    return DatasetManifest(
        domain=domain,
        selector=str(data.get("selector") or ""),
        search_roots=tuple(str(root) for root in roots_raw),
        clips=clips,
        summary=str(data.get("summary") or ""),
    )


def load_manifest(domain: str, *, path: Path | None = None) -> DatasetManifest:
    """Load the shipped YAML for `domain`, or a caller-supplied file."""
    source = path if path is not None else manifest_path(domain)
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Dataset manifest {source} is not a mapping.")
    manifest = parse_manifest(loaded)
    if path is None and manifest.domain != domain:
        raise ValueError(
            f"Manifest {source} declares domain {manifest.domain!r}, not {domain!r}."
        )
    return manifest


def resolve_roots(manifest: DatasetManifest, *, extra: Sequence[Path] = ()) -> tuple[Path, ...]:
    """Search roots, with relative entries resolved against the repository root."""
    roots: list[Path] = [Path(root) for root in extra]
    for raw in manifest.search_roots:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = _REPO_ROOT / candidate
        roots.append(candidate)
    # Preserve order, drop duplicates (resolved, so a symlink and its target collide).
    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots:
        key = root
        try:
            key = root.resolve()
        except OSError:
            pass
        if key in seen:
            continue
        seen.add(key)
        ordered.append(root)
    return tuple(ordered)


def _existing_source(spec: ClipSpec, roots: Sequence[Path]) -> Path | None:
    for root in roots:
        candidate = root / spec.path if not Path(spec.path).is_absolute() else Path(spec.path)
        if spec.kind == "video" and candidate.is_file():
            return candidate
        if spec.kind == "frames" and candidate.is_dir():
            return candidate
    return None


def _expected_paths(spec: ClipSpec, roots: Sequence[Path]) -> tuple[Path, ...]:
    expected: list[Path] = []
    for root in roots:
        if Path(spec.path).is_absolute():
            expected.append(Path(spec.path))
            break
        expected.append(root / spec.path)
    return tuple(expected)


def list_frames(source: Path, spec: ClipSpec) -> tuple[Path, ...]:
    """Frame paths under a clip directory, in name order."""
    if spec.kind != "frames":
        return ()
    pattern = spec.pattern or "*"
    return tuple(sorted(path for path in source.glob(pattern) if path.is_file()))


def resolve_clip(
    spec: ClipSpec,
    roots: Sequence[Path],
    *,
    domain: str,
    missing: MissingPolicy = "skip",
) -> DatasetItem | None:
    """Turn a clip spec into a runner item, or skip/fail if the files are absent."""
    source = _existing_source(spec, roots)
    if source is None:
        if missing == "error":
            raise DatasetMissingError(domain, spec.id, _expected_paths(spec, roots))
        return None
    frames = list_frames(source, spec)
    if spec.kind == "frames" and not frames:
        if missing == "error":
            raise DatasetMissingError(domain, spec.id, (source,))
        return None
    return DatasetItem(
        domain=domain,
        clip_id=spec.id,
        kind=spec.kind,
        source=source,
        frames=frames,
        summary=spec.summary,
    )


def iter_dataset(
    domain: str,
    *,
    manifest: DatasetManifest | None = None,
    extra_roots: Sequence[Path] = (),
    missing: MissingPolicy = "skip",
) -> Iterator[DatasetItem]:
    """Yield clips for `domain`, tagged with that domain name.

    Default `missing="skip"` lets a runner start on whatever is present.
    `missing="error"` fails on the first absent clip, naming the paths checked.
    """
    loaded = manifest if manifest is not None else load_manifest(domain)
    roots = resolve_roots(loaded, extra=extra_roots)
    for spec in loaded.clips:
        item = resolve_clip(spec, roots, domain=loaded.domain, missing=missing)
        if item is not None:
            yield item


def first_sample(
    domain: str,
    *,
    manifest: DatasetManifest | None = None,
    extra_roots: Sequence[Path] = (),
    missing: MissingPolicy = "skip",
) -> DatasetItem | None:
    """One clip from the domain's minimal set, or None if nothing is on disk."""
    for item in iter_dataset(
        domain, manifest=manifest, extra_roots=extra_roots, missing=missing
    ):
        return item
    if missing == "error":
        loaded = manifest if manifest is not None else load_manifest(domain)
        first = loaded.clips[0]
        raise DatasetMissingError(
            loaded.domain, first.id, _expected_paths(first, resolve_roots(loaded, extra=extra_roots))
        )
    return None


def smoke(
    *,
    domains: Sequence[str] = _REGISTERED_DATASETS,
    extra_roots: Sequence[Path] = (),
    require: bool = False,
) -> dict[str, DatasetItem]:
    """Load one sample from each profile's minimal set.

    When files are missing, the domain is omitted unless `require=True`, which
    raises `DatasetMissingError` naming the first absent clip. This is not an
    encode — it only proves the iterator can hand a runner a path.
    """
    policy: MissingPolicy = "error" if require else "skip"
    found: dict[str, DatasetItem] = {}
    absent: list[str] = []
    for domain in domains:
        item = first_sample(domain, extra_roots=extra_roots, missing="skip")
        if item is None:
            absent.append(domain)
        else:
            found[domain] = item
    if policy == "error" and absent:
        # Re-run the first missing domain with error policy so the message is
        # about a real clip path, not a generic "nothing found".
        first_sample(absent[0], extra_roots=extra_roots, missing="error")
    return found
