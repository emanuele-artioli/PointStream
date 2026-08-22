"""Minimal-dataset plumbing for registered domain profiles."""

from src.components.domain.datasets.catalog import (
    ClipSpec,
    DatasetItem,
    DatasetManifest,
    DatasetMissingError,
    first_sample,
    iter_dataset,
    load_manifest,
    manifest_path,
    parse_manifest,
    smoke,
)

__all__ = [
    "ClipSpec",
    "DatasetItem",
    "DatasetManifest",
    "DatasetMissingError",
    "first_sample",
    "iter_dataset",
    "load_manifest",
    "manifest_path",
    "parse_manifest",
    "smoke",
]
