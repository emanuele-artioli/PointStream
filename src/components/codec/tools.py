"""Resolve encoder binaries by path and version, not by name.

This host has carried two SVT-AV1 builds where only one exposed
``--roi-map-file``. Testing the other one reads as "region control does not
work" for a reason that has nothing to do with region control. Every encode
records the path it actually ran and that binary's self-reported version.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Final

ENV_FFMPEG: Final = "FFMPEG_BIN"
ENV_KVAZAAR: Final = "KVAZAAR_BIN"
ENV_SVTAV1: Final = "SVTAV1_BIN"

_DEFAULT_NAMES: Final[dict[str, str]] = {
    ENV_FFMPEG: "ffmpeg",
    ENV_KVAZAAR: "kvazaar",
    ENV_SVTAV1: "SvtAv1EncApp",
}


@dataclass(frozen=True)
class ResolvedTool:
    """One executable, located and identified."""

    name: str
    path: str
    version: str
    features: frozenset[str] = frozenset()

    def has(self, feature: str) -> bool:
        """Whether ``feature`` was observed in the binary's help text."""
        return feature in self.features


def resolve_tool(env_var: str, binary_name: str | None = None) -> ResolvedTool:
    """Locate ``binary_name``, honouring ``env_var`` as an explicit path.

    ``env_var`` wins even when it is not on ``PATH``, which is how a session
    pins the SVT-AV1 build that actually has region control.
    """
    name = binary_name or _DEFAULT_NAMES.get(env_var, env_var)
    explicit = os.environ.get(env_var)
    if explicit:
        path = str(Path(explicit).expanduser())
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{env_var}={path!r} does not exist. Point it at a real {name} binary."
            )
    else:
        found = shutil.which(name)
        if found is None:
            raise FileNotFoundError(
                f"Required binary {name!r} was not found on PATH. "
                f"Set {env_var} to the executable path (not just the name)."
            )
        path = found

    version = probe_version(path)
    features = probe_features(path, name)
    return ResolvedTool(name=name, path=path, version=version, features=features)


def resolve_ffmpeg() -> ResolvedTool:
    """The ffmpeg used for conversion, muxing, and the ffmpeg-driven rungs."""
    return resolve_tool(ENV_FFMPEG, "ffmpeg")


def resolve_encoder(codec_name: str) -> ResolvedTool:
    """The encoder binary for ``codec_name`` (config key: avc/hevc/av1/vvc).

    ffmpeg-driven rungs return ffmpeg. Binary rungs return kvazaar or
    SvtAv1EncApp. The caller still needs ffmpeg for y4m conversion.
    """
    if codec_name in {"avc", "vvc"}:
        return resolve_ffmpeg()
    if codec_name == "hevc":
        return resolve_tool(ENV_KVAZAAR, "kvazaar")
    if codec_name == "av1":
        return resolve_tool(ENV_SVTAV1, "SvtAv1EncApp")
    raise FileNotFoundError(f"No encoder binary mapping for codec {codec_name!r}.")


def probe_version(path: str) -> str:
    """The first line of ``--version`` / ``-version``, or ``unknown``."""
    for flag in ("--version", "-version"):
        result = subprocess.run(
            [path, flag],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (result.stdout or result.stderr or "").strip()
        if text:
            return text.splitlines()[0].strip()
    return "unknown"


def probe_features(path: str, name: str) -> frozenset[str]:
    """Flags whose presence we have to know, not assume.

    ``roi-map-file`` is the only one today: SVT-AV1 grew it in 1.8, and an
    older binary on this host accepted every other flag while ignoring the
    region map (by not having the flag at all).
    """
    if name != "SvtAv1EncApp":
        return frozenset()
    result = subprocess.run(
        [path, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    help_text = f"{result.stdout}\n{result.stderr}"
    features: set[str] = set()
    if "--roi-map-file" in help_text:
        features.add("roi-map-file")
    return frozenset(features)
