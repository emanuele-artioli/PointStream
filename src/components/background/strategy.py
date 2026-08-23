"""Transmission strategies for the background model.

Three registered names, matching ``src.contracts.domain``:

- ``panorama-full`` — encode the plate once per chunk.
- ``panorama-delta`` — encode a full plate on the first chunk of a scene, and
  a signed diff against the previously *decoded* plate on later chunks of the
  same scene. A single-chunk run therefore sends the same bytes as full; that
  is the correct result for that harness, not a bug.
- ``none`` — send nothing; the residual carries the background.

Sidecar codec is an independent constructor argument, not a strategy name.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.components.background.delta import apply_delta, compute_delta
from src.components.background.plate import build_plate
from src.components.background.sidecar import SidecarCodec, build_sidecar, normalize_sidecar
from src.components.background.types import (
    MODE_DELTA,
    MODE_FULL,
    MODE_NONE,
    BackgroundArtifact,
)
from src.contracts.config import PointstreamConfig
from src.contracts.domain import (
    BACKGROUND_NONE,
    BACKGROUND_PANORAMA_DELTA,
    BACKGROUND_PANORAMA_FULL,
    DomainProfile,
    profile as resolve_profile,
)


def _resolve_domain(domain: DomainProfile | str | None) -> DomainProfile | None:
    if domain is None:
        return None
    if isinstance(domain, DomainProfile):
        return domain
    return resolve_profile(domain)


class BackgroundModel:
    """Shared construction: sidecar choice plus optional domain check."""

    method: str = BACKGROUND_NONE
    sends_panorama: bool = False

    def __init__(
        self,
        codec: str = "jpeg",
        jpeg_quality: int = 50,
        png_compression: int = 3,
        roi_crf: int = 30,
        roi_preset: str = "veryfast",
        domain: DomainProfile | str | None = None,
    ) -> None:
        self.codec_name = normalize_sidecar(codec)
        resolved = _resolve_domain(domain)
        if resolved is not None:
            resolved.assert_background_valid(self.method)
        self._sidecar: SidecarCodec | None
        if self.sends_panorama:
            self._sidecar = build_sidecar(
                self.codec_name,
                jpeg_quality=jpeg_quality,
                png_compression=png_compression,
                roi_crf=roi_crf,
                roi_preset=roi_preset,
            )
        else:
            self._sidecar = None

    @property
    def codec_id(self) -> str:
        if self._sidecar is None:
            return "none"
        return self._sidecar.codec_id

    def transmit(
        self,
        plate: np.ndarray,
        *,
        previous_decoded: np.ndarray | None = None,
        scene_id: str | None = None,
        previous_scene_id: str | None = None,
        chunk_id: str = "",
        homographies: tuple[tuple[float, ...], ...] = (),
    ) -> BackgroundArtifact:
        """Encode one already-built plate under this strategy.

        ``previous_decoded`` must be the plate the *client* will have after
        decoding the previous sidecar — not the pre-codec pixels — so a later
        residual is computed against what both sides actually share.
        """
        if not self.sends_panorama or self._sidecar is None:
            return BackgroundArtifact(
                method=self.method,
                codec=self.codec_name,
                codec_id=self.codec_id,
                mode=MODE_NONE,
                payload=b"",
                scene_id=scene_id,
                chunk_id=chunk_id,
                deferred_to_residual=True,
            )

        array = np.asarray(plate, dtype=np.uint8)
        height, width = int(array.shape[0]), int(array.shape[1])
        use_delta = (
            self.method == BACKGROUND_PANORAMA_DELTA
            and previous_decoded is not None
            and scene_id is not None
            and scene_id == previous_scene_id
            and np.asarray(previous_decoded).shape == array.shape
        )
        if use_delta:
            image = compute_delta(array, np.asarray(previous_decoded, dtype=np.uint8))
            mode = MODE_DELTA
        else:
            image = array
            mode = MODE_FULL
        payload = self._sidecar.encode(image)
        return BackgroundArtifact(
            method=self.method,
            codec=self.codec_name,
            codec_id=self._sidecar.codec_id,
            mode=mode,
            payload=payload,
            width=width,
            height=height,
            homographies=homographies,
            scene_id=scene_id,
            chunk_id=chunk_id,
            deferred_to_residual=False,
        )

    def encode_frames(
        self,
        frames: np.ndarray,
        *,
        masks: np.ndarray | None = None,
        previous_decoded: np.ndarray | None = None,
        scene_id: str | None = None,
        previous_scene_id: str | None = None,
        chunk_id: str = "",
    ) -> BackgroundArtifact:
        """Build a plate from ``frames`` and transmit it."""
        if not self.sends_panorama:
            return self.transmit(
                np.zeros((1, 1, 3), dtype=np.uint8),
                scene_id=scene_id,
                chunk_id=chunk_id,
            )
        plate, homographies = build_plate(frames, masks=masks)
        return self.transmit(
            plate,
            previous_decoded=previous_decoded,
            scene_id=scene_id,
            previous_scene_id=previous_scene_id,
            chunk_id=chunk_id,
            homographies=homographies,
        )

    def decode_payload(self, artifact: BackgroundArtifact) -> np.ndarray | None:
        """Pixels the client reconstructs from ``artifact``.

        For a delta, this returns the *diff image*, not the reconstructed
        plate. Call ``reconstruct`` to apply it to the previous plate.
        """
        if self._sidecar is None or not artifact.payload:
            return None
        return self._sidecar.decode(artifact.payload)

    def reconstruct(
        self,
        artifact: BackgroundArtifact,
        previous_decoded: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Plate the client holds after this transmission."""
        decoded = self.decode_payload(artifact)
        if decoded is None:
            return None
        if artifact.mode == MODE_DELTA:
            if previous_decoded is None:
                raise ValueError(
                    "A delta panorama needs the previous decoded plate to reconstruct."
                )
            return apply_delta(previous_decoded, decoded)
        return decoded


class PanoramaFull(BackgroundModel):
    method = BACKGROUND_PANORAMA_FULL
    sends_panorama = True


class PanoramaDelta(BackgroundModel):
    method = BACKGROUND_PANORAMA_DELTA
    sends_panorama = True


class BackgroundNone(BackgroundModel):
    method = BACKGROUND_NONE
    sends_panorama = False


def bind(config: PointstreamConfig, **overrides: Any) -> BackgroundModel:
    """Construct the configured background backend, checking the domain.

    This is the path a pipeline should use. ``Registry.build`` without a
    domain still constructs — the Phase B 'every backend constructs' gate
    needs that — but a panorama under a free-moving camera is refused here.
    """
    from src.components.background import REGISTRY

    method = config.background.method if config.lattice.background else BACKGROUND_NONE
    kwargs: dict[str, Any] = {
        "codec": config.background.codec,
        "jpeg_quality": config.background.jpeg_quality,
        "domain": config.profile,
    }
    kwargs.update(overrides)
    built = REGISTRY.build(method, **kwargs)
    if not isinstance(built, BackgroundModel):
        raise TypeError(f"background backend {method!r} did not construct a BackgroundModel")
    return built
