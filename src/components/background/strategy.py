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

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.components.background.delta import apply_delta, compute_delta
from src.components.background.plate import CanonicalCanvas, build_plate, prepare_canonical_context
from src.components.background.sidecar import SidecarCodec, build_sidecar, normalize_sidecar
from src.components.background.stream import scene_groups
from src.components.background.types import (
    MODE_DELTA,
    MODE_FULL,
    MODE_NONE,
    MODE_STREAM,
    BackgroundArtifact,
)
from src.contracts.config import PointstreamConfig
from src.contracts.domain import (
    BACKGROUND_NONE,
    BACKGROUND_PANORAMA_DELTA,
    BACKGROUND_PANORAMA_FULL,
    BACKGROUND_PANORAMA_STREAM,
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

    def export_stream_state(self) -> dict[str, Any] | None:
        """None unless this model holds a cross-scene encoder."""
        return None

    def import_stream_state(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError(f"{self.method} has no stream state to restore")

    def transmit(
        self,
        plate: np.ndarray,
        *,
        previous_decoded: np.ndarray | None = None,
        scene_id: str | None = None,
        previous_scene_id: str | None = None,
        chunk_id: str = "",
        homographies: tuple[tuple[float, ...], ...] = (),
        context_id: str | None = None,
    ) -> BackgroundArtifact:
        """Encode one already-built plate under this strategy.

        ``previous_decoded`` must be the plate the *client* will have after
        decoding the previous sidecar — not the pre-codec pixels — so a later
        residual is computed against what both sides actually share.
        ``context_id`` is consumed by the streaming strategy; other methods
        ignore it.
        """
        _ = context_id
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
        plate, homographies = self.stitch(frames, masks=masks)
        return self.transmit(
            plate,
            previous_decoded=previous_decoded,
            scene_id=scene_id,
            previous_scene_id=previous_scene_id,
            chunk_id=chunk_id,
            homographies=homographies,
        )

    def stitch(
        self,
        frames: np.ndarray,
        masks: np.ndarray | None = None,
        *,
        register: bool = True,
    ) -> tuple[np.ndarray, tuple[tuple[float, ...], ...]]:
        """Build this scene's plate. Independent coding uses a local canvas."""
        return build_plate(frames, masks=masks, register=register)

    def prepare_context(
        self,
        scenes: Sequence[np.ndarray],
        *,
        context_id: str | None = None,
        register: bool = True,
    ) -> CanonicalCanvas | None:
        """Offline union canvas. Independent coding has nothing to prepare."""
        _ = (scenes, context_id, register)
        return None

    def prepare_contexts(
        self,
        scenes: Sequence[np.ndarray],
        context_ids: Sequence[str],
        *,
        register: bool = True,
    ) -> CanonicalCanvas | None:
        """One union canvas per consecutive run of the same context id."""
        _ = (scenes, context_ids, register)
        return None

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


@dataclass(frozen=True)
class PreparedContext:
    """One consecutive run of scenes that share a background context."""

    start: int
    end: int
    context_id: str
    canvas: CanonicalCanvas
    alignments: tuple[np.ndarray, ...]


class PanoramaStream(BackgroundModel):
    """The plate as a low-delay stream across scenes, not a still per scene.

    Stateful on purpose, and it is the only strategy here that is. Every other
    backend codes one plate with no memory, which is what made the background
    88-91% of the payload: each scene paid for its plate from scratch. This one
    carries the previous **reconstruction** across scenes and codes the next
    plate against it. Measured over five videos, 16 scenes each
    (`plans/done/BP30-findings.md` §29): 49.2% +- 6.2% of coding every plate fresh,
    best case 29.4%.

    **The reconstruction is never recomputed, only decoded**, which is what
    keeps encoder and client identical. `decode_payload` returns what the
    transmitter decoded from its own output rather than decoding this scene's
    payload alone — a P-frame in isolation does not decode, and the client's
    copy comes from the same bytes along the same chain. BP30's
    `test_reconstructions_are_bit_identical_across_scenes` is what backs that.

    **One instance is one stream.** The runner builds stages once and reuses the
    closure across chunks, so the sequence of chunks is the sequence of scenes.
    A second run needs a second instance, or it would predict scene 1 of the new
    run from the last scene of the old one.
    """

    method = BACKGROUND_PANORAMA_STREAM
    sends_panorama = True

    def __init__(
        self,
        codec: str = "jpeg",
        jpeg_quality: int = 50,
        png_compression: int = 3,
        roi_crf: int = 30,
        roi_preset: str = "veryfast",
        domain: DomainProfile | str | None = None,
        reference_mode: str = "last",
        keyframe_interval: int = 0,
        stream_codec: str = "av1",
        stream_crf: int = 38,
        context_id: str = "",
        canvas: str = "independent",
    ) -> None:
        super().__init__(
            codec=codec,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
            roi_crf=roi_crf,
            roi_preset=roi_preset,
            domain=domain,
        )
        from src.components.background.stream import BackgroundStreamTransmitter

        if canvas not in {"independent", "canonical"}:
            raise ValueError(
                f"background.canvas must be 'independent' or 'canonical', got {canvas!r}"
            )
        self.context_id = context_id
        self.canvas_mode = canvas
        self._transmitter = BackgroundStreamTransmitter(
            mode=reference_mode,
            codec=stream_codec,
            crf=stream_crf,
            keyframe_interval=keyframe_interval,
        )
        self._active_context: str | None = None
        self._canvas: CanonicalCanvas | None = None
        self._alignments: tuple[np.ndarray, ...] = ()
        self._groups: tuple[PreparedContext, ...] = ()
        self._scene_index = 0

    def prepare_context(
        self,
        scenes: Sequence[np.ndarray],
        *,
        context_id: str | None = None,
        register: bool = True,
    ) -> CanonicalCanvas:
        """Offline union of scene bounds. Must run before the first transmit.

        This sees every scene in the context, so it is a buffered codec mode.
        One id for every scene — a mixed list belongs on ``prepare_contexts``.
        """
        active = context_id if context_id is not None else (self.context_id or "run")
        canvas = self.prepare_contexts(
            scenes,
            tuple(active for _ in scenes),
            register=register,
        )
        if canvas is None:
            raise ValueError("a background context needs at least one scene")
        return canvas

    def prepare_contexts(
        self,
        scenes: Sequence[np.ndarray],
        context_ids: Sequence[str],
        *,
        register: bool = True,
    ) -> CanonicalCanvas | None:
        """One canonical canvas per consecutive run of the same context id.

        Mixing ids in one ``prepare_context`` used to union unrelated cameras
        onto one canvas. A context change is a new independently coded
        background, possibly a different size.
        """
        if len(scenes) != len(context_ids):
            raise ValueError(
                f"prepare_contexts got {len(scenes)} scenes and {len(context_ids)} "
                "context ids; they must be aligned."
            )
        groups: list[PreparedContext] = []
        for start, end in scene_groups(tuple(context_ids)):
            cid = str(context_ids[start])
            canvas, alignments, _bounds = prepare_canonical_context(
                list(scenes[start:end]),
                context_id=cid,
                register=register,
            )
            groups.append(
                PreparedContext(
                    start=start,
                    end=end,
                    context_id=cid,
                    canvas=canvas,
                    alignments=alignments,
                )
            )
        self._groups = tuple(groups)
        self._scene_index = 0
        if not groups:
            self._canvas = None
            self._alignments = ()
            return None
        first = groups[0]
        self._canvas = first.canvas
        self._alignments = first.alignments
        self.context_id = first.context_id
        return first.canvas

    def _prepared_at(self, index: int) -> PreparedContext | None:
        for group in self._groups:
            if group.start <= index < group.end:
                return group
        return None

    def stitch(
        self,
        frames: np.ndarray,
        masks: np.ndarray | None = None,
        *,
        register: bool = True,
    ) -> tuple[np.ndarray, tuple[tuple[float, ...], ...]]:
        group = self._prepared_at(self._scene_index)
        if group is not None:
            alignment = group.alignments[self._scene_index - group.start]
            plate, homographies = build_plate(
                frames,
                masks=masks,
                register=register,
                canvas=group.canvas,
                alignment=alignment,
            )
            self._scene_index += 1
            return plate, homographies
        if self._canvas is None:
            return build_plate(frames, masks=masks, register=register)
        fallback: np.ndarray | None = None
        if self._scene_index < len(self._alignments):
            fallback = self._alignments[self._scene_index]
        plate, homographies = build_plate(
            frames,
            masks=masks,
            register=register,
            canvas=self._canvas,
            alignment=fallback,
        )
        self._scene_index += 1
        return plate, homographies

    @property
    def codec_id(self) -> str:
        spec = self._transmitter.spec
        return (
            f"{spec.name} low-delay crf{self._transmitter.crf} "
            f"ref={self._transmitter.mode} k={self._transmitter.keyframe_interval}"
        )

    def transmit(
        self,
        plate: np.ndarray,
        *,
        previous_decoded: np.ndarray | None = None,
        scene_id: str | None = None,
        previous_scene_id: str | None = None,
        chunk_id: str = "",
        homographies: tuple[tuple[float, ...], ...] = (),
        context_id: str | None = None,
    ) -> BackgroundArtifact:
        """Code this scene's plate against the stream so far.

        ``previous_decoded`` is ignored: the transmitter holds the
        reconstructions itself, and taking one from a caller is how the two
        sides start disagreeing about which picture was predicted from.

        A change of ``context_id`` resets the stream: the next plate is an
        independently coded keyframe, possibly on a different canvas. Prepared
        groups stay; the encoder is what restarts. Wiping the canvas here
        used to drop a later context back onto a local size mid-run.
        """
        active = context_id if context_id is not None else self.context_id
        if self._active_context is not None and active != self._active_context:
            self._transmitter.reset()
            if not self._groups:
                self._canvas = None
                self._alignments = ()
                self._scene_index = 0
        self._active_context = active
        array = np.asarray(plate, dtype=np.uint8)
        payload = self._transmitter.push(array)
        return BackgroundArtifact(
            method=self.method,
            codec=self.codec_name,
            codec_id=self.codec_id,
            # A keyframe really is a whole plate; saying `full` keeps the ledger
            # honest about which scenes were not amortised.
            mode=MODE_FULL if payload.is_keyframe else MODE_STREAM,
            payload=payload.payload,
            width=int(array.shape[1]),
            height=int(array.shape[0]),
            homographies=homographies,
            scene_id=scene_id,
            chunk_id=chunk_id,
            deferred_to_residual=False,
        )

    def decode_payload(self, artifact: BackgroundArtifact) -> np.ndarray | None:
        """The plate the client holds after this scene.

        Not a decode of ``artifact.payload`` on its own — a P-frame needs its
        chain. This is the reconstruction the transmitter decoded from its own
        output, which is bit-identical to what a client decoding the chain gets.
        """
        reconstructions = self._transmitter.reconstructions
        return reconstructions[-1] if reconstructions else None

    def reconstruct(
        self,
        artifact: BackgroundArtifact,
        previous_decoded: np.ndarray | None = None,
    ) -> np.ndarray | None:
        return self.decode_payload(artifact)

    def export_stream_state(self) -> dict[str, Any] | None:
        return {
            "scene_index": int(self._scene_index),
            "active_context": self._active_context,
            "transmitter": self._transmitter.export_state(),
            "context_id": self.context_id,
            "canvas": asdict(self._canvas) if self._canvas is not None else None,
            "alignments": [item.tolist() for item in self._alignments],
            "groups": [
                {"start": group.start, "end": group.end, "context_id": group.context_id,
                 "canvas": asdict(group.canvas),
                 "alignments": [item.tolist() for item in group.alignments]}
                for group in self._groups
            ],
        }

    def import_stream_state(self, state: dict[str, Any]) -> None:
        self._scene_index = int(state["scene_index"])
        self._active_context = state.get("active_context")
        self._transmitter.import_state(state["transmitter"])
        self.context_id = state["context_id"]
        self._canvas = CanonicalCanvas(**state["canvas"]) if state["canvas"] else None
        self._alignments = tuple(np.asarray(item) for item in state["alignments"])
        self._groups = tuple(
            PreparedContext(
                start=group["start"], end=group["end"], context_id=group["context_id"],
                canvas=CanonicalCanvas(**group["canvas"]),
                alignments=tuple(np.asarray(item) for item in group["alignments"]),
            ) for group in state["groups"]
        )


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
    if method == BACKGROUND_PANORAMA_STREAM:
        kwargs.update(
            {
                "reference_mode": config.background.reference_mode,
                "keyframe_interval": config.background.keyframe_interval,
                "stream_codec": config.background.stream_codec,
                "stream_crf": config.background.stream_crf,
                "context_id": config.background.context_id,
                "canvas": config.background.canvas,
            }
        )
    kwargs.update(overrides)
    built = REGISTRY.build(method, **kwargs)
    if not isinstance(built, BackgroundModel):
        raise TypeError(f"background backend {method!r} did not construct a BackgroundModel")
    return built
