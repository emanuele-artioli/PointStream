"""Background-model artifacts and payload accounting.

Phase C residual consumes ``BackgroundArtifact``: ``deferred_to_residual`` is
the signal that the lattice row is off and the residual must carry the
background; ``cost().byte_count`` is the measured sidecar size to add to the
run total. Do not invent a byte estimate when nothing was encoded — that is
how a payload total comes out smaller than it really is.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.contracts.objectstream import WireCost

MODE_FULL: str = "full"
MODE_DELTA: str = "delta"
MODE_NONE: str = "none"

#: One scene of a cross-scene low-delay stream: a P-frame whose reference is a
#: reconstruction both sides already hold. The payload is the *marginal* cost of
#: this scene, not a whole plate — `SizesBytes.panorama` means something
#: different under this mode and a total spanning scenes must still include the
#: first scene's keyframe.
MODE_STREAM: str = "stream"


@dataclass(frozen=True)
class BackgroundArtifact:
    """One chunk's background transmission.

    Args:
        method: The strategy that produced this (``panorama-full``,
            ``panorama-delta``, ``none``).
        codec: Sidecar codec name (``jpeg``, ``png``, ``roi-video``). Unused
            when ``mode`` is ``none``, but still recorded from config so a
            typo is visible rather than silently dropped.
        codec_id: Codec plus its settings, so jpeg q50 and jpeg q90 cannot
            be mistaken for the same arm.
        mode: ``full``, ``delta``, ``stream``, or ``none``. Under
            ``panorama-delta`` a first chunk of a scene is ``full`` — that is
            the correct result, not a fallback leak. Under ``panorama-stream``
            a forced keyframe is also ``full``, because a keyframe really is a
            whole plate and the ledger should not read as though it were
            amortised.
        payload: Encoded sidecar bytes. Empty when the model is off.
        width: Plate width in pixels, 0 when off.
        height: Plate height in pixels, 0 when off.
        homographies: Per-frame 3x3 maps, row-major, from frame to plate.
        scene_id: Scene this plate belongs to. Delta keys on this.
        chunk_id: Chunk this transmission is for.
        deferred_to_residual: True when no background model was sent, so
            the residual has to carry the background.
    """

    method: str
    codec: str
    codec_id: str
    mode: str
    payload: bytes
    width: int = 0
    height: int = 0
    homographies: tuple[tuple[float, ...], ...] = ()
    scene_id: str | None = None
    chunk_id: str = ""
    deferred_to_residual: bool = False

    def cost(self) -> WireCost:
        """Measured sidecar bytes. Zero is a real measurement when off."""
        if self.deferred_to_residual:
            return WireCost(
                values=0,
                byte_count=0,
                exact=True,
                basis="no background model; residual carries the background",
            )
        return WireCost(
            values=None,
            byte_count=len(self.payload),
            exact=True,
            basis=f"{self.method} {self.mode} via {self.codec_id}, measured",
        )
