"""Codec drivers and region-control arms.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("codec")

REGISTRY.register(
    BackendSpec(
        name="avc",
        target="src.components.codec.backend:CodecBackend",
        defaults={"codec_name": "avc"},
        capabilities=frozenset({"standard:avc", "driver:ffmpeg", "roi:addroi", "roi:pixel"}),
        summary="H.264/AVC via ffmpeg libx264 — the speed rung",
    )
)
REGISTRY.register(
    BackendSpec(
        name="hevc",
        target="src.components.codec.backend:CodecBackend",
        defaults={"codec_name": "hevc"},
        capabilities=frozenset({"standard:hevc", "driver:binary", "roi:delta-qp-map"}),
        summary="H.265/HEVC via kvazaar — native delta-QP map",
    )
)
REGISTRY.register(
    BackendSpec(
        name="av1",
        target="src.components.codec.backend:CodecBackend",
        defaults={"codec_name": "av1"},
        capabilities=frozenset({"standard:av1", "driver:binary", "roi:delta-qp-map"}),
        summary="AV1 via SvtAv1EncApp — native --roi-map-file",
    )
)
REGISTRY.register(
    BackendSpec(
        name="vvc",
        target="src.components.codec.backend:CodecBackend",
        defaults={"codec_name": "vvc"},
        capabilities=frozenset({"standard:vvc", "driver:ffmpeg", "roi:pixel"}),
        summary="H.266/VVC via ffmpeg libvvenc — pixel-domain ROI only",
    )
)
