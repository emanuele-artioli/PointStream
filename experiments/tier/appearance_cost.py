"""Is `actor_reference` a transmitted cost, or an array size wearing one?

BP24 listed `actor_reference` in `SizesBytes.raw_parts` unconditionally, with
the reason recorded rather than guessed at: *appearance reports a measured size
and nobody has shown it is a coded one*. Withholding it withholds the
compression ratio, so the paired ladder cannot run until the question is
answered — and the answer has to be driven, not read off the code, because a
flag existing is not a feature working.

Three questions per shipped appearance backend, each with a way to be wrong:

1. **Does the payload exist at all?** ``encode`` must hand back bytes, not just
   a descriptor with a number on it. A descriptor's ``measured_bytes`` with no
   buffer behind it is exactly the shape of an invented size.
2. **Is the declared size the buffer's size?** ``descriptor.cost().byte_count``
   must equal ``len(payload)``. A mismatch means the ledger is counting
   something other than what would be sent.
3. **Does the size respond to the knob that is supposed to move it?** For a
   real encoder, quality changes the bitstream. For a packed array it must
   *not* — the size follows the declared quantization and nothing else, which
   is a different, equally checkable claim.

Question 3 is what separates a coded payload from a packed one, and the
distinction is reported rather than collapsed: both are wire costs, but a
reader comparing a JPEG appearance against a latent one is comparing a
bitstream against an array, and the report has to say so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "bp24-ladder" / "appearance-cost.json"


def _crop(seed: int = 7, height: int = 192, width: int = 128) -> np.ndarray:
    """A compressible crop. Noise would be incompressible and prove nothing.

    `plans/BP24-findings.md` §3: a noise anchor saturates every encoder at every
    quality, so a size that does not move would be uninformative rather than
    evidence.
    """
    rng = np.random.default_rng(seed)
    ramp_y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    ramp_x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    base = np.stack(
        [
            np.broadcast_to(ramp_y, (height, width)),
            np.broadcast_to(ramp_x, (height, width)),
            (ramp_y + ramp_x) / 2.0,
        ],
        axis=2,
    )
    blob_y, blob_x = np.ogrid[:height, :width]
    disc = ((blob_y - height // 3) ** 2 + (blob_x - width // 2) ** 2) < (min(height, width) // 4) ** 2
    base[disc] = (40.0, 200.0, 90.0)
    base += rng.normal(0.0, 3.0, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def _probe(name: str, backend: Any, crop: np.ndarray, knob: dict[str, Any]) -> dict[str, Any]:
    encoded = backend.encode(crop)
    if not (isinstance(encoded, tuple) and len(encoded) == 2):
        return {
            "backend": name,
            "verdict": "RAW — encode returned no payload buffer to check",
            "returned": type(encoded).__name__,
        }
    descriptor, payload = encoded
    cost = descriptor.cost()
    payload_bytes = len(payload) if isinstance(payload, (bytes, bytearray)) else int(np.asarray(payload).nbytes)

    sizes: dict[str, int] = {}
    for label, kwargs in knob.items():
        try:
            _, other = backend.encode(crop, **kwargs)
        except TypeError:
            # Backend does not take the knob; that is itself the answer.
            sizes[label] = -1
            continue
        sizes[label] = len(other)

    moved = len({size for size in sizes.values() if size >= 0}) > 1
    agrees = cost.byte_count == payload_bytes

    if not agrees:
        verdict = (
            f"RAW — declared cost {cost.byte_count} B does not match the "
            f"{payload_bytes} B buffer; the ledger would count something that is not sent"
        )
    elif moved:
        verdict = "CODED — a real bitstream; its size responds to the encoder's quality knob"
    else:
        verdict = (
            "PACKED — the buffer is the declared quantization sent verbatim, with no "
            "coding step configured. A wire cost, but not a coded one"
        )

    return {
        "backend": name,
        "payload_type": type(payload).__name__,
        "payload_bytes": payload_bytes,
        "declared_byte_count": cost.byte_count,
        "declared_equals_payload": agrees,
        "cost_exact": bool(cost.exact),
        "cost_basis": cost.basis,
        "sizes_across_knob": sizes,
        "size_responds_to_knob": moved,
        "verdict": verdict,
    }


def main() -> int:
    from src.components.appearance.compressed import CompressedImageAppearance
    from src.components.appearance.embedding import ImageEmbeddingAppearance
    from src.components.appearance.latent import DiffusionLatentAppearance

    crop = _crop()
    records = [
        _probe(
            "compressed-image",
            CompressedImageAppearance(quality=90),
            crop,
            {"q20": {"quality": 20}, "q60": {"quality": 60}, "q95": {"quality": 95}},
        ),
        _probe("image-embedding", ImageEmbeddingAppearance(), crop, {}),
        _probe("diffusion-latent", DiffusionLatentAppearance(), crop, {}),
    ]

    # The JPEG path is the one the tier configs use, so it gets the extra check
    # a byte count cannot give: the payload has to decode back to the crop.
    jpeg = CompressedImageAppearance(quality=90)
    _, payload = jpeg.encode(crop)
    decoded = jpeg.decode(payload)
    # `decode` returns BGR from cv2; compare per-channel against both orders and
    # take the better, rather than asserting a channel convention that is not
    # the thing under test.
    diff_rgb = float(np.abs(decoded.astype(np.int16) - crop.astype(np.int16)).mean())
    diff_bgr = float(np.abs(decoded[..., ::-1].astype(np.int16) - crop.astype(np.int16)).mean())
    roundtrip = {
        "decoded_shape": list(decoded.shape),
        "mean_abs_diff_best_channel_order": min(diff_rgb, diff_bgr),
        "channel_order": "rgb" if diff_rgb <= diff_bgr else "bgr (cv2 convention)",
        "verdict": (
            "the JPEG payload decodes back to the crop, so those bytes really "
            "carry the appearance"
            if min(diff_rgb, diff_bgr) < 8.0
            else "ALARM — the payload does not decode back to the crop"
        ),
    }

    payload_out = {
        "question": "is actor_reference a transmitted cost or an array size?",
        "why": (
            "BP24 withheld transport_to_source_ratio while actor_reference was "
            "listed raw. The paired ladder needs a rate, so this had to be "
            "settled with evidence rather than cleared."
        ),
        "crop": {"shape": list(crop.shape), "bytes": int(crop.nbytes)},
        "backends": records,
        "jpeg_roundtrip": roundtrip,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload_out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
