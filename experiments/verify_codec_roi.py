"""Does a codec's region-of-interest surface actually move quality?

A flag existing in `--help` is not evidence that it works. The prior project's
region-of-interest table was invalidated by assuming it did, and this project's
own panorama encoder drives ffmpeg's `addroi` filter without anything ever
checking that the output differs from the same encode without it.

This measures the thing directly. Two encodes of the same source at the *same
fixed QP*, differing only in whether a region map is supplied, then per-region
PSNR of each decode against the source. If the surface works, the mapped region
gets better and the rest gets worse. If both regions score the same in both arms,
the flag is decorative and we do not have that ROI arm.

Fixed QP, not CRF or target bitrate, is deliberate: a delta-QP map offsets a base
QP, so under any rate control free to move that base the offsets stop meaning
anything fixed. It is also the matched-rate discipline the codec contract
enforces — the baseline here differs from the ROI arm in exactly one thing.

**What this currently establishes, and what it does not.** On SVT-AV1 1.8.0 the
map is *precisely localized*: a -30 offset over a centred region with zero
elsewhere moved that region +0.35 dB and left the rest unchanged to two decimals.
Bits go where the map says and nowhere else.

On ffmpeg n7.1.1-56 (`/opt/local/bin/ffmpeg`, libx264) native `addroi` at
matched **QP** is a no-op: 20 frames of `assets/real_tennis.mp4` at 640x384,
QP 45, preset veryfast, inside offset -30, produced byte-identical bitstreams
and 0.00 dB PSNR change in the labelled region. The same filter at **CRF** 45
is localized (+17.00 dB inside, 0.00 dB outside). CQP disables the AQ path
libx264 uses for ROI, so AVC has no native arm under the contract's QP
discipline. `--roi-arm auto` correctly stays on the pixel arm.

**Matched QP is not matched rate.** The region arm at the same QP spends more
bytes; more bits buying more quality is not a result. `--match-bitrate` walks a
QP ladder, binary-searches the ROI-arm base QP onto each baseline's byte count,
and reports whether the region gained at the background's expense at equal size.
`assert_matched_rate_control` runs before any cross-arm number is printed.
A single point is not a comparison — the ladder is the comparison.

This is the same trap that invalidated a prior project's region-of-interest
table, where fixed-QP region arms were compared against target-bitrate baselines
on an encoder that overshoots its target by 30-45%.

Two reproducibility notes worth keeping:

**Offsets are in q_index units, not QP units.** AV1's q_index runs 0-255 against
SVT-AV1's `--qp` 0-63, so an offset of -30 is about -7.5 QP steps, not -30. A
prediction made in QP units overestimates the effect roughly fourfold.

**Very large offsets are not simply stronger.** At -120/+60 the differentiation
between regions *shrank* rather than grew, and both regions lost quality.

Run:
    python -m experiments.verify_codec_roi --codec av1
    python -m experiments.verify_codec_roi --codec av1 --inside-offset -30 --outside-offset 0
    python -m experiments.verify_codec_roi --codec av1 --match-bitrate
    python -m experiments.verify_codec_roi --codec av1 --match-bitrate --ladder 40,45,50
    python -m experiments.verify_codec_roi --codec avc --roi-arm native --preset veryfast

AVC: ``--roi-arm auto`` selects the pixel arm (addroi is not a verified
delta-QP map). Native addroi needs ``--roi-arm native`` and an x264 preset
name, not the SVT-AV1 default ``8``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from src.components.codec.encode import (
    BITSTREAM_SUFFIX,
    QP_BOUNDS,
    decode as decode_bitstream,
    encode as encode_bitstream,
    search_qp,
)
from src.components.codec.roi import AV1_BLOCK, BlockRoiMap
from src.components.codec.tools import resolve_encoder, resolve_ffmpeg
from src.contracts.codecs import EncodeRequest, RateControl, assert_matched_rate_control

BLOCK = AV1_BLOCK
"""SVT-AV1 applies ROI QP offsets per 64x64 block, row-major across the frame."""


# --------------------------------------------------------------------------
# y4m
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Y4M:
    """Decoded luma planes and their geometry."""

    width: int
    height: int
    luma: np.ndarray  # (frames, height, width) uint8

    @property
    def frames(self) -> int:
        return int(self.luma.shape[0])


def read_y4m_luma(path: Path) -> Y4M:
    """Read a 4:2:0 y4m, keeping only the luma plane.

    Luma is what PSNR is dominated by and what a QP change moves most visibly,
    so chroma is skipped rather than averaged into a number that would hide the
    effect being measured.
    """
    data = path.read_bytes()
    header_end = data.index(b"\n")
    header = data[:header_end].decode("ascii")

    width = height = 0
    for token in header.split():
        if token.startswith("W"):
            width = int(token[1:])
        elif token.startswith("H"):
            height = int(token[1:])
    if not width or not height:
        raise ValueError(f"{path} has no frame size in its header: {header!r}")

    luma_size = width * height
    chroma_size = (width // 2) * (height // 2) * 2
    frame_size = luma_size + chroma_size

    frames: list[np.ndarray] = []
    offset = header_end + 1
    marker = b"FRAME"
    while offset < len(data):
        if not data.startswith(marker, offset):
            break
        offset = data.index(b"\n", offset) + 1
        plane = np.frombuffer(data, dtype=np.uint8, count=luma_size, offset=offset)
        frames.append(plane.reshape(height, width).copy())
        offset += frame_size

    if not frames:
        raise ValueError(f"{path} contained no frames")
    return Y4M(width=width, height=height, luma=np.stack(frames))


# --------------------------------------------------------------------------
# Region map
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionMap:
    """A rectangle of interest, in 64x64 block coordinates."""

    blocks_wide: int
    blocks_high: int
    col0: int
    col1: int
    row0: int
    row1: int
    inside_offset: int
    outside_offset: int

    @classmethod
    def centred(
        cls,
        width: int,
        height: int,
        *,
        inside_offset: int,
        outside_offset: int,
        fraction: float = 0.4,
    ) -> RegionMap:
        """A centred rectangle covering roughly `fraction` of each axis."""
        blocks_wide = -(-width // BLOCK)
        blocks_high = -(-height // BLOCK)
        span_w = max(1, round(blocks_wide * fraction))
        span_h = max(1, round(blocks_high * fraction))
        col0 = (blocks_wide - span_w) // 2
        row0 = (blocks_high - span_h) // 2
        return cls(
            blocks_wide=blocks_wide,
            blocks_high=blocks_high,
            col0=col0,
            col1=col0 + span_w,
            row0=row0,
            row1=row0 + span_h,
            inside_offset=inside_offset,
            outside_offset=outside_offset,
        )

    def offsets(self) -> list[int]:
        """Per-block QP offsets, row-major — the order SVT-AV1 reads them in."""
        values: list[int] = []
        for row in range(self.blocks_high):
            for col in range(self.blocks_wide):
                inside = self.row0 <= row < self.row1 and self.col0 <= col < self.col1
                values.append(self.inside_offset if inside else self.outside_offset)
        return values

    def pixel_mask(self, width: int, height: int) -> np.ndarray:
        """Boolean mask, True inside the region, at pixel resolution."""
        mask = np.zeros((height, width), dtype=bool)
        y0, y1 = self.row0 * BLOCK, min(self.row1 * BLOCK, height)
        x0, x1 = self.col0 * BLOCK, min(self.col1 * BLOCK, width)
        mask[y0:y1, x0:x1] = True
        return mask

    def write(self, path: Path, frames: int) -> None:
        """Write one line per frame: frame number then every block's offset."""
        offsets = " ".join(str(value) for value in self.offsets())
        path.write_text(
            "".join(f"{index} {offsets}\n" for index in range(frames)),
            encoding="ascii",
        )


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr[-2000:]}"
        )
    return result


def extract_source(
    source: Path,
    target: Path,
    *,
    width: int,
    height: int,
    frames: int,
    ffmpeg: str,
) -> None:
    run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-frames:v", str(frames),
            "-vf", f"scale={width}:{height}",
            "-pix_fmt", "yuv420p",
            str(target),
        ]
    )


def _block_map(region: RegionMap) -> BlockRoiMap:
    return BlockRoiMap(
        block_size=BLOCK,
        blocks_wide=region.blocks_wide,
        blocks_high=region.blocks_high,
        offsets=tuple(region.offsets()),
        inside_offset=region.inside_offset,
        outside_offset=region.outside_offset,
        col0=region.col0,
        col1=region.col1,
        row0=region.row0,
        row1=region.row1,
    )


def _request(codec_name: str, qp: int, preset: str, extra: tuple[str, ...] = ()) -> EncodeRequest:
    return EncodeRequest(
        codec_name=codec_name,
        rate_control=RateControl.QP,
        rate=qp,
        preset=preset,
        pix_fmt="yuv420p",
        extra_args=extra,
    )


def _encode_arm(
    source: Path,
    bitstream: Path,
    request: EncodeRequest,
    *,
    roi: BlockRoiMap | None,
    roi_arm: str,
) -> None:
    encode_bitstream(
        source,
        bitstream,
        request,
        roi=roi,
        roi_arm="none" if roi is None else roi_arm,  # type: ignore[arg-type]
    )


def _decode_arm(bitstream: Path, target: Path, request: EncodeRequest) -> None:
    decode_bitstream(bitstream, target, request)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def masked_psnr(reference: np.ndarray, decoded: np.ndarray, mask: np.ndarray) -> float:
    """Luma PSNR over the masked pixels only, across every frame."""
    frames = min(reference.shape[0], decoded.shape[0])
    ref = reference[:frames][:, mask].astype(np.float64)
    dec = decoded[:frames][:, mask].astype(np.float64)
    mse = float(np.mean((ref - dec) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(255.0**2 / mse))


@dataclass(frozen=True)
class ArmResult:
    label: str
    bytes_: int
    psnr_inside: float
    psnr_outside: float
    psnr_overall: float


def measure(label: str, source: Y4M, decoded_path: Path, bitstream: Path, mask: np.ndarray) -> ArmResult:
    decoded = read_y4m_luma(decoded_path)
    return ArmResult(
        label=label,
        bytes_=bitstream.stat().st_size,
        psnr_inside=masked_psnr(source.luma, decoded.luma, mask),
        psnr_outside=masked_psnr(source.luma, decoded.luma, ~mask),
        psnr_overall=masked_psnr(source.luma, decoded.luma, np.ones_like(mask)),
    )


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

WORKS_THRESHOLD_DB = 0.25
"""Below this the difference is not distinguishable from encoder noise."""


LOCALIZED_THRESHOLD_DB = 0.1
"""A region with a zero offset should not move at all. This is how much drift
still counts as "not moved"."""


def verdict(
    baseline: ArmResult,
    roi: ArmResult,
    *,
    inside_offset: int,
    outside_offset: int,
) -> tuple[bool, str]:
    """Judge each region against what its own offset asked for.

    Judged per region rather than as a single pass/fail, because the informative
    experiment is the one-sided map: setting one region's offset to zero and
    checking it does not move is what distinguishes real region control from an
    encoder simply spending fewer bits everywhere.
    """
    inside_change = roi.psnr_inside - baseline.psnr_inside
    outside_change = roi.psnr_outside - baseline.psnr_outside

    def judge(offset: int, change: float, where: str) -> tuple[bool, str]:
        if offset == 0:
            if abs(change) <= LOCALIZED_THRESHOLD_DB:
                return True, f"{where} unmapped and unmoved ({change:+.2f} dB)"
            return False, f"{where} had no offset but moved {change:+.2f} dB — not localized"
        wanted = "better" if offset < 0 else "worse"
        got_expected = (change > 0) if offset < 0 else (change < 0)
        if abs(change) < WORKS_THRESHOLD_DB:
            return False, f"{where} asked to be {wanted} but barely moved ({change:+.2f} dB)"
        if not got_expected:
            return False, f"{where} asked to be {wanted} and went the other way ({change:+.2f} dB)"
        return True, f"{where} asked to be {wanted} and was ({change:+.2f} dB)"

    inside_ok, inside_note = judge(inside_offset, inside_change, "inside")
    outside_ok, outside_note = judge(outside_offset, outside_change, "outside")

    if inside_ok and outside_ok:
        return True, (
            f"LOCALIZED — {inside_note}; {outside_note}. "
            f"Note this is at matched QP, so it does not yet show that region "
            f"control pays: compare at matched bitrate for that."
        )
    if (
        abs(inside_change) < WORKS_THRESHOLD_DB
        and abs(outside_change) < WORKS_THRESHOLD_DB
        and (inside_offset or outside_offset)
    ):
        return False, (
            "NO EFFECT — supplying the map changed nothing measurable. "
            "The flag is accepted and ignored; this is not a usable ROI arm."
        )
    return False, f"UNEXPECTED — {inside_note}; {outside_note}. Investigate before relying on it."


def redistribution_verdict(baseline: ArmResult, roi: ArmResult) -> tuple[bool, str]:
    """At matched bitrate: did the region gain at the background's expense?"""
    inside = roi.psnr_inside - baseline.psnr_inside
    outside = roi.psnr_outside - baseline.psnr_outside
    rel = abs(roi.bytes_ - baseline.bytes_) / max(baseline.bytes_, 1)
    if inside > WORKS_THRESHOLD_DB and outside < -LOCALIZED_THRESHOLD_DB:
        return True, (
            f"REDISTRIBUTES — inside {inside:+.2f} dB, outside {outside:+.2f} dB "
            f"at {rel:.1%} byte error ({baseline.bytes_} vs {roi.bytes_} bytes)."
        )
    if abs(inside) < WORKS_THRESHOLD_DB and abs(outside) < WORKS_THRESHOLD_DB:
        return False, (
            f"NO REDISTRIBUTION — inside {inside:+.2f} dB, outside {outside:+.2f} dB "
            f"at matched size. The map did not move quality at equal bitrate."
        )
    return False, (
        f"UNEXPECTED at matched size — inside {inside:+.2f} dB, outside {outside:+.2f} dB "
        f"({baseline.bytes_} vs {roi.bytes_} bytes)."
    )


def _ladder(args: argparse.Namespace) -> list[int]:
    lo, hi = QP_BOUNDS[args.codec]
    if args.ladder:
        values = [int(tok.strip()) for tok in args.ladder.split(",") if tok.strip()]
    else:
        values = [args.qp - 4, args.qp, args.qp + 4]
    clipped = [max(lo, min(hi, qp)) for qp in values]
    seen: set[int] = set()
    out: list[int] = []
    for qp in clipped:
        if qp not in seen:
            seen.add(qp)
            out.append(qp)
    return out


def _print_arms(arms: dict[str, ArmResult]) -> None:
    print(f"{'arm':<10} {'bytes':>9}  {'PSNR in':>8}  {'PSNR out':>8}  {'PSNR all':>8}")
    for arm in arms.values():
        print(
            f"{arm.label:<10} {arm.bytes_:>9}  {arm.psnr_inside:>8.2f}  "
            f"{arm.psnr_outside:>8.2f}  {arm.psnr_overall:>8.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--codec", default="av1", choices=["avc", "hevc", "av1", "vvc"])
    parser.add_argument("--source", type=Path, default=Path("assets/real_tennis.mp4"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384, help="A multiple of 64 keeps blocks whole.")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--qp", type=int, default=45, help="High enough that QP offsets have room to act.")
    parser.add_argument(
        "--preset",
        default=None,
        help="Encoder preset in that codec's own vocabulary (svtav1 0-13, x264 names, vvenc names). "
        "Default: 8 for av1, veryfast for avc, ultrafast for hevc, faster for vvc.",
    )
    parser.add_argument("--inside-offset", type=int, default=-30, help="Negative means better quality. AV1: q_index units, ~4 per QP step.")
    parser.add_argument(
        "--outside-offset",
        type=int,
        default=0,
        help="Zero is the sharper test: an unmapped region that does not move proves localization.",
    )
    parser.add_argument("--aq-mode", type=int, default=None, help="SVT-AV1 docs tie ROI to aq-mode 1.")
    parser.add_argument(
        "--encoder-bin",
        default=None,
        help="Override the encoder binary (sets SVTAV1_BIN / KVAZAAR_BIN / FFMPEG_BIN). Version is recorded.",
    )
    parser.add_argument(
        "--roi-arm",
        default="auto",
        choices=["auto", "native", "pixel"],
        help="native = encoder map/addroi; pixel = in-house degradation. auto prefers a verified delta-QP map.",
    )
    parser.add_argument(
        "--match-bitrate",
        action="store_true",
        help="Sweep a QP ladder and compare arms at matched byte count, not matched QP.",
    )
    parser.add_argument(
        "--ladder",
        default=None,
        help="Comma-separated baseline QPs to sweep. Default: qp-4,qp,qp+4.",
    )
    parser.add_argument("--match-tolerance", type=float, default=0.08)
    parser.add_argument("--out", type=Path, default=None, help="Where to write report.json.")
    args = parser.parse_args(argv)
    if args.preset is None:
        args.preset = {"av1": "8", "avc": "veryfast", "hevc": "ultrafast", "vvc": "faster"}[args.codec]

    if args.encoder_bin:
        env_key = {"av1": "SVTAV1_BIN", "hevc": "KVAZAAR_BIN", "avc": "FFMPEG_BIN", "vvc": "FFMPEG_BIN"}[args.codec]
        os.environ[env_key] = args.encoder_bin

    try:
        ffmpeg = resolve_ffmpeg()
        encoder = resolve_encoder(args.codec)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not args.source.exists():
        print(f"source not found: {args.source}", file=sys.stderr)
        return 2

    extra: tuple[str, ...] = ()
    if args.codec == "av1":
        extra = ("--keyint", "-1")
        if args.aq_mode is not None:
            extra += ("--aq-mode", str(args.aq_mode))

    suffix = BITSTREAM_SUFFIX[args.codec]
    roi_arm = args.roi_arm

    with tempfile.TemporaryDirectory(prefix="roi-verify-") as raw_tmp:
        tmp = Path(raw_tmp)
        source_y4m = tmp / "source.y4m"
        extract_source(
            args.source, source_y4m,
            width=args.width, height=args.height, frames=args.frames,
            ffmpeg=ffmpeg.path,
        )
        source = read_y4m_luma(source_y4m)
        region = RegionMap.centred(
            source.width,
            source.height,
            inside_offset=args.inside_offset,
            outside_offset=args.outside_offset,
        )
        block_map = _block_map(region)
        mask = region.pixel_mask(source.width, source.height)

        def run_pair(qp_base: int, qp_roi: int) -> tuple[ArmResult, ArmResult]:
            arms: dict[str, ArmResult] = {}
            for label, qp, roi in (
                ("baseline", qp_base, None),
                ("roi", qp_roi, block_map),
            ):
                bitstream = tmp / f"{label}_qp{qp}{suffix}"
                decoded = tmp / f"{label}_qp{qp}.y4m"
                request = _request(args.codec, qp, str(args.preset), extra)
                _encode_arm(source_y4m, bitstream, request, roi=roi, roi_arm=roi_arm)
                _decode_arm(bitstream, decoded, request)
                arms[label] = measure(label, source, decoded, bitstream, mask)
            assert_matched_rate_control(
                (
                    _request(args.codec, qp_base, str(args.preset), extra),
                    _request(args.codec, qp_roi, str(args.preset), extra),
                )
            )
            return arms["baseline"], arms["roi"]

        print(f"source      : {args.source} @ {source.width}x{source.height}, {source.frames} frames")
        print(
            f"region      : blocks {region.blocks_wide}x{region.blocks_high}, "
            f"cols {region.col0}-{region.col1}, rows {region.row0}-{region.row1}, "
            f"offsets {args.inside_offset} inside / {args.outside_offset} outside"
        )
        print(f"encoder     : {encoder.path} ({encoder.version})")
        print(f"ffmpeg      : {ffmpeg.path} ({ffmpeg.version})")
        print(f"roi_arm     : {roi_arm}")

        report: dict[str, object]
        works: bool

        if args.match_bitrate:
            ladder = _ladder(args)
            print(f"encode      : matched-bitrate sweep ladder={ladder} preset={args.preset}")
            print()
            points = []
            any_redistribute = False
            any_wrong = False
            for qp in ladder:
                baseline_arm, _ignored = run_pair(qp, qp)
                target = baseline_arm.bytes_
                lo, hi = QP_BOUNDS[args.codec]

                def _at(probe_qp: int) -> int:
                    bitstream = tmp / f"search_qp{probe_qp}{suffix}"
                    request = _request(args.codec, probe_qp, str(args.preset), extra)
                    rec = encode_bitstream(
                        source_y4m, bitstream, request, roi=block_map, roi_arm=roi_arm,
                    )
                    return rec.size_bytes

                try:
                    roi_qp, _size = search_qp(_at, target, lo, hi, tolerance=args.match_tolerance)
                except RuntimeError as exc:
                    print(f"qp {qp}: search failed: {exc}")
                    any_wrong = True
                    continue
                baseline_arm, roi_arm_result = run_pair(qp, roi_qp)
                ok, note = redistribution_verdict(baseline_arm, roi_arm_result)
                print(f"--- baseline qp={qp}  roi qp={roi_qp}  ---")
                _print_arms({"baseline": baseline_arm, "roi": roi_arm_result})
                print(note)
                print()
                points.append(
                    {
                        "baseline_qp": qp,
                        "roi_qp": roi_qp,
                        "baseline": {
                            "bytes": baseline_arm.bytes_,
                            "psnr_inside": baseline_arm.psnr_inside,
                            "psnr_outside": baseline_arm.psnr_outside,
                            "psnr_overall": baseline_arm.psnr_overall,
                        },
                        "roi": {
                            "bytes": roi_arm_result.bytes_,
                            "psnr_inside": roi_arm_result.psnr_inside,
                            "psnr_outside": roi_arm_result.psnr_outside,
                            "psnr_overall": roi_arm_result.psnr_overall,
                        },
                        "redistributes": ok,
                        "note": note,
                    }
                )
                any_redistribute = any_redistribute or ok
                any_wrong = any_wrong or (not ok and "UNEXPECTED" in note)
            works = any_redistribute and not any_wrong
            explanation = (
                "REDISTRIBUTES on at least one ladder point at matched bitrate."
                if works
                else "No ladder point showed quality redistribution at matched bitrate."
            )
            print(explanation)
            report = {
                "codec": args.codec,
                "encoder_path": encoder.path,
                "encoder_version": encoder.version,
                "ffmpeg_path": ffmpeg.path,
                "ffmpeg_version": ffmpeg.version,
                "match_bitrate": True,
                "ladder": ladder,
                "points": points,
                "roi_works": works,
                "verdict": explanation,
            }
        else:
            print(f"encode      : qp={args.qp} preset={args.preset} (matched QP, not matched rate)")
            print()
            baseline_arm, roi_arm_result = run_pair(args.qp, args.qp)
            works, explanation = verdict(
                baseline_arm,
                roi_arm_result,
                inside_offset=args.inside_offset,
                outside_offset=args.outside_offset,
            )
            _print_arms({"baseline": baseline_arm, "roi": roi_arm_result})
            print()
            print(explanation)
            report = {
                "codec": args.codec,
                "encoder_path": encoder.path,
                "encoder_version": encoder.version,
                "ffmpeg_path": ffmpeg.path,
                "ffmpeg_version": ffmpeg.version,
                "source": str(args.source),
                "width": source.width,
                "height": source.height,
                "frames": source.frames,
                "qp": args.qp,
                "preset": args.preset,
                "aq_mode": args.aq_mode,
                "inside_offset": args.inside_offset,
                "outside_offset": args.outside_offset,
                "match_bitrate": False,
                "arms": {
                    "baseline": {
                        "bytes": baseline_arm.bytes_,
                        "psnr_inside": baseline_arm.psnr_inside,
                        "psnr_outside": baseline_arm.psnr_outside,
                        "psnr_overall": baseline_arm.psnr_overall,
                    },
                    "roi": {
                        "bytes": roi_arm_result.bytes_,
                        "psnr_inside": roi_arm_result.psnr_inside,
                        "psnr_outside": roi_arm_result.psnr_outside,
                        "psnr_overall": roi_arm_result.psnr_overall,
                    },
                },
                "roi_works": works,
                "verdict": explanation,
            }

        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"\nwrote {args.out}")

    return 0 if works else 1


if __name__ == "__main__":
    raise SystemExit(main())
