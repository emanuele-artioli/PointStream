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

**It does not establish that region control is worth anything**, because both
arms were encoded at matched **QP**, and the region arm therefore also spent more
bytes (31374 against 31374 -> 32341). More bits buying more quality is not a
result. The claim that matters is quality *redistribution at equal bitrate*:
binary-search the base QP until both arms land on the same byte count, then ask
whether the region gained at the background's expense. That is unimplemented —
see `--match-bitrate`, which is the next thing this script needs.

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
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

BLOCK = 64
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


def extract_source(source: Path, target: Path, *, width: int, height: int, frames: int) -> None:
    run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-frames:v", str(frames),
            "-vf", f"scale={width}:{height}",
            "-pix_fmt", "yuv420p",
            str(target),
        ]
    )


def encoder_version(encoder: str) -> str:
    """The encoder's self-reported version.

    Recorded in the report because it decides the answer. `--roi-map-file` only
    exists from SVT-AV1 1.8; an older build rejects the flag outright, which
    reads as "region control does not work" for a reason unrelated to region
    control. This host briefly carried two builds, the older one shadowing the
    newer inside the conda environment.
    """
    result = subprocess.run([encoder, "--version"], capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).splitlines()[0].strip()


def encode_av1(
    source: Path,
    bitstream: Path,
    *,
    encoder: str,
    qp: int,
    preset: int,
    roi_map: Path | None,
    aq_mode: int | None,
) -> None:
    command = [
        encoder,
        "-i", str(source),
        "-b", str(bitstream),
        "--rc", "0",           # constant QP; a delta map needs a base that stays put
        "--qp", str(qp),
        "--preset", str(preset),
        "--keyint", "-1",
    ]
    if aq_mode is not None:
        command += ["--aq-mode", str(aq_mode)]
    if roi_map is not None:
        command += ["--roi-map-file", str(roi_map)]
    run(command)


def decode(bitstream: Path, target: Path) -> None:
    run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(bitstream),
            "-pix_fmt", "yuv420p",
            str(target),
        ]
    )


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--codec", default="av1", choices=["av1"], help="Which ROI surface to test.")
    parser.add_argument("--source", type=Path, default=Path("assets/real_tennis.mp4"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384, help="A multiple of 64 keeps blocks whole.")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--qp", type=int, default=45, help="High enough that QP offsets have room to act.")
    parser.add_argument("--preset", type=int, default=8)
    parser.add_argument("--inside-offset", type=int, default=-30, help="Negative means better quality. q_index units, ~4 per QP step.")
    parser.add_argument(
        "--outside-offset",
        type=int,
        default=0,
        help="Zero is the sharper test: an unmapped region that does not move proves localization.",
    )
    parser.add_argument("--aq-mode", type=int, default=None, help="SVT-AV1 docs tie ROI to aq-mode 1.")
    parser.add_argument(
        "--encoder-bin",
        default="SvtAv1EncApp",
        help="Encoder to drive. Its version is recorded in the report.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Where to write report.json.")
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 2
    if shutil.which(args.encoder_bin) is None and not Path(args.encoder_bin).exists():
        print(f"encoder not found: {args.encoder_bin}", file=sys.stderr)
        return 2
    version = encoder_version(args.encoder_bin)
    if not args.source.exists():
        print(f"source not found: {args.source}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="roi-verify-") as raw_tmp:
        tmp = Path(raw_tmp)
        source_y4m = tmp / "source.y4m"
        extract_source(args.source, source_y4m, width=args.width, height=args.height, frames=args.frames)
        source = read_y4m_luma(source_y4m)

        region = RegionMap.centred(
            source.width,
            source.height,
            inside_offset=args.inside_offset,
            outside_offset=args.outside_offset,
        )
        map_file = tmp / "roi.txt"
        region.write(map_file, source.frames)
        mask = region.pixel_mask(source.width, source.height)

        arms: dict[str, ArmResult] = {}
        for label, roi_map in (("baseline", None), ("roi", map_file)):
            bitstream = tmp / f"{label}.obu"
            decoded = tmp / f"{label}.y4m"
            encode_av1(
                source_y4m,
                bitstream,
                encoder=args.encoder_bin,
                qp=args.qp,
                preset=args.preset,
                roi_map=roi_map,
                aq_mode=args.aq_mode,
            )
            decode(bitstream, decoded)
            arms[label] = measure(label, source, decoded, bitstream, mask)

        works, explanation = verdict(
            arms["baseline"],
            arms["roi"],
            inside_offset=args.inside_offset,
            outside_offset=args.outside_offset,
        )

        print(f"source      : {args.source} @ {source.width}x{source.height}, {source.frames} frames")
        print(
            f"region      : blocks {region.blocks_wide}x{region.blocks_high}, "
            f"cols {region.col0}-{region.col1}, rows {region.row0}-{region.row1}, "
            f"offsets {args.inside_offset} inside / {args.outside_offset} outside"
        )
        print(f"encoder     : {args.encoder_bin} ({version})")
        print(f"encode      : qp={args.qp} preset={args.preset} aq-mode={args.aq_mode}")
        print()
        print(f"{'arm':<10} {'bytes':>9}  {'PSNR in':>8}  {'PSNR out':>8}  {'PSNR all':>8}")
        for arm in arms.values():
            print(
                f"{arm.label:<10} {arm.bytes_:>9}  {arm.psnr_inside:>8.2f}  "
                f"{arm.psnr_outside:>8.2f}  {arm.psnr_overall:>8.2f}"
            )
        print()
        print(explanation)

        report = {
            "codec": args.codec,
            "encoder_bin": args.encoder_bin,
            "encoder_version": version,
            "source": str(args.source),
            "width": source.width,
            "height": source.height,
            "frames": source.frames,
            "qp": args.qp,
            "preset": args.preset,
            "aq_mode": args.aq_mode,
            "inside_offset": args.inside_offset,
            "outside_offset": args.outside_offset,
            "arms": {
                label: {
                    "bytes": arm.bytes_,
                    "psnr_inside": arm.psnr_inside,
                    "psnr_outside": arm.psnr_outside,
                    "psnr_overall": arm.psnr_overall,
                }
                for label, arm in arms.items()
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
