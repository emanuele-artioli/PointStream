"""Run a validated EncodeRequest: convert, optional ROI arm, encode, record.

The command builder in ``command.py`` produces argv. This module is the
execution: tool resolution, y4m conversion, native vs pixel-domain ROI, the
run record (path + version), and the matched-bitrate search that every
cross-arm comparison has to go through.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Literal

from src.contracts.codecs import EncodeRequest, RateControl, assert_matched_rate_control, codec
from src.contracts.errors import CodecConstraintError
from src.components.codec import command as command_builder
from src.components.codec import roi as roi_mod
from src.components.codec import tools as tool_mod
from src.components.codec import y4m as y4m_mod
from src.components.codec.tools import ResolvedTool

RoiArm = Literal["native", "pixel", "auto", "none"]

QP_BOUNDS: dict[str, tuple[int, int]] = {
    "avc": (0, 51),
    "hevc": (0, 51),
    "av1": (1, 63),
    "vvc": (0, 63),
}

BITSTREAM_SUFFIX: dict[str, str] = {
    "avc": ".mp4",
    "hevc": ".hevc",
    "av1": ".ivf",
    "vvc": ".vvc",
}


@dataclass(frozen=True)
class EncodeRecord:
    """What actually ran. Path and version are load-bearing, not decorative."""

    codec_name: str
    output: Path
    size_bytes: int
    encode_seconds: float
    tool_path: str
    tool_version: str
    command: tuple[str, ...]
    rate_control: str
    rate: int | None
    preset: str | None
    pix_fmt: str
    roi_arm: str | None
    ffmpeg_path: str
    ffmpeg_version: str


def encode(
    source: Path,
    dest: Path,
    request: EncodeRequest,
    *,
    roi: roi_mod.BlockRoiMap | None = None,
    roi_arm: RoiArm = "auto",
    frames: int | None = None,
    work_dir: Path | None = None,
) -> EncodeRecord:
    """Encode ``source`` to ``dest`` according to ``request``.

    ``roi_arm``:
        * ``native`` — encoder delta-QP map or ffmpeg ``addroi``.
        * ``pixel`` — degrade non-salient blocks, then encode without a map.
          This is the VVC arm, and the AVC arm until ``addroi`` is shown to
          move quality.
        * ``auto`` — native when the codec's ROI is a verified delta-QP map,
          otherwise pixel if a map was supplied.
        * ``none`` — ignore ``roi`` and ``request.roi_map``.
    """
    request.validate()
    arm = resolve_roi_arm(request.codec_name, roi_arm, has_map=roi is not None or request.is_roi_arm)
    if arm == "native" and not codec(request.codec_name).supports_roi:
        raise CodecConstraintError(
            request.codec_name,
            "roi_arm",
            "native",
            ["pixel — this codec has no region map"],
        )
    if arm == "pixel" and request.pix_fmt != "yuv420p":
        raise CodecConstraintError(
            request.codec_name,
            "pix_fmt for pixel ROI",
            request.pix_fmt,
            ["yuv420p"],
        )

    ffmpeg = tool_mod.resolve_ffmpeg()
    encoder = tool_mod.resolve_encoder(request.codec_name)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    own_tmp: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        own_tmp = tempfile.TemporaryDirectory(prefix="ps-codec-")
        work_dir = Path(own_tmp.name)
    try:
        return _encode_in(
            Path(source),
            dest,
            request,
            roi=roi,
            arm=arm,
            frames=frames,
            work_dir=work_dir,
            ffmpeg=ffmpeg,
            encoder=encoder,
        )
    finally:
        if own_tmp is not None:
            own_tmp.cleanup()


def decode(
    bitstream: Path,
    dest: Path,
    request: EncodeRequest,
    *,
    ffmpeg: ResolvedTool | None = None,
) -> None:
    """Decode ``bitstream`` to ``dest`` (typically a y4m) through the same builder."""
    request.validate()
    ffmpeg = ffmpeg or tool_mod.resolve_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = command_builder.build_command(
        "decode",
        request,
        source=bitstream,
        dest=dest,
        encoder=ffmpeg,
        ffmpeg=ffmpeg,
    )
    _run(argv, dest)


def resolve_roi_arm(codec_name: str, requested: RoiArm, *, has_map: bool) -> str | None:
    """Pick native vs pixel given what the codec actually honours."""
    if requested == "none" or not has_map:
        return None
    if requested in {"native", "pixel"}:
        return requested
    caps = codec(codec_name)
    if caps.roi_is_verified:
        return "native"
    return "pixel"


def search_qp(
    encode_at_qp: Callable[[int], int],
    target_bytes: int,
    lo: int,
    hi: int,
    *,
    tolerance: float = 0.05,
) -> tuple[int, int]:
    """Binary-search a QP whose encode lands near ``target_bytes``.

    Higher QP means fewer bytes. Returns ``(qp, size_bytes)`` for the closest
    landing, re-encoding the winner if the last probe was not it.

    This is deliberately a pure search over a byte oracle so a unit test can
    drive it with a fake encoder. Real matched-rate comparisons go through
    ``match_rate``, which calls ``assert_matched_rate_control`` first.
    """
    if target_bytes <= 0:
        raise ValueError(f"target_bytes must be positive, got {target_bytes}.")
    if lo > hi:
        raise ValueError(f"empty QP range [{lo}, {hi}].")

    best_qp = lo
    best_size = encode_at_qp(lo)
    best_diff = abs(best_size - target_bytes)
    last_qp = lo
    low, high = lo, hi
    while low <= high:
        mid = (low + high) // 2
        size = encode_at_qp(mid)
        last_qp = mid
        diff = abs(size - target_bytes)
        if diff < best_diff or (diff == best_diff and mid > best_qp):
            # Prefer the higher QP on a tie — fewer bytes for the same error,
            # which is the direction a rate-matched ROI arm has to go.
            best_qp, best_size, best_diff = mid, size, diff
        if size > target_bytes:
            low = mid + 1
        else:
            high = mid - 1
    if last_qp != best_qp:
        best_size = encode_at_qp(best_qp)
    rel = abs(best_size - target_bytes) / target_bytes
    if rel > tolerance:
        raise RuntimeError(
            f"QP search landed {best_size} bytes at qp={best_qp}, "
            f"target {target_bytes} ({rel:.1%} off, tolerance {tolerance:.0%})."
        )
    return best_qp, best_size


def match_rate(
    source: Path,
    dest_dir: Path,
    baseline: EncodeRequest,
    roi_request: EncodeRequest,
    *,
    roi: roi_mod.BlockRoiMap,
    roi_arm: RoiArm = "auto",
    target_bytes: int | None = None,
    frames: int | None = None,
    tolerance: float = 0.05,
) -> tuple[EncodeRecord, EncodeRecord]:
    """Encode both arms onto the same byte count.

    Matched QP is not matched rate: the ROI arm at the same QP simply spends
    more bytes, and more bits buying more quality is not a result. Call this
    (which calls ``assert_matched_rate_control``) before any cross-arm number.

    If ``target_bytes`` is omitted, the baseline is encoded at ``baseline.rate``
    and the ROI arm's QP is searched to that size.
    """
    assert_matched_rate_control((baseline, roi_request))
    if baseline.rate_control is not RateControl.QP:
        raise CodecConstraintError(
            "comparison",
            "rate_control for matched-rate ROI",
            baseline.rate_control.value,
            ["qp"],
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = BITSTREAM_SUFFIX[baseline.codec_name]
    base_record = encode(
        source,
        dest_dir / f"baseline{suffix}",
        baseline,
        frames=frames,
    )
    target = target_bytes if target_bytes is not None else base_record.size_bytes
    lo, hi = QP_BOUNDS[roi_request.codec_name]

    def _at(qp: int) -> int:
        record = encode(
            source,
            dest_dir / f"roi_qp{qp}{suffix}",
            roi_request.replace_rate(qp),
            roi=roi,
            roi_arm=roi_arm,
            frames=frames,
        )
        return record.size_bytes

    qp, _size = search_qp(_at, target, lo, hi, tolerance=tolerance)
    roi_record = encode(
        source,
        dest_dir / f"roi{suffix}",
        roi_request.replace_rate(qp),
        roi=roi,
        roi_arm=roi_arm,
        frames=frames,
    )
    return base_record, roi_record


def sweep_qp(
    source: Path,
    dest_dir: Path,
    request: EncodeRequest,
    qps: Sequence[int],
    *,
    roi: roi_mod.BlockRoiMap | None = None,
    roi_arm: RoiArm = "auto",
    frames: int | None = None,
) -> list[EncodeRecord]:
    """Encode ``request`` at every QP in ``qps``. Comparisons need a curve."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = BITSTREAM_SUFFIX[request.codec_name]
    records: list[EncodeRecord] = []
    for qp in qps:
        records.append(
            encode(
                source,
                dest_dir / f"{request.codec_name}_qp{qp}{suffix}",
                request.replace_rate(int(qp)),
                roi=roi,
                roi_arm=roi_arm if roi is not None else "none",
                frames=frames,
            )
        )
    return records


def _encode_in(
    source: Path,
    dest: Path,
    request: EncodeRequest,
    *,
    roi: roi_mod.BlockRoiMap | None,
    arm: str | None,
    frames: int | None,
    work_dir: Path,
    ffmpeg: ResolvedTool,
    encoder: ResolvedTool,
) -> EncodeRecord:
    y4m_source = _ensure_y4m(source, work_dir / "source.y4m", request, ffmpeg, frames=frames)
    info = y4m_mod.read(y4m_source)
    n_frames = info.frames
    encode_input = y4m_source
    roi_file: Path | None = None
    addroi: str | None = None
    encode_request = request

    if arm == "pixel":
        if roi is None:
            raise ValueError("pixel ROI arm needs a BlockRoiMap")
        degraded = y4m_mod.Y4M(
            width=info.width,
            height=info.height,
            fps=info.fps,
            luma=roi_mod.degrade_video(info.luma, roi),
            chroma=info.chroma,
        )
        encode_input = work_dir / "degraded.y4m"
        y4m_mod.write(encode_input, degraded)
        encode_request = request.without_roi()
    elif arm == "native":
        if roi is None and request.roi_map:
            roi_file = Path(request.roi_map)
        elif roi is not None:
            if request.codec_name == "av1":
                roi_file = work_dir / "roi_av1.txt"
                roi_mod.write_svtav1(roi, roi_file, n_frames)
                encode_request = _with_roi_path(request, roi_file)
            elif request.codec_name == "hevc":
                roi_file = work_dir / "roi_kvazaar.bin"
                roi_mod.write_kvazaar(roi, roi_file, n_frames)
                encode_request = _with_roi_path(request, roi_file)
            elif request.codec_name == "avc":
                addroi = roi_mod.addroi_filter(roi, info.width, info.height) or None
                # addroi is a filter, not a map file. Keep request.roi_map as
                # given so validate() still sees an ROI arm under QP.
            else:
                raise CodecConstraintError(
                    request.codec_name,
                    "native roi",
                    request.codec_name,
                    ["av1", "hevc", "avc"],
                )
        if request.codec_name == "av1" and roi_file is not None and not encoder.has("roi-map-file"):
            raise CodecConstraintError(
                "av1",
                "roi-map-file",
                f"{encoder.path} ({encoder.version})",
                ["an SvtAv1EncApp build that lists --roi-map-file in --help"],
            )

    if request.codec_name == "hevc":
        # Kvazaar's y4m reader skips FRAME headers; raw yuv is the reliable input.
        raw = work_dir / "input.yuv"
        _y4m_to_raw(encode_input, raw, ffmpeg, pix_fmt=request.pix_fmt)
        encode_input = raw

    argv = command_builder.build_command(
        "encode",
        encode_request,
        source=encode_input,
        dest=dest,
        encoder=encoder,
        ffmpeg=ffmpeg,
        roi_file=roi_file,
        addroi=addroi,
        width=info.width,
        height=info.height,
        fps=info.fps,
    )
    started = time.perf_counter()
    _run(argv, dest)
    elapsed = time.perf_counter() - started
    return EncodeRecord(
        codec_name=request.codec_name,
        output=dest,
        size_bytes=dest.stat().st_size,
        encode_seconds=elapsed,
        tool_path=encoder.path,
        tool_version=encoder.version,
        command=tuple(argv),
        rate_control=request.rate_control.value,
        rate=encode_request.rate,
        preset=request.preset,
        pix_fmt=request.pix_fmt,
        roi_arm=arm,
        ffmpeg_path=ffmpeg.path,
        ffmpeg_version=ffmpeg.version,
    )


def _with_roi_path(request: EncodeRequest, path: Path) -> EncodeRequest:
    return EncodeRequest(
        codec_name=request.codec_name,
        rate_control=request.rate_control,
        rate=request.rate,
        preset=request.preset,
        pix_fmt=request.pix_fmt,
        roi_map=str(path),
        extra_args=request.extra_args,
    )


def _ensure_y4m(
    source: Path,
    dest: Path,
    request: EncodeRequest,
    ffmpeg: ResolvedTool,
    *,
    frames: int | None,
) -> Path:
    if source.suffix.lower() in {".y4m", ".yuv4mpeg"}:
        if frames is None:
            shutil.copyfile(source, dest)
            return dest
        # Still re-wrap if a frame cap was requested.
    argv = [
        ffmpeg.path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
    ]
    if frames is not None:
        argv += ["-frames:v", str(frames)]
    argv += ["-pix_fmt", request.pix_fmt, str(dest)]
    _run(argv, dest)
    return dest


def _y4m_to_raw(source: Path, dest: Path, ffmpeg: ResolvedTool, *, pix_fmt: str) -> None:
    argv = [
        ffmpeg.path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-pix_fmt",
        pix_fmt,
        "-f",
        "rawvideo",
        str(dest),
    ]
    _run(argv, dest)


def _run(
    argv: list[str], dest: Path, *, attempts: int = 3
) -> subprocess.CompletedProcess[str]:
    """Run ``argv``. Judge the file: Kvazaar can crash after a valid write,
    and libvvenc can exit 0 after a 0-byte 4K QP-48 bitstream."""
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        last = subprocess.run(argv, capture_output=True, text=True, check=False)
        if dest.exists() and dest.stat().st_size > 0:
            if attempt > 1:
                print(f"encode retry {attempt}/{attempts} wrote {dest}", flush=True)
            return last
        print(
            f"encode attempt {attempt}/{attempts} left empty {dest} "
            f"(exit {last.returncode})",
            flush=True,
        )
    detail = ((last.stderr or last.stdout or "") if last else "").strip()[-2000:]
    code = last.returncode if last is not None else "n/a"
    raise RuntimeError(f"command failed ({code}): {' '.join(argv)}\n{detail}")
