"""BP29 §1.1 — what the plate's codec knob costs, and what it buys.

`background.codec` accepts `{jpeg, png, roi-video}`. Two of those three have
never been driven: the axis reached nothing at all until BP24 wired
`make_background` (`plans/BP24-findings.md` §6), and every sweep since has moved
`background.jpeg_quality`, which is a knob on the codec that was already chosen.
A config axis only ever set to one value is indistinguishable from a constant.

This module asks two questions and keeps them separate, because they have
different failure modes:

**1. What does each codec cost on the plate, at matched fidelity?**
`sidecar_sweep()` drives `build_sidecar` directly on the exact array the runner
transmits (`source[0]`), over each codec's own quality knob, and reports bytes
and plate PSNR per rung. Matched fidelity is then read by interpolating each
curve at a target PSNR — matched *knob* is meaningless across a JPEG quality, a
PNG compression level and an x264 CRF.

**2. Does the knob reach the encoder, and does the plate reach the payload?**
Two checks, neither of which a byte count alone can pass:

* every payload is identified from its own bytes (JPEG SOI, PNG signature, MP4
  `ftyp`) and, for the video route, by `ffprobe` reporting `codec_name`. A codec
  that silently falls back to another one is exactly the failure this stream
  exists to catch, and it would otherwise read as a plausible number;
* the end-to-end arm's `sizes.panorama` is compared against the sidecar byte
  count measured here for the same settings. Equality is the proof that the
  plate the runner sent is the plate this module encoded.

**What this deliberately does not do.** No BD-rate, no paired ladder. Four
streams are moving the plate at once and four ladders against four half-finished
levers is four numbers nobody can combine. The question here is what the plate
costs and what it buys.

Bounds were written before the first encode:
`outputs/bp29-plate-codec/bounds-before-run.json`. The alarms in
`check_bounds()` are those bounds evaluated in the run, rather than left in a
JSON file beside the result where they get skipped exactly when the number is
exciting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.tier.clip import TierClip, load_tier_clip
from src.components.background.sidecar import (
    FFMPEG_BIN,
    SIDECAR_JPEG,
    SIDECAR_PNG,
    SIDECAR_ROI_VIDEO,
    build_sidecar,
)
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp29-plate-codec"

BOUNDS_FILE = "outputs/bp29-plate-codec/bounds-before-run.json"

#: Resolved by path and version, not by name — this host has carried two ffmpeg
#: builds with different capabilities. `ffprobe` sits beside the `ffmpeg` the
#: sidecar itself resolves, so the identification uses the same build that did
#: the encoding.
FFPROBE_BIN = FFMPEG_BIN.with_name("ffprobe")

#: Each codec's own quality knob. Not comparable across routes — the comparison
#: is made at matched fidelity, never at matched knob.
JPEG_QUALITIES: tuple[int, ...] = (10, 30, 50, 75, 90, 95)
PNG_COMPRESSIONS: tuple[int, ...] = (0, 3, 6, 9)
ROI_CRFS: tuple[int, ...] = (12, 18, 23, 28, 30, 35, 40)

#: The CRF `roi-video` gets on the runner path. `BackgroundConfig` carries
#: `codec` and `jpeg_quality` and nothing else, and `strategy.bind` forwards
#: only those two, so `RoiVideoSidecar`'s constructor defaults stand: crf 30,
#: preset veryfast. Selecting `roi-video` in a config therefore selects one
#: fixed operating point. That is a finding about the axis, not a knob this
#: module may set behind the config's back — so the end-to-end arm runs at 30
#: and the matched-fidelity answer comes from the sidecar sweep.
RUNNER_ROI_CRF = 30
RUNNER_ROI_PRESET = "veryfast"
RUNNER_PNG_COMPRESSION = 3

#: Fidelity targets to read every curve at, in dB. 42.8 is the jpeg75 operating
#: point the BP24 ladder's reference rung sits on; 38 and 40 are where a plate
#: would rather operate, and `plans/BP24-findings.md` §16 measured the intra
#: routes' advantage widening as the target gets cheaper. 35 is where `roi-video`
#: actually lands on the runner path, because config pins it to crf 30 — the
#: comparison has to include the operating point the system really uses, not only
#: the ones that read well.
FIDELITY_TARGETS: tuple[float, ...] = (35.0, 38.0, 40.0, 42.8)

#: What the payload's first bytes must look like, per route. A byte count cannot
#: tell a JPEG from an MP4; this can.
_MAGIC: tuple[tuple[str, int, bytes], ...] = (
    ("jpeg", 0, b"\xff\xd8\xff"),
    ("png", 0, b"\x89PNG\r\n\x1a\n"),
    ("mp4", 4, b"ftyp"),
)


def pooled_psnr(reference: np.ndarray, candidate: np.ndarray, *, luma: bool = False) -> float:
    """One PSNR over the whole image's MSE — the ladder's convention, imported.

    Deliberately not a second implementation: BP23 found two PSNR conventions
    inside one ladder disagreeing by 0.65 dB. This is a thin re-export so a
    reader of this file can see which convention is in force.
    """
    from experiments.tier.ladder import pooled_psnr as ladder_pooled_psnr

    return float(ladder_pooled_psnr(reference, candidate, luma=luma))


def container_kind(payload: bytes) -> str:
    """What the payload actually is, read off its own first bytes."""
    for name, offset, signature in _MAGIC:
        if payload[offset : offset + len(signature)] == signature:
            return name
    return "unknown"


def probe_video(payload: bytes) -> dict[str, Any]:
    """`ffprobe` on the payload: which encoder actually produced it.

    The container says MP4; only the stream says libx264 rather than something
    ffmpeg fell back to. `plans/BP24-findings.md` §14 is the case where an
    unnamed codec silently became a second encoder and capped every quality it
    touched.
    """
    if not FFPROBE_BIN.is_file():
        return {"probed": False, "reason": f"no ffprobe at {FFPROBE_BIN}"}
    with tempfile.TemporaryDirectory(prefix="ps_bp29_probe_") as tmp_dir:
        target = Path(tmp_dir) / "payload.mp4"
        target.write_bytes(payload)
        completed = subprocess.run(
            [
                str(FFPROBE_BIN),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,profile,width,height,pix_fmt,nb_frames",
                "-of",
                "json",
                str(target),
            ],
            capture_output=True,
            text=True,
        )
    if completed.returncode != 0:
        return {"probed": False, "reason": completed.stderr.strip()[:400]}
    streams = json.loads(completed.stdout or "{}").get("streams") or [{}]
    return {"probed": True, **{str(k): v for k, v in streams[0].items()}}


@dataclass(frozen=True)
class SidecarRung:
    """One codec at one setting on one plate."""

    codec: str
    knob: str
    codec_id: str
    payload_bytes: int
    psnr_rgb_db: float
    psnr_y_db: float
    bit_identical: bool
    container: str
    seconds: float
    detail: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "knob": self.knob,
            "codec_id": self.codec_id,
            "bytes": self.payload_bytes,
            "psnr_rgb_dB": self.psnr_rgb_db,
            "psnr_y_dB": self.psnr_y_db,
            "bit_identical": self.bit_identical,
            "container": self.container,
            "seconds": round(self.seconds, 2),
            **self.detail,
        }


def sidecar_rung(plate: np.ndarray, codec: str, knob: str, **kwargs: Any) -> SidecarRung:
    """Encode the plate once through one sidecar, and score what came back.

    The plate is handed over exactly as `make_background` hands it over — an RGB
    array into a sidecar whose docstring says BGR. That swap costs a little in
    the chroma planes and is why the runner's jpeg75 plate is 463,334 B where
    `plate_probe`'s channel-corrected one is 461,771 B. Reproducing the runner's
    number is the point here, so the swap is kept rather than quietly fixed.
    """
    sidecar = build_sidecar(codec, **kwargs)
    started = time.time()
    payload = sidecar.encode(plate)
    decoded = sidecar.decode(payload)
    seconds = time.time() - started
    reference = np.asarray(plate)
    got = np.asarray(decoded)
    # roi-video crops an odd dimension to even before encoding; score on the
    # region that survives rather than failing on a shape mismatch.
    height = min(int(reference.shape[0]), int(got.shape[0]))
    width = min(int(reference.shape[1]), int(got.shape[1]))
    cropped_ref = reference[:height, :width]
    cropped_got = got[:height, :width]
    detail: dict[str, Any] = {
        "decoded_shape": [int(got.shape[0]), int(got.shape[1])],
        "plate_shape": [int(reference.shape[0]), int(reference.shape[1])],
    }
    if codec == SIDECAR_ROI_VIDEO:
        detail["ffprobe"] = probe_video(payload)
    return SidecarRung(
        codec=codec,
        knob=knob,
        codec_id=sidecar.codec_id,
        payload_bytes=len(payload),
        psnr_rgb_db=pooled_psnr(cropped_ref, cropped_got),
        psnr_y_db=pooled_psnr(cropped_ref, cropped_got, luma=True),
        bit_identical=bool(np.array_equal(cropped_ref, cropped_got)),
        container=container_kind(payload),
        seconds=seconds,
        detail=detail,
    )


def sidecar_sweep(plate: np.ndarray) -> list[SidecarRung]:
    """Every codec over its own knob, on one plate. Prints each rung as it lands."""
    rungs: list[SidecarRung] = []
    plans: list[tuple[str, str, dict[str, Any]]] = []
    plans += [(SIDECAR_JPEG, f"q{q}", {"jpeg_quality": q}) for q in JPEG_QUALITIES]
    plans += [(SIDECAR_PNG, f"z{z}", {"png_compression": z}) for z in PNG_COMPRESSIONS]
    plans += [
        (SIDECAR_ROI_VIDEO, f"crf{c}", {"roi_crf": c, "roi_preset": RUNNER_ROI_PRESET})
        for c in ROI_CRFS
    ]
    for codec, knob, kwargs in plans:
        rung = sidecar_rung(plate, codec, knob, **kwargs)
        rungs.append(rung)
        print(
            f"  {rung.codec:<10} {rung.knob:>6}  {rung.payload_bytes:>10} B  "
            f"rgb {rung.psnr_rgb_db:6.2f} dB  Y {rung.psnr_y_db:6.2f} dB  "
            f"[{rung.container}] {rung.seconds:5.1f}s",
            flush=True,
        )
    return rungs


def bytes_at_fidelity(
    rungs: Sequence[SidecarRung], target_db: float, *, axis: str = "rgb"
) -> dict[str, Any]:
    """Bytes this codec needs to hit `target_db`, by interpolation on its curve.

    Interpolates in (PSNR, log10 bytes), which is the shape a rate-distortion
    curve actually has. **Never extrapolates**: a target outside the measured
    range comes back as `None` with the range that was measured, because a
    matched-fidelity claim built on an extrapolation is a claim about a fit.
    """
    points = [
        (rung.psnr_rgb_db if axis == "rgb" else rung.psnr_y_db, float(rung.payload_bytes))
        for rung in rungs
        if np.isfinite(rung.psnr_rgb_db if axis == "rgb" else rung.psnr_y_db)
    ]
    points.sort()
    if len(points) < 2:
        return {"bytes": None, "reason": "fewer than two finite-PSNR rungs"}
    lowest, highest = points[0][0], points[-1][0]
    if not lowest <= target_db <= highest:
        return {
            "bytes": None,
            "reason": f"target {target_db} dB outside measured {lowest:.2f}-{highest:.2f} dB",
            "measured_range_dB": [lowest, highest],
        }
    qualities = np.array([item[0] for item in points], dtype=np.float64)
    log_bytes = np.log10(np.array([item[1] for item in points], dtype=np.float64))
    interpolated = float(np.interp(target_db, qualities, log_bytes))
    return {
        "bytes": int(round(10.0**interpolated)),
        "interpolated": True,
        "measured_range_dB": [lowest, highest],
    }


def matched_fidelity_table(rungs: Sequence[SidecarRung], *, axis: str = "rgb") -> dict[str, Any]:
    """Every codec read at the same fidelity, with the ratio against JPEG."""
    by_codec: dict[str, list[SidecarRung]] = {}
    for rung in rungs:
        by_codec.setdefault(rung.codec, []).append(rung)
    table: dict[str, Any] = {"axis": f"{axis}-PSNR", "targets": {}}
    for target in FIDELITY_TARGETS:
        row: dict[str, Any] = {}
        for codec, codec_rungs in by_codec.items():
            row[codec] = bytes_at_fidelity(codec_rungs, target, axis=axis)
        jpeg_bytes = row.get(SIDECAR_JPEG, {}).get("bytes")
        for codec, entry in row.items():
            if jpeg_bytes and entry.get("bytes"):
                entry["times_smaller_than_jpeg"] = round(jpeg_bytes / entry["bytes"], 3)
        table["targets"][f"{target:.1f} dB"] = row
    return table


def check_bounds(rungs: Sequence[SidecarRung], arms: Sequence[dict[str, Any]]) -> list[str]:
    """The pre-run bounds, evaluated here rather than by whoever reads the table."""
    alarms: list[str] = []
    by_codec: dict[str, list[SidecarRung]] = {}
    for rung in rungs:
        by_codec.setdefault(rung.codec, []).append(rung)

    expected_container = {SIDECAR_JPEG: "jpeg", SIDECAR_PNG: "png", SIDECAR_ROI_VIDEO: "mp4"}
    for rung in rungs:
        want = expected_container[rung.codec]
        if rung.container != want:
            alarms.append(
                f"{rung.codec} {rung.knob}: payload is a {rung.container}, not a {want}. "
                "The codec named in the config is not the codec that ran."
            )
        if rung.codec == SIDECAR_ROI_VIDEO:
            probe = rung.detail.get("ffprobe", {})
            if probe.get("probed") and probe.get("codec_name") != "h264":
                alarms.append(
                    f"roi-video {rung.knob}: ffprobe reports codec_name="
                    f"{probe.get('codec_name')!r}, not h264. A silent fallback caps "
                    "every quality it returns (plans/BP24-findings.md §14)."
                )

    for rung in by_codec.get(SIDECAR_PNG, []):
        if not rung.bit_identical:
            alarms.append(
                f"png {rung.knob} is not bit-identical to the plate. PNG is lossless; "
                "if it is not, it is not running."
            )
        if rung.payload_bytes < 1_000_000:
            alarms.append(
                f"png {rung.knob} coded {rung.payload_bytes} B. A lossless 4K plate "
                "cannot be that small — png is not running."
            )
    jpeg_max = max((r.payload_bytes for r in by_codec.get(SIDECAR_JPEG, [])), default=0)
    png_min = min((r.payload_bytes for r in by_codec.get(SIDECAR_PNG, [])), default=0)
    if jpeg_max and png_min and png_min <= jpeg_max:
        alarms.append(
            f"png's cheapest rung ({png_min} B) is not larger than jpeg's dearest "
            f"({jpeg_max} B). Lossless must cost more; it does not, so one of them "
            "is not running."
        )

    for codec, knob_order in (
        (SIDECAR_JPEG, sorted(by_codec.get(SIDECAR_JPEG, []), key=lambda r: r.psnr_rgb_db)),
        (
            SIDECAR_ROI_VIDEO,
            sorted(by_codec.get(SIDECAR_ROI_VIDEO, []), key=lambda r: r.psnr_rgb_db),
        ),
    ):
        for previous, current in zip(knob_order, knob_order[1:]):
            if current.payload_bytes <= previous.payload_bytes:
                alarms.append(
                    f"{codec}: {current.knob} scored higher than {previous.knob} but cost "
                    f"{current.payload_bytes} B against {previous.payload_bytes} B. Better "
                    "quality must cost more; if it does not, the knob is not reaching "
                    "the encoder."
                )

    seen: dict[int, SidecarRung] = {}
    for rung in rungs:
        twin = seen.get(rung.payload_bytes)
        if twin is not None and twin.codec != rung.codec:
            alarms.append(
                f"{rung.codec} {rung.knob} and {twin.codec} {twin.knob} returned the "
                f"same {rung.payload_bytes} B. Two codecs do not agree to the byte; "
                "suspect one encoder being measured twice."
            )
        seen[rung.payload_bytes] = rung

    for arm in arms:
        if arm.get("error"):
            continue
        plate_bytes = arm.get("plate_bytes")
        sidecar_bytes = arm.get("sidecar_bytes_for_same_settings")
        if plate_bytes is not None and sidecar_bytes is not None and plate_bytes != sidecar_bytes:
            alarms.append(
                f"{arm['arm']}: the runner sent {plate_bytes} B of plate where this "
                f"module's sidecar produced {sidecar_bytes} B for the same settings. "
                "The plate the runner sent is not the plate measured here."
            )
        if arm.get("arm") == "jpeg:75" and plate_bytes not in (None, 463334):
            alarms.append(
                f"jpeg:75 sent {plate_bytes} B of plate where the BP24 ladder's "
                "reference rung sent 463,334 B. The harness does not reproduce the "
                "run every other number here is compared against."
            )
        if not arm.get("is_rate", True):
            alarms.append(f"{arm['arm']}: the ledger withheld its ratio, so that total is not a rate.")
    return alarms


def end_to_end_arm(
    clip: TierClip,
    *,
    label: str,
    codec: str,
    jpeg_quality: int,
    sidecar_bytes: int | None,
    tier: str = "balanced",
    residual_rate: int = 38,
) -> dict[str, Any]:
    """One full run with `background.codec` set, everything else held fixed.

    The residual is pinned to the BP24 ladder's reference rung — av1, QP 38, the
    preset `src.components.codec.measure.PRESETS` gives it — so the only thing
    moving between arms is the plate's codec. Quality is scored on
    `delivered_frames`, never on `RunResult.frames`: the latter is the
    reconstruction *before* `residual.codec` ran, and pairing it with a byte
    count puts the rate and the quality at different operating points
    (`plans/BP24-findings.md` §8).
    """
    from dataclasses import replace as dc_replace

    from experiments.tier.run import run_config
    from src.components.codec.measure import PRESETS
    from src.contracts.codecs import RateControl
    from src.runner.config_io import load_tier

    base = load_tier(tier)
    config = base.with_(
        background=dc_replace(base.background, codec=codec, jpeg_quality=int(jpeg_quality)),
        residual=dc_replace(
            base.residual,
            codec="av1",
            preset=PRESETS["av1"],
            rate_control=RateControl.QP,
            rate=int(residual_rate),
        ),
    )
    started = time.time()
    try:
        outcome = run_config(f"bp29/{label}", config, clip)
    except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
        print(f"  {label:<16} FAILED {exc!r}", flush=True)
        return {"arm": label, "background_codec": codec, "error": repr(exc)}
    seconds = time.time() - started
    sizes = outcome.result.sizes
    delivered = outcome.result.delivered_frames
    record = {
        "arm": label,
        "background_codec": codec,
        "background_jpeg_quality": int(jpeg_quality),
        "residual": {
            "codec": config.residual.codec,
            "preset": config.residual.preset,
            "rate_control": config.residual.rate_control.value,
            "rate": config.residual.rate,
        },
        "plate_bytes": int(sizes.panorama),
        "sidecar_bytes_for_same_settings": sidecar_bytes,
        "total_bytes": int(sizes.transport_total),
        "parts": {
            "residual": int(sizes.residual),
            "panorama": int(sizes.panorama),
            "actor_reference": int(sizes.actor_reference),
            "metadata": int(sizes.metadata),
        },
        "plate_share_of_payload": (
            round(sizes.panorama / sizes.transport_total, 4) if sizes.transport_total else None
        ),
        "delivered_psnr_y_dB": pooled_psnr(clip.frames, delivered, luma=True),
        "delivered_psnr_rgb_dB": pooled_psnr(clip.frames, delivered),
        "is_rate": bool(sizes.is_rate),
        "raw_parts": list(sizes.raw_parts),
        "seconds": round(seconds, 1),
    }
    print(
        f"  {label:<16} plate {record['plate_bytes']:>10} B  total "
        f"{record['total_bytes']:>10} B  Y {record['delivered_psnr_y_dB']:6.2f} dB  "
        f"{seconds:6.1f}s",
        flush=True,
    )
    return record


def match_jpeg_quality(
    plate: np.ndarray, target_db: float, *, axis: str = "rgb"
) -> dict[str, Any]:
    """The JPEG quality whose plate PSNR sits closest to `target_db`.

    A bisection over the quality scale, so the end-to-end arms can be compared
    at matched plate fidelity rather than at matched knob. Each probe is a real
    encode; the ones it runs are recorded so the search is auditable.
    """
    low, high = 1, 100
    probes: list[dict[str, Any]] = []
    best: tuple[float, int, SidecarRung] | None = None
    while low <= high:
        quality = (low + high) // 2
        rung = sidecar_rung(plate, SIDECAR_JPEG, f"q{quality}", jpeg_quality=quality)
        score = rung.psnr_rgb_db if axis == "rgb" else rung.psnr_y_db
        probes.append({"quality": quality, "psnr_dB": score, "bytes": rung.payload_bytes})
        distance = abs(score - target_db)
        if best is None or distance < best[0]:
            best = (distance, quality, rung)
        if score < target_db:
            low = quality + 1
        else:
            high = quality - 1
    assert best is not None
    _distance, quality, rung = best
    return {
        "target_dB": target_db,
        "axis": f"{axis}-PSNR",
        "quality": quality,
        "psnr_rgb_dB": rung.psnr_rgb_db,
        "psnr_y_dB": rung.psnr_y_db,
        "bytes": rung.payload_bytes,
        "probes": probes,
    }


def png_diagnostics(plate: np.ndarray, clip: TierClip) -> dict[str, Any]:
    """Why a lossless 4K plate came in at 3.3 MB when the bound said 8-24.9 MB.

    The pre-run bound gave png [8,000,000, 24,883,200] B on the basis that "PNG
    on natural broadcast content typically gives 1.2-2.0x". The measurement came
    in at 3,272,798 B — 7.6x — which is outside the bound and therefore an alarm
    to investigate rather than a number to report.

    Four checks, three of which have a known answer:

    * **an independent PNG encoder** on the same pixels (ffmpeg's `png`, not
      OpenCV's). Two libraries agreeing rules out one of them being lazy;
    * **the file the frame was extracted into**, written months earlier by a
      third path. The same order of magnitude means the plate really is this
      compressible;
    * **smoothness statistics**, which say *why*: a decoded video frame has been
      through a transform codec already, so its high frequencies are gone;
    * **a noise control**: add +-2 grey levels of uniform noise and re-encode.
      Noise is incompressible, so if PNG is working the size must jump toward
      raw. If it does not, the encoder is not doing what it claims.
    """
    from experiments.tier.clip import BP21_CLIPS

    height, width = int(plate.shape[0]), int(plate.shape[1])
    raw_bytes = int(plate.size)

    ffmpeg_bytes: int | None = None
    ffmpeg_error: str | None = None
    with tempfile.TemporaryDirectory(prefix="ps_bp29_png_") as tmp_dir:
        target = Path(tmp_dir) / "independent.png"
        completed = subprocess.run(
            [
                str(FFMPEG_BIN),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-i",
                "-",
                "-frames:v",
                "1",
                # Name the codec. An unnamed one is how a decode became a
                # re-encode in `plans/BP24-findings.md` §14.
                "-c:v",
                "png",
                "-compression_level",
                "3",
                "-y",
                str(target),
            ],
            input=np.ascontiguousarray(plate).tobytes(),
            capture_output=True,
        )
        if completed.returncode == 0 and target.is_file():
            ffmpeg_bytes = target.stat().st_size
        else:
            ffmpeg_error = completed.stderr.decode("utf-8", "replace")[:400]

    window = BP21_CLIPS / clip.video / clip.scene / "window"
    source_png = window / f"frame_{clip.frame_ids[0]:06d}.png"
    source_bytes = source_png.stat().st_size if source_png.is_file() else None

    grey = np.asarray(plate, dtype=np.int16).mean(axis=2)
    horizontal = np.abs(np.diff(grey, axis=1))
    rng = np.random.default_rng(1337)
    noisy = np.clip(
        np.asarray(plate, dtype=np.int16) + rng.integers(-2, 3, plate.shape), 0, 255
    ).astype(np.uint8)
    noisy_rung = sidecar_rung(noisy, SIDECAR_PNG, "z3+noise", png_compression=3)

    return {
        "question": "png measured 3,272,798 B against a pre-run bound of 8,000,000-24,883,200 B",
        "raw_bytes": raw_bytes,
        "independent_encoder": {
            "tool": f"{FFMPEG_BIN} -c:v png -compression_level 3",
            "bytes": ffmpeg_bytes,
            "error": ffmpeg_error,
        },
        "frame_as_extracted_months_earlier": {
            "path": str(source_png),
            "bytes": source_bytes,
        },
        "smoothness": {
            "mean_abs_horizontal_gradient_grey_levels": float(horizontal.mean()),
            "fraction_of_neighbours_within_1_level": float((horizontal <= 1).mean()),
            "reading": (
                "The plate is a decoded 4K broadcast frame, so it has already been "
                "through a transform codec: its high frequencies were quantised away "
                "before PNG ever saw it. That, not the encoder, is why it compresses "
                "far past what a camera-original photograph would."
            ),
        },
        "noise_control": {
            "what": "the same plate with +-2 grey levels of uniform noise, png z3",
            "bytes": noisy_rung.payload_bytes,
            "bit_identical": noisy_rung.bit_identical,
            "expected": (
                "noise is incompressible, so this must jump toward the raw "
                f"{raw_bytes} B. If it does not, png is not really compressing."
            ),
        },
    }


#: The x264 controls. The shipped sidecar is one point in this grid — crf 30,
#: veryfast, four addroi regions — and the grid says how much of its behaviour
#: is x264 and how much is that configuration.
ROI_CONTROL_GRID: tuple[tuple[int, str, bool], ...] = (
    (0, "veryfast", False),
    (12, "veryfast", False),
    (12, "slow", False),
    (12, "veryfast", True),
    (18, "slow", False),
    (30, "veryfast", False),
    (30, "slow", False),
)


def roi_video_diagnostics(plate: np.ndarray) -> dict[str, Any]:
    """Calibrate the x264 route before believing that it loses to JPEG.

    The shipped `roi-video` sidecar fixes three things at once — libx264,
    `veryfast`, and four `addroi` regions at qoffset -0.4 — and the sweep
    measured all three together. This separates them, and it starts with the
    anchor whose answer is known: **crf 0 is lossless in YUV**, so whatever PSNR
    it returns is the ceiling the RGB -> yuv420p -> RGB round trip imposes,
    not something the quantiser could ever have reached. A comparison run above
    that ceiling would be measuring the colour conversion.
    """
    from src.components.background.sidecar import RoiVideoSidecar

    rows: list[dict[str, Any]] = []
    for crf, preset, roi_on in ROI_CONTROL_GRID:
        sidecar = RoiVideoSidecar(crf=crf, preset=preset, regions=None if roi_on else ())
        started = time.time()
        payload = sidecar.encode(plate)
        decoded = sidecar.decode(payload)
        seconds = time.time() - started
        height = min(int(plate.shape[0]), int(decoded.shape[0]))
        width = min(int(plate.shape[1]), int(decoded.shape[1]))
        reference, got = plate[:height, :width], decoded[:height, :width]
        row = {
            "crf": crf,
            "preset": preset,
            "addroi": roi_on,
            "bytes": len(payload),
            "psnr_rgb_dB": pooled_psnr(reference, got),
            "psnr_y_dB": pooled_psnr(reference, got, luma=True),
            "container": container_kind(payload),
            "ffprobe_codec": probe_video(payload).get("codec_name"),
            "seconds": round(seconds, 1),
        }
        rows.append(row)
        print(
            f"  x264 crf{crf:<3} {preset:<9} roi={str(roi_on):<5} {row['bytes']:>10} B  "
            f"rgb {row['psnr_rgb_dB']:6.2f} dB  Y {row['psnr_y_dB']:6.2f} dB  {seconds:5.1f}s",
            flush=True,
        )
    ceiling = next((row for row in rows if row["crf"] == 0), None)
    return {
        "question": (
            "roi-video costs more than jpeg above ~40 dB. Is that x264, or is it "
            "the sidecar's fixed preset and addroi steering?"
        ),
        "grid": rows,
        "round_trip_ceiling": ceiling,
        "how_to_read_the_ceiling": (
            "crf 0 is lossless in YUV, so its PSNR is what the RGB -> yuv420p -> RGB "
            "conversion alone permits. Any matched-fidelity target at or above it is "
            "unreachable by this route at any bitrate, and a comparison there would be "
            "measuring the colour format rather than the codec."
        ),
    }


def _x264_roundtrip(
    plate: np.ndarray,
    *,
    encode_vf: str,
    decode_vf: str | None,
    crf: int = 0,
    preset: str = "veryfast",
) -> dict[str, Any]:
    """One libx264 round trip with the colour handling stated explicitly.

    Both halves name their codec (`-c:v libx264` out, `-c:v png` back), because
    an ffmpeg step that names none picks the muxer's default and quietly becomes
    a second encoder — `plans/BP24-findings.md` §14.
    """
    import cv2

    image = np.ascontiguousarray(np.asarray(plate, dtype=np.uint8))
    height = int(image.shape[0]) - int(image.shape[0]) % 2
    width = int(image.shape[1]) - int(image.shape[1]) % 2
    image = image[:height, :width]
    with tempfile.TemporaryDirectory(prefix="ps_bp29_ceiling_") as tmp_dir:
        tmp = Path(tmp_dir)
        source, coded, back = tmp / "src.png", tmp / "coded.mp4", tmp / "back.png"
        if not cv2.imwrite(str(source), image):
            raise RuntimeError("could not write the intermediate PNG")
        encode = [
            str(FFMPEG_BIN), "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-i", str(source), "-frames:v", "1",
            "-vf", encode_vf, "-c:v", "libx264", "-crf", str(crf),
            "-preset", preset, "-y", str(coded),
        ]
        subprocess.run(encode, check=True, capture_output=True, text=True)
        decode = [str(FFMPEG_BIN), "-hide_banner", "-loglevel", "error", "-i", str(coded)]
        if decode_vf:
            decode += ["-vf", decode_vf]
        decode += ["-vframes", "1", "-c:v", "png", "-y", str(back)]
        subprocess.run(decode, check=True, capture_output=True, text=True)
        decoded = cv2.imread(str(back), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("could not read the decoded PNG back")
        payload_bytes = coded.stat().st_size
    return {
        "encode_vf": encode_vf,
        "decode_vf": decode_vf,
        "crf": crf,
        "preset": preset,
        "bytes": int(payload_bytes),
        "psnr_rgb_dB": pooled_psnr(image, decoded),
        "psnr_y_dB": pooled_psnr(image, decoded, luma=True),
    }


def ceiling_controls(plate: np.ndarray) -> dict[str, Any]:
    """Where the roi-video route's ~44 dB ceiling comes from.

    `crf 0` is lossless coding, so anything the round trip still loses happened
    in the colour handling around the codec, not in it. Three variants separate
    the two suspects:

    * the sidecar's own chain (`format=yuv420p`, whatever range ffmpeg picks);
    * the same chain with full-range conversion stated on both halves — which
      isolates the limited-range (16-235) squeeze;
    * `yuv444p`, which keeps full chroma resolution — which isolates 4:2:0.

    This matters beyond `roi-video`: any sidecar built on the same interface
    inherits whichever of these is responsible.
    """
    variants = (
        ("sidecar's own chain", "format=yuv420p", None),
        (
            "full-range 4:2:0",
            "scale=in_range=full:out_range=full,format=yuv420p",
            "scale=in_range=full:out_range=full",
        ),
        (
            "full-range 4:4:4",
            "scale=in_range=full:out_range=full,format=yuv444p",
            "scale=in_range=full:out_range=full",
        ),
    )
    rows: list[dict[str, Any]] = []
    for label, encode_vf, decode_vf in variants:
        row = {"variant": label, **_x264_roundtrip(plate, encode_vf=encode_vf, decode_vf=decode_vf)}
        rows.append(row)
        print(
            f"  {label:<22} {row['bytes']:>10} B  rgb {row['psnr_rgb_dB']:6.2f} dB  "
            f"Y {row['psnr_y_dB']:6.2f} dB",
            flush=True,
        )
    return {
        "question": (
            "roi-video at crf 0 — lossless coding — still returns only ~44 dB. "
            "Lossless cannot lose anything, so the loss is in the colour handling."
        ),
        "variants": rows,
        "how_to_read_it": (
            "Every row codes losslessly, so the differences between them are the "
            "colour conversion alone. A row that scores far above the sidecar's own "
            "chain names what the sidecar is giving away before the codec even runs."
        ),
    }


def range_forensics(plate: np.ndarray) -> dict[str, Any]:
    """Why the range half of the ceiling costs 4.6 dB when quantisation predicts 1-3.

    The pre-written bound for the full-range 4:2:0 control was [44.0, 48.0] dB, on
    the reasoning that recovering a 16-235 squeeze is worth 1-3 dB. It measured
    48.83, so the bound was wrong and the mechanism was not understood. Squeezing
    255 levels into 219 and expanding them back is a quantisation whose error sits
    around 57 dB; that cannot account for what was measured, so something else is
    happening - a systematic gain, or clipping of values that fall outside the
    legal range.

    This separates them. A range mismatch is a **gain and offset**: fit one per
    channel and see whether removing it recovers the quality. Clipping shows up as
    decoded pixels pinned at 0 or 255 that were not pinned before.
    """
    import cv2

    image = np.ascontiguousarray(np.asarray(plate, dtype=np.uint8))
    height = int(image.shape[0]) - int(image.shape[0]) % 2
    width = int(image.shape[1]) - int(image.shape[1]) % 2
    image = image[:height, :width]
    with tempfile.TemporaryDirectory(prefix="ps_bp29_range_") as tmp_dir:
        tmp = Path(tmp_dir)
        source, coded, back = tmp / "src.png", tmp / "coded.mp4", tmp / "back.png"
        cv2.imwrite(str(source), image)
        subprocess.run(
            [
                str(FFMPEG_BIN), "-hide_banner", "-loglevel", "error",
                "-loop", "1", "-i", str(source), "-frames:v", "1",
                "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "0",
                "-preset", "veryfast", "-y", str(coded),
            ],
            check=True, capture_output=True, text=True,
        )
        tags = probe_video(coded.read_bytes())
        subprocess.run(
            [
                str(FFMPEG_BIN), "-hide_banner", "-loglevel", "error", "-i", str(coded),
                "-vframes", "1", "-c:v", "png", "-y", str(back),
            ],
            check=True, capture_output=True, text=True,
        )
        decoded = cv2.imread(str(back), cv2.IMREAD_COLOR)

    original = image.astype(np.float64)
    got = np.asarray(decoded, dtype=np.float64)
    channels = []
    corrected = np.empty_like(got)
    for index in range(3):
        x = original[..., index].ravel()
        y = got[..., index].ravel()
        gain, offset = np.polyfit(x, y, 1)
        corrected[..., index] = (got[..., index] - offset) / gain
        channels.append({"channel": index, "gain": float(gain), "offset": float(offset)})
    return {
        "stream_tags": {
            key: tags.get(key)
            for key in ("codec_name", "pix_fmt", "color_range", "color_space", "profile")
        },
        "per_channel_fit_of_decoded_against_original": channels,
        "psnr_as_measured_dB": pooled_psnr(image, decoded),
        "psnr_after_removing_gain_and_offset_dB": pooled_psnr(
            original, np.clip(corrected, 0.0, 255.0)
        ),
        "clipping": {
            "decoded_pixels_at_0_or_255": float(((got <= 0) | (got >= 255)).mean()),
            "original_pixels_at_0_or_255": float(((original <= 0) | (original >= 255)).mean()),
            "original_pixels_outside_16_235": float(
                ((original < 16) | (original > 235)).mean()
            ),
        },
        "how_to_read_it": (
            "A gain far from 1.0 means the two halves of the round trip disagree about "
            "colour range, and the PSNR after removing it says how much of the loss that "
            "disagreement accounts for. Clipping that appears only in the decoded image "
            "is range squeezing destroying values that were legal before."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=("sidecar", "end-to-end", "diagnose", "ceiling", "all"),
        default="all",
        help=(
            "diagnose: the follow-ups the sweep's own result demanded — why a "
            "lossless plate landed under its bound, and whether x264's loss at "
            "high fidelity is the codec or the sidecar's fixed configuration. "
            "Writes its own file and does not touch the sweep's."
        ),
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--video", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--tier", default="balanced")
    parser.add_argument("--residual-rate", type=int, default=38)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    kwargs: dict[str, Any] = {"n_frames": args.frames}
    if args.video:
        kwargs["video"] = args.video
    if args.scene:
        kwargs["scene"] = args.scene
    clip = load_tier_clip(**kwargs)
    plate = np.asarray(clip.frames[0])
    print(
        f"clip {clip.video}/{clip.scene} {clip.describe()['resolution']} x{args.frames}; "
        f"plate = frame {clip.frame_ids[0]}",
        flush=True,
    )

    if args.part == "ceiling":
        print("--- where the roi-video route's ceiling comes from ---", flush=True)
        answer = ceiling_controls(plate)
        print("--- range forensics: gain, offset, clipping ---", flush=True)
        forensics = range_forensics(plate)
        print(
            f"  tags {forensics['stream_tags']}; as measured "
            f"{forensics['psnr_as_measured_dB']:.2f} dB, after removing gain/offset "
            f"{forensics['psnr_after_removing_gain_and_offset_dB']:.2f} dB",
            flush=True,
        )
        destination = Path(args.out) if args.out else OUT_DIR / "plate-codec-ceiling.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "brief": "BP29 §1.1 — the roi-video route's quality ceiling",
                    "bounds_written_before_measurement": BOUNDS_FILE,
                    "clip": clip.describe(),
                    "ceiling_controls": answer,
                    "range_forensics": forensics,
                    "bounds_for_this_control": (
                        "outputs/bp29-plate-codec/bounds-before-ceiling-control.json"
                    ),
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {destination}", flush=True)
        return 0

    if args.part == "diagnose":
        print("--- png: settling a bound that came in under its floor ---", flush=True)
        png_answer = png_diagnostics(plate, clip)
        print(
            f"  independent ffmpeg png {png_answer['independent_encoder']['bytes']} B; "
            f"frame as extracted {png_answer['frame_as_extracted_months_earlier']['bytes']} B; "
            f"same plate + noise {png_answer['noise_control']['bytes']} B "
            f"(raw {png_answer['raw_bytes']} B)",
            flush=True,
        )
        print("--- x264 controls: codec, preset, or addroi? ---", flush=True)
        roi_answer = roi_video_diagnostics(plate)
        diagnostics = {
            "brief": "BP29 §1.1 — follow-ups the sweep's own numbers demanded",
            "bounds_written_before_measurement": BOUNDS_FILE,
            "clip": clip.describe(),
            "png_bound_investigation": png_answer,
            "roi_video_controls": roi_answer,
        }
        destination = Path(args.out) if args.out else OUT_DIR / "plate-codec-diagnostics.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(diagnostics, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"wrote {destination}", flush=True)
        return 0

    rungs: list[SidecarRung] = []
    matched: dict[str, Any] = {}
    jpeg_match: dict[str, Any] = {}
    if args.part in ("sidecar", "all"):
        print("--- sidecar sweep (plate only) ---", flush=True)
        rungs = sidecar_sweep(plate)
        matched = {
            "rgb": matched_fidelity_table(rungs, axis="rgb"),
            "luma": matched_fidelity_table(rungs, axis="luma"),
        }

    arms: list[dict[str, Any]] = []
    if args.part in ("end-to-end", "all"):
        roi_reference = next(
            (
                rung
                for rung in rungs
                if rung.codec == SIDECAR_ROI_VIDEO and rung.knob == f"crf{RUNNER_ROI_CRF}"
            ),
            None,
        )
        if roi_reference is not None:
            print(
                f"--- matching a jpeg quality to roi-video crf{RUNNER_ROI_CRF} "
                f"({roi_reference.psnr_rgb_db:.2f} dB rgb) ---",
                flush=True,
            )
            jpeg_match = match_jpeg_quality(plate, roi_reference.psnr_rgb_db)
            print(
                f"  jpeg q{jpeg_match['quality']} at {jpeg_match['psnr_rgb_dB']:.2f} dB rgb, "
                f"{jpeg_match['bytes']} B",
                flush=True,
            )

        def sidecar_bytes_for(codec: str, knob: str) -> int | None:
            for rung in rungs:
                if rung.codec == codec and rung.knob == knob:
                    return rung.payload_bytes
            return None

        print("--- end to end (residual held fixed) ---", flush=True)
        plan: list[tuple[str, str, int, int | None]] = [
            ("jpeg:75", SIDECAR_JPEG, 75, sidecar_bytes_for(SIDECAR_JPEG, "q75")),
            (
                f"png:{RUNNER_PNG_COMPRESSION}",
                SIDECAR_PNG,
                75,
                sidecar_bytes_for(SIDECAR_PNG, f"z{RUNNER_PNG_COMPRESSION}"),
            ),
            (
                f"roi-video:crf{RUNNER_ROI_CRF}",
                SIDECAR_ROI_VIDEO,
                75,
                sidecar_bytes_for(SIDECAR_ROI_VIDEO, f"crf{RUNNER_ROI_CRF}"),
            ),
        ]
        if jpeg_match.get("quality") is not None:
            quality = int(jpeg_match["quality"])
            plan.insert(
                1,
                (
                    f"jpeg:{quality} (matched to roi crf{RUNNER_ROI_CRF})",
                    SIDECAR_JPEG,
                    quality,
                    int(jpeg_match["bytes"]),
                ),
            )
        for label, codec, quality, sidecar_bytes in plan:
            arms.append(
                end_to_end_arm(
                    clip,
                    label=label,
                    codec=codec,
                    jpeg_quality=quality,
                    sidecar_bytes=sidecar_bytes,
                    tier=args.tier,
                    residual_rate=args.residual_rate,
                )
            )

    alarms = check_bounds(rungs, arms)
    for alarm in alarms:
        print(f"  ALARM {alarm}", flush=True)

    payload: dict[str, Any] = {
        "brief": "BP29 §1.1 — sweep background.codec over {jpeg, png, roi-video}",
        "bounds_written_before_measurement": BOUNDS_FILE,
        "question": (
            "What does the plate cost under each codec, and what does it buy? "
            "Not a BD-rate: the paired ladder is deliberately left to one run "
            "once every stream's lever has landed."
        ),
        "clip": clip.describe(),
        "plate_is": (
            "the clip's first frame — what make_background transmits today. The "
            "plate is still a single source frame rather than a stitched panorama "
            "(plans/BP24-findings.md §6)."
        ),
        "ffmpeg": {"path": str(FFMPEG_BIN), "ffprobe": str(FFPROBE_BIN)},
        "config_axis_reach": (
            "BackgroundConfig carries `codec` and `jpeg_quality` only, and "
            "strategy.bind forwards only those two. png_compression, roi_crf and "
            "roi_preset therefore keep RoiVideoSidecar's and PngSidecar's "
            "constructor defaults on every runner path: png z3, roi-video "
            f"libx264 crf{RUNNER_ROI_CRF} {RUNNER_ROI_PRESET}. Selecting a codec "
            "in a config selects one fixed operating point on it."
        ),
        "sidecar_rungs": [rung.record() for rung in rungs],
        "matched_fidelity": matched,
        "jpeg_quality_matched_to_roi": jpeg_match,
        "end_to_end_arms": arms,
        "bound_alarms": alarms,
    }
    destination = Path(args.out) if args.out else OUT_DIR / "plate-codec-sweep.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
