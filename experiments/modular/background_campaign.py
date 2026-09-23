"""Background campaign encodes.

Four representations against a same-pipeline VVC source, with encode and
decode clocks split. The record is
``docs/workflow/session/evaluation-campaign/20260923-background-campaign.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from experiments.modular.background_arms import (
    _intra_still,
    _render_panorama,
    _repeat,
    _score,
    choose_background,
)
from experiments.modular.measured_ladder import (
    _as_clip_rgb,
    _direct_vvc_encode,
    _rgb_to_bgr,
    _rgb_to_yuv420,
    _yuv420_to_rgb,
    load_sequence,
)
from scripts.background_probe import (
    build_common_cleaned_stack,
    charge_side_data,
    pack_panorama_side_data,
    pack_still_or_video_side_data,
)
from src.components.background.still import select_best_background_frame
from src.components.codec.encode import BITSTREAM_SUFFIX, decode, encode
from src.components.codec.frames import even_size
from src.components.codec.tools import resolve_encoder
from src.components.codec.y4m import Y4M, read, write
from src.contracts.codecs import EncodeRequest, RateControl

OUT_DIR = Path("/home/itec/emanuele/pointstream-data/outputs/modular/background-arms")
CLIPS_ROOT = Path("/home/itec/emanuele/pointstream-data/outputs/bp46-long-scenes/clips")
PRESET = "faster"
CODEC = "vvc"
FG_WEIGHT = 0.70
BG_WEIGHT = 0.30
MIN_BUDGET = 8_000
PART1 = OUT_DIR / "federer007.json"
CAMPAIGN_VVC_QP46 = {
    "bytes": 112_295,
    "psnr_fg": 21.69,
    "psnr_bg": 31.362980885416793,
    "psnr_weighted": 24.58878360131408,
}


def required_foreground(weighted_anchor: float, psnr_bg: float) -> float:
    """Foreground PSNR that ties the anchor's weighted score at this background."""
    return (float(weighted_anchor) - BG_WEIGHT * float(psnr_bg)) / FG_WEIGHT


def choose_setup(
    rows: list[dict[str, object]],
    anchor_bytes: int,
    weighted_anchor: float,
    anchor_fg: float,
) -> dict[str, object]:
    """Lowest required foreground among arms that leave at least 8 kB."""
    eligible = [
        row
        for row in rows
        if row.get("representation") != "source"
        and row.get("psnr_bg") is not None
        and anchor_bytes - int(row["total_bytes"]) >= MIN_BUDGET
    ]
    if not eligible:
        return {"fits": False, "reason": "no arm leaves 8 kB", "pipeline_setup": False}
    chosen = min(eligible, key=lambda row: required_foreground(weighted_anchor, float(row["psnr_bg"])))
    required = required_foreground(weighted_anchor, float(chosen["psnr_bg"]))
    budget = anchor_bytes - int(chosen["total_bytes"])
    gap = required - float(anchor_fg)
    return {
        "fits": True,
        "representation": chosen["representation"],
        "qp": chosen["qp"],
        "total_bytes": chosen["total_bytes"],
        "psnr_bg": chosen["psnr_bg"],
        "foreground_budget_bytes": budget,
        "required_fg": required,
        "anchor_fg": anchor_fg,
        "required_fg_minus_anchor_fg": gap,
        "pipeline_setup": gap <= 4.0,
        "reason": "lowest required foreground among arms with at least 8 kB left",
    }


def background_rate_win(psnr_bg: float, anchor_bg: float, budget: int) -> bool:
    """Within 0.5 dB of the source background and at least 8 kB under it."""
    return float(anchor_bg) - float(psnr_bg) <= 0.5 and int(budget) >= MIN_BUDGET


def long_window_gate(setup: dict[str, object]) -> bool:
    """48-frame setup still has room and the required foreground is within 6 dB."""
    if not setup.get("fits"):
        return False
    gap = setup.get("required_fg_minus_anchor_fg")
    if gap is None:
        return False
    return int(setup["foreground_budget_bytes"]) >= MIN_BUDGET and float(gap) <= 6.0


def _fps(n_frames: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return n_frames / seconds


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _timed_vvc(
    frames_rgb: np.ndarray,
    qp: int,
) -> tuple[bytes, np.ndarray, float, float, str, str]:
    """Encode RGB frames and return payload, decoded RGB, encode seconds, decode seconds."""
    frames = _as_clip_rgb(frames_rgb)
    with tempfile.TemporaryDirectory(prefix="ps_bg2_") as tmp:
        root = Path(tmp)
        clip = even_size(frames)
        luma, chroma = _rgb_to_yuv420(clip)
        source = root / "input.y4m"
        write(
            source,
            Y4M(
                width=int(luma.shape[2]),
                height=int(luma.shape[1]),
                fps=25.0,
                luma=luma,
                chroma=chroma,
            ),
        )
        request = EncodeRequest(
            codec_name=CODEC,
            rate_control=RateControl.QP,
            rate=int(qp),
            preset=PRESET,
            pix_fmt="yuv420p",
        )
        bitstream = root / f"{CODEC}_qp{int(qp)}{BITSTREAM_SUFFIX[CODEC]}"
        encoder = resolve_encoder(CODEC)
        tool_path, tool_version = encoder.path, encoder.version
        try:
            record = encode(source, bitstream, request, work_dir=root)
            encode_s = float(record.encode_seconds)
            tool_path, tool_version = record.tool_path, record.tool_version
            if not bitstream.is_file() or bitstream.stat().st_size == 0:
                raise RuntimeError("empty VVC bitstream")
        except RuntimeError:
            started = time.perf_counter()
            tool_path, tool_version = _direct_vvc_encode(source, bitstream, request)
            encode_s = time.perf_counter() - started
        payload = bitstream.read_bytes()
        started = time.perf_counter()
        decoded_path = root / "decoded.y4m"
        decode(bitstream, decoded_path, request)
        decoded = read(decoded_path)
        if decoded.chroma is None:
            raise RuntimeError(f"{decoded_path} decoded without chroma")
        decoded_rgb = _yuv420_to_rgb(decoded.luma, decoded.chroma)
        decode_s = time.perf_counter() - started
    if decoded_rgb.shape[0] != frames.shape[0]:
        decoded_rgb = decoded_rgb[: frames.shape[0]]
    if decoded_rgb.shape[1] != frames.shape[1] or decoded_rgb.shape[2] != frames.shape[2]:
        canvas = np.zeros(frames.shape, dtype=np.uint8)
        height = min(decoded_rgb.shape[1], frames.shape[1])
        width = min(decoded_rgb.shape[2], frames.shape[2])
        canvas[:, :height, :width] = decoded_rgb[:, :height, :width]
        decoded_rgb = canvas
    return payload, decoded_rgb, encode_s, decode_s, tool_path, tool_version


def _load_or_build(
    cache: Path,
    frames_rgb: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple, int, float, float]:
    if cache.is_file():
        data = np.load(cache)
        homographies = tuple(np.asarray(row, dtype=np.float64) for row in data["homographies"])
        print(f"cache {cache}", flush=True)
        return (
            data["cleaned"],
            data["plate"],
            homographies,
            int(data["best_index"]),
            float(data["best_mse"]),
            float(data["build_seconds"]),
        )
    print("building plate-inpainted stack", flush=True)
    started = time.perf_counter()
    cleaned, plate, homographies, _prep = build_common_cleaned_stack(
        frames_rgb,
        mask,
        removal="on",
        register=True,
    )
    build_s = time.perf_counter() - started
    best_index, best_mse = select_best_background_frame(frames_rgb, mask)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        cleaned=cleaned,
        plate=plate,
        homographies=np.stack([np.asarray(h, dtype=np.float64).reshape(3, 3) for h in homographies]),
        best_index=np.int32(best_index),
        best_mse=np.float64(best_mse),
        build_seconds=np.float64(build_s),
    )
    print(f"plate {build_s:.1f}s peak_gb={_peak_gb():.1f} best={best_index}", flush=True)
    return cleaned, plate, homographies, best_index, best_mse, build_s


def _arm_row(
    name: str,
    qp: int,
    payload: bytes,
    side_bytes: int,
    rendered: np.ndarray,
    frames_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    encode_s: float,
    decode_s: float,
    tool_path: str,
    tool_version: str,
    plate_s: float,
    extra: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "representation": name,
        "qp": qp,
        "codec": CODEC,
        "preset": PRESET,
        "tool_path": tool_path,
        "tool_version": tool_version,
        "encoded_payload_bytes": len(payload),
        "side_data_bytes": side_bytes,
        "total_bytes": len(payload) + side_bytes,
        "encode_seconds": round(encode_s, 3),
        "decode_seconds": round(decode_s, 3),
        "plate_build_seconds": 0.0 if name == "source" else round(plate_s, 3),
    }
    row.update(_score(frames_rgb, rendered, mask))
    row.update(extra)
    return row


def _attach_clocks(rows: list[dict[str, object]], n_frames: int) -> None:
    sources = {int(row["qp"]): row for row in rows if row["representation"] == "source"}
    for row in rows:
        encode_s = float(row["encode_seconds"])
        decode_s = float(row["decode_seconds"])
        plate_s = float(row["plate_build_seconds"])
        row["encode_fps"] = _fps(n_frames, encode_s)
        row["decode_fps"] = _fps(n_frames, decode_s)
        sender = encode_s if row["representation"] == "source" else plate_s + encode_s
        row["sender_seconds"] = round(sender, 3)
        row["client_seconds"] = round(decode_s, 3)
        row["sender_fps"] = _fps(n_frames, sender)
        row["client_fps"] = _fps(n_frames, decode_s)
        source = sources.get(int(row["qp"]))
        if source is None or row["representation"] == "source":
            continue
        source_encode = float(source["encode_seconds"])
        source_decode = float(source["decode_seconds"])
        row["encode_fps_ratio_vs_source"] = (source_encode / encode_s) if encode_s > 0 else None
        row["decode_fps_ratio_vs_source"] = (source_decode / decode_s) if decode_s > 0 else None
        row["sender_fps_ratio_vs_source"] = (source_encode / sender) if sender > 0 else None


def _choices_for_qp(
    rows: list[dict[str, object]],
    qp: int,
    anchor: dict[str, object],
) -> dict[str, object]:
    same = [row for row in rows if int(row["qp"]) == qp and row["representation"] != "source"]
    anchor_bytes = int(anchor["bytes"])
    quality = choose_background(same, anchor_bytes)
    setup = choose_setup(same, anchor_bytes, float(anchor["psnr_weighted"]), float(anchor["psnr_fg"]))
    wins = []
    for row in same:
        budget = anchor_bytes - int(row["total_bytes"])
        if background_rate_win(float(row["psnr_bg"]), float(anchor["psnr_bg"]), budget):
            wins.append(
                {
                    "representation": row["representation"],
                    "qp": qp,
                    "total_bytes": row["total_bytes"],
                    "psnr_bg": row["psnr_bg"],
                    "foreground_budget_bytes": budget,
                }
            )
    return {"quality": quality, "setup": setup, "background_rate_wins": wins}


def _source_anchor(row: dict[str, object]) -> dict[str, object]:
    return {
        "bytes": int(row["total_bytes"]),
        "psnr_fg": row["psnr_fg"],
        "psnr_bg": row["psnr_bg"],
        "psnr_weighted": row["psnr_weighted"],
        "encode_seconds": row["encode_seconds"],
        "decode_seconds": row["decode_seconds"],
    }


def run_clip(
    *,
    clip_id: str,
    frames_dir: Path,
    mask_path: Path,
    n_frames: int,
    arm_qps: tuple[int, ...],
    source_qps: tuple[int, ...],
    clock_only_qps: tuple[int, ...],
    out_path: Path,
    arms: tuple[str, ...] = ("still_frame0", "best_frame", "registered_panorama", "cleaned_video"),
) -> dict[str, object]:
    wall = time.perf_counter()
    frames_rgb, mask = load_sequence(frames_dir, mask_path, n_frames)
    print(
        f"{clip_id} {tuple(frames_rgb.shape)} foreground_fraction={float(mask.mean()):.6f}",
        flush=True,
    )
    cache = OUT_DIR / "cache" / f"{clip_id}-n{n_frames}.npz"
    cleaned, plate, homographies, best_index, best_mse, plate_s = _load_or_build(cache, frames_rgb, mask)
    if plate_s > 1800 and clip_id == "federer001":
        print(f"{clip_id} plate build {plate_s:.0f}s exceeded 30 minutes", flush=True)
        return {"clip_id": clip_id, "stopped": True, "choices": {}}
    if n_frames > 48 and (plate_s > 45 * 60 or _peak_gb() > 200):
        print(f"{clip_id} plate {plate_s:.0f}s peak {_peak_gb():.1f} GB exceeded the long-window stop", flush=True)
        return {"clip_id": clip_id, "stopped": True, "choices": {}}
    height, width = int(frames_rgb.shape[1]), int(frames_rgb.shape[2])
    frame_shape = (height, width)
    still_side = pack_still_or_video_side_data(frame_shape=frame_shape, n_frames=n_frames, fps=25.0)
    best_side = still_side + int(best_index).to_bytes(2, "little")
    pano_side = pack_panorama_side_data(
        homographies,
        plate_shape=(int(plate.shape[0]), int(plate.shape[1])),
        frame_shape=frame_shape,
        fps=25.0,
    )
    pano_detail = charge_side_data(
        "registered_panorama",
        n_frames,
        plate_shape=(int(plate.shape[0]), int(plate.shape[1])),
        frame_shape=frame_shape,
        homographies=homographies,
    )
    pano_side_bytes = int(pano_detail["total_side_data_bytes"])
    encoder = resolve_encoder(CODEC)
    rows: list[dict[str, object]] = []
    document: dict[str, object] = {
        "clip_id": clip_id,
        "source": str(frames_dir),
        "mask": str(mask_path),
        "n_frames": n_frames,
        "foreground_fraction": float(mask.mean()),
        "best_frame": {"index": best_index, "background_mse": best_mse},
        "plate_shape": [int(plate.shape[0]), int(plate.shape[1])],
        "plate_build_seconds": round(plate_s, 3),
        "peak_rss_gb": round(_peak_gb(), 3),
        "encoder_resolved": {"path": encoder.path, "version": encoder.version},
        "rows": rows,
    }
    _write(out_path, document)

    for qp in source_qps:
        print(f"{clip_id} source qp {qp}", flush=True)
        payload, decoded, enc_s, dec_s, path, version = _timed_vvc(frames_rgb, qp)
        rows.append(
            _arm_row(
                "source",
                qp,
                payload,
                0,
                decoded,
                frames_rgb,
                mask,
                encode_s=enc_s,
                decode_s=dec_s,
                tool_path=path,
                tool_version=version,
                plate_s=plate_s,
                extra={"role": "same-pipeline anchor"},
            )
        )
        print(rows[-1], flush=True)
        _write(out_path, document)

    arm_plan: list[tuple[int, tuple[str, ...]]] = [(qp, arms) for qp in arm_qps]
    for qp in clock_only_qps:
        arm_plan.append((qp, ("registered_panorama", "cleaned_video")))

    for qp, names in arm_plan:
        for name in names:
            print(f"{clip_id} {name} qp {qp}", flush=True)
            if name == "cleaned_video":
                payload, decoded, enc_s, dec_s, path, version = _timed_vvc(cleaned, qp)
                side = len(still_side)
                extra = {}
            elif name == "registered_panorama":
                payload, decoded_plate, path, version, enc_s, dec_s = _intra_still(plate, qp)
                started = time.perf_counter()
                decoded = _render_panorama(decoded_plate, pano_side, n_frames, frame_shape)
                dec_s += time.perf_counter() - started
                side = pano_side_bytes
                extra = {"plate_shape": [int(plate.shape[0]), int(plate.shape[1])]}
            else:
                image = cleaned[0] if name == "still_frame0" else cleaned[best_index]
                payload, decoded_still, path, version, enc_s, dec_s = _intra_still(image, qp)
                decoded = _repeat(decoded_still, n_frames, frame_shape)
                side = len(still_side) if name == "still_frame0" else len(best_side)
                extra = {"frame_index": 0 if name == "still_frame0" else best_index}
            rows.append(
                _arm_row(
                    name,
                    qp,
                    payload,
                    side,
                    decoded,
                    frames_rgb,
                    mask,
                    encode_s=enc_s,
                    decode_s=dec_s,
                    tool_path=path,
                    tool_version=version,
                    plate_s=plate_s,
                    extra=extra,
                )
            )
            print(rows[-1], flush=True)
            _write(out_path, document)

    _attach_clocks(rows, n_frames)
    anchors = {f"vvc_qp{int(row['qp'])}": _source_anchor(row) for row in rows if row["representation"] == "source"}
    document["anchors"] = anchors
    document["choices"] = {
        name: _choices_for_qp(rows, int(name.removeprefix("vvc_qp")), anchor)
        for name, anchor in anchors.items()
    }
    if clip_id == "federer007" and PART1.is_file():
        part1_rows = json.loads(PART1.read_text())["rows"]
        combined = list(part1_rows) + [row for row in rows if row["representation"] != "source"]
        document["against_campaign_vvc_qp46"] = _choices_for_qp(combined, 46, CAMPAIGN_VVC_QP46)
        # Coarser new rows are also eligible against the fixed campaign anchor.
        document["against_campaign_vvc_qp46_all_qps"] = choose_setup(
            combined,
            int(CAMPAIGN_VVC_QP46["bytes"]),
            float(CAMPAIGN_VVC_QP46["psnr_weighted"]),
            float(CAMPAIGN_VVC_QP46["psnr_fg"]),
        )
    document["wall_seconds"] = round(time.perf_counter() - wall, 3)
    _write(out_path, document)
    print(f"wrote {out_path}", flush=True)
    print(document["choices"], flush=True)
    return document


def _summary(documents: list[dict[str, object]]) -> dict[str, object]:
    choices = []
    for document in documents:
        for name, choice in document.get("choices", {}).items():
            setup = choice["setup"]
            choices.append(
                {
                    "clip_id": document["clip_id"],
                    "anchor": name,
                    "foreground_fraction": document["foreground_fraction"],
                    "setup": setup,
                    "quality": choice["quality"],
                    "background_rate_wins": choice["background_rate_wins"],
                }
            )
    gaps = [
        float(item["setup"]["required_fg_minus_anchor_fg"])
        for item in choices
        if item["setup"].get("fits") and item["anchor"].endswith("46") and item["clip_id"] != "perricard002-192"
    ]
    mean_gap = sum(gaps) / len(gaps) if gaps else None
    return {
        "choices": choices,
        "labeled_mean_required_fg_gap_at_qp46": mean_gap,
        "mean_is_not_the_claim": True,
    }


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only == "alcaraz000":
        run_clip(
            clip_id="alcaraz000",
            frames_dir=CLIPS_ROOT / "alcaraz_highlights/scene_000/window_48",
            mask_path=CLIPS_ROOT / "alcaraz_highlights/scene_000/masks_48.npz",
            n_frames=48,
            arm_qps=(46, 50),
            source_qps=(46, 50),
            clock_only_qps=(),
            out_path=OUT_DIR / "alcaraz000.json",
        )
        return
    documents: list[dict[str, object]] = []
    specs = [
        (
            "federer007",
            CLIPS_ROOT / "federer_djokovic/scene_007/window_48",
            CLIPS_ROOT / "federer_djokovic/scene_007/masks_48.npz",
            48,
            (50, 54),
            (46, 50, 54),
            (46,),
            OUT_DIR / "federer007-matched.json",
        ),
        (
            "federer001",
            CLIPS_ROOT / "federer_djokovic/scene_001/window_48",
            CLIPS_ROOT / "federer_djokovic/scene_001/masks_48.npz",
            48,
            (46, 50),
            (46, 50),
            (),
            OUT_DIR / "federer001.json",
        ),
        (
            "perricard002",
            CLIPS_ROOT / "alcaraz_perricard/scene_002/window_48",
            CLIPS_ROOT / "alcaraz_perricard/scene_002/masks_48.npz",
            48,
            (46, 50),
            (46, 50),
            (),
            OUT_DIR / "perricard002.json",
        ),
    ]
    for spec in specs:
        if only not in ("all", spec[0]):
            continue
        documents.append(
            run_clip(
                clip_id=spec[0],
                frames_dir=spec[1],
                mask_path=spec[2],
                n_frames=spec[3],
                arm_qps=spec[4],
                source_qps=spec[5],
                clock_only_qps=spec[6],
                out_path=spec[7],
            )
        )
        if spec[0] == "federer001" and _peak_gb() > 80:
            print("stopping before perricard: peak RSS exceeded 80 GB", flush=True)
            break
        if documents[-1].get("stopped"):
            break

    if documents:
        _write(OUT_DIR / "summary.json", _summary(documents))


if __name__ == "__main__":
    main()
