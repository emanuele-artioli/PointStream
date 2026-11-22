"""Reproduce the existing paid joint at 160 and test fixed lower base sizes by bytes alone."""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from experiments.tennis import common_player_packet as players, w8_payload as wire
from src.codecs import svtav1
from src.pointstream import ball_stream


def digest(b):
    return sha256(b).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("clean", "grabcut_delivery", "ball", "out", "receipt"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--final-64", action="store_true")
    a = parser.parse_args()
    prior = json.loads(
        (Path(__file__).resolve().parents[1] / "joint-trajectory-20261122/result.json").read_text()
    )
    assert digest(a.clean.read_bytes()) == prior["clean_sha256"]
    original = (a.grabcut_delivery / "players.pscp").read_bytes()
    assert digest(original) == "25242f4db9c081a75472762db8b409c827c7f84626d073e7838d40bef724910c"
    parts, _, _ = players.unpack(original)
    profile = players._embedded_profile(original)
    blob = a.ball.read_bytes()
    assert digest(blob) == prior["ball_source"]
    symbols = ball_stream.unpack(blob).ball_payload
    runtime = json.loads((a.grabcut_delivery / "receiver.manifest").read_text())
    assert runtime.pop("carrier_sha256") == digest(original)
    assert runtime.pop("composition") == "decoded-base,farthest-to-nearest-players,optional-ball"
    cache = wire.pack(
        {"type": "joint-initialization-v1", "runtime": runtime, "resources": profile["resources"]},
        [(k, "reference", v) for k, v in parts.items() if k.startswith("ref")],
    )
    assert digest(cache) == prior["cache_sha256"]
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "initial.psr").write_bytes(cache)
    tools = {k: svtav1.tool(k) for k in ("SvtAv1EncApp", "dav1d", "ffmpeg")}
    if any(tools[k]["sha256"] != prior["tools"][k]["sha256"] for k in tools):
        raise ValueError("native tool differs from original")
    rows = []
    sizes = [(160, 96), (64, 64)] if a.final_64 else [(160, 96), (128, 80), (96, 64)]
    for width, height in sizes:
        target = a.out / f"w{width}"
        target.mkdir()
        raw = target / "input.yuv"
        subprocess.run(
            [
                tools["ffmpeg"]["path"],
                "-v",
                "error",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "-pixel_format",
                "yuv420p",
                "-video_size",
                "960x544",
                "-framerate",
                "50",
                "-i",
                str(a.clean),
                "-vf",
                f"scale={width}:{height}:flags=area",
                "-filter_threads",
                "1",
                "-frames:v",
                "240",
                "-f",
                "rawvideo",
                str(raw),
            ],
            check=True,
            timeout=90,
        )
        coded = svtav1.code(
            raw,
            target,
            width=width,
            height=height,
            fps=Fraction(50),
            frames=240,
            crf=70,
            preset=4,
            threads=1,
            full_range=False,
            encoder=tools["SvtAv1EncApp"]["path"],
            decoder=tools["dav1d"]["path"],
            timeout=150,
        )
        assert coded["decoded_frames"] == coded["temporal_units"] == 240
        native = Path(coded["stream"]).read_bytes()
        profile["coded_size"] = [width, height]
        manifest = {
            "kind": "joint-trajectory-v1",
            "player": {k: v for k, v in profile.items() if k != "resources"},
            "cache_sha256": digest(cache),
            "ball_canvas": [960, 544],
            "display": [1920, 1080],
            "order": ["clean_base", "ball", "far_player", "near_player"],
        }
        payload = wire.pack(
            manifest,
            [
                ("base", "frame", native),
                ("poses", "frame", parts["keypoints"]),
                ("trajectory", "frame", symbols),
            ],
        )
        file = target / "joint.pscp"
        file.write_bytes(payload)
        parsed = wire.unpack(file.read_bytes(), manifest)[0]
        assert parsed == {"base": native, "poses": parts["keypoints"], "trajectory": symbols}
        checked = target / "independent.ivf"
        checked.write_bytes(parsed["base"])
        proof = target / "independent.yuv"
        subprocess.run(
            [
                tools["dav1d"]["path"],
                "-i",
                str(checked),
                "-o",
                str(proof),
                "--muxer",
                "yuv",
                "--threads",
                "1",
            ],
            check=True,
            capture_output=True,
            timeout=90,
        )
        assert proof.stat().st_size == 240 * width * height * 3 // 2
        assert digest(proof.read_bytes()) == digest(Path(coded["decoded"]).read_bytes())
        row = dict(
            width=width,
            height=height,
            crf=70,
            preset=4,
            bytes=len(payload),
            kbps=len(payload) / 600,
            cold_bytes=len(payload) + len(cache),
            fits=len(payload) <= 12240,
            base_bytes=len(native),
            pose_bytes=len(parsed["poses"]),
            ball_bytes=len(symbols),
            framing_bytes=len(payload) - len(native) - len(parsed["poses"]) - len(symbols),
            wire_sha256=digest(payload),
            base_sha256=digest(native),
            base_decoded_sha256=digest(proof.read_bytes()),
            decoded_base_frames=240,
        )
        if width == 160:
            assert (
                row["wire_sha256"] == prior["streams"]["20"]["wire_sha256"]
                and row["bytes"] == 12927
            )
        rows.append(row)
        if row["fits"]:
            break
    result = dict(
        source_sha256=prior["source_clip_sha256"],
        clean_sha256=prior["clean_sha256"],
        cache_sha256=digest(cache),
        cache_bytes=len(cache),
        source_ball_sha256=digest(blob),
        tools={k: {f: v[f] for f in ["sha256", "version"]} for k, v in tools.items()},
        frame_count=240,
        fps=[50, 1],
        ceiling_bytes=12240,
        rows=rows,
        scope="Native base recompressed from authenticated original donor-filled YUV, same original player and ball channels, fully serialized packet; full new RGB joint display requires separate receiver replay.",
    )
    with a.receipt.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print([(r["width"], r["base_bytes"], r["bytes"]) for r in rows])


if __name__ == "__main__":
    main()
