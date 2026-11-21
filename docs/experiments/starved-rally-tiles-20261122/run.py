"""Size-match real ball-tile deliveries before scoring any decoded image."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import cv2
import numpy as np

from experiments.tennis import common_player_ball as ball, common_player_packet as player
from src.codecs import svtav1
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clean-court-regional-screen-20261118"))
import crops  # noqa: E402

PRIOR = crops.ROOT / "ps-3rate-ball-20261118/48"
SECONDS = 4.8
TICKS = list(range(10, 240, 20))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(args):
    cv2.setNumThreads(1)
    root = args.output
    root.mkdir(exist_ok=False)
    tools = {name: svtav1.tool(name) for name in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    assert (
        sha(PRIOR / "native.pscp")
        == "b60466f988bdd4cc852134371f208d77bf7a53adc5d153551c761d64612da6eb"
    )
    assert (
        sha(PRIOR / "ball.sidecar")
        == "205faa1fa276a8a43f431bc28636483673258e0a0797d445c01efa72b0bec233"
    )
    assert (
        tools["SvtAv1EncApp"]["sha256"]
        == "db044194faf94d553c2509c8e033ecd05a373ba54d919d0fb49a3a13149d1687"
    )
    coords, tiles, size = ball._open(
        (PRIOR / "native.pscp").read_bytes(),
        (PRIOR / "ball.sidecar").read_bytes(),
        tools["dav1d"]["sha256"],
    )
    assert (
        len(coords) == 240
        and size == (1920, 1080)
        and args.raw.stat().st_size == 240 * 240 * 136 * 3 // 2
    )
    profile = dict(
        mode="native_only",
        recipe=player.NATIVE_RECIPE,
        resources={"native_decoder": tools["ffmpeg"]["sha256"]},
        coded_size=[240, 136],
        canvas_size=list(size),
        cuts=[],
        runs=[],
        order=[],
    )
    rows = {}

    def evaluate(q, shared=None):
        if q in rows:
            return rows[q]
        dest = root / f"q{q:g}"
        dest.mkdir()
        if shared:
            stream = shared
        else:
            native = svtav1.code(
                args.raw,
                dest,
                width=240,
                height=136,
                fps=Fraction(50),
                frames=240,
                crf=q,
                preset=4,
                threads=2,
                full_range=False,
                encoder=tools["SvtAv1EncApp"]["path"],
                decoder=tools["dav1d"]["path"],
                timeout=60,
            )
            (dest / "native.json").write_text(json.dumps(native, indent=2))
            assert native["decoded_frames"] == 240
            stream = Path(native["stream"])
        base = stream.read_bytes()
        carrier = player.build(profile, base, None, {})
        side = ball.pack(carrier, coords, tiles, decoder_sha256=tools["dav1d"]["sha256"])
        assert player.unpack(carrier)[0]["base"] == base
        assert ball._open(carrier, side, tools["dav1d"]["sha256"]) == (coords, tiles, size)
        (dest / "native.pscp").write_bytes(carrier)
        (dest / "ball.sidecar").write_bytes(side)
        row = dict(
            crf=q,
            stream=str(stream),
            base_sha256=sha(stream),
            plain_bytes=len(base),
            tile_bytes=len(carrier) + len(side),
            carrier_bytes=len(carrier),
            sidecar_bytes=len(side),
            carrier_sha256=sha(dest / "native.pscp"),
            sidecar_sha256=sha(dest / "ball.sidecar"),
        )
        rows[q] = row
        return row

    for q, stream in zip((36, 48, 60), args.prior, strict=True):
        # A full independent native decode must agree with the original experiment.
        row = evaluate(q, stream)
        assert row["base_sha256"] == args.hashes[(36, 48, 60).index(q)]

    def tune(key, target, steps, tol):
        def closest():
            return min(rows.values(), key=lambda r: (abs(r[key] - target), r["crf"]))

        if min(r[key] for r in rows.values()) > target * (1 + tol):
            evaluate(63)
        if max(r[key] for r in rows.values()) < target * (1 - tol):
            evaluate(24)
        for _ in range(steps):
            best = closest()
            if abs(best[key] / target - 1) <= tol:
                break
            brackets = [
                (a, b)
                for a, b in zip(sorted(rows)[:-1], sorted(rows)[1:])
                if (rows[a][key] - target) * (rows[b][key] - target) < 0 and b - a >= 0.5
            ]
            if not brackets:
                break
            a, b = min(
                brackets,
                key=lambda pair: abs(rows[pair[0]][key] - target)
                + abs(rows[pair[1]][key] - target),
            )
            evaluate(round((a + b) * 2) / 4)
        return closest()

    pairs = []
    for target in [50, 63, 125]:
        if target == 63 and pairs and pairs[0].get("matched"):
            continue
        tiled = tune("tile_bytes", target * 600, 4, 0.06)
        if abs(tiled["tile_bytes"] / (target * 600) - 1) > 0.1:
            pairs.append(
                dict(target=target, near_tile_kbps=tiled["tile_bytes"] / 600, feasible=False)
            )
            continue
        plain = tune("plain_bytes", tiled["tile_bytes"], 10, 0.005)
        pairs.append(
            dict(
                target=target,
                plain_crf=plain["crf"],
                tile_crf=tiled["crf"],
                plain_kbps=plain["plain_bytes"] / 600,
                tile_kbps=tiled["tile_bytes"] / 600,
                matched=abs(plain["plain_bytes"] - tiled["tile_bytes"]) / plain["plain_bytes"]
                <= 0.005,
                feasible=True,
            )
        )
    result = dict(
        pairs=pairs,
        rows=list(rows.values()),
        tools=tools,
        source_yuv_sha256=sha(args.raw),
        script_sha256=sha(Path(__file__)),
        repo_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        placements=sum(p is not None for p in coords),
    )
    (root / "sizes.json").write_text(json.dumps(result, indent=2))
    print(pairs, flush=True)
    if not any(p["feasible"] for p in pairs):
        return
    src_zip = crops.ROOT / "ps-development-source-refs-20261118-native/source-only-review.zip"
    labels = crops.ROOT / "foreground-human-reviewed-20261118" / f"{crops.LABEL}.json"
    assert sha(src_zip) == crops.BUNDLE and sha(labels) == crops.LABEL
    frames = json.loads(labels.read_text())["frames"]

    def add(z, key, rgb):
        image = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, data = cv2.imencode(".png", image)
        assert ok
        z.writestr(key + ".png", data.tobytes())

    def canvases(base):
        for t, yuv in enumerate(base):
            yield (
                t,
                cv2.resize(yuv_to_rgb(yuv, 240, 136, False), size, interpolation=cv2.INTER_CUBIC),
            )

    with zipfile.ZipFile(src_zip) as source, zipfile.ZipFile(root / "crops.zip", "x") as cropzip:
        truth = {}
        for f in frames:
            t = f["tick"]
            png = source.read(f"frame-{t:03d}/source.png")
            assert hashlib.sha256(png).hexdigest() == f["source_png_sha256"]
            truth[t] = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            for region, y, x, mask in crops.supports(f):
                add(cropzip, f"{t:03d}/{region}/source", truth[t][y, x])
                add(cropzip, f"{t:03d}/{region}/mask", mask)
        for pair in pairs:
            if not pair["feasible"]:
                continue
            for arm in ("plain", "tile"):
                q = pair[arm + "_crf"]
                dest = root / f"q{q:g}"
                path = root / f"{pair['target']}-{arm}.yuv"
                # Decode the payload actually read back from disk.
                carrier = (dest / "native.pscp").read_bytes()
                base = (
                    player.unpack(carrier)[0]["base"]
                    if arm == "tile"
                    else Path(rows[q]["stream"]).read_bytes()
                )
                subprocess.run(
                    [
                        tools["ffmpeg"]["path"],
                        "-v",
                        "error",
                        "-c:v",
                        "libdav1d",
                        "-threads",
                        "1",
                        "-f",
                        "ivf",
                        "-i",
                        "pipe:0",
                        "-pix_fmt",
                        "yuv420p",
                        "-f",
                        "rawvideo",
                        str(path),
                    ],
                    input=base,
                    check=True,
                    timeout=60,
                )
                native = np.memmap(path, "uint8", "r", shape=(240, 204, 240))
                assert path.stat().st_size == native.nbytes
                output = canvases(native)
                if arm == "tile":
                    output = ball.render(
                        carrier,
                        (dest / "ball.sidecar").read_bytes(),
                        output,
                        scratch=root,
                        decoder=tools["dav1d"]["path"],
                    )
                error = 0.0
                count = 0
                for t, rgb in output:
                    assert t == count
                    count += 1
                    if t not in truth:
                        continue
                    error += np.square(rgb.astype(np.float32) - truth[t], dtype=np.float32).sum(
                        dtype=np.float64
                    )
                    f = frames[TICKS.index(t)]
                    for region, y, x, _ in crops.supports(f):
                        add(cropzip, f"{t:03d}/{region}/{arm}{pair['target']}", rgb[y, x])
                assert count == 240
                pair[arm + "_rgb_psnr_12"] = 10 * math.log10(255**2 * 12 * 1920 * 1080 * 3 / error)
    result.update(
        source_zip_sha256=crops.BUNDLE,
        labels_sha256=crops.LABEL,
        crops_sha256=sha(root / "crops.zip"),
    )
    (root / "render.json").write_text(json.dumps(result, indent=2))
    print(pairs)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--raw", type=Path, required=True)
    cli.add_argument("--prior", type=Path, nargs=3, required=True)
    cli.add_argument("--hashes", nargs=3, required=True)
    cli.add_argument("--output", type=Path, required=True)
    run(cli.parse_args())
