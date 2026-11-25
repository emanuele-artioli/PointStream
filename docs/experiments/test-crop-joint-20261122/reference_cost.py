"""Price the fixed two-reference Shanghai startup ladder and verify its decoded image crops."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from experiments.tennis import common_player_packet as packet, k6b, w8_payload as wire


def sha(b):
    return hashlib.sha256(b).hexdigest()


def encode(level, png, out, tools):
    bgr = cv2.imread(str(png))
    if level.startswith("jpeg"):
        ok, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(level[4:])])
        assert ok
        return data.tobytes(), cv2.imdecode(data, cv2.IMREAD_COLOR)[:, :, ::-1]
    if level == "webp":
        encoded = out.with_suffix(".webp")
        subprocess.run(
            [
                tools["cwebp"],
                "-lossless",
                "-exact",
                "-m",
                "6",
                "-metadata",
                "none",
                png,
                "-o",
                encoded,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            [tools["dwebp"], encoded, "-o", out], check=True, capture_output=True, timeout=30
        )
    else:
        encoded = out.with_suffix(".avif")
        subprocess.run(
            [tools["avifenc"], "-q", level[4:], "--speed", "8", "--jobs", "1", png, encoded],
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            [tools["avifdec"], "--jobs", "1", encoded, out],
            check=True,
            capture_output=True,
            timeout=30,
        )
    return encoded.read_bytes(), np.asarray(Image.open(out).convert("RGB"))


def psnr(error, pixels):
    return float(10 * np.log10(255**2 * pixels / error)) if error else None


def main(a):
    from score_native import get_support

    start = time.process_time()
    cv2.setNumThreads(1)
    tools = {
        name: Path(shutil.which(name)).resolve()
        for name in ("cwebp", "dwebp", "avifenc", "avifdec")
    }
    tools_sha = {n: sha(p.read_bytes()) for n, p in tools.items()}
    assert (
        tools_sha["avifenc"] == "f9cd38ec2804b56174344fd178ea919f9bff41ec7fc4a471f0ff9065b1b5d076"
    )
    assert (
        tools_sha["avifdec"] == "635e3acc8cc93212ca5cf1cc867b9bb6f3b738574e92107ed24be48a4a75e6bd"
    )
    cvlib = next(Path(cv2.__file__).resolve().parent.glob("cv2*.so"))
    tools_sha["opencv_native"] = sha(cvlib.read_bytes())
    paid = json.loads(Path(__file__).with_name("match102-sender.json").read_text())
    row = next(r for r in paid["rates"] if r["crf"] == paid["selected_crf"])
    cache = (a.paid / "initial.psr").read_bytes()
    joint = (a.paid / "rally.psj").read_bytes()
    assert sha(cache) == row["init_sha256"] and sha(joint) == row["joint_sha256"]
    cprof = packet._embedded_profile(cache)
    jprof = packet._embedded_profile(joint)
    refs, _ = wire.unpack(cache, cprof)
    stream, _ = wire.unpack(joint, jprof)
    sections, _, _ = packet.unpack((a.paid / "players.pscp").read_bytes())
    supports, support_sha = get_support(a.features)
    a.out.mkdir(exist_ok=False, parents=True)
    metadata = {}
    sources = {}
    masks = {}
    for name, data in refs.items():
        assert data == sections[name]
        m, png = packet._bounded_transport(data, b"REF1", packet.MAX_REF)
        assert m["reference_frame"] == 0 and m["codec"] == ".png"
        metadata[name] = m
        f = a.out / (name + ".png")
        f.write_bytes(png)
        rgb = np.asarray(Image.open(f).convert("RGB"))
        assert np.array_equal(packet._bounded_reference(data)[1], rgb)
        sources[name] = rgb
        (x0, y0, x1, y1), small = supports[0][int(m["unit"].endswith("_far"))]
        mask = np.zeros((1080, 1920, 3), np.uint8)
        mask[y0:y1, x0:x1] = small[:, :, None] * 255
        masks[name] = k6b.crop_image(mask, np.array(m["reference_box"]))[:, :, 0] >= 128
        assert masks[name].sum() > 50
    results = []
    for level in ("png", "webp", "avif65", "avif78", "avif90", "jpeg40", "jpeg70"):
        all_bytes = {}
        arms = {}
        E = P = 0
        F = N = 0
        for key, src in sources.items():
            if level == "png":
                b = (a.out / (key + ".png")).read_bytes()
                decoded = np.asarray(Image.open(a.out / (key + ".png")).convert("RGB"))
            else:
                b, decoded = encode(
                    level, a.out / (key + ".png"), a.out / (level + "-" + key + ".png"), tools
                )
            delta = src.astype(np.int32) - decoded.astype(np.int32)
            err = int(np.square(delta).sum())
            fg = int(np.square(delta[masks[key]]).sum())
            n = 3 * int(masks[key].sum())
            N += n
            F += fg
            E += err
            P += src.size
            all_bytes[key] = b
            arms[key] = dict(
                unit=metadata[key]["unit"],
                image_bytes=len(b),
                image_sha256=sha(b),
                rgb_sha256=sha(decoded.tobytes()),
                source_rgb_sha256=sha(src.tobytes()),
                rgb_exact=err == 0,
                crop_psnr=psnr(err, src.size),
                source_body_psnr=psnr(fg, n),
                body_pixels=n // 3,
            )
        if level == "png":
            wire_cache = cache
        else:
            dec_key = (
                "avifdec"
                if level.startswith("avif")
                else ("dwebp" if level == "webp" else "opencv_native")
            )
            label = level.replace("avif", "avif-q")
            new_profile = dict(
                cprof,
                reference_codec=label,
                reference_container="images-with-ref1-metadata",
                reference_decoder_sha256=tools_sha[dec_key],
                reference_metadata=metadata,
            )
            wire_cache = wire.pack(new_profile, [(n, "reference", b) for n, b in all_bytes.items()])
            assert wire.unpack(wire_cache, packet._embedded_profile(wire_cache))[0] == all_bytes
        prof = dict(jprof, cache_sha256=sha(wire_cache))
        wire_joint = wire.pack(prof, [(n, "frame", b) for n, b in stream.items()])
        assert wire.unpack(wire_joint, prof)[0] == stream and len(wire_joint) == len(joint)
        if level == "png":
            assert wire_joint == joint
        if level in ("png", "webp"):
            assert E == 0
        (a.out / (level + ".psr")).write_bytes(wire_cache)
        running = row["kbps_cached"]
        results.append(
            dict(
                setting=level,
                init_bytes=len(wire_cache),
                init_sha256=sha(wire_cache),
                joint_bytes=len(wire_joint),
                cold_kbps=(len(wire_cache) + len(wire_joint)) * float(30000 / 1001) / 30000,
                seconds_to_fit_20kbps=8 * len(wire_cache) / (1000 * (20 - running)),
                image_crop_psnr=psnr(E, P),
                source_body_reference_psnr=psnr(F, N),
                rgb_lossless=E == 0,
                references=arms,
            )
        )
    result = dict(
        source=paid["source_sha256"],
        source_mask=support_sha,
        actual_original=sha(cache),
        rate_target_kbps=20,
        warm_run_kbps=row["kbps_cached"],
        image_metrics_only=True,
        inputs="two original frozen source references",
        fps=paid["native_fps"],
        tools=tools_sha,
        cpu_seconds=time.process_time() - start,
        rows=results,
    )
    assert result["cpu_seconds"] < 300
    a.result.write_text(json.dumps(result, indent=2) + "\n")
    print(
        [
            (
                r["setting"],
                r["init_bytes"],
                round(r["seconds_to_fit_20kbps"], 1),
                r["source_body_reference_psnr"],
            )
            for r in results
        ]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("paid", "features", "out", "result"):
        p.add_argument("--" + k, type=Path, required=True)
    main(p.parse_args())
