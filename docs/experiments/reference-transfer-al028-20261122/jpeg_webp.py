"""Prepare alternative JPEG and exactly lossless WebP envelopes from the same receiver images."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

import prepare as avif
from experiments.tennis import common_player_packet as packet, common_receiver_manifest as manifest
from experiments.tennis import w8_payload as wire


def main(a):
    sha = avif.sha
    original, cache = a.carrier.read_bytes(), a.cache.read_bytes()
    assert sha(original) == avif.ORIGINAL and sha(cache) == avif.CACHE
    profile = packet._embedded_profile(original)
    sections, _, runs = packet.unpack(original)
    init_profile = packet._embedded_profile(cache)
    refs, _ = wire.unpack(cache, init_profile)
    a.output.mkdir(exist_ok=False, parents=True)
    enc, dec = [shutil.which(x) for x in ("cwebp", "dwebp")]
    cvlib = next(Path(cv2.__file__).parent.glob("cv2*.so"))
    totals = {}
    for level in ("jpeg40", "jpeg70", "webp"):
        p = a.output / level
        p.mkdir()
        stored, meta, internal, rows = {}, {}, {}, {}
        for name in sorted(refs):
            m, image = packet._bounded_transport(refs[name], b"REF1", packet.MAX_REF)
            assert (m, image) == packet._bounded_transport(sections[name], b"REF1", packet.MAX_REF)
            pix = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            if level == "webp":
                raw, mid, decoded = (
                    p / (name + ".png"),
                    p / (name + ".webp"),
                    p / (name + "-decoded.png"),
                )
                raw.write_bytes(image)
                subprocess.run(
                    [
                        enc,
                        "-lossless",
                        "-exact",
                        "-m",
                        "6",
                        "-metadata",
                        "none",
                        str(raw),
                        "-o",
                        str(mid),
                    ],
                    check=True,
                    capture_output=True,
                )
                subprocess.run([dec, str(mid), "-o", str(decoded)], check=True, capture_output=True)
                blob = mid.read_bytes()
                decoded_bgr = cv2.imread(str(decoded))
                assert np.array_equal(pix, decoded_bgr)
            else:
                q = int(level.removeprefix("jpeg"))
                ok, coded = cv2.imencode(".jpg", pix, [cv2.IMWRITE_JPEG_QUALITY, q])
                assert ok
                blob = coded.tobytes()
                decoded_bgr = cv2.imdecode(coded, cv2.IMREAD_COLOR)
            stored[name], meta[name] = blob, m
            ok, png = cv2.imencode(".png", decoded_bgr)
            assert ok
            rgb_hash = sha(decoded_bgr[:, :, ::-1].tobytes())
            internal[name] = avif.refblob(m, png.tobytes())
            rows[name] = dict(
                image_bytes=len(blob), image_sha256=sha(blob), decoded_rgb_sha256=rgb_hash
            )
        envelope_profile = dict(
            init_profile,
            reference_codec=level,
            reference_container="images-with-ref1-metadata",
            reference_decoder_sha256=sha(Path(dec if level == "webp" else cvlib).read_bytes()),
            reference_metadata=meta,
        )
        envelope = wire.pack(envelope_profile, [(k, "reference", v) for k, v in stored.items()])
        recovered, _ = wire.unpack(envelope, packet._embedded_profile(envelope))
        assert recovered == stored
        (p / "initial.psr").write_bytes(envelope)
        carrier = packet.build(profile, sections["base"], sections["keypoints"], internal)
        checked, _, cond = packet.unpack(carrier)
        assert checked["base"] == sections["base"] and checked["keypoints"] == sections["keypoints"]
        for name, rec in meta.items():
            u = rec["unit"]
            one, other = runs[u][0], cond[u][0]
            assert sha(other["reference"].tobytes()) == rows[name]["decoded_rgb_sha256"]
            assert one["seed"] == other["seed"] and all(
                np.array_equal(one[k], other[k]) for k in ("t", "boxes", "kp", "reference_box")
            )
            assert all(
                np.array_equal(packet.pose_for_tick(one, t), packet.pose_for_tick(other, t))
                for t in range(240)
            )
        mf = manifest.encode(sha(carrier))
        (p / "players.pscp").write_bytes(carrier)
        (p / "players.manifest").write_bytes(mf)
        manifest.verify(p / "players.manifest", sha(carrier))
        totals[level] = dict(
            rows=rows,
            image_bytes=sum(z["image_bytes"] for z in rows.values()),
            envelope_bytes=len(envelope),
            envelope_sha256=sha(envelope),
            carrier_sha256=sha(carrier),
            manifest_sha256=sha(mf),
        )
    oracle = json.loads((Path(__file__).parent / "image-oracle.json").read_text())
    lo, mid, hi = (
        totals["jpeg40"]["image_bytes"],
        totals["jpeg70"]["image_bytes"],
        oracle["avif_image_bytes"],
    )
    admit70 = lo + 0.15 * (hi - lo) <= mid <= hi - 0.15 * (hi - lo) and lo < hi
    result = dict(
        rows=totals,
        admit_jpeg70=admit70,
        bytes_q40_q70_q90=[lo, mid, hi],
        source_sha256=avif.ORIGINAL,
        opencv=cv2.__version__,
        opencv_lib_sha256=sha(cvlib.read_bytes()),
        cwebp_sha256=sha(Path(enc).read_bytes()),
        dwebp_sha256=sha(Path(dec).read_bytes()),
        code_sha256=sha(Path(__file__).read_bytes()),
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["bytes_q40_q70_q90"], admit70, totals["webp"]["image_bytes"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("carrier", "cache", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    main(p.parse_args())
