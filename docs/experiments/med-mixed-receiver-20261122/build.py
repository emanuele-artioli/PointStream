"""Build a complete 240-tick broadcast with one charged 120-tick two-player interval."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zlib

import cv2
import numpy as np
import zstandard

from experiments.tennis import common_player_packet as packet, common_receiver_manifest as manifest
from experiments.tennis import k6b, k6b_rate_streams as rates, w2_pose_rate
from src.codecs import svtav1
from src.pointstream import ball

HERE = Path(__file__).resolve().parent
SOURCES = {
    "000": "3bf7ddaae0db5c48b1072fccb15f9f162e41fa5999886b9a5bde58dc0495c630",
    "019": "1c13479eb3b6f8d175eb6f5020c629896e6b01f719014ec995a06b4c6c5a3e9c",
    "024": "4d602291ebea0fcb9698d86d3f93ce4c3076a13005e33079a82975b8b64605f2",
}
PREP = "39ddd9b9942131c9d283ed5be52e53a979406dd6b9c325fd18b1bddb2ceb5f1c"
TRANS = "70cfa5b37cbb579be788202148cf3fa88fafca0c45252ac02b91fd25f44fa0e2"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def portable(blob):
    magic, size, _, checksum = rates.HEADER.unpack_from(blob)
    raw = zstandard.ZstdDecompressor().decompress(blob[rates.HEADER.size :])
    assert len(raw) == size and hashlib.sha256(raw).digest() == checksum
    compressed = zlib.compress(raw, 6)
    return rates.HEADER.pack(magic, size, len(compressed), checksum) + compressed


def reference_image(source, tick):
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    try:
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 50
        for _ in range(tick + 1):
            ok, frame = cap.read()
            assert ok
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def poses(a):
    assert (
        sha(a.source) == SOURCES[a.clip] and sha(a.prepared) == PREP and sha(a.transport) == TRANS
    )
    prefix = f"publish/packets/medvedev_struff__{a.clip}@120/"
    with tarfile.open(a.transport) as archive:
        metadata, old = rates.decode_keypoints(archive.extractfile(prefix + "keypoints.kps").read())
        order = metadata["compositing_order"]
        by_suffix = {u["name"].split("_")[-1]: u for u in old}
        ordered = [by_suffix[k.split(":")[-1]] for k in order]
        assert metadata["fps"] == [50, 1] and len(ordered) == 2
        common = set.intersection(*(set(u["t"]) for u in ordered))
        start = next(i for i in range(121) if set(range(i, i + 120)) <= common)
        subset = []
        with tarfile.open(a.prepared) as prep:
            for u in ordered:
                with np.load(
                    io.BytesIO(prep.extractfile("publish/units/" + u["name"] + ".npz").read()),
                    allow_pickle=False,
                ) as z:
                    assert np.array_equal(z["t"], u["t"])
                mask = (u["t"] >= start) & (u["t"] < start + 120)
                subset.append(dict(name=u["name"], t=u["t"][mask], kp=u["kp"][mask]))
        encoded = w2_pose_rate.encode_window(
            subset,
            w2_pose_rate.Setting(q=4, key=4, predictor="previous", interp="linear", vis="all"),
        )
        names = [u["name"] for u in subset]
        expected = [
            dict(name=name, t=row["t"], kp=row["kp"])
            for name, row in zip(names, encoded.expected, strict=True)
        ]
        coded = rates.encode_keypoints_kps2({"native_frames": 240}, expected)
        _, actual = rates.decode_keypoints(coded)
        assert all(np.array_equal(a["kp"], b["kp"]) for a, b in zip(actual, expected, strict=True))
        refs = {}
        rows = []
        geometry = {}
        for i, (src, u) in enumerate(zip(subset, actual, strict=True)):
            name = u["name"]
            oldref = archive.extractfile(prefix + name + ".png.ref").read()
            rm, image = rates.decode_reference(oldref)
            assert bool(np.all((src["kp"][:, :, 2] > 0.5) == (u["kp"][:, :, 2] > 0.5)))
            boxes = k6b.run_boxes(list(u["kp"]))
            geometry[name] = [{"boxes": boxes}]
            if rm["reference_frame"] != start:
                pix = k6b.crop_image(reference_image(a.source, start), boxes[0])
                info = {k: v for k, v in rm.items() if k not in ("color", "codec", "quality")}
                info.update(reference_frame=start, reference_box=boxes[0].tolist())
                oldref = rates.encode_reference(pix, info)
                _, image = rates.decode_reference(oldref)
                assert np.array_equal(image, pix)
            refs[f"ref{i}"] = oldref
            err = np.linalg.norm(src["kp"][:, :, :2] - u["kp"][:, :, :2], axis=2)
            valid = src["kp"][:, :, 2] > 0.5
            rows.append(
                dict(
                    unit=name,
                    real_ticks=120,
                    first=start,
                    last=start + 119,
                    visible_coordinate_95=float(np.percentile(err[valid], 95)),
                    visible_coordinate_max=float(err[valid].max()),
                    reference_rgb_sha256=digest(np.ascontiguousarray(image)),
                    reference_bytes=len(oldref),
                )
            )
    return start, names, coded, refs, geometry, rows


def load_pilot():
    p = HERE.parent / "clean-court-rally-20261118/pilot.py"
    spec = importlib.util.spec_from_file_location("med_donor_pilot", p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cleaned(a, start, geometry, pos):
    pilot = load_pilot()
    ff = svtav1.tool("ffmpeg")["path"]
    source = a.out / "source.yuv"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(a.source),
            "-vf",
            "scale=960:544:flags=area,format=yuv420p",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            "240",
            "-f",
            "rawvideo",
            "-n",
            str(source),
        ],
        check=True,
        timeout=180,
    )
    v = np.memmap(source, np.uint8, "r", shape=(240, 816, 960))
    active = v[start : start + 120]
    coords = np.asarray(
        [
            (-1, -1)
            if p is None
            else (max(0, 2 * p[0] - 16), max(0, round(p[1] * 1080 / 544) - 16))
            for p in pos[start : start + 120]
        ]
    )
    masks = np.stack([pilot.mask_at(t, active[t], geometry, coords) for t in range(120)])
    orb = cv2.ORB_create(nfeatures=8000)
    keys0, d0 = orb.detectAndCompute(
        cv2.cvtColor(active[0], cv2.COLOR_YUV2BGR_I420), cv2.bitwise_not(masks[0])
    )
    if d0 is None:
        raise ValueError("insufficient court features on first source tick")

    def align(ticks):
        hs = []
        rows = []
        for t in ticks:
            keys, desc = orb.detectAndCompute(
                cv2.cvtColor(active[t], cv2.COLOR_YUV2BGR_I420), cv2.bitwise_not(masks[t])
            )
            if desc is None:
                raise ValueError("insufficient donor features")
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, d0, k=2)
            good = [
                x
                for pair in pairs
                if len(pair) == 2
                for x, y in [pair]
                if x.distance < 0.7 * y.distance
            ]
            a = np.float32([keys[x.queryIdx].pt for x in good])
            b = np.float32([keys0[x.trainIdx].pt for x in good])
            h, valid = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
            if h is None or valid is None or int(valid.sum()) < 50:
                raise ValueError("court motion minimum inlier count failed")
            hs.append(h / h[2, 2])
            rows.append(dict(local_tick=int(t), pairs=len(good), inliers=int(valid.sum())))
        return hs, rows

    tick = np.linspace(0, 119, 3).round().astype(int)
    before = time.monotonic()
    h, _ = align(tick)
    luma, _ = pilot.plate_for(
        [pilot.planes(active[t])[0] for t in tick], [masks[t] for t in tick], h, 1
    )
    forecast = 1.5 * (time.monotonic() - before) * 17 / 3 * 3 + 120
    if luma.shape != (544, 960) or forecast > 900:
        raise ValueError("donor pilot gate")
    ticks = np.linspace(0, 119, 17).round().astype(int)
    hom, rows = align(ticks)
    plates = []
    holes = []
    for plane, scale in enumerate((1, 0.5, 0.5)):
        p, miss = pilot.plate_for(
            [pilot.planes(active[t])[plane] for t in ticks], [masks[t] for t in ticks], hom, scale
        )
        plates.append(p)
        holes.append(miss)
    path = a.out / "mixed.yuv"
    mixed = np.memmap(path, np.uint8, "w+", shape=v.shape)
    mixed[:] = v
    for t in range(120):
        h = np.array(
            [[np.interp(t, ticks, [x[y, x0] for x in hom]) for x0 in range(3)] for y in range(3)]
        )
        inv = np.linalg.inv(h)
        for plane, (dst, plate, scale) in enumerate(
            zip(pilot.planes(mixed[start + t]), plates, (1, 0.5, 0.5), strict=True)
        ):
            matrix = np.diag([scale, scale, 1])
            support = (
                masks[t] > 0
                if scale == 1
                else cv2.resize(masks[t], (480, 272), interpolation=cv2.INTER_AREA) > 0
            )
            warped = cv2.warpPerspective(
                plate,
                matrix @ inv @ np.linalg.inv(matrix),
                (int(960 * scale), int(544 * scale)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            dst[support] = warped[support]
            assert np.array_equal(dst[~support], pilot.planes(active[t])[plane][~support])
    mixed.flush()
    assert np.array_equal(mixed[:start], v[:start]) and np.array_equal(
        mixed[start + 120 :], v[start + 120 :]
    )
    return dict(
        source_yuv_sha256=sha(source),
        mixed_yuv_sha256=sha(path),
        donors=rows,
        holes=holes,
        forecast_seconds=forecast,
    )


def main(a):
    cv2.setNumThreads(1)
    a.out.mkdir(parents=True, exist_ok=False)
    start, names, coded, refs, geom, rows = poses(a)
    if a.gate_only:
        print(json.dumps(dict(start=start, people=rows, keys_bytes=len(coded))))
        return
    s = json.loads(a.sender.read_text())
    payload = a.ball.read_bytes()
    assert (
        s["source_sha256"] == SOURCES[a.clip]
        and s["payload_sha256"] == digest(payload)
        and s["fps"] == "50"
    )
    pos, apps = ball.decode_frames(payload, 240, 960, 544)
    assert ball.encode_frames(pos, apps, 960, 544) == payload
    q = cleaned(a, start, geom, pos)
    pos = [p if start <= i < start + 120 else None for i, p in enumerate(pos)]
    apps = [p if start <= i < start + 120 else None for i, p in enumerate(apps)]
    symbols = ball.encode_frames(pos, apps, 960, 544)
    (a.out / "active.ball").write_bytes(symbols)
    ff = svtav1.tool("ffmpeg")["path"]
    sub = a.out / "base.yuv"
    subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            "960x544",
            "-r",
            "50",
            "-i",
            str(a.out / "mixed.yuv"),
            "-vf",
            "scale=160:96:flags=area",
            "-frames:v",
            "240",
            "-f",
            "rawvideo",
            "-n",
            str(sub),
        ],
        check=True,
        timeout=180,
    )
    profile = dict(
        mode="players",
        canvas_size=[1920, 1080],
        cuts=[],
        resources=json.loads((HERE / "installed-resources.json").read_text()),
    )
    profile.update(
        recipe=packet.RECIPE.copy(),
        coded_size=[160, 96],
        order=names,
        runs=[
            dict(unit=n, start=start, end=start + 120, reference=f"ref{i}")
            for i, n in enumerate(names)
        ],
    )
    profile["resources"]["native_decoder"] = manifest.NATIVE
    coded = portable(coded)
    refs = {key: portable(value) for key, value in refs.items()}
    original = tuple(refs.items())
    ffbin = svtav1.tool("SvtAv1EncApp")
    dbin = svtav1.tool("dav1d")
    configs = []
    for crf in (70, 60, 50):
        p = a.out / f"crf{crf}"
        p.mkdir()
        native = svtav1.code(
            sub,
            p,
            width=160,
            height=96,
            fps=Fraction(50),
            frames=240,
            crf=crf,
            preset=4,
            threads=1,
            full_range=False,
            encoder=ffbin["path"],
            decoder=dbin["path"],
            timeout=300,
        )
        base = (p / "stream.ivf").read_bytes()
        unnormalized = packet.build(profile, base, coded, refs)
        runtime = json.loads(manifest.encode(digest(unnormalized)))
        assert runtime.pop("carrier_sha256") == digest(unnormalized)
        assert (
            runtime.pop("composition") == "decoded-base,farthest-to-nearest-players,optional-ball"
        )
        cache_profile = dict(
            type="joint-initialization-v1",
            runtime=runtime,
            resources=profile["resources"],
        )
        cache = packet.wire.pack(cache_profile, [(k, "reference", v) for k, v in original])
        joint_profile = dict(
            kind="joint-trajectory-v1",
            player={k: v for k, v in profile.items() if k != "resources"},
            cache_sha256=digest(cache),
            ball_canvas=[960, 544],
            display=[1920, 1080],
            order=["clean_base", "ball", "far_player", "near_player"],
        )
        joint = packet.wire.pack(
            joint_profile,
            [("base", "frame", base), ("poses", "frame", coded), ("trajectory", "frame", symbols)],
        )
        deliv, _ = packet.wire.unpack(joint, joint_profile)
        assets, _ = packet.wire.unpack(cache, cache_profile)
        assert packet.build(profile, deliv["base"], deliv["poses"], assets) == unnormalized
        carrier = packet.build(profile, deliv["base"], deliv["poses"], assets)
        assert carrier == unnormalized
        _, _, port = packet.unpack(carrier)
        for n in names:
            assert np.array_equal(port[n][0]["t"], np.arange(start, start + 120))
        (p / "players.pscp").write_bytes(carrier)
        (p / "players.manifest").write_bytes(manifest.encode(digest(carrier)))
        (p / "initial.psr").write_bytes(cache)
        (p / "rally.psj").write_bytes(joint)
        configs.append(
            dict(
                crf=crf,
                kbps_cached=len(joint) / 600,
                kbps_cold=(len(joint) + len(cache)) / 600,
                base_bytes=len(base),
                pose_bytes=len(coded),
                ball_bytes=len(symbols),
                init_bytes=len(cache),
                joint_bytes=len(joint),
                carrier_sha256=digest(carrier),
                manifest_sha256=sha(p / "players.manifest"),
                base_sha256=digest(base),
                decoded_sha256=sha(Path(native["decoded"])),
                joint_sha256=digest(joint),
                init_sha256=digest(cache),
                native=native,
            )
        )
    chosen = min(configs, key=lambda c: (abs(c["kbps_cached"] - 21.5), c["kbps_cached"]))
    result = dict(
        source_sha256=SOURCES[a.clip],
        clip=a.clip,
        frames=240,
        native_fps=50,
        active=[start, start + 120],
        foreground=rows,
        donor=q,
        rates=configs,
        selected_crf=chosen["crf"],
        verified_source_denial_pending=True,
        tools={"svt": ffbin, "dav1d": dbin, "ffmpeg": svtav1.tool("ffmpeg")},
    )
    (a.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected": chosen["crf"], "kbps": chosen["kbps_cached"], "start": start}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", choices=SOURCES, required=True)
    p.add_argument("--gate-only", action="store_true")
    for key in ("source", "transport", "prepared", "sender", "ball", "out"):
        p.add_argument("--" + key, type=Path, required=key not in ("ball", "sender"))
    main(p.parse_args())
