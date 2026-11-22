"""Build and independently decode the selected official TEST rally, preserving its native cadence."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

import cv2
import numpy as np

from experiments.tennis import common_player_packet as packet, common_receiver_manifest as manifest
from experiments.tennis import (
    k6b,
    k6b_rate_streams as rates,
    med_mixed_channel as mixed,
    w2_pose_rate,
)
from experiments.tennis.k4b_pose import player_masks
from src.codecs import svtav1
from src.pointstream import ball
from src.segmentation.masks import ClipMasks

from price_tracks import portable

HERE = Path(__file__).resolve().parent
FPS = Fraction(30000, 1001)
TOTAL = 240


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def input_config(event):
    if event == "alcaraz031":
        folder = HERE.parent / "six-match-source-panel-20261122"
        return dict(
            event=event,
            fps=Fraction(25),
            recipe=packet.RECIPE_25,
            receipt=folder / "alcaraz031-native-crop.json",
            roles={"far": "person:000000000006", "near": "person:000000000002"},
            units={
                "far": "alcaraz_hurkacz__028_at120__person_000000000004",
                "near": "alcaraz_hurkacz__028_at120__person_000000000006",
            },
        )
    if event == "match102":
        return dict(
            event=event,
            fps=FPS,
            recipe=packet.RECIPE_2997,
            receipt=HERE / "match102-result.json",
            roles={"far": "person:000000000001", "near": "person:000000000000"},
            units={"far": "match102_far", "near": "match102_near"},
        )
    raise ValueError("unsupported source case")


def signal(a, case):
    receipt = json.loads(case["receipt"].read_text())
    assert (
        sha(a.source) == receipt["source_sha256"]
        and sha(a.poses) == receipt["output_archive_sha256"]
    )
    assert sha(a.features) == receipt["prior_feature_archive_sha256"]
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    ok, first = cap.read()
    assert ok and abs(cap.get(cv2.CAP_PROP_FPS) - float(case["fps"])) < 0.001
    cap.release()
    rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    units = []
    with tarfile.open(a.poses) as packed:
        assert set(case["roles"].values()) == set(receipt["selected_ids"])
        for role, track in case["roles"].items():
            data = packed.extractfile("publish/" + track.replace(":", "_") + ".npz").read()
            assert digest(data) == receipt["tracks"][track]["npz_sha256"]
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                assert z["fitted"].all() and np.array_equal(z["t"], np.arange(TOTAL))
                units.append(dict(name=case["units"][role], t=z["t"], kp=z["kp"]))
    setting = w2_pose_rate.Setting(q=4, key=4, predictor="previous", interp="linear", vis="all")
    out = w2_pose_rate.encode_window(units, setting).expected
    names = [u["name"] for u in units]
    coded = portable(
        rates.encode_keypoints_kps2(
            {"native_frames": TOTAL}, [dict(name=n, **u) for n, u in zip(names, out, strict=True)]
        )
    )
    _, paid = packet._bounded_keypoints(coded, names)
    refs = {}
    for i, u in enumerate(paid):
        assert np.array_equal(out[i]["kp"], u["kp"])
        box = k6b.run_boxes(list(u["kp"]))[0]
        crop = k6b.crop_image(rgb, box)
        metadata = dict(
            unit=u["name"],
            reference_frame=0,
            reference_box=box.tolist(),
            pixels="full_prepared_conditioning_crop",
        )
        refs[f"ref{i}"] = portable(rates.encode_reference(crop, metadata))
        assert np.array_equal(packet._bounded_reference(refs[f"ref{i}"])[1], crop)
    return names, coded, refs, receipt


def raster_source(a):
    ff = svtav1.tool("ffmpeg")["path"]
    p = a.out / "source.yuv"
    subprocess.run(
        [
            ff,
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(a.source),
            "-vf",
            "scale=960:544:flags=area",
            "-pix_fmt",
            "yuv420p",
            "-frames:v",
            "240",
            "-f",
            "rawvideo",
            "-n",
            str(p),
        ],
        check=True,
        timeout=180,
    )
    assert p.stat().st_size == TOTAL * 960 * 816
    return np.memmap(p, np.uint8, "r", shape=(TOTAL, 816, 960))


def spatial_masks(a, receipt, pos):
    pre = f"publish/test__{receipt['event']}@120/"
    with tarfile.open(a.features) as archive:
        root = json.load(archive.extractfile(pre + "features.json"))
        masks = archive.extractfile(pre + "sam-masks.rle").read()
    path = a.out / "source-sam.rle"
    path.write_bytes(masks)
    clip = ClipMasks.load(path)
    tracks = receipt["selected_ids"]
    dst = np.empty((TOTAL, 544, 960), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for t in range(TOTAL):
        people = player_masks(clip.frames[t], root["aliases"], tracks)
        assert set(people) == set(tracks) and all(m.any() for m in people.values())
        m = (people[tracks[0]] | people[tracks[1]]).astype(np.uint8) * 255
        m = cv2.resize(m, (960, 544), interpolation=cv2.INTER_NEAREST)
        m = cv2.dilate(m, kernel)
        if pos[t] is not None:
            xy = tuple(int(k) for k in pos[t])
            cv2.circle(m, xy, 8, 255, -1)
        dst[t] = m
    return dst


def donor_base(a, src, masks):
    spec = importlib.util.spec_from_file_location(
        "donor", HERE.parent / "clean-court-rally-20261118/pilot.py"
    )
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    orb = cv2.ORB_create(nfeatures=8000)
    key0, desc0 = orb.detectAndCompute(cv2.cvtColor(src[0], cv2.COLOR_YUV2BGR_I420), 255 - masks[0])
    assert desc0 is not None

    def alignment(ticks):
        homs, counts = [], []
        for t in ticks:
            keys, desc = orb.detectAndCompute(
                cv2.cvtColor(src[t], cv2.COLOR_YUV2BGR_I420), 255 - masks[t]
            )
            assert desc is not None
            pair = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, desc0, k=2)
            valid = [m for m, n in pair if m.distance < 0.7 * n.distance]
            assert len(valid) >= 50
            h, keep = cv2.findHomography(
                np.float32([keys[m.queryIdx].pt for m in valid]),
                np.float32([key0[m.trainIdx].pt for m in valid]),
                cv2.RANSAC,
                2.5,
            )
            assert h is not None and int(keep.sum()) >= 50
            homs.append(h / h[2, 2])
            counts.append([int(t), len(valid), int(keep.sum())])
        return np.stack(homs), counts

    tick = np.linspace(0, 239, 3).round().astype(int)
    start = time.monotonic()
    hs, smoke = alignment(tick)
    test, _ = pilot.plate_for([pilot.planes(src[t])[0] for t in tick], masks[tick], hs, 1.0)
    forecast = (time.monotonic() - start) * 1.5 * 17 + 90
    assert test.shape == (544, 960) and forecast < 900
    tick = np.linspace(0, 239, 17).round().astype(int)
    hs, counts = alignment(tick)
    plates, holes = [], []
    for plane, scale in enumerate((1, 0.5, 0.5)):
        plate, unseen = pilot.plate_for(
            [pilot.planes(src[t])[plane] for t in tick], masks[tick], hs, scale
        )
        plates.append(plate)
        holes.append(unseen)
    target = np.memmap(a.out / "clean.yuv", np.uint8, "w+", shape=src.shape)
    target[:] = src
    written = []
    for t in range(TOTAL):
        h = np.array([[np.interp(t, tick, hs[:, y, x]) for x in range(3)] for y in range(3)])
        inv = np.linalg.inv(h)
        written.append(int(np.count_nonzero(masks[t])))
        for chan, (dst, plate, scale) in enumerate(
            zip(pilot.planes(target[t]), plates, (1, 0.5, 0.5), strict=True)
        ):
            bounds = tuple(reversed(dst.shape))
            mask = (
                masks[t]
                if scale == 1
                else cv2.resize(masks[t], bounds, interpolation=cv2.INTER_AREA)
            ) > 0
            mat = np.diag([scale, scale, 1.0])
            warped = cv2.warpPerspective(
                plate,
                mat @ inv @ np.linalg.inv(mat),
                bounds,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            dst[mask] = warped[mask]
            assert np.array_equal(dst[~mask], pilot.planes(src[t])[chan][~mask])
    target.flush()
    return dict(
        smoke=smoke,
        donors=counts,
        holes=holes,
        painted_area=written,
        forecast_s=forecast,
        mask_sha256=digest(masks.tobytes()),
        clean_yuv_sha256=sha(a.out / "clean.yuv"),
    )


def main(a, case=None):
    a.out.mkdir(exist_ok=False, parents=True)
    cv2.setNumThreads(1)
    case = case if case is not None else input_config(getattr(a, "event", "match102"))
    fps = case["fps"]
    names, coded, refs, receipt = signal(a, case)
    payload = a.ball.read_bytes()
    ball_receipt = getattr(a, "ball_receipt", None)
    if ball_receipt is None:
        if case["event"] != "match102":
            raise ValueError("new inputs require their independent source ball receipt")
        ball_receipt = (
            HERE.parent / "cohort-expansion-20261122/test-physical/match102-sender-retry.json"
        )
    br = json.loads(ball_receipt.read_text())
    assert (
        digest(payload) == br["payload_sha256"]
        and br["source_sha256"] == receipt["source_sha256"]
        and br["fps"] == str(fps)
    )
    pos, app = ball.decode_frames(payload, TOTAL, 960, 544)
    assert (
        ball.encode_frames(pos, app, 960, 544) == payload
        and sum(p is not None for p in pos) == br["present"]
    )
    source = raster_source(a)
    footprint = spatial_masks(a, receipt, pos)
    donor = donor_base(a, source, footprint)
    ff = svtav1.tool("ffmpeg")
    sub = a.out / "low.yuv"
    subprocess.run(
        [
            ff["path"],
            "-nostdin",
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
            str(fps),
            "-i",
            str(a.out / "clean.yuv"),
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
    resources = json.loads(
        (HERE.parent / "med-mixed-receiver-20261122/installed-resources.json").read_text()
    )
    resources["native_decoder"] = manifest.NATIVE
    profile = dict(
        mode="players",
        canvas_size=[1920, 1080],
        coded_size=[160, 96],
        recipe=case["recipe"],
        order=names,
        cuts=[],
        runs=[dict(unit=n, start=0, end=240, reference=f"ref{i}") for i, n in enumerate(names)],
        resources=resources,
    )
    rows = []
    enc, dec = svtav1.tool("SvtAv1EncApp"), svtav1.tool("dav1d")
    for crf in (70, 60, 50):
        d = a.out / f"crf{crf}"
        d.mkdir()
        native = svtav1.code(
            sub,
            d,
            width=160,
            height=96,
            fps=fps,
            frames=240,
            crf=crf,
            preset=4,
            threads=1,
            full_range=False,
            encoder=enc["path"],
            decoder=dec["path"],
            timeout=300,
        )
        base = (d / "stream.ivf").read_bytes()
        carrier = packet.build(profile, base, coded, refs)
        runtime = json.loads(manifest.encode(digest(carrier)))
        del runtime["carrier_sha256"], runtime["composition"]
        cache = packet.wire.pack(
            dict(type="joint-initialization-v1", runtime=runtime, resources=resources),
            [(k, "reference", v) for k, v in refs.items()],
        )
        jp = dict(
            kind="joint-trajectory-v1",
            player={k: v for k, v in profile.items() if k != "resources"},
            cache_sha256=digest(cache),
            ball_canvas=[960, 544],
            display=[1920, 1080],
            order=["clean_base", "ball", "far_player", "near_player"],
        )
        joint = packet.wire.pack(
            jp,
            [("base", "frame", base), ("poses", "frame", coded), ("trajectory", "frame", payload)],
        )
        for k, v in (
            ("players.pscp", carrier),
            ("players.manifest", manifest.encode(digest(carrier))),
            ("rally.psj", joint),
            ("initial.psr", cache),
        ):
            (d / k).write_bytes(v)
        checked = mixed.decode(
            d / "rally.psj", d / "initial.psr", d / "players.pscp", digest(carrier)
        )
        assert (
            checked["start"] == 0
            and checked["stop"] == 240
            and checked["base"] == base
            and checked["pos"] == pos
        )
        rows.append(
            dict(
                crf=crf,
                kbps_cached=len(joint) * float(fps) / 30000,
                kbps_cold=(len(joint) + len(cache)) * float(fps) / 30000,
                base_bytes=len(base),
                pose_bytes=len(coded),
                ball_bytes=len(payload),
                joint_bytes=len(joint),
                init_bytes=len(cache),
                reference_bytes={k: len(v) for k, v in refs.items()},
                carrier_sha256=digest(carrier),
                manifest_sha256=sha(d / "players.manifest"),
                base_sha256=digest(base),
                decoded_sha256=sha(Path(native["decoded"])),
                joint_sha256=digest(joint),
                init_sha256=digest(cache),
                native=native,
            )
        )
    chosen = min(rows, key=lambda r: (abs(r["kbps_cached"] - 20), r["kbps_cached"]))
    report = dict(
        source_sha256=receipt["source_sha256"],
        native_fps=str(fps),
        frames=240,
        foreground=[dict(unit=n) for n in names],
        donor=donor,
        rates=rows,
        selected_crf=chosen["crf"],
        tools=dict(svt=enc, dav1d=dec, ffmpeg=ff),
        source_opencv=cv2.__version__,
    )
    (a.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(chosen["crf"], chosen["kbps_cached"], chosen["kbps_cold"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("out", "features", "source", "poses", "ball"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--event", choices=("match102", "alcaraz031"), default="match102")
    ap.add_argument("--ball-receipt", type=Path)
    main(ap.parse_args())
