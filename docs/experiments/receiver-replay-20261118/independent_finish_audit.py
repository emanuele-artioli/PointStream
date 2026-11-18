"""Authenticate a published source-denied player finish and rebuild its pixels on CPU."""

import argparse
import hashlib
import io
import json
import tarfile
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from independent_crop_audit import (
    CARRIER,
    MANIFEST,
    array_frames,
    array_sha,
    file_sha,
    members,
    read_small,
    require,
    tar_data,
)

BASE = "7490edf9b678db805c23baae49a2a3b5e9d7b037b18dd216320aad47a061beff"
SAM = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
NATIVE = "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2"
SAM_ENV = "32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661"
UNITS = ["nadal_kyrgios__000_at120__person_" + s for s in ["000000000001", "000000000000"]]


def base_canvas(frame):
    # Independent evaluation of the specified limited-range BT.709 transform, including rounding.
    height, width = 272, 480
    y = (frame[:height].astype(np.float32) - 16) / 219
    uv = frame[height:].reshape(2, height // 2, width // 2).astype(np.float32)
    u, v = ((uv - 128) / 224).repeat(2, axis=1).repeat(2, axis=2)
    red = y + 2 * (1 - 0.2126) * v
    blue = y + 2 * (1 - 0.0722) * u
    green = (y - 0.2126 * red - 0.0722 * blue) / (1 - 0.2126 - 0.0722)
    small = np.rint(np.stack((red, green, blue), axis=-1) * 255).clip(0, 255).astype(np.uint8)
    return cv2.resize(small, (1920, 1080), interpolation=cv2.INTER_CUBIC)


def overlay(canvas, support, crop, box, mask):
    x, y, w, h = (round(float(v)) for v in box)
    w, h = max(1, w), max(1, h)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    count = 0
    if x0 < x1 and y0 < y1 and mask.any():
        inside = np.s_[y0 - y : y1 - y, x0 - x : x1 - x]
        region = np.s_[y0:y1, x0:x1]
        keep = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)[
            inside
        ].astype(bool)
        pixels = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)[inside]
        np.copyto(canvas[region], pixels, where=keep[:, :, None])
        support[region] |= keep
        count = int(keep.sum())
    status = "composed" if count else "offscreen" if mask.any() else "empty_mask"
    return count, status


def bind_generation(stack, archive_path, audit_path, report):
    audited = json.loads(audit_path.read_text())
    unit = audited["unit"]
    require(
        audited["stage"] == "full" and audited["exact_conditioning_and_geometry"],
        "generation requires audited 240 native ticks",
    )
    require(file_sha(archive_path) == audited["new_archive_sha256"], "generation archive identity")
    with tarfile.open(archive_path) as archive:
        raw = tar_data(archive, "publish/" + unit + ".npz")
        receipt = tar_data(archive, "publish/generation.json")
    require(
        hashlib.sha256(raw).hexdigest()
        == audited["new_npz_sha256"]
        == report["generated"][unit]["sha256"],
        "paid crop identity",
    )
    require(
        hashlib.sha256(receipt).hexdigest()
        == audited["new_receipt_sha256"]
        == report["generation_receipts"][unit]["sha256"],
        "paid generation receipt",
    )
    zipped = stack.enter_context(zipfile.ZipFile(io.BytesIO(raw)))
    boxes = read_small(zipped, "box")
    require(
        array_sha(boxes) == audited["box_sha256"]
        and np.array_equal(read_small(zipped, "t"), np.arange(240)),
        "paid crop placement",
    )
    return unit, (
        boxes,
        stack.enter_context(array_frames(zipped, "crops", (240, 784, 512, 3), np.uint8)),
    )


def audit(args):
    cpu, start = time.process_time(), time.monotonic()
    cv2.setNumThreads(1)
    require(file_sha(args.archive) == args.archive_sha256, "finish archive identity")
    ticks = [0, 11, 23, 43] if args.stage == "smoke" else list(range(240))
    checked = []
    with tarfile.open(args.archive) as tar, ExitStack() as stack:
        names = {
            "publish/finish.json",
            "publish/base.yuv",
            "publish/frames.npy",
            *("publish/" + u + "-masks.npz" for u in UNITS),
        }
        require(set(members(tar)) == names, "finish archive membership")
        report = json.loads(tar_data(tar, "publish/finish.json"))
        isolation = report["isolation"]
        require(
            report["ticks"] == ticks
            and [r["tick"] for r in report["rows"]] == ticks
            and report["order"] == UNITS,
            "paid timeline and far-to-near order",
        )
        require(
            (
                report["carrier_sha256"],
                report["manifest_sha256"],
                report["carrier_bytes"],
                report["manifest_bytes"],
                report["total_receiver_bytes"],
            )
            == (CARRIER, MANIFEST, 391798, 625, 392423),
            "paid receiver bytes",
        )
        require(
            report["sam_weight"] == SAM and report["native_sha256"] == NATIVE and report["kernels"],
            "SAM, native decoder and kernels",
        )
        require(
            isolation["runtime_archive_sha256"] == SAM_ENV
            and isolation["manifest_sha256"] == MANIFEST
            and isolation["interfaces"] == ["lo"]
            and isolation["landlock_abi"] >= 3
            and isolation["mutation_denied"]
            and not isolation["extra_fds"]
            and isolation["pre_sandbox_threads"] == 1
            and len(isolation["denials"]) >= 4
            and set(isolation["denials"].values()) <= {1, 13},
            "source denial and manifest",
        )
        require(
            report["gpu"]["uuid"] == report["gpu"]["visible_uuid"]
            and (report["gpu"]["major"], report["gpu"]["minor"]) == (8, 9),
            "physical Ada",
        )
        generations = dict(
            bind_generation(stack, archive, audit_path, report)
            for archive, audit_path in args.generation
        )
        require(set(generations) == set(UNITS), "two audited distinct units")
        native = tar_data(tar, "publish/base.yuv")
        require(
            hashlib.sha256(native).hexdigest() == report["yuv_sha256"] == BASE,
            "native YUV identity",
        )
        native = np.frombuffer(native, np.uint8).reshape(240, 408, 480)
        masks = {}
        for unit in UNITS:
            data = tar_data(tar, "publish/" + unit + "-masks.npz")
            require(
                hashlib.sha256(data).hexdigest() == report["mask_files"][unit], "SAM mask identity"
            )
            zipped = stack.enter_context(zipfile.ZipFile(io.BytesIO(data)))
            require(
                set(zipped.namelist()) == {"masks.npy", "t.npy"}
                and np.array_equal(read_small(zipped, "t"), ticks),
                "SAM tick coverage",
            )
            masks[unit] = stack.enter_context(
                array_frames(zipped, "masks", (len(ticks), 784, 512), np.bool_)
            )
        with tar.extractfile("publish/frames.npy") as stream:
            require(
                hashlib.file_digest(stream, "sha256").hexdigest() == report["frames_file_sha256"],
                "published canvas file identity",
            )
        with tar.extractfile("publish/frames.npy") as stream:
            require(np.lib.format.read_magic(stream) == (1, 0), "canvas NPY header")
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            require(
                shape == (len(ticks), 1080, 1920, 3) and not fortran and dtype == np.uint8,
                "canvas layout",
            )
            row_map = {r["tick"]: r for r in report["rows"]}
            for t in range(ticks[-1] + 1):
                crops = {u: next(generations[u][1]) for u in UNITS}
                if t not in row_map:
                    continue
                row = row_map[t]
                base = base_canvas(native[t])
                require(
                    array_sha(base) == row["base_sha256"],
                    f"native RGB differs at tick {t} on OpenCV {cv2.__version__}",
                )
                canvas = np.copy(base)
                support = np.zeros(base.shape[:2], bool)
                status, projected = {}, {}
                for unit in UNITS:
                    projected[unit], status[unit] = overlay(
                        canvas, support, crops[unit], generations[unit][0][t], next(masks[unit])
                    )
                require(
                    row["projected_pixels"] == projected
                    and row["status"] == status
                    and row["coverage_pixels"] == int(support.sum()),
                    f"mask geometry or fallback at {t}",
                )
                actual = np.frombuffer(stream.read(base.nbytes), np.uint8).reshape(base.shape)
                require(array_sha(actual) == row["frame_sha256"], f"frame hash at {t}")
                require(np.array_equal(actual, canvas), f"independent RGB recomposition at {t}")
                checked.append(
                    {
                        "tick": t,
                        "union_pixels": int(support.sum()),
                        "projected_pixels": projected,
                        "status": status,
                        "frame_sha256": row["frame_sha256"],
                    }
                )
            require(not stream.read(1), "extra canvas bytes")
    result = {
        "job_id": args.job,
        "archive_sha256": args.archive_sha256,
        "stage": args.stage,
        "scope": "Sealed source-denied player-only canvas integrity and independent pixel recomposition",
        "ticks": len(ticks),
        "all_frames_exact": True,
        "outside_masks_changed_pixels": 0,
        "rgb_values_compared": len(ticks) * 1080 * 1920 * 3,
        "carrier_sha256": CARRIER,
        "manifest_sha256": MANIFEST,
        "native_yuv_sha256": BASE,
        "sam_sha256": SAM,
        "rows": checked,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.monotonic() - start,
        "audit_script_sha256": file_sha(Path(__file__)),
    }
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"{len(ticks)} exact frames, {result['rgb_values_compared']} RGB values, {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument(
        "--generation",
        type=Path,
        nargs=2,
        action="append",
        metavar=("SEALED_TAR", "INDEPENDENT_FULL_CROP_AUDIT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())
