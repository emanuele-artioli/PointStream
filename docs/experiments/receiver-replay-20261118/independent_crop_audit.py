"""Source-aligned independent checks for the selected lossless crop receiver."""

import argparse
import contextlib
import hashlib
import io
import json
import resource
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
STAGE_TICKS = {"smoke": 44, "full": 240}
CARRIER = "3dd4aa753dfa3d15a6fdbd1b368d6b48929b4eb8c933e2217a33bed87f735bfd"
MANIFEST = "b9eae3cef99a5139cf8db4745809db40345ec2aaebc5d82f3164c18d46f020a8"


def require(value, label):
    if not value:
        raise ValueError(label)


def file_sha(path):
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def array_sha(array):
    value = hashlib.sha256(f"{array.dtype.str}:{array.shape}:".encode())
    value.update(np.ascontiguousarray(array))
    return value.hexdigest()


def members(tar):
    entries = tar.getmembers()
    regular = [m for m in entries if m.isfile()]
    require(all(m.isfile() or m.isdir() for m in entries), "archive contains non-regular target")
    require(len({m.name for m in regular}) == len(regular), "repeated member")
    require(all(0 <= m.size <= 2_000_000_000 for m in regular), "archive member bounds")
    return {m.name: m for m in regular}


def tar_data(tar, name):
    with tar.extractfile(name) as member:
        return member.read()


def read_small(zipped, key):
    with zipped.open(key + ".npy") as f:
        return np.lib.format.read_array(f, allow_pickle=False)


@contextlib.contextmanager
def array_frames(zipped, key, expected_shape, expected_type):
    with zipped.open(key + ".npy") as f:
        require(np.lib.format.read_magic(f) == (1, 0), "NPY header version")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        require(not fortran and shape == expected_shape and dtype == expected_type, key + " layout")
        frame_bytes = int(np.prod(shape[1:])) * dtype.itemsize

        def frames():
            for _ in range(shape[0]):
                row = f.read(frame_bytes)
                require(len(row) == frame_bytes, "truncated " + key)
                yield np.frombuffer(row, dtype=dtype).reshape(shape[1:])
            require(not f.read(1), "extra " + key + " data")

        yield frames()


def old_crop(previous, unit, stage, player):
    record_path = EXPERIMENTS / "player-motion-amendment-20261117" / f"player-{player}-audit.json"
    record = json.loads(record_path.read_text())
    frozen = record["stages"][stage]
    require(file_sha(previous) == frozen["archive_sha256"], "original archive hash")
    old_name = f"{unit}@tennis_only.npz"
    with tarfile.open(previous) as archive:
        sealed = members(archive)
        for name, item in frozen["files"].items():
            require(sealed["publish/" + name].size == item["bytes"], "original size " + name)
        crop_data = tar_data(archive, "publish/" + old_name)
        receipt_data = tar_data(archive, "publish/receipt.json")
        require(
            hashlib.sha256(crop_data).hexdigest() == frozen["files"][old_name]["sha256"],
            "original NPZ hash",
        )
        require(
            hashlib.sha256(receipt_data).hexdigest() == frozen["files"]["receipt.json"]["sha256"],
            "original receipt",
        )
    receipt = json.loads(receipt_data)
    (entry,) = receipt["plan"]["units"]
    require(
        entry["unit"] == unit and entry["profile"] == "tennis_only"
        if "profile" in entry
        else entry["unit"] == unit,
        "original unit",
    )
    require(
        receipt["profile"] == "tennis_only" and receipt["native_frames"] == STAGE_TICKS[stage],
        "original run context",
    )
    return record["job_id"], entry, crop_data


def audit(args):
    cpu, wall = time.process_time(), time.monotonic()
    n = STAGE_TICKS[args.stage]
    unit = "nadal_kyrgios__000_at120__person_" + (
        "000000000000" if args.player == 1 else "000000000001"
    )
    history_job, original, old_bytes = old_crop(args.previous, unit, args.stage, args.player)
    shape = (n, 784, 512, 3)
    old_npz_sha = hashlib.sha256(old_bytes).hexdigest()
    with zipfile.ZipFile(io.BytesIO(old_bytes)) as old_zip:
        old_t = read_small(old_zip, "t")
        old_box = read_small(old_zip, "box")
        require(
            np.array_equal(old_t, np.arange(n)) and array_sha(old_box) == original["box_sha256"],
            "original tick/box join",
        )
        require(set(old_zip.namelist()) == {"t.npy", "box.npy", "crops.npy"}, "old NPZ membership")
        report = {
            "unit": unit,
            "stage": args.stage,
            "ticks": n,
            "original_job": history_job,
            "original_archive_sha256": file_sha(args.previous),
            "original_npz_sha256": old_npz_sha,
            "reference_sha256": original["reference_sha256"],
            "pose_sha256": original["pose_sha256"],
            "box_sha256": array_sha(old_box),
            "t_sha256": array_sha(old_t),
            "seed": original["seed"],
        }
        if args.archive is None:
            with array_frames(old_zip, "crops", shape, np.uint8) as old:
                hashes = [array_sha(a) for a in old]
            report.update(scope="original evidence authentication", original_frame_sha256=hashes)
        else:
            require(
                bool(args.job and args.archive_sha256),
                "new job and trusted stage archive hash required",
            )
            require(file_sha(args.archive) == args.archive_sha256, "new archive hash")
            with tarfile.open(args.archive) as fresh:
                require(
                    set(members(fresh)) == {"publish/generation.json", f"publish/{unit}.npz"},
                    "fresh archive membership",
                )
                raw = tar_data(fresh, "publish/generation.json")
                seal = json.loads(raw)
                new_bytes = tar_data(fresh, f"publish/{unit}.npz")
            (row,) = seal["outputs"]
            npz_sha = hashlib.sha256(new_bytes).hexdigest()
            require(
                row["unit"] == unit
                and row["seed"] == original["seed"]
                and row["crops_file_sha256"] == npz_sha,
                "fresh unit, seed or NPZ hash",
            )
            require(
                seal["carrier_sha256"] == CARRIER
                and seal["isolation"]["manifest_sha256"] == MANIFEST
                and seal["frames"] == n,
                "fresh lossless carrier scope",
            )
            require(
                seal["isolation"]["interfaces"] == ["lo"]
                and len(seal["isolation"]["denials"]) >= 4
                and all(e in (1, 13) for e in seal["isolation"]["denials"].values()),
                "fresh source denial",
            )
            expected = {k: report[k + "_sha256"] for k in ("pose", "reference", "t", "box")}
            require(
                row["array_sha256"] == expected and row["kernels"]["kernel_launches"] > 0,
                "conditioning or execution provenance differs",
            )
            diffs = []
            pixel_hashes = []
            total = 0
            with zipfile.ZipFile(io.BytesIO(new_bytes)) as new_zip:
                require(set(new_zip.namelist()) == set(old_zip.namelist()), "fresh NPZ membership")
                require(
                    np.array_equal(old_t, read_small(new_zip, "t"))
                    and np.array_equal(old_box, read_small(new_zip, "box")),
                    "fresh tick/box join",
                )
                with (
                    array_frames(old_zip, "crops", shape, np.uint8) as old,
                    array_frames(new_zip, "crops", shape, np.uint8) as new,
                ):
                    for t, (a, b) in enumerate(zip(old, new, strict=True)):
                        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
                        counts = int(np.count_nonzero(delta))
                        summed = int(delta.sum(dtype=np.int64))
                        total += summed
                        diffs.append([t, counts, summed, int(delta.max())])
                        pixel_hashes.append(array_sha(b))
            changed = sum(row[1] for row in diffs)
            exact = sum(row[1] == 0 for row in diffs)
            report.update(
                scope="new source-denied crop pixels against original same-length render",
                job_id=args.job,
                new_archive_sha256=args.archive_sha256,
                new_receipt_sha256=hashlib.sha256(raw).hexdigest(),
                new_npz_sha256=npz_sha,
                ticks_with_exact_rgb=exact,
                channels_compared=int(np.prod(shape)),
                different_channel_values=changed,
                mean_absolute_channel_delta=total / int(np.prod(shape)),
                maximum_channel_delta=max(row[3] for row in diffs),
                exact_conditioning_and_geometry=True,
                all_pixels_equal=exact == n,
                per_tick_delta_columns=[
                    "tick",
                    "differing_rgb_values",
                    "absolute_error_sum",
                    "max_absolute_error",
                ],
                per_tick_deltas=diffs,
                new_frame_sha256=pixel_hashes,
            )
    report.update(
        wall_seconds=time.monotonic() - wall,
        cpu_seconds=time.process_time() - cpu,
        peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        audit_script_sha256=file_sha(Path(__file__)),
    )
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(args.output, report.get("ticks_with_exact_rgb", n), "/", n, report["scope"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGE_TICKS, required=True)
    p.add_argument("--player", type=int, choices=(1, 2), required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    p.add_argument("--archive-sha256")
    p.add_argument("--job")
    audit(p.parse_args())
