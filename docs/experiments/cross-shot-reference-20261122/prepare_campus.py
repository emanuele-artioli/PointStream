"""Prepare fresh and reused-reference /002 conditioning on campus."""
import hashlib
import io
import json
import resource
import shutil
import struct
import tarfile
import time
import zlib
from pathlib import Path

import numpy as np
import zstandard
from PIL import Image

PREPARED = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T162058Z-c9b2c47b/full/published.tar")
TRANSPORT = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T171359Z-aba3845f/full/published.tar")
TEMPLATE = Path("/home/itec/emanuele/Datasets/pointstream-data/tennis/player-reference-common-avif-q65-20261122/config.json")
ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data/tennis/cross-shot-reference-20261122")
SCRATCH = Path("/local/users/emanuele/pointstream/crossshot-reference-preparation-20261122")
PREPARED_SHA = "94a5d6318fc113557fb24c8a32f2edffdf368f7a289484fddd4b19a749302bfa"
TRANSPORT_SHA = "d80fac2ab880743e100a90a0dbb46af6173bc2479389a6d8f75663578b9db017"
TEMPLATE_SHA = "0cd63fe70ce618be8ee9457ed69c266e19a50fa9ade751fbcdd7d74995c33869"
TARGETS = ["nadal_kyrgios__002_at120__person_000000000000",
           "nadal_kyrgios__002_at120__person_000000000002"]
SOURCES = ["nadal_kyrgios__000_at120__person_000000000000",
           "nadal_kyrgios__000_at120__person_000000000001"]
HEADER = struct.Struct(">4sII32s")


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def array_sha(array):
    data = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{data.dtype.str}:{data.shape}:".encode())
    h.update(data.tobytes())
    return h.hexdigest()


def read_member(tar, name):
    member = tar.extractfile(name)
    if member is None:
        raise ValueError(f"Missing {name}")
    return member.read()


def decode_ref(blob):
    magic, raw_size, coded_size, expected = HEADER.unpack_from(blob)
    assert magic == b"REF1" and len(blob) == HEADER.size + coded_size
    coded = blob[HEADER.size:]
    raw = zstandard.ZstdDecompressor().decompress(coded, max_output_size=raw_size) if coded.startswith(b"\x28\xb5\x2f\xfd") else zlib.decompress(coded)
    assert len(raw) == raw_size and hashlib.sha256(raw).digest() == expected
    n = struct.unpack_from(">I", raw)[0]
    meta = json.loads(raw[4:4+n])
    assert meta["codec"] == ".png"
    rgb = np.asarray(Image.open(io.BytesIO(raw[4+n:])).convert("RGB"))
    assert rgb.shape == (784, 512, 3)
    return meta, rgb


def main():
    begun = time.monotonic()
    for path, expected in ((PREPARED, PREPARED_SHA), (TRANSPORT, TRANSPORT_SHA), (TEMPLATE, TEMPLATE_SHA)):
        if sha_file(path) != expected:
            raise ValueError(f"Input changed: {path}")
    SCRATCH.mkdir(parents=True, exist_ok=False)
    output = {}
    refs = {}
    with tarfile.open(PREPARED) as prepared, tarfile.open(TRANSPORT) as transport:
        for name in SOURCES + TARGETS:
            group = "000" if name in SOURCES else "002"
            prefix = f"publish/packets/nadal_kyrgios__{group}@120/"
            blob = read_member(transport, prefix + name + ".png.ref")
            meta, rgb = decode_ref(blob)
            with np.load(io.BytesIO(read_member(prepared, f"publish/units/{name}.npz")), allow_pickle=False) as arrays:
                if not np.array_equal(arrays["reference"], rgb):
                    raise ValueError(f"Packet decode differs from W7 conditioning for {name}")
            refs[name] = rgb
            output[name] = {"reference_packet_bytes": len(blob), "packet_sha256": sha_bytes(blob),
                            "decoded_array_sha256": array_sha(rgb), "metadata": meta}
        pair_rows = []
        for condition in ("fresh", "reuse"):
            (SCRATCH / condition / "publish").mkdir(parents=True)
        for target, source in zip(TARGETS, SOURCES):
            raw = read_member(prepared, f"publish/units/{target}.npz")
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            unchanged = {key: array_sha(value) for key, value in arrays.items() if key != "reference"}
            fresh = SCRATCH / "fresh" / "publish" / (target + ".npz")
            fresh.write_bytes(raw)
            assert np.array_equal(arrays["reference"], refs[target])
            arrays["reference"] = refs[source]
            reuse = SCRATCH / "reuse" / "publish" / (target + ".npz")
            np.savez_compressed(reuse, **arrays)
            with np.load(reuse, allow_pickle=False) as verify:
                assert all(array_sha(verify[key]) == expected for key, expected in unchanged.items())
                assert np.array_equal(verify["reference"], refs[source])
                assert np.array_equal(verify["t"][:24], np.arange(24))
                assert np.all(verify["run"][:24] == 0)
            pair_rows.append({"target": target, "source": source, "fresh_npz_sha256": sha_file(fresh),
                              "reuse_npz_sha256": sha_file(reuse), "unchanged": unchanged})
    configs = {}
    for condition in ("fresh", "reuse"):
        base = SCRATCH / condition
        archive = base / "conditioning.tar"
        with tarfile.open(archive, "w") as tar:
            for path in sorted((base / "publish").iterdir()):
                tar.add(path, arcname=f"publish/{path.name}")
        config = json.loads(TEMPLATE.read_text())
        # A truthy reference_pilot enables the production reference smoke cost gate.
        config["reference_pilot"] = {"kind": "cross-shot", "condition": condition}
        config["source"].update(window="nadal_kyrgios/002@120",
            source_video_sha256="f0551443eaff4e4b3539f21cd0c6ac66f55601d480ad099a571e8626db735e4a",
            provenance="W7 Tier2 /002, broadcast frames 87691–87930, opened development.")
        config["conditioning"]["provenance"] = (
            f"Paired cross-shot {condition} condition. Target W7 /002 pose, boxes, times and seeds are identical. "
            "Reference pixels are decoded from archived PNG REF1 packets; the reused reference is from /000."
        )
        config["units"] = [{"unit": row["target"],
                            "path": f'staged:conditioning/publish/{row["target"]}.npz',
                            "sha256": row[f"{condition}_npz_sha256"],
                            "review_label": f"Track {i + 1}", "run": 0, "start_t": 0}
                           for i, row in enumerate(pair_rows)]
        config_path = base / "config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        configs[condition] = {"config_sha256": sha_file(config_path), "conditioning_sha256": sha_file(archive),
                              "conditioning_bytes": archive.stat().st_size}
    ROOT.mkdir(parents=True, exist_ok=False)
    for condition in ("fresh", "reuse"):
        target_dir = ROOT / condition
        target_dir.mkdir()
        for name in ("config.json", "conditioning.tar"):
            shutil.copyfile(SCRATCH / condition / name, target_dir / name)
            if sha_file(target_dir / name) != sha_file(SCRATCH / condition / name):
                raise ValueError("Staging hash mismatch")
    fresh_bytes = sum(output[name]["reference_packet_bytes"] for name in TARGETS)
    reuse_bytes = sum(output[name]["reference_packet_bytes"] for name in SOURCES)
    result = {"prepared_archive_sha256": PREPARED_SHA, "transport_archive_sha256": TRANSPORT_SHA,
              "template_sha256": TEMPLATE_SHA, "units": output, "pairs": pair_rows, "configs": configs,
              "fresh_pair_ref1_bytes": fresh_bytes, "previous_pair_ref1_bytes": reuse_bytes,
              "mapping_signal": "Not yet serialized; 1 byte is a modeled two-track permutation.",
              "wall_seconds": time.monotonic()-begun,
              "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime}
    (ROOT / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
