"""Serialize and independently replay the two-shot W7 appearance-reference channel."""
import hashlib
import io
import json
import struct
import sys
import tarfile
import zlib
from pathlib import Path

import numpy as np
import zstandard
from PIL import Image

ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data/tennis/cross-shot-reference-20261122")
TRANSPORT = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T171359Z-aba3845f/full/published.tar")
TRANSPORT_SHA = "d80fac2ab880743e100a90a0dbb46af6173bc2479389a6d8f75663578b9db017"
OUTPUT = ROOT / "reference-channel-v1"
SOURCES = ["nadal_kyrgios__000_at120__person_000000000000",
           "nadal_kyrgios__000_at120__person_000000000001"]
TARGETS = ["nadal_kyrgios__002_at120__person_000000000000",
           "nadal_kyrgios__002_at120__person_000000000002"]
REF_HEADER = struct.Struct(">4sII32s")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pack(meta, payloads):
    metadata = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    packet = b"PSR1" + struct.pack(">I", len(metadata)) + metadata
    for payload in payloads:
        packet += struct.pack(">I", len(payload)) + payload
    return packet + hashlib.sha256(packet).digest()


def unpack(packet):
    if len(packet) < 40 or packet[:4] != b"PSR1" or hashlib.sha256(packet[:-32]).digest() != packet[-32:]:
        raise ValueError("Reference channel integrity failure")
    size = struct.unpack_from(">I", packet, 4)[0]
    if size > 8192 or 8 + size > len(packet) - 32:
        raise ValueError("Invalid channel header")
    meta = json.loads(packet[8:8 + size])
    position = 8 + size
    payloads = []
    for _ in range(meta["payload_count"]):
        if position + 4 > len(packet) - 32:
            raise ValueError("Missing payload length")
        n = struct.unpack_from(">I", packet, position)[0]
        position += 4
        if n > 4 * 1024 * 1024 or position + n > len(packet) - 32:
            raise ValueError("Invalid reference length")
        payloads.append(packet[position:position+n])
        position += n
    if position != len(packet) - 32 or meta["version"] != 1 or meta["decoder"] != "REF1-PNG-RGB8":
        raise ValueError("Invalid packet ending or decoder contract")
    if [sha(payload) for payload in payloads] != meta["payload_sha256"]:
        raise ValueError("Reference payload hash differs")
    return meta, payloads


def ref_decode(blob):
    magic, raw_size, coded_size, expected = REF_HEADER.unpack_from(blob)
    if magic != b"REF1" or len(blob) != REF_HEADER.size + coded_size or raw_size > 8 * 1024 * 1024:
        raise ValueError("Invalid REF1 framing")
    coded = blob[REF_HEADER.size:]
    raw = zstandard.ZstdDecompressor().decompress(coded, max_output_size=raw_size) if coded.startswith(b"\x28\xb5\x2f\xfd") else zlib.decompress(coded)
    if len(raw) != raw_size or hashlib.sha256(raw).digest() != expected:
        raise ValueError("Invalid REF1 payload integrity")
    n = struct.unpack_from(">I", raw)[0]
    meta = json.loads(raw[4:4+n])
    if meta["codec"] != ".png":
        raise ValueError("REF1 decoder contract requires PNG")
    pixels = np.asarray(Image.open(io.BytesIO(raw[4+n:])).convert("RGB"))
    if pixels.shape != (784, 512, 3) or pixels.dtype != np.uint8:
        raise ValueError("Invalid reference dimensions")
    return meta, pixels


def member(tar, name):
    source = tar.extractfile(name)
    if source is None:
        raise ValueError(f"Missing member {name}")
    return source.read()


def refs_for(tar, group, names):
    return [member(tar, f"publish/packets/nadal_kyrgios__{group}@120/{name}.png.ref") for name in names]


def metadata(kind, tracks, payloads, **extra):
    return dict(version=1, kind=kind, tracks=tracks, payload_count=len(payloads),
                payload_sha256=[sha(payload) for payload in payloads], decoder="REF1-PNG-RGB8", **extra)


def build():
    if sha(TRANSPORT.read_bytes()) != TRANSPORT_SHA:
        raise ValueError("Archived reference transport changed")
    with tarfile.open(TRANSPORT) as tar:
        source = refs_for(tar, "000", SOURCES)
        target = refs_for(tar, "002", TARGETS)
    initial = pack(metadata("initialize", SOURCES, source), source)
    reuse = pack(metadata("reuse", TARGETS, [], initial_sha256=sha(initial), source_slots=[0, 1]), [])
    refresh = pack(metadata("refresh", TARGETS, target, initial_sha256=sha(initial)), target)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, payload in (("initial.psr", initial), ("return-reuse.psr", reuse), ("return-refresh.psr", refresh)):
        (OUTPUT / name).write_bytes(payload)
    print(json.dumps({"initial_bytes": len(initial), "reuse_bytes": len(reuse), "refresh_bytes": len(refresh)}))


def read_conditioning(condition, unit):
    archive = ROOT / condition / "conditioning.tar"
    with tarfile.open(archive) as tar:
        data = member(tar, f"publish/{unit}.npz")
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def audit():
    initial_bytes = (OUTPUT / "initial.psr").read_bytes()
    reuse_bytes = (OUTPUT / "return-reuse.psr").read_bytes()
    refresh_bytes = (OUTPUT / "return-refresh.psr").read_bytes()
    initial, slots = unpack(initial_bytes)
    reuse, reuse_payload = unpack(reuse_bytes)
    refresh, fresh = unpack(refresh_bytes)
    assert initial["kind"] == "initialize" and initial["tracks"] == SOURCES and len(slots) == 2
    assert reuse["kind"] == "reuse" and reuse["tracks"] == TARGETS and not reuse_payload
    assert reuse["initial_sha256"] == refresh["initial_sha256"] == sha(initial_bytes)
    assert refresh["kind"] == "refresh" and refresh["tracks"] == TARGETS and len(fresh) == 2
    assert reuse["source_slots"] == [0, 1]
    source_refs = [ref_decode(blob) for blob in slots]
    fresh_refs = [ref_decode(blob) for blob in fresh]
    assert [meta["unit"] for meta, _ in source_refs] == SOURCES
    assert [meta["unit"] for meta, _ in fresh_refs] == TARGETS
    decoded_slots = [pixels for _, pixels in source_refs]
    decoded_fresh = [pixels for _, pixels in fresh_refs]
    for i, unit in enumerate(TARGETS):
        cold = read_conditioning("fresh", unit)
        warm = read_conditioning("reuse", unit)
        assert np.array_equal(decoded_slots[reuse["source_slots"][i]], warm["reference"])
        assert np.array_equal(decoded_fresh[i], cold["reference"])
        assert set(cold) == set(warm)
        assert all(np.array_equal(cold[key], warm[key]) for key in cold if key != "reference")
    altered = bytearray(reuse_bytes)
    altered[-33] ^= 1
    try:
        unpack(bytes(altered))
    except ValueError:
        pass
    else:
        raise ValueError("Corrupted mapping packet was accepted")
    assert [len(initial_bytes), len(reuse_bytes), len(refresh_bytes)] == [
        (OUTPUT / name).stat().st_size for name in ("initial.psr", "return-reuse.psr", "return-refresh.psr")]
    seconds = 48 / 50
    result = {"passed": True, "initial_bytes": len(initial_bytes), "return_reuse_bytes": len(reuse_bytes),
              "return_refresh_bytes": len(refresh_bytes),
              "total_reuse_bytes": len(initial_bytes) + len(reuse_bytes),
              "total_fresh_bytes": len(initial_bytes) + len(refresh_bytes),
              "observed_shot_seconds": seconds,
              "reference_only_reuse_kbps": (len(initial_bytes) + len(reuse_bytes)) * 8 / seconds / 1000,
              "reference_only_fresh_kbps": (len(initial_bytes) + len(refresh_bytes)) * 8 / seconds / 1000,
              "separation_seconds_between_starts": (87691 - 76626) / 50,
              "initial_sha256": sha(initial_bytes), "reuse_sha256": sha(reuse_bytes),
              "refresh_sha256": sha(refresh_bytes), "corrupt_mapping_rejected": True,
              "target_units": TARGETS, "transport_archive_sha256": TRANSPORT_SHA}
    (OUTPUT / "audit-v2.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    {"build": build, "audit": audit}[sys.argv[1]]()
