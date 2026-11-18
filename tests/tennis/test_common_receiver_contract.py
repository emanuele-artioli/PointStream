"""Regression coverage for the new paid manifest and complete-rally loader."""

import hashlib
import json
import struct

import numpy as np
import pytest

from experiments.tennis import common_player_packet as packet
from experiments.tennis import common_receiver_generate as generator
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import k6b_rate_streams as rates
from experiments.tennis import w8_cuda_system_leaves as policy


def make_rally(tmp_path):
    name = "player"
    profile = {
        "mode": "players",
        "recipe": packet.RECIPE,
        "coded_size": [64, 64],
        "canvas_size": [1920, 1080],
        "resources": {n: "a" * 64 for n in packet.RESOURCES},
        "order": [name],
        "cuts": [],
        "runs": [{"unit": name, "start": 0, "end": 240, "reference": "ref"}],
    }
    base = struct.pack("<4sHH4sHHIIII", b"DKIF", 0, 32, b"AV01", 64, 64, 50, 1, 240, 0)
    base += b"".join(struct.pack("<IQ", 1, t) + b"x" for t in range(240))
    ticks = np.arange(240)
    pose = np.ones((240, 17, 3))
    pose[:, :, 0] = 432
    pose[:, :, 1] = np.arange(17) * 8 + 300
    kps = rates.encode_keypoints_kps2(
        {"native_frames": 240}, [{"name": name, "t": ticks, "kp": pose}]
    )
    image = np.zeros((784, 512, 3), np.uint8)
    ref = rates.encode_reference(
        image,
        {
            "unit": name,
            "reference_frame": 0,
            "reference_box": [0, 0, 100, 100],
            "pixels": "full_prepared_conditioning_crop",
        },
    )
    blob = packet.build(profile, base, kps, {"ref": ref})
    path = tmp_path / "window.pscp"
    path.write_bytes(blob)
    return path, hashlib.sha256(blob).hexdigest()


def test_conditioning_is_packet_only_and_every_tick(tmp_path):
    carrier, digest = make_rally(tmp_path)
    ((name, seed, result),) = generator.conditioning(carrier, digest, 24, ["player"])
    assert (
        name == "player" and seed == int(hashlib.sha256(name.encode()).hexdigest()[:7], 16) * 1009
    )
    assert list(result["t"]) == list(range(24)) and result["pose"].shape == (24, 784, 512, 3)
    assert set(result) == {"reference", "pose", "t", "box"}
    for wrong_hash, frames, units in (
        (digest, 25, ["player"]),
        ("0" * 64, 24, ["player"]),
        (digest, 24, ["other"]),
    ):
        with pytest.raises(ValueError):
            next(generator.conditioning(carrier, wrong_hash, frames, units))


def test_manifest_rejects_different_peer_or_resource(tmp_path):
    path = tmp_path / "receiver.manifest"
    digest = "a" * 64
    path.write_bytes(manifest.encode(digest))
    assert manifest.verify(path, digest)["manifest_bytes"] == len(path.read_bytes())
    with pytest.raises(ValueError):
        manifest.verify(path, "b" * 64)
    fields = json.loads(path.read_text())
    fields["sam_weight"] = "c" * 64
    path.write_text(json.dumps(fields, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError):
        manifest.verify(path, digest)
    with pytest.raises(ValueError):
        policy.restrict([tmp_path], tmp_path, executables=(tmp_path,))
    with pytest.raises(ValueError):
        policy.restrict([tmp_path], tmp_path, listing_only=(path,))


def test_receiver_checks_the_common_replay_fit_and_preserves_w8_pin(tmp_path, monkeypatch):
    from experiments.tennis import w8_receiver_generate as legacy

    assert generator.producer_sources() == {
        "generator": legacy.GENERATOR_CODE_SHA,
        "common_fit": "8f4a78fd7f443de0f9e7454a31c4900e56f73dc737d1be03684932d654f74dc7",
    }
    assert legacy.FIT_CODE_SHA == "5e76dd363174d370572b3bab73e2d5a533f274371ebdc612cc764a947f2a9c69"
    altered = tmp_path / "k6b_fit.py"
    altered.write_bytes(b"other implementation")
    monkeypatch.setattr(generator.k6b_fit, "__file__", str(altered))
    with pytest.raises(ValueError, match="common_fit implementation hash"):
        generator.producer_sources()
