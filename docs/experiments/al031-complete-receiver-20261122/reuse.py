"""Price the return shot by referring to the exact cache delivered on the earlier shot."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import common_receiver_manifest as receiver
from experiments.tennis import med_mixed_channel as mixed

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(a):
    proven = json.loads((HERE / "reference-identity.json").read_text())
    assert not proven["ambiguous"] and proven["assignments"][0][1] == [0, 1]
    cache = a.cache.read_bytes()
    assert digest(cache) == proven["old_cache_sha256"]
    cp = packet._embedded_profile(cache)
    prior, _ = packet.wire.unpack(cache, cp)
    book = json.loads((a.fresh / "result.json").read_text())
    assert book["native_fps"] == "25" and book["source_sha256"] == proven["source_sha256"]
    rows = []
    a.out.mkdir(exist_ok=False)
    for row in book["rates"]:
        crf = row["crf"]
        d = a.fresh / f"crf{crf}"
        out = a.out / f"crf{crf}"
        out.mkdir()
        orig = (d / "rally.psj").read_bytes()
        fresh = (d / "initial.psr").read_bytes()
        priorcar = (d / "players.pscp").read_bytes()
        assert (
            digest(orig) == row["joint_sha256"]
            and digest(fresh) == row["init_sha256"]
            and digest(priorcar) == row["carrier_sha256"]
        )
        assert packet._embedded_profile(fresh) == cp and fresh != cache
        fresh_fields, _ = packet.wire.unpack(fresh, cp)
        jp = packet._embedded_profile(orig)
        fields, _ = packet.wire.unpack(orig, jp)
        profile = packet._embedded_profile(priorcar)
        assert (
            profile["order"] == proven["old_paid_units"]
            and profile["recipe"] == packet.RECIPE_25
            and profile["resources"] == cp["resources"]
        )
        for i, n in enumerate(profile["order"]):
            k = f"ref{i}"
            f, imgf = packet._bounded_reference(fresh_fields[k])
            r, imgr = packet._bounded_reference(prior[k])
            assert (
                f["unit"] == r["unit"] == n
                and r["reference_frame"] == f["reference_frame"] == 0
                and not np.array_equal(imgf, imgr)
            )
        carrier = packet.build(profile, fields["base"], fields["poses"], prior)
        manifest = receiver.encode(digest(carrier))
        revised = {**jp, "cache_sha256": digest(cache)}
        wire = packet.wire.pack(
            revised, [(k, "frame", fields[k]) for k in ["base", "poses", "trajectory"]]
        )
        check, _ = packet.wire.unpack(wire, revised)
        assert check == fields
        for f, blob in [
            ("rally.psj", wire),
            ("initial.psr", cache),
            ("players.pscp", carrier),
            ("players.manifest", manifest),
        ]:
            (out / f).write_bytes(blob)
        parsed = mixed.decode(
            out / "rally.psj", out / "initial.psr", out / "players.pscp", digest(carrier)
        )
        assert parsed["start"] == 0 and parsed["stop"] == 240
        _, _, fresh_units = packet.unpack(priorcar)
        _, _, reuse_units = packet.unpack(carrier)
        for n in profile["order"]:
            f, r = fresh_units[n][0], reuse_units[n][0]
            assert f["seed"] == r["seed"] and all(
                np.array_equal(f[k], r[k]) for k in ("t", "kp", "boxes")
            )
        rows.append(
            {
                **row,
                "kbps_cached": len(wire) / 1200,
                "kbps_cold": (len(wire) + len(cache)) / 1200,
                "joint_bytes": len(wire),
                "init_bytes": len(cache),
                "reference_bytes": {k: len(v) for k, v in prior.items()},
                "carrier_sha256": digest(carrier),
                "manifest_sha256": digest(manifest),
                "joint_sha256": digest(wire),
                "init_sha256": digest(cache),
            }
        )
    book.update(
        rates=rows,
        reference_mode="Previously delivered Al028 cache, byte-identical; physical mapping inside new paid object IDs and cache selector.",
        reference_age_seconds=proven["age_seconds"],
    )
    (a.out / "result.json").write_text(json.dumps(book, indent=2) + "\n")
    print(
        "old cache",
        len(cache),
        "age",
        book["reference_age_seconds"],
        "charged",
        [(r["crf"], r["joint_bytes"]) for r in rows],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for v in ("cache", "fresh", "out"):
        p.add_argument("--" + v, type=Path, required=True)
    run(p.parse_args())
