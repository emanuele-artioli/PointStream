"""Authenticate source times, anonymous identities and the exact earlier receiver cache."""

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from experiments.tennis import common_player_packet as packet
from experiments.tennis import common_receiver_manifest as runtime

HERE = Path(__file__).resolve().parent


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(a):
    scripts = HERE.parent / "test-crop-joint-20261122"
    sys.path.insert(0, str(scripts))
    sender = load_script("paired_native_sender", scripts / "build_joint.py")
    prior_rule = load_script(
        "nadal_reuse_rule", HERE.parent / "cross-shot-reference-20261122/source_audit.py"
    )
    case = sender.input_config("alcaraz031")
    _, _, fresh, r = sender.signal(
        SimpleNamespace(source=a.source, poses=a.poses, features=a.features), case
    )
    old = a.earlier.read_bytes()
    profile = packet._embedded_profile(old)
    old_roles = profile["order"]
    assets, _ = packet.wire.unpack(old, profile)
    current_resources = json.loads(
        (HERE.parent / "med-mixed-receiver-20261122/installed-resources.json").read_text()
    )
    current_resources["native_decoder"] = runtime.NATIVE
    assert current_resources == profile["resources"] and profile["recipe"] == packet.RECIPE_25
    rt = json.loads(a.manifest.read_text())
    check = json.loads(runtime.encode(packet.wire.sha(old)))
    assert check == rt
    del rt["composition"], rt["carrier_sha256"]
    cache = packet.wire.pack(
        dict(type="joint-initialization-v1", runtime=rt, resources=current_resources),
        [(k, "reference", assets[k]) for k in ("ref0", "ref1")],
    )
    assert (
        packet.wire.sha(cache) == "91bd3a3fd9b18db301348c98e5af1536b859f2e51dde71e7de56c3beee1c1af2"
    )
    a.cache.write_bytes(cache)

    def h(ref):
        return prior_rule.histogram(packet._bounded_reference(ref)[1])

    dist = [
        [
            float(
                np.sqrt(
                    0.5
                    * np.square(np.sqrt(h(fresh[f"ref{i}"])) - np.sqrt(h(assets[f"ref{j}"]))).sum()
                )
            )
            for j in (0, 1)
        ]
        for i in (0, 1)
    ]
    orders = sorted(
        [
            (sum(dist[i][j] for i, j in enumerate(p)), list(p))
            for p in itertools.permutations((0, 1))
        ]
    )
    margin = orders[1][0] - orders[0][0]
    times = json.loads(a.times.read_text())
    rows = times["rows"]
    assert (
        times["index_sha256"] == "40cc90152baafa65c0db7b7b1b6f886bde5e957beb534b2e9252923eaf4c3a9d"
        and len(rows) == 21
        and all(x["cadence"] == "25" and x["source_frames"] == x["indexed_frames"] for x in rows)
    )
    from fractions import Fraction

    t = {row["clip"]: row for row in rows}
    assert (
        t[31]["sha256"] == r["source_sha256"]
        and t[28]["sha256"] == "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934"
    )
    result = dict(
        source_sha256=t[31]["sha256"],
        earlier_source_sha256=t[28]["sha256"],
        cut_index_sha256=times["index_sha256"],
        all_available_clip_lengths_match=len(rows),
        start_seconds={str(i): float(Fraction(t[i]["start"])) for i in (28, 31)},
        age_seconds=float(Fraction(t[31]["start"]) - Fraction(t[28]["start"])),
        old_carrier_sha256=packet.wire.sha(old),
        old_cache_sha256=packet.wire.sha(cache),
        old_cache_bytes=len(cache),
        old_paid_units=old_roles,
        source_track_ids=case["roles"],
        matrix=dist,
        assignments=orders,
        margin=margin,
        threshold=0.10,
        ambiguous=margin < 0.10,
        source_contact_sheet="Both source views show far role in red with red shoes, near role in grey/blue with yellow shoes.",
        reference_sha256={k: hashlib.sha256(assets[k]).hexdigest() for k in ("ref0", "ref1")},
        compatible_resources=True,
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print("identities", orders, "margin", margin, "age", result["age_seconds"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ["source", "poses", "features", "earlier", "manifest", "cache", "times", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    run(p.parse_args())
