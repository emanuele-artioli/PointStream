"""Match complete AV1 and joint bytes to the frozen 25-fps neural display."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource

from experiments.tennis import common_player_packet as player
from experiments.tennis import common_receiver_manifest as manifest
from experiments.tennis import w8_payload as transport
from src.codecs import svtav1

HERE = Path(__file__).resolve().parent
TMP = Path("/private/tmp")
SOURCE = {
    "ordinary": TMP / "ps-al028-av1-ladder-20261122/320.yuv",
    "joint": TMP / "ps-al028-donor-rate-20261122/input.yuv",
}
OLD = TMP / "ps-al028-joint25-crf52-20261122"
TARGET = 24494


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cpu():
    return sum(
        r.ru_utime + r.ru_stime
        for r in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def run(out):
    out.mkdir(exist_ok=False)
    origin = json.loads(
        (HERE.parent / "joint-replication-feasibility-20261122/al028-joint-21kbps.json").read_text()
    )
    cache = (OLD / "initial.psr").read_bytes()
    previous = (OLD / "rally.psj").read_bytes()
    assert digest(cache) == origin["cache_sha256"] and digest(previous) == origin["joint_sha256"]
    profile = player._embedded_profile(previous)
    parts, _ = transport.unpack(previous, profile)
    cache_profile = player._embedded_profile(cache)
    refs, _ = transport.unpack(cache, cache_profile)
    baseline = json.loads(
        (HERE.parent / "joint-replication-feasibility-20261122/al028-av1-target.json").read_text()
    )["grid"][0]
    assert baseline["width"] == 320 and baseline["crf"] == 52
    expected = {"joint": digest(parts["base"]), "ordinary": baseline["stream_sha256"]}
    start = cpu()
    tools = {k: svtav1.tool(k) for k in ("SvtAv1EncApp", "dav1d")}
    old_tools = json.loads((TMP / "ps-al028-av1-ladder-20261122/result.json").read_text())["tools"]
    assert all(v["sha256"] == old_tools[k]["sha256"] for k, v in tools.items())
    records = {arm: [] for arm in SOURCE}

    def probe(arm, q):
        dest = out / f"{arm}-{q:g}"
        dest.mkdir()
        r = svtav1.code(
            SOURCE[arm],
            dest,
            width=320,
            height=192,
            fps=Fraction(25),
            frames=240,
            crf=q,
            preset=4,
            threads=1,
            full_range=False,
            encoder=tools["SvtAv1EncApp"]["path"],
            decoder=tools["dav1d"]["path"],
            timeout=55,
        )
        data = Path(r["stream"]).read_bytes()
        assert r["decoded_frames"] == r["temporal_units"] == 240
        if q == 52:
            assert digest(data) == expected[arm]
        delivered = data
        if arm == "joint":
            delivered = transport.pack(
                profile,
                [
                    (role, "frame", data if role == "base" else parts[role])
                    for role in ("base", "poses", "trajectory")
                ],
            )
            paid, _ = transport.unpack(delivered, profile)
            assert (
                paid["poses"] == parts["poses"]
                and paid["trajectory"] == parts["trajectory"]
                and paid["base"] == data
            )
            carrier = player.build(
                profile["player"] | {"resources": cache_profile["resources"]},
                data,
                paid["poses"],
                refs,
            )
            player._ivf(data, 320, 192, (25, 1))
            if q == 52:
                assert (
                    delivered == previous and digest(carrier) == origin["receiver_carrier_sha256"]
                )
            (dest / "rally.psj").write_bytes(delivered)
            (dest / "players.pscp").write_bytes(carrier)
            (dest / "players.manifest").write_bytes(manifest.encode(digest(carrier)))
        row = {
            "crf": q,
            "bytes": len(delivered),
            "kbps": len(delivered) / 1200,
            "percent_from_neural": 100 * (len(delivered) / TARGET - 1),
            "stream_sha256": digest(delivered),
            "ivf_sha256": digest(data),
            "ivf_bytes": len(data),
            "raw_sha256": digest(Path(r["decoded"]).read_bytes()),
            "path": str(dest),
        }
        records[arm].append(row)
        return row

    for arm in SOURCE:
        probe(arm, 52)
    forecast = (cpu() - start) * (2 + 6) * 1.5 + 35
    if forecast > 600:
        raise RuntimeError(f"full-size controls forecast {forecast:.1f} CPU seconds")

    def nearest(rows):
        return min(rows, key=lambda v: (abs(v["bytes"] - TARGET), v["bytes"]))

    best = {}
    for arm, rows in records.items():
        for q in (53, 53.5, 54):
            probe(arm, q)
        for _ in range(3):
            if abs(nearest(rows)["percent_from_neural"]) <= 0.5:
                break
            small = [r for r in rows if r["bytes"] < TARGET]
            big = [r for r in rows if r["bytes"] > TARGET]
            if not small or not big:
                break
            q = (nearest(small)["crf"] + nearest(big)["crf"]) / 2
            if any(r["crf"] == q for r in rows):
                break
            probe(arm, q)
        best[arm] = nearest(rows)
    totals = [r["bytes"] for r in best.values()] + [TARGET]
    result = {
        "neural_bytes": TARGET,
        "selection": "nearest complete bytes; smaller packet breaks ties",
        "cache_sha256": digest(cache),
        "cache_bytes": len(cache),
        "all_pose_and_ball_bytes_identical_to_21_298": True,
        "reference_rate_kbps": TARGET / 1200,
        "max_pairwise_percent": 100 * (max(totals) / min(totals) - 1),
        "within_half_percent_of_neural": all(
            abs(r["percent_from_neural"]) <= 0.5 for r in best.values()
        ),
        "selected": best,
        "rows": records,
        "native_tools": tools,
        "cpu_seconds": cpu() - start,
        "forecast_cpu": forecast,
        "input_sha256": {k: digest(v.read_bytes()) for k, v in SOURCE.items()},
    }
    assert result["cpu_seconds"] < 600
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: (r["crf"], r["kbps"]) for k, r in best.items()}, result["max_pairwise_percent"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    run(ap.parse_args().out)
