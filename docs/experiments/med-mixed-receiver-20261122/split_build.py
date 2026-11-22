"""Encode separate active and conventional intervals, charging their real IDRs and headers."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from av1 import cpu
from build import sha
from experiments.tennis import common_player_packet as packet
from src.codecs import svtav1


def main(a):
    orig = json.loads((a.build / "result.json").read_text())
    selected = next(r for r in orig["rates"] if r["crf"] == orig["selected_crf"])
    cap = selected["joint_bytes"]
    first, last = orig["active"]
    parts = [
        dict(name=f"v{i}", begin=b, end=e, mode=m, width=w, height=h)
        for i, (b, e, m, w, h) in enumerate(
            (
                p
                for p in [
                    (0, first, "conventional", 320, 192),
                    (first, last, "objects", 160, 96),
                    (last, 240, "conventional", 320, 192),
                ]
                if p[1] > p[0]
            )
        )
    ]
    assert (
        sum(p["end"] - p["begin"] for p in parts) == 240
        and sum(p["end"] - p["begin"] for p in parts if p["mode"] == "objects") == 120
    )
    original = a.build / f"crf{selected['crf']}"
    o = (original / "rally.psj").read_bytes()
    cache = (original / "initial.psr").read_bytes()
    assert (
        packet.wire.sha(o) == selected["joint_sha256"]
        and packet.wire.sha(cache) == selected["init_sha256"]
    )
    profile = packet._embedded_profile(o)
    shared, _ = packet.wire.unpack(o, profile)
    profile = {**profile, "kind": "joint-split-trajectory-v1", "parts": parts}
    a.out.mkdir(exist_ok=False)
    ff = svtav1.tool("ffmpeg")
    enc = svtav1.tool("SvtAv1EncApp")
    dec = svtav1.tool("dav1d")
    inputs = {}
    results = {}
    trial_records = []
    began = cpu()
    for p in parts:
        src = a.build / ("mixed.yuv" if p["mode"] == "objects" else "source.yuv")
        required = orig["donor"][
            "mixed_yuv_sha256" if p["mode"] == "objects" else "source_yuv_sha256"
        ]
        assert sha(src) == required
        raw = a.out / (p["name"] + ".yuv")
        command = [
            ff["path"],
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
            str(src),
            "-vf",
            f"trim=start_frame={p['begin']}:end_frame={p['end']},scale={p['width']}:{p['height']}:flags=area",
            "-frames:v",
            str(p["end"] - p["begin"]),
            "-f",
            "rawvideo",
            "-n",
            str(raw),
        ]
        subprocess.run(command, check=True, timeout=140)
        assert raw.stat().st_size == (p["end"] - p["begin"]) * p["width"] * p["height"] * 3 // 2
        inputs[p["name"]] = raw

    def encode(p, q, frames=None):
        n = frames or (p["end"] - p["begin"])
        key = (p["name"], n, q)
        if key in results:
            return results[key]
        d = a.out / f"{p['name']}-n{n}-q4{q}"
        d.mkdir()
        r = svtav1.code(
            inputs[p["name"]],
            d,
            width=p["width"],
            height=p["height"],
            fps=Fraction(50),
            frames=n,
            crf=q / 4,
            preset=4,
            threads=1,
            full_range=False,
            encoder=enc["path"],
            decoder=dec["path"],
            timeout=240,
        )
        assert (
            r["temporal_units"] == r["decoded_frames"] == n
            and Path(r["decoded"]).stat().st_size == n * p["width"] * p["height"] * 3 // 2
        )
        row = dict(
            section=p["name"],
            frames=n,
            crf=q / 4,
            bytes=r["file_bytes"],
            stream=r["stream"],
            stream_sha256=sha(Path(r["stream"])),
            decoded=r["decoded"],
            native_sha256=sha(Path(r["decoded"])),
        )
        results[key] = row
        return row

    longest = max(
        (p for p in parts if p["mode"] == "conventional"), key=lambda p: p["end"] - p["begin"]
    )
    t = cpu()
    encode(longest, 65 * 4, 24)
    smoke = cpu() - t
    projected = 1.5 * smoke * (120 / 24) * 13 + 45
    if projected > 1800:
        raise ValueError(f"partial native CPU forecast {projected}s exceeds 1800s")

    def measure(qa, qf):
        rs = [encode(p, (qa if p["mode"] == "objects" else qf)) for p in parts]
        sec = [
            (p["name"], "frame", Path(r["stream"]).read_bytes())
            for p, r in zip(parts, rs, strict=True)
        ]
        sec += [("poses", "frame", shared["poses"]), ("trajectory", "frame", shared["trajectory"])]
        payload = packet.wire.pack(profile, sec)
        unpacked, roles = packet.wire.unpack(payload, profile)
        assert sum(roles.values()) == len(payload) and all(unpacked[k] == v for k, _, v in sec)
        record = dict(
            active_crf=qa / 4,
            fallback_crf=qf / 4,
            total_bytes=len(payload),
            kbps=len(payload) / 600,
            wire_sha256=packet.wire.sha(payload),
            pieces=rs,
            roles=roles,
        )
        trial_records.append(record)
        return record, payload

    chosen = None
    seen = {}
    for qa in dict.fromkeys((selected["crf"] * 4, 70 * 4)):
        row, payload = measure(qa, 70 * 4)
        if row["total_bytes"] > cap:
            continue
        lo, hi = 40 * 4, 70 * 4
        seen = {70 * 4: (row, payload), 40 * 4: measure(qa, 40 * 4)}
        if seen[40 * 4][0]["total_bytes"] > cap:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                seen[mid] = measure(qa, mid)
                if seen[mid][0]["total_bytes"] <= cap:
                    hi = mid
                else:
                    lo = mid
        chosen = max(
            (v for v in seen.values() if v[0]["total_bytes"] <= cap),
            key=lambda v: v[0]["total_bytes"],
        )
        break
    record = dict(
        source_sha256=orig["source_sha256"],
        original_wire_sha256=selected["joint_sha256"],
        price_cap=cap,
        active_interval=orig["active"],
        cache_sha256=selected["init_sha256"],
        cache_bytes=len(cache),
        parts=parts,
        smoke_cpu_seconds=smoke,
        smoke_forecast_cpu=projected,
        cpu_seconds=cpu() - began,
        tools=dict(ffmpeg=ff, encoder=enc, decoder=dec),
        trials=trial_records,
        feasible=chosen is not None,
    )
    if chosen is not None:
        row, payload = chosen
        (a.out / "split.psj").write_bytes(payload)
        record["selected"] = row
    (a.out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                feasible=record["feasible"],
                cap=cap,
                attempts=[
                    (r["active_crf"], r["fallback_crf"], r["total_bytes"]) for r in trial_records
                ],
                chosen=record.get("selected", {}).get("total_bytes"),
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
