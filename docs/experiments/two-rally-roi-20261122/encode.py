"""Physical ROI byte search and unchanged-Q ordinary twin on original development footage."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from src.codecs import svtav1


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cpu():
    return sum(
        r.ru_utime + r.ru_stime
        for r in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def bank(a):
    config = json.loads((a.root / "sources.json").read_text())[a.event][str(a.width)]
    directory = a.root / f"native-{a.event}-{a.width}-{a.strength}"
    directory.mkdir(exist_ok=False)
    source = Path(config["source"])
    maskpath = Path(config["maps"][str(a.strength)]["path"])
    assert (
        sha(source) == config["sha256"]
        and sha(maskpath) == config["maps"][str(a.strength)]["sha256"]
    )
    w = a.width
    h = config["height"]
    cap = config["cap"]
    fps = Fraction(config["fps"])
    tools = {k: svtav1.tool(k) for k in ("SvtAv1EncApp", "dav1d")}
    baseline = json.loads(
        (
            Path("/private/tmp/ps-al028-av1-ladder-20261122/result.json")
            if a.event == "al028"
            else Path(f"/private/tmp/ps-joint-trajectory-controls-20261122/20-{w}/result.json")
        ).read_text()
    )
    assert all(x["sha256"] == baseline["tools"][k]["sha256"] for k, x in tools.items())
    begin = cpu()
    trials = {}

    def encode(quarter, n, roi):
        path = directory / f"{'roi' if roi else 'twin'}-q{quarter}-n{n}.ivf"
        command = svtav1.encode_command(
            tools["SvtAv1EncApp"]["path"],
            source,
            path,
            width=w,
            height=h,
            fps=fps,
            frames=n,
            crf=quarter / 4,
            preset=4,
            full_range=False,
            extra=["--roi-map-file", str(maskpath)] if roi else None,
        )
        subprocess.run(command, check=True, capture_output=True, timeout=180)
        assert len(svtav1.ivf_frames(path.read_bytes())) == n
        return {
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "path": str(path),
            "command": command,
        }

    def decode(row, out, n):
        c = svtav1.decode_command(tools["dav1d"]["path"], Path(row["path"]), out, threads=1)
        subprocess.run(c, check=True, capture_output=True, timeout=180)
        assert out.stat().st_size == n * w * h * 3 // 2
        return sha(out)

    with TemporaryDirectory(dir=directory) as tmp:
        tmp = Path(tmp)
        smoke = encode(60 * 4, 24, True)
        decode(smoke, tmp / "smoke.yuv", 24)
        smoke_cost = cpu() - begin
        assert 1.5 * smoke_cost * 10 * 11 + 30 < 900, (
            f"Out of per-arm CPU forecast at {smoke_cost=}"
        )

        def probe(q):
            if q not in trials:
                trials[q] = encode(q, 240, True)
            return trials[q]["bytes"]

        lo, hi = 32 * 4, 70 * 4
        if probe(hi) <= cap:
            if probe(lo) > cap:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if probe(mid) <= cap:
                        hi = mid
                    else:
                        lo = mid
            best = min(
                (q for q, r in trials.items() if r["bytes"] <= cap),
                key=lambda q: (q, -trials[q]["bytes"]),
            )
            chosen = trials[best].copy()
            display = directory / "decoded.yuv"
            chosen["decoded_sha256"] = decode(chosen, display, 240)
            twin = encode(best, 240, False)
            twin["decoded_sha256"] = decode(twin, tmp / "twin.yuv", 240)
            A = np.memmap(display, np.uint8, "r", shape=(240, h * 3 // 2, w))
            B = np.memmap(tmp / "twin.yuv", np.uint8, "r", shape=A.shape)
            changed = sum(int(np.count_nonzero(u[:h] != v[:h])) for u, v in zip(A, B, strict=True))
            chosen.update(
                crf=best / 4,
                decoded=str(display),
                same_q_ordinary=twin,
                changed_luma=changed,
                effective=bool(changed),
            )
        else:
            chosen = None
    receipt = dict(
        event=a.event,
        width=w,
        height=h,
        offset=a.strength,
        cap=cap,
        fps=int(fps),
        selected=chosen,
        tools=tools,
        source_sha256=sha(source),
        mask_sha256=sha(maskpath),
        code_sha256=sha(Path(__file__)),
        trials=trials,
        smoke=smoke,
        cpu_s=cpu() - begin,
        smoke_cpu=smoke_cost,
    )
    (directory / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        a.event,
        w,
        a.strength,
        "affordable" if chosen else "over cap",
        None if not chosen else chosen["bytes"],
        round(receipt["cpu_s"], 2),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("event")
    p.add_argument("width", type=int)
    p.add_argument("strength", type=int)
    bank(p.parse_args())
