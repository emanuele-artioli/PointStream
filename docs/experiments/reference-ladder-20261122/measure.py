"""Compare shared reference quantizers at fixed two-image payload budgets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import tempfile
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "common-reference-20261122" / "receipt.json"
CAPS = (25_000, 45_000, 65_000)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cpu_time():
    return sum(r.ru_stime + r.ru_utime for r in
               (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def encode(codec, quality, path, rgb, tmp, tools):
    if codec == "jpeg":
        ok, array = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        restored = cv2.imdecode(array, cv2.IMREAD_COLOR)[:, :, ::-1]
        payload = array.tobytes()
    else:
        encoded, decoded = tmp / "ref.bin", tmp / "decoded.png"
        if codec == "webp":
            enc = [tools["cwebp"], "-quiet", "-q", str(quality), "-m", "6", "-metadata", "none", path, "-o", encoded]
            dec = [tools["dwebp"], "-quiet", encoded, "-o", decoded]
        else:
            enc = [tools["avifenc"], "-q", str(quality), "--speed", "8", "--jobs", "1", path, encoded]
            dec = [tools["avifdec"], encoded, decoded]
        for command in (enc, dec):
            subprocess.run(command, check=True, capture_output=True, timeout=30)
        payload = encoded.read_bytes()
        restored = cv2.imread(str(decoded))[:, :, ::-1]
    assert restored.shape == rgb.shape == (784, 512, 3)
    error = np.square(restored.astype(np.float64) - rgb).mean()
    assert 0 < error <= 255**2
    return dict(bytes=len(payload), sha256=digest(payload), decoded_rgb_sha256=digest(restored.tobytes()),
                psnr_db=10 * math.log10(255**2 / error))


def measure(input_dir, output):
    start_cpu, start_wall = cpu_time(), time.monotonic()
    prior = json.loads(PRIOR.read_bytes())
    tools = {}
    assert cv2.__version__ == prior["opencv"]["version"]
    assert digest(cv2.getBuildInformation().encode()) == prior["opencv"]["build_sha256"]
    for name, item in prior["tools"].items():
        exe = Path(item["path"])
        assert digest(exe.read_bytes()) == item["sha256"]
        tools[name] = exe
    sources = []
    for idx in range(2):
        old = next(r for r in prior["rows"] if r["section"] == f"ref{idx}")
        path = input_dir / (old["unit"] + "-source.png")
        rgb = cv2.imread(str(path))[:, :, ::-1]
        assert digest(rgb.tobytes()) == old["source_rgb_sha256"]
        sources.append((path, rgb))
    rows, chosen = {}, []
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory() as scratch:
        def probe(codec, quality):
            key = (codec, quality)
            if key not in rows:
                if cpu_time()-start_cpu > 175 or time.monotonic()-start_wall > 290:
                    raise RuntimeError("Local experiment budget exhausted")
                pairs = [encode(codec, quality, path, rgb, Path(scratch), tools) for path, rgb in sources]
                rows[key] = dict(codec=codec, quality=quality, total_bytes=sum(p["bytes"] for p in pairs), players=pairs)
            return rows[key]

        for codec in ("jpeg", "webp", "avif"):
            smoke = probe(codec, 65)
            if codec != "jpeg":
                for i, pair in enumerate(smoke["players"]):
                    old = next(r for r in prior["rows"] if (r["section"], r["codec"], r["quality"]) == (f"ref{i}", codec, 65))
                    assert (pair["sha256"], pair["decoded_rgb_sha256"]) == (old["payload_sha256"], old["decoded_rgb_sha256"])
            for cap in CAPS:
                low, high = 0, 100
                best = None
                while low <= high:
                    q = (low + high) // 2
                    row = probe(codec, q)
                    if row["total_bytes"] <= cap:
                        best, low = q, q + 1
                    else:
                        high = q - 1
                if best is None:
                    raise ValueError(f"No {codec} operating point at {cap} bytes")
                chosen.append(dict(codec=codec, budget=cap, quality=best))
    for selected in chosen:
        fitting = [q for (c, q), row in rows.items() if c == selected["codec"] and row["total_bytes"] <= selected["budget"]]
        selected["quality"] = max(fitting)
    elapsed = time.monotonic() - start_wall
    used_cpu = cpu_time() - start_cpu
    assert elapsed < 300 and used_cpu < 180
    result = dict(caps_bytes=CAPS, prior_sha256=digest(PRIOR.read_bytes()), script_sha256=digest(Path(__file__).read_bytes()),
                  source_rgb_sha256=[digest(rgb.tobytes()) for _, rgb in sources], tools=prior["tools"], opencv=prior["opencv"],
                  platform=platform.platform(), numpy=np.__version__, device="CPU", cpu_seconds=used_cpu, wall_seconds=elapsed,
                  selected=chosen, probes=[rows[key] for key in sorted(rows)])
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for row in chosen:
        measured = rows[row["codec"],row["quality"]]
        print(row, measured["total_bytes"], [round(p["psnr_db"],2) for p in measured["players"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    measure(args.inputs, args.output)
