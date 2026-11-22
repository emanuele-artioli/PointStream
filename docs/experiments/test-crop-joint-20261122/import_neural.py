"""Build scorer manifests from published neural streams with four exact native passes."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
SOURCE = "fc5d3ee431f84417d2de3412402f4e8efb0546418ce96f1f0a1e2f1887a88b1a"


def main(a):
    rows = []
    for path in a.archives:
        with tarfile.open(path) as tar:
            result = json.load(tar.extractfile("publish/result.json"))
            reports = json.load(tar.extractfile("publish/reports.json"))
            stream = tar.extractfile("publish/stream.psdc").read()
        n = result["native"]
        assert result["source_sha256"] == SOURCE and result["native_fps"] == [30000, 1001]
        assert result["frames"] == 240 and n["accepted"] and len(result["yuv_frames"]) == 240
        passes = [p for group in reports["decode"].values() for p in group["passes"]]
        assert len(passes) == 4 and all(
            p["yuv420_sha256"] == result["yuv_frames"] and p["frame_sha256"] == result["yuv_frames"]
            for p in passes
        )
        assert (
            len(stream) == n["bytes"]
            and hashlib.sha256(stream).hexdigest() == n["container_sha256"]
        )
        w, h = n["size"]
        rows.append(
            dict(
                name=f"dcvc-{w}-q{n['qp']}",
                width=w,
                height=h,
                bytes=n["bytes"],
                kind="yuv_tar",
                archive=str(path),
                member="publish/decoded.yuv",
                frame_sha256=result["yuv_frames"],
                stream_sha256=n["container_sha256"],
                decoded_sha256=result["raw_sha256"],
            )
        )
    a.manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print([(r["name"], r["bytes"]) for r in rows])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives", nargs="+", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    main(p.parse_args())
