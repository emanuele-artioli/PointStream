"""Add PNG, AV1 intra, and VVC intra to the JPEG/WebP still-image probe."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from experiments.modular.image_codec_probe import _psnr, _signals
from src.components.background.sidecar import build_sidecar

OUT = Path("/home/itec/emanuele/pointstream-data/outputs/image-codec-probe/federer007.json")
INTRA = (("av1", (32, 42, 50)), ("vvc", (32, 40, 46)))


def main() -> None:
    signals = _signals()
    report = json.loads(OUT.read_text())
    rows = report["rows"]
    for name, image in signals.items():
        sidecar = build_sidecar("png", png_compression=3)
        payload = sidecar.encode(image)
        decoded = sidecar.decode(payload)
        row = {
            "signal": name,
            "codec": "png",
            "quality": 3,
            "bytes": len(payload),
            "psnr": round(_psnr(image, decoded), 4),
        }
        rows.append(row)
        print(row, flush=True)
        for codec, qps in INTRA:
            for qp in qps:
                try:
                    coder = build_sidecar(codec, intra_qp=qp)
                    payload = coder.encode(image)
                    decoded = coder.decode(payload)
                    if decoded.shape != image.shape:
                        decoded = decoded[: image.shape[0], : image.shape[1]]
                    row = {
                        "signal": name,
                        "codec": codec,
                        "quality": qp,
                        "bytes": len(payload),
                        "psnr": round(_psnr(image, decoded), 4),
                    }
                except Exception as exc:
                    row = {"signal": name, "codec": codec, "quality": qp, "error": str(exc)}
                    traceback.print_exc()
                rows.append(row)
                print(row, flush=True)
    report["rows"] = rows
    OUT.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
