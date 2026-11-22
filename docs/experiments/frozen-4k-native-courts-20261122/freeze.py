"""Derive the six spatial-rendition source jobs directly from the existing frozen census."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT.parent / "six-match-source-panel-20261122"


def main():
    frozen = json.loads((NATIVE / "frozen.json").read_text())
    metadata = json.loads((NATIVE / "source-verification.json").read_text())["sources"]
    original = {r["source_sha256"]: r for r in metadata}
    template = json.loads((NATIVE / "alcaraz031-court.json").read_text())
    cases = []
    for r in frozen["original_files"]:
        if r["event"] not in ["djokovic_federer", "djokovic_zverev"]:
            continue
        row = original[r["sha256"]]
        m = row["meta"]
        assert (
            m["width"] == 3840
            and m["height"] == 2160
            and m["r_frame_rate"] == m["avg_frame_rate"]
            and m["pix_fmt"] == "yuv420p"
            and int(m["nb_frames"]) >= 240
        )
        tag = r["event"] + "-" + row["shot"]
        path = str(Path(frozen["archive"]) / r["path"])
        spec = {
            **template,
            "entrypoint": ["-m", "experiments.tennis.native_4k_court"],
            "validator": [
                "{python}",
                "-m",
                "experiments.tennis.native_4k_court",
                "--validate",
                "--full-seconds",
                "1800",
            ],
            "deadline": "2026-11-22T21:01:00Z",
            "budget_seconds": 2170,
            "smoke": dict(
                seconds=240,
                representative_basis="First 12 native frames from the same 4K source with precisely the same 1080p spatial filter and real court registration.",
            ),
            "full": {"seconds": 1800},
            "arguments": [
                "--source",
                "{staged:original}",
                "--event",
                tag,
                "--sha",
                r["sha256"],
                "--fps",
                m["avg_frame_rate"],
                "--length",
                m["nb_frames"],
                "--frames",
                "{frames}",
            ],
            "staged_inputs": [dict(name="original", path=path, sha256=r["sha256"], extract=False)],
            "inputs": [dict(path=path, sha256=r["sha256"])],
        }
        (ROOT / (tag + ".json")).write_text(json.dumps(spec, indent=2) + "\n")
        cases.append(
            dict(
                tag=tag,
                rank=r["rank"],
                native_fps=m["avg_frame_rate"],
                length=int(m["nb_frames"]),
                source_sha256=r["sha256"],
                bytes=r["bytes"],
                archive=path,
            )
        )
    assert len(cases) == 6
    (ROOT / "sources.json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
