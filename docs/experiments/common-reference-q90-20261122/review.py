"""Build a two-condition author review from verified 24-frame receiver outputs."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import shutil

from experiments.tennis.player_reference_review import LABELS, UNITS

HERE = Path(__file__).resolve().parent
OLD = Path("/Users/manu/Documents/PointStream-review/2026-11-18-reference-appearance/q70-q40")
OLD_RECEIPT = HERE.parent / "appearance-resume-20261118/review-delivery.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(new, output):
    original = json.loads(OLD_RECEIPT.read_text())["files"]
    previous_key = json.loads((OLD / "reveal.json").read_text())["mappings"]
    fresh = json.loads((new / "receipt.json").read_text())
    prepared = json.loads((HERE / "prepared.json").read_text())
    assert fresh["config_sha256"] == prepared["config_sha256"] and fresh["repeated_player_equal"]
    assert all(row["quality"] == 90 for row in prepared["rows"])
    output.mkdir(exist_ok=False)
    sections = []
    mapping = {}
    rng = random.Random(2026112290)
    for unit in UNITS:
        order = ["q90", "lossless"]
        rng.shuffle(order)
        mapping[unit] = dict(zip("AB", order, strict=True))
        loss = next(k for k, v in previous_key[unit].items() if v == "lossless")
        tracks = []
        for fps in (50, 10):
            panels = []
            for label, arm in {"Source": "source", **mapping[unit]}.items():
                source = (
                    new / f"{unit}-{fps}.mp4"
                    if arm == "q90"
                    else OLD / f"{unit}-{loss if arm == 'lossless' else 'source'}-{fps}.mp4"
                )
                ledger = fresh["units"][unit] if arm == "q90" else original
                assert sha(source) == ledger[source.name]["sha256"]
                file = f"{unit}-{label}-{fps}.mp4"
                shutil.copyfile(source, output / file)
                panels.append(
                    f'<figure><figcaption>{label}</figcaption><video muted loop playsinline controls src="{file}"></video></figure>'
                )
            name = "Original speed" if fps == 50 else "Slow motion"
            tracks.append(f'<h3>{name}</h3><div class="row">{"".join(panels)}</div>')
        sections.append(f"<section><h2>{html.escape(LABELS[unit])}</h2>{''.join(tracks)}</section>")
    page = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Player appearance review</title><style>body{font:16px system-ui;max-width:1000px;margin:24px auto;padding:0 12px;color:#1f2937}h1,h2{margin-top:1.3em}.row{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}video{width:100%;max-height:35vh;background:#111}figcaption{font-weight:650}button{padding:.5em 1em}</style>
<h1>Player appearance review</h1><p>For each player, judge A and B relative to the source for body, clothing and pose. Each row shows the same 24 frames. Choose acceptable, almost acceptable or unacceptable for each version.</p>
<button onclick="for(const v of document.querySelectorAll('video')){v.pause();v.currentTime=0;v.play().catch(()=>{})}">Replay together</button>"""
    (output / "review.html").write_text(page + "".join(sections))
    result = dict(
        mappings=mapping,
        source_review_sha256=sha(OLD_RECEIPT),
        receipt_sha256=sha(new / "receipt.json"),
        files={p.name: sha(p) for p in output.iterdir()},
    )
    (output / "reveal.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("clips", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    build(args.clips, args.output)
