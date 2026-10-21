"""Content-blind K4b marking inputs; CPU only, archive opened read-only.

Run as ``python -m tools.datasets.k4b_inputs --datasets ROOT --dest NEW_DIR``.
The global hash prefix is extended only to meet the predeclared diversity minima.
Frame indices are zero-based decimal integers; clip names are stems, as in K4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from experiments.tennis.windows import convert_command
from tools.datasets.k4_inputs import ARCHIVE, MATCHES, entry, file_sha256, probe

SEED = "pointstream-k4b"


def frame_hash(match: str, clip: str, frame: int) -> str:
    return hashlib.sha256(f"{SEED}:{match}/{clip}/{frame}".encode()).hexdigest()


def select_frames(inventory: list[dict[str, Any]], excluded: set[str]) -> list[dict[str, Any]]:
    ranked = sorted((dict(match=c["match"], clip=c["clip"], frame=t,
                          rank=frame_hash(c["match"], c["clip"], t))
                     for c in inventory if f"broadcast/{c['match']}/{c['clip']}" not in excluded
                     for t in range(c["frames"])), key=lambda r: r["rank"])
    chosen: list[dict[str, Any]] = []
    for row in ranked:
        chosen.append(row)
        if (len(chosen) >= 48 and len({(r["match"], r["clip"]) for r in chosen}) >= 8
                and {r["match"] for r in chosen} == set(MATCHES)):
            return chosen
    raise ValueError("Archive cannot meet K4b's frame/clip/match minima")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--selection", type=Path, help="reuse an already frozen frame selection")
    p.add_argument("--context-frames", type=int, default=8)
    args = p.parse_args()
    dest = args.dest.resolve()
    dest.relative_to((args.datasets / "pointstream-data").resolve())
    dest.mkdir(parents=True, exist_ok=False)
    k4 = args.datasets / "pointstream-data/tennis/k4-2026-10-13/windows.json"
    excluded = {w["id"] for w in json.loads(k4.read_text())["windows"] if w["dataset"] == "broadcast"}
    if args.context_frames < 3:
        raise ValueError("SAM's temporal confirmation needs at least three real frames")
    if args.selection:
        frozen = json.loads(args.selection.read_text())
        inventory = frozen["inventory"]
        if set(frozen["excluded"]) != excluded:
            raise ValueError("K4 exclusion list changed")
    else:
        inventory = []
        for match, sub in MATCHES.items():
            for path in sorted((args.datasets / ARCHIVE / match / sub).glob("*.mp4")):
                inventory.append({"match": match, "clip": path.stem, "path": str(path), **probe(path)})
    chosen = select_frames(inventory, excluded)
    (dest / "selection-plan.json").write_text(json.dumps({"inventory": inventory, "excluded": sorted(excluded), "frames": chosen}, indent=1) + "\n")
    sources = []
    with tempfile.TemporaryDirectory(prefix="k4b-", dir="/tmp") as temp:
        scratch = Path(temp)
        with tarfile.open(dest / "frames.tar", "w") as archive:
            for c in inventory:
                rows = sorted((r for r in chosen if r["match"] == c["match"] and r["clip"] == c["clip"]), key=lambda r: r["frame"])
                if not rows:
                    continue
                source = Path(c["path"])
                contexts = []
                for r in rows:
                    start = max(0, min(r["frame"] - args.context_frames // 2, c["frames"] - args.context_frames))
                    contexts.append(list(range(start, min(c["frames"], start + args.context_frames))))
                indices = sorted({t for context in contexts for t in context})
                window = {"size": [1920, 1080], "source": {"kind": "video"}, "fps": c["fps"]}
                out = scratch / "selected.yuv"
                command = convert_command(args.ffmpeg, window, source, out, len(indices))
                vfi = command.index("-vf") + 1
                command[vfi] += ",select=" + "+".join(f"eq(n\\,{t})" for t in indices)
                command[command.index("-f"):command.index("-f")] = ["-fps_mode", "passthrough"]
                subprocess.run(command, check=True, capture_output=True, timeout=600)
                if out.stat().st_size != len(indices) * 1920 * 1080 * 3 // 2:
                    raise RuntimeError(f"Wrong converted frame count: {c}")
                member = f"{c['match']}__{c['clip']}.yuv"
                archive.add(out, arcname=member)
                for r, context in zip(rows, contexts):
                    r.update(member=member, offset=indices.index(r["frame"]), fps=c["fps"], context_frames=context,
                             context_offsets=[indices.index(t) for t in context])
                sources.append({**c, "sha256": file_sha256(source), "conversion": command,
                                "converted_sha256": file_sha256(out), "member": member, "selected_frames": len(indices)})
                print(f"prepared {c['match']}/{c['clip']}: {len(rows)} frames", flush=True)
    manifest: dict[str, Any] = {"schema": 1, "seed": SEED, "indexing": "zero-based decimal; clip stem",
                "selection": "smallest global hash prefix meeting >=48 frames, >=8 clips and all 5 matches; no content filter",
                "k4_windows_sha256": file_sha256(k4), "excluded": sorted(excluded),
                "review_context_frames": args.context_frames,
                "previous_selection_sha256": file_sha256(args.selection) if args.selection else None,
                "inventory": inventory, "frames": chosen, "sources": sources,
                "archive": entry("frames", dest / "frames.tar", extract=True)}
    (dest / "selection.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"frames": len(chosen), "clips": len(sources), "matches": sorted({r['match'] for r in chosen}),
                      "selection_sha256": file_sha256(dest / "selection.json"), "archive_sha256": manifest["archive"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
