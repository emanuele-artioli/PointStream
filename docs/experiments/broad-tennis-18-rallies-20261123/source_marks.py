"""Build private original-only ball and named-player annotation pages from verified TenniSet points."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
ATHLETES = {
    "v006": ["Serena Williams", "Maria Sharapova"],
    "v007": ["Novak Djokovic", "Jo-Wilfried Tsonga"],
}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, record):
    with path.open("x") as f:
        json.dump(record, f, indent=2)
        f.write("\n")


def choose(n, shots):
    ticks = {(2 * j + 1) * n // 16: "primary" for j in range(8)}
    if len(ticks) != 8 or [t for shot in shots for t in range(shot["start"], shot["stop"])] != list(
        range(n)
    ):
        raise ValueError("point timing or shot coverage mismatch")
    for s in shots:
        if s["predicted"] == "non_court" and all(t < s["start"] or t >= s["stop"] for t in ticks):
            ticks[(s["start"] + s["stop"]) // 2] = "fallback-diagnostic"
    return ticks


def sample(source, row, route, native, output):
    rid = row["rally_id"]
    start, stop = row["native_range"]
    n = stop - start
    if (
        row["lossless_extract_sha256"] != file_sha(source)
        or route["source_sha256"] != row["lossless_extract_sha256"]
        or native["native_range"] != row["native_range"]
    ):
        raise ValueError("source, route or extraction lineage mismatch")
    ticks = choose(n, route["shots"])
    assets = output / "assets"
    assets.mkdir(parents=True)
    hashes, frames, selections = [], [], []
    sheet = Image.new("RGB", (4 * 320, ((len(ticks) + 3) // 4) * 202), "white")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "2",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-pix_fmt",
        "rgb24",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-",
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as proc:
        for t in range(n):
            data = proc.stdout.read(1280 * 720 * 3)
            if len(data) != 1280 * 720 * 3:
                raise ValueError("truncated source")
            rgb_sha = digest(data)
            hashes.append(rgb_sha)
            if t not in ticks:
                continue
            im = Image.frombytes("RGB", (1280, 720), data)
            png = assets / f"{rid}-{t:03d}.png"
            im.save(png)
            if im.tobytes() != Image.open(png).tobytes():
                raise ValueError("PNG changed a source pixel")
            ix = len(frames)
            x, y = ix % 4 * 320, ix // 4 * 202
            sheet.paste(im.resize((320, 180), Image.Resampling.LANCZOS), (x, y))
            ImageDraw.Draw(sheet).text(
                (x + 4, y + 183), f"{rid}   native tick {t}   broadcast {t + start}", fill="black"
            )
            shot = next(s for s in route["shots"] if s["start"] <= t < s["stop"])
            frames.append(
                dict(
                    id=f"{rid}-{t}",
                    frame_index=t,
                    absolute_tick=start + t,
                    source="assets/" + png.name,
                    source_sha256=file_sha(png),
                    source_rgb_sha256=rgb_sha,
                    players=[{"id": name} for name in ATHLETES[rid[:4]]],
                )
            )
            selections.append(
                dict(
                    rally_id=rid,
                    tick=t,
                    absolute_tick=t + start,
                    sample=ticks[t],
                    route=shot["predicted"],
                    shot=[shot["start"], shot["stop"]],
                    rgb_sha256=rgb_sha,
                )
            )
        if proc.stdout.read(1) or proc.wait():
            raise ValueError("extra or undecodable original frame")
    if digest("\n".join(hashes).encode()) != native["proof"]["rgb24"]["ordered_hash_sha256"]:
        raise ValueError("source pixel timeline differs from the authenticated extraction")
    sheet.save(output / "contact-sheet.jpg", quality=90)
    return frames, selections


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("source_dir", type=Path)
    arg.add_argument("packet", type=Path)
    arg.add_argument("receipt", type=Path)
    a = arg.parse_args()
    a.packet.mkdir(exist_ok=False, parents=True)
    route_path = HERE / "route-summary.json"
    route = json.loads(route_path.read_text())["rows"]
    manifest = json.loads((HERE / "manifest.json").read_text())["events"]
    template = (
        HERE.parent / "joint-replication-feasibility-20261122/review_template.html"
    ).read_text()
    items, links, selections = [], [], []
    for ev in manifest:
        if not ev["held_out"]:
            continue
        for row in ev["rallies"]:
            rid = row["rally_id"]
            original = next(
                x
                for x in json.loads((HERE / row["extraction_receipt"]).read_text())["rallies"]
                if x["rally_id"] == rid
            )
            frames, picks = sample(
                a.source_dir / original["filename"],
                row,
                next(x for x in route if x["rally_id"] == rid),
                original,
                a.packet / rid,
            )
            title = f"{' and '.join(ATHLETES[rid[:4]])} · point {int(rid.split('_p')[1])}"
            review = dict(
                version=1,
                title=title,
                draw_from_source=True,
                athletes=ATHLETES[rid[:4]],
                width=1280,
                height=720,
                source_video_sha256=row["lossless_extract_sha256"],
                original_video_sha256=row["source_sha256"],
                review_id=rid,
                source_group=rid,
                source_video_fps=[25, 1],
                frames=frames,
                storage_prefix="pointstream-tenniset-source-",
                download_name=rid + "-source-marks.json",
            )
            review["review_sha256"] = digest(
                json.dumps(review, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            )
            write_json(a.packet / rid / "review.json", review)
            (a.packet / rid / "index.html").write_text(
                template.replace("__REVIEW_JSON__", json.dumps(review, separators=(",", ":")))
            )
            items.append(
                dict(rally_id=rid, review_sha256=review["review_sha256"], frames=len(frames))
            )
            links.append(
                f'<li><a href="{rid}/index.html">{title}</a> · <a href="{rid}/contact-sheet.jpg">contact sheet</a></li>'
            )
            selections.extend(picks)
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>London 2012 source marks</title><style>body{font:18px system-ui;max-width:50em;margin:2em auto;padding:1em}li{margin:.8em 0}</style>"
        "<h1>London 2012 source marks</h1><p>Each page contains eight original broadcast frames. Mark the ball and both named athletes, then export the decisions for that point.</p><ul>"
        + "".join(links)
        + "</ul></html>"
    )
    (a.packet / "index.html").write_text(html)
    receipt = dict(
        schema=1,
        selection="Eight equally spaced bin midpoints per point, then uncovered fallback-shot midpoints separately.",
        route_summary_sha256=file_sha(route_path),
        reviewed_frames=0,
        pages=items,
        route_counts=dict(Counter(s["route"] for s in selections)),
        frames=selections,
    )
    write_json(a.receipt, receipt)
    print(a.packet, len(selections), receipt["route_counts"])


if __name__ == "__main__":
    main()
