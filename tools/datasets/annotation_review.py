"""Source-only, static foreground-union annotation review (no inference).

Input JSON: {"version": 1, "review_id": "pilot-1", "frames": [{"id": "a",
"source_group": "match-1", "frame_index": 12, "blind": false,
"source": {"path": "frame.png", "sha256": "..."},
"proposal": {"path": "mask.png", "sha256": "..."}}]}.
Paths resolve relative to the input manifest. Proposals must be binary PNGs,
with values 0/1 or 0/255 at source resolution. Blind assignments are frozen in
this manifest, before reviewing codec outcomes. Each item labels the complete
foreground union, not just whether a proposed instance is a player.

python -m tools.datasets.annotation_review prepare --manifest INPUT --out NEW_DIR
python tools/datasets/k8_review_server.py --root NEW_DIR
python -m tools.datasets.annotation_review import --review NEW_DIR --decisions DOWNLOAD --out NEW_DIR_2

The browser saves a hash-bound local draft after each edit and restores it on
reload, including unfinished polygons. Its status banner reports storage errors.
It exports decisions and never writes back to the server. Imported
records keep rejected/uncertain/missing-object cases. Approved corrections
replace the entire union with the union of drawn polygons. Blind items require
polygons or an explicit empty-foreground confirmation. Only approved rows get
reference masks, and unreviewed propagated frames gain no human provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact(value: Any, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid {name} fields")


def nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid {name}")


def asset(value: Any, root: Path) -> tuple[Path, bytes]:
    exact(value, {"path", "sha256"}, "asset")
    nonempty(value["path"], "asset path")
    if not isinstance(value["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", value["sha256"]):
        raise ValueError("invalid asset SHA256")
    path = (root / value["path"]).resolve()
    data = path.read_bytes()
    if digest(data) != value["sha256"]:
        raise ValueError("asset checksum mismatch")
    return path, data


def prepare(manifest: Path, output: Path) -> dict[str, Any]:
    raw = json.loads(manifest.read_text())
    exact(raw, {"version", "review_id", "frames"}, "manifest")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("unsupported manifest version")
    nonempty(raw["review_id"], "review ID")
    if not isinstance(raw["frames"], list) or not raw["frames"]:
        raise ValueError("empty frame selection")
    rows, assets, ids = [], [], set()
    for i, row in enumerate(raw["frames"]):
        exact(row, {"id", "source_group", "frame_index", "blind", "source", "proposal"}, "frame")
        nonempty(row["id"], "frame ID")
        nonempty(row["source_group"], "source group")
        if row["id"] in ids or type(row["frame_index"]) is not int or row["frame_index"] < 0 or type(row["blind"]) is not bool:
            raise ValueError("duplicate ID or invalid frame index/blinding")
        ids.add(row["id"])
        source, _ = asset(row["source"], manifest.parent)
        proposal, _ = asset(row["proposal"], manifest.parent)
        with Image.open(source) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("source must be PNG or JPEG")
            width, height = image.size
            suffix = ".png" if image.format == "PNG" else ".jpg"
        with Image.open(proposal) as image:
            mask = np.asarray(image)
            if image.format != "PNG" or mask.ndim != 2 or image.size != (width, height):
                raise ValueError("proposal must be source-resolution grayscale PNG")
            values = set(np.unique(mask).tolist())
            if not (values <= {0, 1} or values <= {0, 255}):
                raise ValueError("proposal must be binary")
            alpha = (mask != 0).astype(np.uint8) * 100
        source_name, overlay_name = f"source-{i}{suffix}", f"overlay-{i}.png"
        public = {k: row[k] for k in ("id", "source_group", "frame_index", "blind")}
        public.update(width=width, height=height, source=source_name,
                      source_sha256=row["source"]["sha256"], proposal_sha256=row["proposal"]["sha256"],
                      overlay=None if row["blind"] else overlay_name)
        overlay_data = None
        if not row["blind"]:
            rgba = np.zeros((*alpha.shape, 4), np.uint8)
            rgba[:, :, 1], rgba[:, :, 3] = 255, alpha
            buffer = io.BytesIO()
            Image.fromarray(rgba).save(buffer, format="PNG")
            overlay_data = buffer.getvalue()
        public["overlay_sha256"] = digest(overlay_data) if overlay_data is not None else None
        rows.append(public)
        assets.append((source, source_name, overlay_name, overlay_data))
    record = {"version": 1, "review_id": raw["review_id"], "frames": rows}
    record["review_sha256"] = digest(canonical(record))
    output.mkdir(parents=True, exist_ok=False)
    for source, source_name, overlay_name, overlay_data in assets:
        shutil.copyfile(source, output / source_name)
        if overlay_data is not None:
            (output / overlay_name).write_bytes(overlay_data)
    (output / "review.json").write_bytes(canonical(record))
    encoded = json.dumps(record).replace("<", "\\u003c")
    (output / "index.html").write_text(HTML.replace("__REVIEW__", encoded))
    return record


def load_review(root: Path) -> dict[str, Any]:
    review = json.loads((root / "review.json").read_text())
    expected = review.pop("review_sha256")
    if digest(canonical(review)) != expected:
        raise ValueError("review checksum mismatch")
    review["review_sha256"] = expected
    for row in review["frames"]:
        if digest((root / row["source"]).read_bytes()) != row["source_sha256"]:
            raise ValueError("review source changed")
        if row["overlay"] is not None and digest((root / row["overlay"]).read_bytes()) != row["overlay_sha256"]:
            raise ValueError("review proposal overlay changed")
    return review


def validate_decisions(review: dict[str, Any], decisions: Any) -> list[dict[str, Any]]:
    exact(decisions, {"version", "review_sha256", "reviewer", "elapsed_seconds", "frames"}, "decisions")
    if type(decisions["version"]) is not int or decisions["version"] != 1 or decisions["review_sha256"] != review["review_sha256"]:
        raise ValueError("wrong review identity")
    nonempty(decisions["reviewer"], "reviewer")
    elapsed = decisions["elapsed_seconds"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid review duration")
    if not isinstance(decisions["frames"], list):
        raise ValueError("invalid decisions")
    selected = {r["id"]: r for r in review["frames"]}
    rows, seen = [], set()
    for row in decisions["frames"]:
        exact(row, {"id", "source_sha256", "proposal_sha256", "status", "missing_object", "confirmed_empty", "polygons", "note"}, "decision")
        if not isinstance(row["id"], str) or row["id"] not in selected or row["id"] in seen:
            raise ValueError("unknown or duplicate decision")
        seen.add(row["id"])
        frame = selected[row["id"]]
        if any(row[k] != frame[k] for k in ("source_sha256", "proposal_sha256")):
            raise ValueError("frame/proposal identity mismatch")
        if row["status"] not in ("approved", "rejected", "uncertain") or type(row["missing_object"]) is not bool or type(row["confirmed_empty"]) is not bool or not isinstance(row["note"], str):
            raise ValueError("invalid annotation decision")
        polygons = row["polygons"]
        if not isinstance(polygons, list):
            raise ValueError("invalid polygons")
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError("polygon requires three vertices")
            for point in polygon:
                if not isinstance(point, list) or len(point) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
                    raise ValueError("invalid polygon vertex")
                if not (0 <= point[0] < frame["width"] and 0 <= point[1] < frame["height"]):
                    raise ValueError("polygon outside source")
            area = sum(polygon[i][0] * polygon[(i + 1) % len(polygon)][1] - polygon[(i + 1) % len(polygon)][0] * polygon[i][1] for i in range(len(polygon)))
            if abs(area) < 1:
                raise ValueError("degenerate polygon")
        if polygons and row["confirmed_empty"]:
            raise ValueError("nonempty polygons contradict empty foreground")
        if row["status"] == "approved" and (frame["blind"] or row["missing_object"]) and not (polygons or row["confirmed_empty"]):
            raise ValueError("blind/missing-object approval requires complete correction or confirmed empty")
        provenance = "human_without_proposal" if frame["blind"] else "human_reviewed_model_assisted"
        rows.append({**row, "source_group": frame["source_group"], "frame_index": frame["frame_index"], "provenance": provenance})
    if seen != set(selected):
        raise ValueError("every selected frame requires a decision, including failures")
    return rows


def import_decisions(root: Path, download: Path, output: Path) -> dict[str, Any]:
    review = load_review(root)
    data = download.read_bytes()
    decisions = json.loads(data)
    rows = validate_decisions(review, decisions)
    frames = {r["id"]: r for r in review["frames"]}
    masks = []
    for i, row in enumerate(rows):
        if row["status"] != "approved":
            continue
        frame = frames[row["id"]]
        mask = Image.new("L", (frame["width"], frame["height"]), 0)
        if row["polygons"]:
            draw = ImageDraw.Draw(mask)
            for polygon in row["polygons"]:
                draw.polygon([tuple(p) for p in polygon], fill=255)
        elif not row["confirmed_empty"]:
            with Image.open(root / frame["overlay"]) as overlay:
                alpha = np.asarray(overlay)[:, :, 3]
                mask = Image.fromarray((alpha > 0).astype(np.uint8) * 255)
        name = f"reference-{i}.png"
        row["reference_mask"] = name
        masks.append((name, mask))
    result = {"version": 1, "review_sha256": review["review_sha256"], "decisions_sha256": digest(data),
              "reviewer": decisions["reviewer"], "elapsed_seconds": decisions["elapsed_seconds"], "frames": rows}
    output.mkdir(parents=True, exist_ok=False)
    (output / "decisions-original.json").write_bytes(data)
    for name, mask in masks:
        mask.save(output / name)
    (output / "annotations.json").write_bytes(canonical(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--out", type=Path, required=True)
    importer = commands.add_parser("import")
    importer.add_argument("--review", type=Path, required=True)
    importer.add_argument("--decisions", type=Path, required=True)
    importer.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = prepare(args.manifest, args.out) if args.command == "prepare" else import_decisions(args.review, args.decisions, args.out)
    print(json.dumps({"frames": len(record["frames"]), "output": str(args.out)}))
    return 0


HTML = r'''<!doctype html><meta charset="utf-8"><title>Source annotation review</title>
<style>body{font:16px sans-serif;margin:24px;background:#eee}canvas{max-width:100%;background:#111;cursor:crosshair}button,input,select{font:inherit;margin:6px}#info{white-space:pre-wrap}</style>
<h1>Source foreground review</h1><p>Review the complete foreground union. Polygon corrections replace the entire union.
For a blind frame, draw all foreground or confirm it is empty. No codec outputs are shown.</p>
<p id="draft-status" role="status"></p><div id="info"></div><canvas id="view"></canvas><br>
<button id="previous">Previous</button><button id="next">Next</button>
<button id="toggle">Toggle proposal</button><button id="close">Finish polygon</button><button id="clear">Clear polygons</button><br>
<label>Decision <select id="status"><option value="">Unreviewed</option><option>approved</option><option>rejected</option><option>uncertain</option></select></label>
<label><input id="missing" type="checkbox">Missing object in proposal</label><label><input id="empty" type="checkbox">Confirm empty foreground</label><br>
<label>Note <input id="note" size="60"></label><label>Reviewer <input id="reviewer"></label><button id="download">Download decisions</button>
<script>
const review=__REVIEW__, started=Date.now(), records=review.frames.map(f=>({id:f.id,source_sha256:f.source_sha256,proposal_sha256:f.proposal_sha256,status:'',missing_object:false,confirmed_empty:false,polygons:[],note:''}));
let index=0, vertices=[], show=true, generation=0, priorSeconds=0, draftBlocked=false;const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
const field=id=>document.getElementById(id);
const draftKey='pointstream-annotation-review:'+review.review_sha256;
function elapsed(){return priorSeconds+(Date.now()-started)/1000;}
function validPoints(points,f){return Array.isArray(points)&&points.every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&p[0]>=0&&p[0]<f.width&&p[1]>=0&&p[1]<f.height);}
function validDraft(d){return d&&d.version===1&&d.review_sha256===review.review_sha256&&Number.isInteger(d.index)&&d.index>=0&&d.index<records.length&&typeof d.reviewer==='string'&&Number.isFinite(d.elapsed_seconds)&&d.elapsed_seconds>=0&&validPoints(d.vertices,review.frames[d.index])&&Array.isArray(d.records)&&d.records.length===records.length&&d.records.every((r,i)=>r&&r.id===records[i].id&&r.source_sha256===records[i].source_sha256&&r.proposal_sha256===records[i].proposal_sha256&&['','approved','rejected','uncertain'].includes(r.status)&&typeof r.missing_object==='boolean'&&typeof r.confirmed_empty==='boolean'&&typeof r.note==='string'&&Array.isArray(r.polygons)&&r.polygons.every(p=>p.length>=3&&validPoints(p,review.frames[i])));}
function persist(){if(draftBlocked)return;try{localStorage.setItem(draftKey,JSON.stringify({version:1,review_sha256:review.review_sha256,records,index,vertices,reviewer:field('reviewer').value,elapsed_seconds:elapsed()}));field('draft-status').textContent='Draft saved in this browser, including unfinished polygons. Keep this review URL to resume.';}catch(e){field('draft-status').textContent='Draft NOT saved: browser storage unavailable or full. Keep this tab open; do not reload until decisions are downloaded.';}}
function restore(){try{const raw=localStorage.getItem(draftKey);if(raw===null){field('draft-status').textContent='New review. Edits will be saved in this browser.';return;}const d=JSON.parse(raw);if(!validDraft(d))throw Error('invalid draft');d.records.forEach((r,i)=>Object.assign(records[i],r));index=d.index;vertices=d.vertices;field('reviewer').value=d.reviewer;priorSeconds=d.elapsed_seconds;field('draft-status').textContent='Saved draft restored, including unfinished polygons.';}catch(e){draftBlocked=true;field('draft-status').textContent='Draft could not be restored. Existing stored data was preserved; do not edit this tab if recovery is needed.';}}
function save(){const r=records[index];r.status=field('status').value;r.missing_object=field('missing').checked;r.confirmed_empty=field('empty').checked;r.note=field('note').value;persist();}
async function picture(path){return new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=reject;i.src=path;});}
async function paint(){const current=++generation,f=review.frames[index],r=records[index];const source=await picture(f.source);if(current!==generation)return;canvas.width=f.width;canvas.height=f.height;ctx.drawImage(source,0,0);if(f.overlay&&show){const overlay=await picture(f.overlay);if(current!==generation)return;ctx.drawImage(overlay,0,0);}ctx.lineWidth=Math.max(1,f.width/500);ctx.strokeStyle='#ff00ff';for(const p of [...r.polygons,vertices]){if(!p.length)continue;ctx.beginPath();p.forEach((v,i)=>i?ctx.lineTo(...v):ctx.moveTo(...v));if(p!==vertices)ctx.closePath();ctx.stroke();}}
function load(keepVertices=false){const f=review.frames[index],r=records[index];field('info').textContent=`${index+1}/${records.length}: ${f.id} | ${f.source_group} frame ${f.frame_index} | ${f.blind?'BLIND: no proposal available':'proposal review'}`;field('status').value=r.status;field('missing').checked=r.missing_object;field('empty').checked=r.confirmed_empty;field('note').value=r.note;field('toggle').disabled=f.blind;if(!keepVertices)vertices=[];show=true;paint();}
canvas.onclick=e=>{const b=canvas.getBoundingClientRect();vertices.push([Math.max(0,Math.min(canvas.width-1,(e.clientX-b.left)*canvas.width/b.width)),Math.max(0,Math.min(canvas.height-1,(e.clientY-b.top)*canvas.height/b.height))]);save();paint();};
field('close').onclick=()=>{if(vertices.length<3){alert('At least three vertices required');return;}records[index].polygons.push(vertices);vertices=[];save();paint();};
field('clear').onclick=()=>{records[index].polygons=[];vertices=[];save();paint();};field('toggle').onclick=()=>{show=!show;paint();};
field('previous').onclick=()=>{if(vertices.length){alert('Finish or clear the open polygon first');return;}save();index=Math.max(0,index-1);load();persist();};field('next').onclick=()=>{if(vertices.length){alert('Finish or clear the open polygon first');return;}save();index=Math.min(records.length-1,index+1);load();persist();};
field('download').onclick=()=>{save();if(vertices.length){alert('Finish or clear the open polygon first');return;}if(records.some(r=>!r.status)||!field('reviewer').value.trim()){alert('Review every frame and enter reviewer name');return;}const value={version:1,review_sha256:review.review_sha256,reviewer:field('reviewer').value,elapsed_seconds:elapsed(),frames:records};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));a.download='annotation-decisions.json';a.click();URL.revokeObjectURL(a.href);};
for(const id of ['status','missing','empty','note','reviewer'])field(id).addEventListener('input',save);
window.addEventListener('beforeunload',save);restore();load(true);
</script>'''

if __name__ == "__main__":
    raise SystemExit(main())
