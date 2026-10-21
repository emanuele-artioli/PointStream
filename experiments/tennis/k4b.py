"""K4b.1 marking stage: K4 conversion and SAM path, then a self-contained review page.

No player rule is applied here. Short real temporal SAM sessions supply
the review instances; later clip-level candidates are matched to these masks by IoU.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.tennis import k4, windows as tw
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.segmentation.masks import ClipMasks
from tools.datasets.k4b_inputs import frame_hash, select_frames

PROMPTS = {"person": "person", "player": "tennis player"}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def numbered(clip: ClipMasks) -> list[dict[str, Any]]:
    """Stable display IDs, duplicate prompt masks grouped at IoU >= .8.

    All raw masks remain in ClipMasks; grouping affects only review numbering.
    A group's first mask is the review/scoring mask; its source is recorded.
    """
    out: list[dict[str, Any]] = []
    for inst in sorted(clip.frames[0], key=lambda x: (x.class_name, x.track_id)):
        mask = inst.mask().astype(bool)
        if not mask.any():
            continue
        source = {"prompt": PROMPTS[inst.class_name], "track": int(inst.track_id), "score": float(inst.score)}
        same = next((r for r in out if iou(mask, r["mask"]) >= .8), None)
        if same is not None:
            same["sources"].append(source)
        else:
            out.append({"id": len(out) + 1, "mask": mask, "sources": [source]})
    return out


def data_image(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def review_frame(rgb: np.ndarray, groups: list[dict[str, Any]]) -> dict[str, Any]:
    import cv2

    painted = rgb.copy()
    boxes = []
    for row in groups:
        mask = k4.upsample(row["mask"], (1920, 1080))
        ys, xs = np.nonzero(mask)
        colour = np.array([(row["id"] * 97) % 190 + 60, (row["id"] * 47) % 190 + 60, (row["id"] * 137) % 190 + 60])
        painted[mask] = (painted[mask] * .65 + colour * .35).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(painted, contours, -1, tuple(int(c) for c in colour), 2)
        x, y = int(xs.mean()), max(32, int(ys.min()))
        label = str(row["id"])
        cv2.putText(painted, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 7)
        cv2.putText(painted, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        boxes.append({"id": row["id"], "sources": row["sources"],
                      "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]})
    return {"image": data_image(painted), "original": data_image(rgb), "instances": boxes}


def page(rows: list[dict[str, Any]], selection_sha: str, result_sha: str) -> str:
    payload = json.dumps({"schema": 1, "selection_sha256": selection_sha, "result_sha256": result_sha,
                          "frames": rows}, separators=(",", ":")).replace("<", "\\u003c")
    return HTML.replace("__PAYLOAD__", payload)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>PointStream K4b — mark the players</title>
<style>body{font:16px system-ui;margin:20px auto;max-width:1450px;background:#15191f;color:#eee}
button,select,input{font:inherit;padding:9px;margin:4px;border-radius:5px}button{cursor:pointer}
button.selected{background:#9ef0b5}img{width:100%;display:block}header{position:sticky;top:0;background:#15191f;padding:8px;z-index:1}
#status{color:#a8e4bd}fieldset{border:1px solid #556;margin:8px 0}small{color:#bbc}#title{font-weight:bold}</style>
<header><div id="title"></div><button id="prev">← Previous</button><button id="next">Next →</button>
<select id="jump" aria-label="Choose frame"></select><button id="download">Download marks JSON</button>
<label>Import saved marks <input id="import" type="file" accept=".json"></label><div id="status"></div></header>
<p>Mark the tennis players, excluding officials, ball kids and spectators. Click the numbered people below the image.
Then set how many players are actually visible (two, one or none) and confirm. If a visible player has no numbered mask,
select the available player and tick “Missing instance”. Do not substitute an official. No predictions are shown.</p>
<button id="toggle">Show original image</button><img id="image" alt="Broadcast frame with numbered person masks">
<fieldset><legend>Click the player instances (at most two)</legend><div id="people"></div></fieldset>
<label>Players visible <select id="visible"><option value="">Choose…</option><option value="2">Two</option><option value="1">One</option><option value="0">None</option></select></label>
<label><input id="missing" type="checkbox">Missing instance for a visible player</label>
<label>Notes <input id="notes" size="50"></label><button id="confirm">Confirm frame</button><p id="error" role="alert"></p>
<small>Progress is saved in this browser when storage is available. Download the JSON before closing; it can be imported to continue.
Frame numbers are zero-based. Marks remain local; this page makes no network requests.</small>
<script id="data" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),frames=data.frames;
const key='pointstream-k4b:'+data.selection_sha256+':'+data.result_sha256;
let marks={},pos=0,original=false; const $=id=>document.getElementById(id);
try{marks=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function persist(){try{localStorage.setItem(key,JSON.stringify(marks))}catch(e){$('error').textContent='Browser storage unavailable. Download marks before closing.'}}
function draft(){const id=frames[pos].id; if(!marks[id])marks[id]={selected:[],visible:null,missing_instance:false,notes:'',confirmed:false};return marks[id]}
function status(){$('status').textContent=Object.values(marks).filter(m=>m.confirmed).length+' / '+frames.length+' frames confirmed';}
function dirty(){draft().confirmed=false;persist();status()}
frames.forEach((f,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' · '+f.id;$('jump').append(o)});
function show(){const f=frames[pos],m=draft();$('title').textContent='K4b · '+(pos+1)+' / '+frames.length+' · '+f.id;
$('jump').value=pos;$('image').src=original?f.original:f.image;$('people').replaceChildren();
for(const inst of f.instances){const b=document.createElement('button');b.textContent='Person '+inst.id;b.className=m.selected.includes(inst.id)?'selected':'';
b.setAttribute('aria-pressed',m.selected.includes(inst.id));b.onclick=()=>{if(m.selected.includes(inst.id))m.selected=m.selected.filter(x=>x!==inst.id);else if(m.selected.length<2)m.selected.push(inst.id);dirty();show()};$('people').append(b)}
if(!f.instances.length)$('people').textContent='No person instances found. Set visibility and missing-instance flag as appropriate.';
$('visible').value=m.visible===null?'':m.visible;$('missing').checked=m.missing_instance;$('notes').value=m.notes;
$('toggle').textContent=original?'Show numbered masks':'Show original image';$('error').textContent='';status()}
$('visible').onchange=()=>{draft().visible=$('visible').value===''?null:Number($('visible').value);dirty()};
$('missing').onchange=()=>{draft().missing_instance=$('missing').checked;dirty()};$('notes').oninput=()=>{draft().notes=$('notes').value;dirty()};
$('confirm').onclick=()=>{const m=draft();if(m.visible===null||m.selected.length>m.visible||(!m.missing_instance&&m.selected.length!==m.visible)||(m.missing_instance&&m.selected.length>=m.visible)){$('error').textContent='Choose visibility and the corresponding instances; mark a missing instance only if a visible player has no selectable mask.';return}m.confirmed=true;persist();if(pos<frames.length-1)pos++;show()};
$('prev').onclick=()=>{pos=Math.max(0,pos-1);show()};$('next').onclick=()=>{pos=Math.min(frames.length-1,pos+1);show()};$('jump').onchange=()=>{pos=Number($('jump').value);show()};$('toggle').onclick=()=>{original=!original;show()};
$('download').onclick=()=>{const body={schema:1,selection_sha256:data.selection_sha256,result_sha256:data.result_sha256,exported_at:new Date().toISOString(),marks};const url=URL.createObjectURL(new Blob([JSON.stringify(body,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='pointstream-k4b-marks.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
$('import').onchange=async()=>{try{const b=JSON.parse(await $('import').files[0].text());if(b.schema!==1||b.selection_sha256!==data.selection_sha256||b.result_sha256!==data.result_sha256)throw Error('Marks belong to another review page');for(const [id,m]of Object.entries(b.marks)){const f=frames.find(f=>f.id===id);if(!f||!Array.isArray(m.selected)||new Set(m.selected).size!==m.selected.length||m.selected.some(n=>!f.instances.some(i=>i.id===n))||m.selected.length>2||![null,0,1,2].includes(m.visible))throw Error('Invalid marks')}marks=b.marks;persist();show()}catch(e){$('error').textContent=e.message}};show();
</script></html>'''


def run(args: argparse.Namespace) -> int:
    import torch

    from src.segmentation.sam31 import Sam31SequenceSegmenter
    from src.segmentation.sam31_tracker import runtime

    manifest = json.loads(Path(args.selection).read_text())
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir(exist_ok=True)
    for source in manifest["sources"]:
        if file_sha256(Path(args.frames) / source["member"]) != source["converted_sha256"]:
            raise ValueError(f"Converted source identity mismatch: {source['member']}")
    limit = os.environ.get("PS_GPU_MEMORY_LIMIT_MIB")
    if limit:
        torch.cuda.set_per_process_memory_fraction(int(limit) * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    segmenter = Sam31SequenceSegmenter(checkpoint_path=args.sam_checkpoint)
    provenance = segmenter.provenance("offline_bidirectional")
    sam = {"runtime": runtime(), "sdpa_backend_policy": segmenter.sdpa_backend_policy,
           "model_revision": provenance.model_revision, "checkpoint_sha256": provenance.checkpoint_sha256,
           "config_sha256": provenance.config_sha256, "prob_threshold": segmenter.prob_threshold, "prompts": PROMPTS}
    if not 0 < int(args.count) <= len(manifest["frames"]):
        raise ValueError("Count must be a nonempty prefix of the frozen selection")
    selected = manifest["frames"][:int(args.count)]
    rows = []
    for i, row in enumerate(selected):
        source = Path(args.frames) / row["member"]
        context = []
        for offset in row["context_offsets"]:
            raw = np.memmap(source, dtype=np.uint8, mode="r", offset=offset * 1920 * 1080 * 3 // 2, shape=(1080 * 3 // 2, 1920))
            context.append(tw.yuv_to_rgb(raw, 1920, 1080))
        target = row["context_frames"].index(row["frame"])
        rgb = context[target]
        work = scratch / f"frame-{i}"
        sequence = segmenter.segment_frames(k4.write_jpegs(np.stack(context), work), PROMPTS, policy="offline_bidirectional", fps=row["fps"])
        if len(sequence.frames) != len(context):
            raise RuntimeError("SAM must return every real context frame")
        clip = ClipMasks(sequence.classes, sequence.height, sequence.width, sequence.fps,
                         frames=[sequence.frames[target]], meta={"context_frames": row["context_frames"], "target": target})
        mask_path = publish / f"masks-{i:03d}.json.zst"
        clip.save(mask_path)
        groups = numbered(clip)
        review = review_frame(rgb, groups)
        rows.append({**row, "id": f"{row['match']}/{row['clip']}/{row['frame']}",
                     "masks": mask_path.name, "masks_sha256": file_sha256(mask_path), **review})
        write_json(publish / f"frame-{i:03d}.json", rows[-1])
        progress(i + 1)
    record = {"schema": 1, "sam": sam, "selection_sha256": file_sha256(Path(args.selection)),
              "input_archive_sha256": manifest["archive"]["sha256"], "count": len(rows),
              "session": "eight real adjacent frames per selected frame; only the selected frame is reviewed; clip-level rules are not evaluated",
              "frames": [{k: v for k, v in r.items() if k not in ("image", "original")} for r in rows],
              "converted_inputs_verified": True,
              "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    write_json(publish / "instances.json", record)
    (publish / "index.html").write_text(page(rows, record["selection_sha256"], file_sha256(publish / "instances.json")))
    write_json(stage_dir() / "k4b.json", {"count": len(rows), "sam": sam,
                                           "selection": manifest, "record_sha256": file_sha256(publish / "instances.json")})
    return 0


def validate_directory(directory: Path, summary: Path) -> dict[str, Any]:
    meta = json.loads(summary.read_text())
    record = json.loads((directory / "instances.json").read_text())
    selection = meta["selection"]
    expected = select_frames(selection["inventory"], set(selection["excluded"]))
    frames = record["frames"]
    sam = record["sam"]
    checks = {
        "selection_recomputed_from_inventory": [{k: r[k] for k in ("match", "clip", "frame", "rank")} for r in selection["frames"]] == expected,
        "no_k4_clip": all(f"broadcast/{r['match']}/{r['clip']}" not in selection["excluded"] for r in frames),
        "global_hashes": all(r["rank"] == frame_hash(r["match"], r["clip"], r["frame"]) for r in frames),
        "expected_frame_count": len(frames) == meta["count"] == record["count"] and len(frames) > 0,
        "expected_prefix": [r["rank"] for r in frames] == [r["rank"] for r in expected[:len(frames)]],
        "cuda_supported": sam["runtime"]["capability"][0] >= 8 and bool(sam["runtime"]["cuda"]),
        "native_flash": sam["sdpa_backend_policy"] == "native_flash",
        "pinned_checkpoint": sam["checkpoint_sha256"] == "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        "both_prompts": sam["prompts"] == PROMPTS,
        "record_identity": file_sha256(directory / "instances.json") == meta["record_sha256"],
        "nonempty_model_output": any(r["instances"] for r in frames),
        "real_temporal_context": all(len(r["context_frames"]) >= 3 and r["frame"] in r["context_frames"]
                                     and r["context_frames"] == list(range(r["context_frames"][0], r["context_frames"][-1] + 1)) for r in frames),
        "converted_inputs_verified": record["converted_inputs_verified"] is True,
        "mask_integrity": True, "instance_numbering": True, "page_payload": True,
    }
    html = (directory / "index.html").read_text()
    embedded = json.loads(html.split('<script id="data" type="application/json">')[1].split('</script>')[0])
    checks["page_payload"] = (embedded["result_sha256"] == meta["record_sha256"] and
                               embedded["selection_sha256"] == record["selection_sha256"] and len(embedded["frames"]) == len(frames))
    for i, row in enumerate(frames):
        path = directory / row["masks"]
        clip = ClipMasks.load(path)
        groups = numbered(clip)
        checks["mask_integrity"] &= file_sha256(path) == row["masks_sha256"] and len(clip.frames) == 1 and (clip.height, clip.width) == (540, 960)
        checks["instance_numbering"] &= row["instances"] == review_frame(np.zeros((1080, 1920, 3), np.uint8), groups)["instances"]
        checks["page_payload"] &= embedded["frames"][i]["id"] == row["id"] and embedded["frames"][i]["instances"] == row["instances"]
        for name in ("image", "original"):
            im = Image.open(io.BytesIO(base64.b64decode(embedded["frames"][i][name].split(",", 1)[1])))
            im.load()
            checks["page_payload"] &= im.size == (1920, 1080)
    return {"passed": all(checks.values()), "checks": checks, "frames": len(frames)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--selection", required=True)
    r.add_argument("--frames", required=True)
    r.add_argument("--sam-checkpoint", required=True)
    r.add_argument("--count", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--directory", type=Path)
    v.add_argument("--summary", type=Path)
    args = p.parse_args()
    if args.command == "run":
        return run(args)
    summary = args.summary or stage_dir() / "k4b.json"
    if args.directory:
        result = validate_directory(args.directory, summary)
    else:
        # The fleet publishes and removes host-local scratch before validation.
        with tempfile.TemporaryDirectory(prefix="k4b-validate-", dir="/tmp") as temp:
            with tarfile.open(stage_dir() / "published.tar") as archive:
                archive.extractall(temp, filter="data")
            result = validate_directory(Path(temp) / "publish", summary)
    if os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(os.environ["PS_VALIDATION_PATH"]), result)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
