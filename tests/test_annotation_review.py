"""Source-only annotation review: frozen identities and no omitted failures."""
import copy
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from tools.datasets.annotation_review import import_decisions, load_review, prepare, validate_decisions


def setup_review(tmp_path):
    source = tmp_path / "source.png"
    proposal = tmp_path / "mask.png"
    Image.new("RGB", (8, 6), (100, 120, 140)).save(source)
    Image.fromarray(np.asarray([[0] * 8] * 3 + [[255] * 8] * 3, np.uint8)).save(proposal)
    def item(path):
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"version": 1, "review_id": "pilot", "frames": [
        {"id": str(i), "source_group": "match-1", "frame_index": i, "blind": bool(i),
         "source": item(source), "proposal": item(proposal)} for i in range(2)]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    root = tmp_path / "review"
    review = prepare(path, root)
    return path, root, review


def decisions(review):
    return {"version": 1, "review_sha256": review["review_sha256"], "reviewer": "reviewer", "elapsed_seconds": 60,
            "frames": [{"id": f["id"], "source_sha256": f["source_sha256"], "proposal_sha256": f["proposal_sha256"],
                        "status": "approved", "missing_object": False, "confirmed_empty": f["blind"],
                        "polygons": [], "note": ""} for f in review["frames"]]}


def test_prepared_package_hides_blind_proposals_and_forbids_overwrite(tmp_path):
    path, root, review = setup_review(tmp_path)
    assert review["frames"][1]["overlay"] is None
    assert not (root / "overlay-1.png").exists()
    assert (root / "index.html").exists()
    assert load_review(root) == review
    with pytest.raises(FileExistsError):
        prepare(path, root)


def test_import_keeps_failures_and_only_approved_masks(tmp_path):
    _, root, review = setup_review(tmp_path)
    value = decisions(review)
    value["frames"][0].update(status="uncertain", missing_object=True, note="racket missing")
    download = tmp_path / "decisions.json"
    download.write_text(json.dumps(value))
    output = tmp_path / "annotations"
    result = import_decisions(root, download, output)
    assert len(result["frames"]) == 2
    assert result["frames"][0]["missing_object"]
    assert "reference_mask" not in result["frames"][0]
    assert result["frames"][1]["provenance"] == "human_without_proposal"
    assert not np.asarray(Image.open(output / "reference-1.png")).any()
    assert (output / "decisions-original.json").read_bytes() == download.read_bytes()
    with pytest.raises(FileExistsError):
        import_decisions(root, download, output)


def test_approved_proposal_and_polygon_replacement(tmp_path):
    _, root, review = setup_review(tmp_path)
    value = decisions(review)
    value["frames"][1].update(confirmed_empty=False, polygons=[[[1, 1], [4, 1], [4, 4], [1, 4]]])
    download = tmp_path / "decisions.json"
    download.write_text(json.dumps(value))
    result = import_decisions(root, download, tmp_path / "annotations")
    assert result["frames"][0]["provenance"] == "human_reviewed_model_assisted"
    assert np.asarray(Image.open(tmp_path / "annotations/reference-0.png")).sum() == 24 * 255
    assert np.asarray(Image.open(tmp_path / "annotations/reference-1.png")).sum() == 16 * 255


@pytest.mark.parametrize("change", ["omit", "duplicate", "review", "source", "proposal", "blind", "missing", "outside", "degenerate", "nan"])
def test_invalid_decisions_are_rejected(tmp_path, change):
    _, _, review = setup_review(tmp_path)
    value = decisions(review)
    if change == "omit":
        value["frames"].pop()
    elif change == "duplicate":
        value["frames"].append(copy.deepcopy(value["frames"][0]))
    elif change == "review":
        value["review_sha256"] = "0" * 64
    elif change in ("source", "proposal"):
        value["frames"][0][change + "_sha256"] = "0" * 64
    elif change == "blind":
        value["frames"][1]["confirmed_empty"] = False
    elif change == "missing":
        value["frames"][0]["missing_object"] = True
    elif change == "outside":
        value["frames"][0]["polygons"] = [[[0, 0], [9, 0], [0, 4]]]
    elif change == "degenerate":
        value["frames"][0]["polygons"] = [[[0, 0], [1, 1], [2, 2]]]
    else:
        value["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError):
        validate_decisions(review, value)


def test_manifest_rejects_codec_fields_and_asset_changes(tmp_path):
    path, _, _ = setup_review(tmp_path)
    value = json.loads(path.read_text())
    value["frames"][0]["codec_quality"] = 0.1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fields"):
        prepare(path, tmp_path / "bad")
    del value["frames"][0]["codec_quality"]
    value["frames"][0]["source"]["sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        prepare(path, tmp_path / "bad")


def test_import_detects_changed_overlay(tmp_path):
    _, root, _ = setup_review(tmp_path)
    (root / "overlay-0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="overlay changed"):
        load_review(root)


def test_browser_draft_restores_open_polygon_and_rejects_other_review(tmp_path):
    """Execute the generated script with a tiny DOM; no browser dependency needed."""
    import shutil
    import subprocess
    from tools.datasets.annotation_review import HTML

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable for generated JavaScript execution")
    _, _, review = setup_review(tmp_path)
    script = HTML.split("<script>")[1].split("</script>")[0].replace("__REVIEW__", json.dumps(review))
    test = r'''
const vm=require('node:vm'),assert=require('node:assert/strict');
const script=SCRIPT;
function environment(storage){
 const elements={};
 const context={console,Date,JSON,Math,Number,Array,Object,Promise,Error,
 localStorage:storage,window:{addEventListener(){}},Image:class {},
 document:{getElementById(id){return elements[id]??=(id==='view'?{getContext(){return {}},width:8,height:6}:{value:'',checked:false,textContent:'',addEventListener(){}})}}};
 vm.createContext(context);vm.runInContext(script,context);return {context,elements};
}
const entries={};const storage={getItem:k=>entries[k]??null,setItem:(k,v)=>entries[k]=v};
let a=environment(storage);
vm.runInContext("field('reviewer').value='Alice';field('note').value='partial boundary';vertices=[[1,1],[3,1]];save()",a.context);
assert.match(a.elements['draft-status'].textContent,/Draft saved/);
let b=environment(storage);
assert.equal(b.elements.reviewer.value,'Alice');assert.equal(b.elements.note.value,'partial boundary');
assert.equal(vm.runInContext('vertices.length',b.context),2);
assert.match(b.elements['draft-status'].textContent,/restored/);
const key=Object.keys(entries)[0],original=entries[key];
let bad=JSON.parse(original);bad.review_sha256='wrong';entries[key]=JSON.stringify(bad);
let c=environment(storage);const preserved=entries[key];vm.runInContext('save()',c.context);
assert.equal(entries[key],preserved);assert.equal(c.elements.note.value,'');
assert.match(c.elements['draft-status'].textContent,/could not be restored/);
let d=environment({getItem:()=>null,setItem(){throw Error('quota')}});
vm.runInContext('save()',d.context);assert.match(d.elements['draft-status'].textContent,/NOT saved/);
'''.replace("SCRIPT", json.dumps(script))
    path = tmp_path / "draft-test.js"
    path.write_text(test)
    subprocess.run([node, str(path)], check=True, capture_output=True, text=True, timeout=15)


def test_v2_visible_proposals_require_final_correction_without_blind_controls(tmp_path):
    path, _, _ = setup_review(tmp_path)
    manifest = json.loads(path.read_text())
    manifest['version'] = 2
    manifest['review_id'] = 'final-mask-v2'
    for row in manifest['frames']:
        row.update(blind=False, requires_complete_correction=True)
    path.write_text(json.dumps(manifest))
    root = tmp_path / 'v2'
    review = prepare(path, root)
    assert review['version'] == 2
    assert all(f['overlay'] is not None and not f['blind'] for f in review['frames'])
    assert 'Rough polygons are NOT reference masks' in (root / 'index.html').read_text()
    value = decisions(review)
    with pytest.raises(ValueError, match='identity'):
        validate_decisions(review, value)
    value['version'] = 2
    with pytest.raises(ValueError, match='complete correction'):
        validate_decisions(review, value)
    for row in value['frames']:
        row['status'] = 'rejected'
        row['polygons'] = [[[1, 1], [4, 1], [4, 4], [1, 4]]]
    assert all(r['status'] == 'rejected' for r in validate_decisions(review, value))
    download = tmp_path / 'human-v2.json'
    download.write_text(json.dumps(value))
    output = tmp_path / 'v2-import'
    result = import_decisions(root, download, output)
    assert result['version'] == 2
    assert not list(output.glob('reference-*'))


def test_v2_final_union_preview_replaces_original_overlay(tmp_path):
    import shutil
    import subprocess
    from tools.datasets.annotation_review import HTML
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node unavailable')
    paint = next(line for line in HTML.splitlines() if line.startswith('async function paint()'))
    script = '''const assert=require('node:assert/strict');
let generation=0,index=0,show=true,vertices=[];
const calls=[];
const ctx={drawImage(){calls.push('image')},beginPath(){},moveTo(){},lineTo(){},closePath(){},fill(){calls.push('fill')},stroke(){}};
const canvas={};const picture=async path=>path;
const review={version:2,frames:[{source:'source',overlay:'old',width:8,height:6}]};
const records=[{confirmed_empty:false,polygons:[[[1,1],[4,1],[4,4]]]}];
PAINT
(async()=>{
await paint();assert.deepEqual(calls,['image','fill']);
calls.length=0;records[0].polygons=[];records[0].confirmed_empty=true;
await paint();assert.deepEqual(calls,['image']);
calls.length=0;review.version=1;records[0].confirmed_empty=false;
await paint();assert.deepEqual(calls,['image','image']);
})().catch(e=>{console.error(e);process.exitCode=1});
'''.replace('PAINT',paint)
    path=tmp_path/'preview.js'
    path.write_text(script)
    subprocess.run([node,str(path)],check=True,capture_output=True,text=True,timeout=15)
