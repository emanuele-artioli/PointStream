"""A raw instance may be ignored by the union rule but remains preserved."""
import hashlib
import io
import json
import tarfile

import numpy as np
from PIL import Image

from tools.datasets.annotation_review import prepare, load_review
from tools.datasets import visor_action_revision as revision


def test_fixed_union_and_standalone_keep_identities(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (1920, 1080), (30, 40, 50)).save(source)
    source_sha = revision.sha(source)
    def png(x):
        pixels = np.zeros((1080, 1920), np.uint8)
        pixels[0, x] = 255
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format='PNG')
        return output.getvalue()
    official = tmp_path / 'official.png'
    official.write_bytes(png(0))
    manifest = {'version': 1, 'review_id': 'old', 'frames': [
        dict(id=str(i), source_group='P35_105', frame_index=i, blind=False,
             source=dict(path=str(source), sha256=source_sha),
             proposal=dict(path=str(official), sha256=revision.sha(official))) for i in range(18)]}
    path = tmp_path / 'old-manifest.json'
    path.write_text(json.dumps(manifest))
    old_root = tmp_path / 'old'
    old = prepare(path, old_root)
    human = dict(version=1, review_sha256=old['review_sha256'], reviewer='test', elapsed_seconds=1,
                 frames=[dict(id=r['id'],source_sha256=r['source_sha256'],proposal_sha256=r['proposal_sha256'],
                              status='rejected',missing_object=False,confirmed_empty=False,
                              polygons=[[[20,20],[30,20],[30,30]]],note='rough') for r in old['frames']])
    human_path = tmp_path / 'decisions.json'
    human_path.write_text(json.dumps(human))
    monkeypatch.setattr(revision, 'EXPECTED_HUMAN', revision.sha(human_path))
    filtered = dict(frames=[dict(id=str(i),video='P35_105',native_index_0based=i,
                    source_png_sha256=source_sha,proposal='official.png',proposal_sha256=revision.sha(official),
                    instance_inventory=[dict(name='saucepan',keep_candidate=True,native_annotation_canvas_pixels=int(i>0))]) for i in range(18)])
    (tmp_path/'filtered-proposals.json').write_text(json.dumps(filtered))
    records=[]
    archive_path = tmp_path/'sam.tar'
    with tarfile.open(archive_path, 'w') as archive:
        for i in range(18):
            for prompt, x in [('person',1),('hand',2),('object',3)]:
                name=f'{i}-{prompt}.png'
                mask=png(x)
                info=tarfile.TarInfo('masks/'+name)
                info.size=len(mask)
                archive.addfile(info,io.BytesIO(mask))
                records.append(dict(id=f'P35_105_native_{i}',prompt=prompt,source_sha256=source_sha,
                                    instances=[dict(mask=name,mask_sha256=hashlib.sha256(mask).hexdigest(),
                                                    status='present',score=0.9)]))
        raw=json.dumps(dict(count=18,records=records)).encode()
        info=tarfile.TarInfo('result.json')
        info.size=len(raw)
        archive.addfile(info,io.BytesIO(raw))
    out=tmp_path/'new'
    result=revision.combine(tmp_path,old_root,human_path,out,sam_archive=archive_path,sam_sha256=revision.sha(archive_path))
    assert len(result['frames'][0]['included_SAM']) == 3
    assert len(result['frames'][1]['ignored_SAM']) == 1
    assert (out/'raw-SAM-proposals.tar').read_bytes()==archive_path.read_bytes()
    pixels=np.asarray(Image.open(out/'inputs/combined-0.png'))
    assert pixels.sum()==4*255
    assert not pixels[20:31,20:31].any()  # Rejected rough polygon is never a mask.
    pixels=np.asarray(Image.open(out/'inputs/combined-1.png'))
    assert pixels.sum()==3*255
    review=load_review(out/'review')
    assert review['version']==2 and all(not r['blind'] for r in review['frames'])
    assert result['new_reference_annotations']==0
    html=(out/'review/review-standalone.html').read_text()
    assert html.startswith('<!doctype html>')
    assert 'data:image/png;base64,' in html
    assert 'i.src=portableReviewAssets[path]||path;' in html
    assert 'actively moved or manipulated' in html
    assert review==load_review(out/'review')  # Portable page leaves canonical metadata/assets unchanged.
