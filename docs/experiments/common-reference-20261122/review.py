"""Build a blinded 24-frame AVIF player review from authenticated videos."""
import hashlib
import html
import json
import random
import shutil
from pathlib import Path

PREV = Path('/Users/manu/Documents/PointStream-review/2026-11-18-reference-appearance/q70-q40')
OLD = Path('docs/experiments/appearance-resume-20261118/review-delivery.json')
NEW = Path('/private/tmp/ps-common-avif-q65-review-clips-20261122')
DEST = Path('/Users/manu/.codex/visualizations/2026/11/18/01a1edc1-dd4e-7ab3-b819-fbd88da8db05/common-avif-q65-appearance-review')
UNITS = ['nadal_kyrgios__000_at120__person_000000000001',
         'nadal_kyrgios__000_at120__person_000000000000']


def sha(p):
    return hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()


def main():
    old = json.loads(OLD.read_text())['files']
    new_receipt = json.loads((NEW / 'receipt.json').read_text())
    prepared = json.loads(Path('docs/experiments/common-reference-20261122/prepared.json').read_text())
    assert new_receipt['config_sha256'] == prepared['config_sha256']
    assert all(row['quality'] == 65 for row in prepared['rows'])
    new = new_receipt['units']
    mapping = json.loads((PREV / 'reveal.json').read_text())['mappings']
    choices = {}
    rng = random.Random(20261122)
    for unit in UNITS:
        order = ['lossless', 'jpeg40', 'avif']
        rng.shuffle(order)
        choices[unit] = dict(zip('ABC', order))
    DEST.mkdir(parents=True, exist_ok=False)
    sections = []
    for index, unit in enumerate(UNITS):
        def old_file(label, suffix):
            source = PREV / f'{unit}-{label}{suffix}'
            assert sha(source) == old[source.name]['sha256']
            return source
        lossless = next(k for k, v in mapping[unit].items() if v == 'lossless')
        jpeg = next(k for k, v in mapping[unit].items() if v == 'quality-40')
        controls = {'lossless': lossless, 'jpeg40': jpeg}
        for fps in (10, 50):
            src = old_file('source', f'-{fps}.mp4')
            shutil.copyfile(src, DEST / f'{unit}-source-{fps}.mp4')
        panels = []
        for label, selected in choices[unit].items():
            for suffix in ('-10.mp4', '-50.mp4', '.png'):
                if selected == 'avif':
                    source = NEW / f'{unit}{suffix}'
                    assert sha(source) == new[unit][source.name]['sha256']
                else:
                    source = old_file(controls[selected], suffix)
                shutil.copyfile(source, DEST / f'{unit}-{label}{suffix}')
            panels.append(f'<div><strong>{label}</strong><video controls loop src="{unit}-{label}-50.mp4"></video>'
                          f'<p>10 fps</p><video controls loop src="{unit}-{label}-10.mp4"></video>'
                          f'<img src="{unit}-{label}.png"></div>')
        sections.append(f'<h2>Player {2-index}</h2><div class="grid"><div><strong>Source</strong>'
                        f'<video controls loop src="{unit}-source-50.mp4"></video><p>10 fps</p>'
                        f'<video controls loop src="{unit}-source-10.mp4"></video></div>{"".join(panels)}</div>')
    page = ('<!doctype html><meta charset="utf-8"><title>Player appearance review</title>'
            '<style>.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1em}'
            'video,img{width:100%}</style><h1>Player appearance review</h1>'
            '<p>24 frames. The 10 fps clips are slowed. For each player, assess each candidate’s '
            'body, clothing, identity and pose relative to the source. Record your judgments before opening '
            'the reveal file.</p>' + ''.join(sections))
    (DEST/'review.html').write_text(page)
    receipt = {'seed': 20261122, 'mappings': choices,
               'avif_receipt_sha256': sha(NEW/'receipt.json'),
               'source_review_receipt_sha256': sha(OLD), 'files': {}}
    for path in sorted(DEST.iterdir()):
        receipt['files'][path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    (DEST/'reveal.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(html.escape(str(DEST/'review.html')))


if __name__ == '__main__':
    main()
