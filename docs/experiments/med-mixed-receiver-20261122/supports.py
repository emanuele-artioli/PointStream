"""Extract source-only W7 SAM supports before measuring compressed outputs."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np

from build import TRANS, sha
from experiments.tennis import k6b_rate_streams as rates


def main(args):
    assert sha(args.transport) == TRANS
    with tarfile.open(args.transport) as archive:
        directory = f"publish/packets/medvedev_struff__{args.clip}@120/"
        blob = archive.extractfile(directory + "exact-packed.matte").read()
        sent = archive.extractfile(directory + "keypoints.kps").read()
    header, masks = rates.decode_masks(blob)
    kpmeta, _ = rates.decode_keypoints(sent)
    assert (
        masks.dtype == bool
        and masks.shape == (240, 2, 540, 960)
        and header["native_grid_factor"] == 1
    )
    assert kpmeta["clip_local_source_start"] == 0 and kpmeta["fps"] == [50, 1]
    args.out.mkdir(exist_ok=False)
    ticks = list(range(10, 240, 20))
    rows = []
    for tick in ticks:
        for index, m in enumerate(masks[tick]):
            support = np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)
            area = int(support.sum())
            row = dict(tick=tick, object_index=index, area=area)
            if area:
                x, y, w, h = cv2.boundingRect(support.astype("uint8"))
                # Fixed body ROI: tight source support plus 32 native pixels on every side.
                x0, y0, x1, y1 = (
                    max(0, x - 32),
                    max(0, y - 32),
                    min(1920, x + w + 32),
                    min(1080, y + h + 32),
                )
                path = args.out / f"{tick:03d}-p{index}.png"
                roi = support[y0:y1, x0:x1]
                assert cv2.imwrite(str(path), roi.astype("uint8") * 255) and int(roi.sum()) == area
                row.update(xyxy=[x0, y0, x1, y1], mask=path.name, sha256=sha(path))
            rows.append(row)
    record = dict(
        ticks=ticks,
        rows=rows,
        source_start=0,
        native_display=[1920, 1080],
        matte_sha256=hashlib.sha256(blob).hexdigest(),
        producer_archive_sha256=TRANS,
        method="exact original W7 source-SAM matte for each person, replicated from 960x540 to native, tight box plus 32 pixels. Model-derived, no independent human inspection.",
        nonempty=sum(r["area"] > 0 for r in rows),
    )
    (args.out / "supports.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record["nonempty"], "of", len(rows), "nonempty source supports")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--transport", type=Path, required=True)
    main(p.parse_args())
