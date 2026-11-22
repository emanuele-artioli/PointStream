"""Audit source-only W7 references and rank anonymous cross-shot matches."""
import hashlib
import io
import itertools
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ARCHIVE = Path("/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261031T162058Z-c9b2c47b/full/published.tar")
ARCHIVE_SHA = "94a5d6318fc113557fb24c8a32f2edffdf368f7a289484fddd4b19a749302bfa"
OUT = Path("/local/users/emanuele/pointstream/crossshot-reference-audit-20261122")
UNITS = {
    "source": ["nadal_kyrgios__000_at120__person_000000000000",
               "nadal_kyrgios__000_at120__person_000000000001"],
    "target": ["nadal_kyrgios__002_at120__person_000000000000",
               "nadal_kyrgios__002_at120__person_000000000002"],
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def array_sha(array):
    h = hashlib.sha256(f"{array.dtype.str}:{array.shape}:".encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def histogram(rgb):
    h, w = rgb.shape[:2]
    crop = rgb[int(h * .12):int(h * .68), int(w * .15):int(w * .85)]
    parts = [np.histogram(crop[:, :, c], bins=16, range=(0, 256))[0].astype(float)
             for c in range(3)]
    bins = np.concatenate(parts)
    return bins / bins.sum()


def main():
    if sha(ARCHIVE.read_bytes()) != ARCHIVE_SHA:
        raise ValueError("W7 prepared archive changed")
    units = {}
    with tarfile.open(ARCHIVE) as tar:
        for group, names in UNITS.items():
            units[group] = []
            for name in names:
                member = tar.extractfile(f"publish/units/{name}.npz")
                if member is None:
                    raise ValueError("Missing conditioned unit")
                raw = member.read()
                with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                    ref = data["reference"]
                    info = {key: {"sha256": array_sha(data[key]), "shape": list(data[key].shape),
                                  "dtype": str(data[key].dtype)} for key in data.files}
                if ref.shape != (784, 512, 3) or ref.dtype != np.uint8:
                    raise ValueError("Reference geometry differs")
                units[group].append({"name": name, "npz_sha256": sha(raw),
                                     "arrays": info, "reference": ref})
    distances = [[float(np.sqrt(0.5 * np.square(np.sqrt(histogram(source["reference"])) -
                                                 np.sqrt(histogram(target["reference"]))).sum()))
                  for source in units["source"]] for target in units["target"]]
    mappings = []
    for order in itertools.permutations((0, 1)):
        mappings.append({"source_index_for_targets": list(order),
                         "distance": sum(distances[i][order[i]] for i in range(2))})
    mappings.sort(key=lambda entry: entry["distance"])
    ambiguous = mappings[1]["distance"] - mappings[0]["distance"] < 0.10
    OUT.mkdir(parents=True, exist_ok=False)
    sheet = Image.new("RGB", (4 * 192, 315), "white")
    draw = ImageDraw.Draw(sheet)
    for i, item in enumerate(units["source"] + units["target"]):
        thumbnail = Image.fromarray(item["reference"])
        thumbnail.thumbnail((192, 294))
        sheet.paste(thumbnail, (192 * i, 21))
        draw.text((192 * i + 3, 3), ("source " if i < 2 else "target ") + str(i % 2), fill="black")
    sheet.save(OUT / "source-only-thumbnails.png")
    result = {"archive_sha256": ARCHIVE_SHA, "distances": distances, "mappings": mappings,
              "ambiguity_threshold": 0.10, "ambiguous": ambiguous,
              "units": {group: [{key: value for key, value in item.items() if key != "reference"}
                                for item in values] for group, values in units.items()}}
    (OUT / "source-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
