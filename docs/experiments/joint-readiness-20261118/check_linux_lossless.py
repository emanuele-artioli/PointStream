import hashlib
import io
import json
import pathlib
import sys
import tarfile

root = pathlib.Path("/local/users/emanuele/pointstream/joint-native-finish-20261118-a")
sys.path.insert(0, str(root / "source"))
from experiments.tennis import common_receiver_isolation as iso, player_requalification as replay  # noqa: E402
import numpy as np  # noqa: E402

reg = json.loads((root / "linux-carriers.json").read_text())["carriers"]["png"]
auth = json.loads((root / "prepared-manifest.json").read_text())["40"]
packet = root / "qpng.pscp"
manifest = root / "qpng.manifest"
units = [r["unit"] for r in auth["units"]]
archive = pathlib.Path(
    "/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261101T093351Z-24914a4d/full/published.tar"
)
denied = [pathlib.Path(s) for s in json.loads((root / "denied.json").read_text())] + [archive]
proof = iso.launch(
    source_root=root / "source",
    output=root / "lossless-proof",
    worker_config={
        "carrier": str(packet),
        "manifest": str(manifest),
        "expected_sha256": reg["sha256"],
        "frames": 240,
        "units": units,
    },
    inputs=[packet, manifest],
    denied=denied,
    seconds=90,
    kind="conditioning",
)
raw = archive.read_bytes()
assert (
    hashlib.sha256(raw).hexdigest()
    == "3b80afe8c159b6d572acf24e879e4c924b4c81ced06d33185a83f77b6649fe10"
)
checked = {}
with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
    for row in proof["outputs"]:
        n = row["unit"]
        f = next(m for m in tar.getmembers() if m.name.endswith("/" + n + ".npz"))
        d = tar.extractfile(f).read()
        assert hashlib.sha256(d).hexdigest() == next(
            v["source_npz_sha256"] for v in auth["units"] if v["unit"] == n
        )
        with np.load(io.BytesIO(d), allow_pickle=False) as npz:
            checked[n] = {k: replay.array_sha(npz[k]) == v for k, v in row["array_sha256"].items()}
        assert all(checked[n].values())
with (root / "lossless.json").open("x") as fh:
    json.dump({"worker": proof, "checks": checked}, fh, indent=2)
print(checked)
