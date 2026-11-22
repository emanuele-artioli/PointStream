"""Compare the new judge inputs and reproduce all source detections before scoring."""

import argparse
import importlib.util
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "seg_score_65", HERE.parent / "starved-player-segmentation-20261122/score.py"
)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def predictions(path):
    with tarfile.open(path) as archive:
        return json.load(archive.extractfile("publish/predictions.json"))


def compare(args):
    old = predictions(args.prior)
    new = predictions(args.predictions)
    keys = json.loads(args.key.read_text())
    sources = [identity for identity, (arm, _) in keys.items() if arm == "source"]
    assert len(sources) == 12 and all(old["rows"][k] == new["rows"][k] for k in sources)
    assert all(old[k] == new[k] for k in ("weight", "ultralytics", "torch"))
    previous.main(args.predictions, args.key, args.labels, args.output)
    receipt = json.loads(args.output.read_text())
    baseline = json.loads(
        (HERE.parent / "starved-player-segmentation-20261122/result.json").read_text()
    )
    for arm in ("source", "receiver_mask"):
        assert receipt["means"][arm] == baseline["means"][arm]
    print(json.dumps(receipt["means"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("prior", "predictions", "key", "labels", "output"):
        parser.add_argument(arg, type=Path)
    compare(parser.parse_args())
