"""Bind the selected physical split and its native decoded hashes before campus replay."""

import argparse
import json
from pathlib import Path

from specs import DATA


def main(a):
    r = json.loads((a.built / "result.json").read_text())
    assert r["feasible"]
    chosen = r["selected"]
    orig = json.loads(a.reference.read_text())
    args = orig["arguments"]
    assert "--reference" in args
    args.remove("--reference")
    remote = DATA / "tennis" / f"med{a.clip}-split-{chosen['wire_sha256'][:12]}.psj"
    inputs = dict(name="split", path=str(remote), sha256=chosen["wire_sha256"], extract=False)
    orig["staged_inputs"].append(inputs)
    orig["inputs"].append({k: inputs[k] for k in ("path", "sha256")})
    args += ["--split", "{staged:split}"]
    args[args.index("--native-sha") + 1] = ",".join(p["native_sha256"] for p in chosen["pieces"])
    a.out.write_text(json.dumps(orig, indent=2) + "\n")
    print(remote)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    for k in ["reference", "built", "out"]:
        p.add_argument("--" + k, type=Path, required=True)
    main(p.parse_args())
