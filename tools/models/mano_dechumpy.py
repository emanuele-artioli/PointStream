"""Write chumpy-free copies of the MANO model pickles.

The official MANO_{LEFT,RIGHT}.pkl store some arrays as chumpy objects, so
unpickling them imports chumpy, which does not run on Python 3.12 or numpy 1.24+.
This reads them with a stub for every chumpy class, replaces each stub by its
value array, and writes a pickle of plain numpy and scipy objects that smplx,
HaMeR, WiLoR and hand_tracking_toolkit load unchanged.

    python -m tools.models.mano_dechumpy SOURCE_DIR TARGET_DIR

TARGET_DIR receives the two pickles, the licence and MANIFEST.json (sha256 of
every source and output file and of each converted array).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

NAMES = ("MANO_LEFT.pkl", "MANO_RIGHT.pkl")


class _ChumpyStub:
    """Stands in for a chumpy class; keeps its name and pickled state."""

    kind = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.state: Any = None

    def __setstate__(self, state: Any) -> None:
        self.state = state


def _stub(module: str, name: str) -> type:
    return type(name, (_ChumpyStub,), {"kind": f"{module}.{name}"})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "chumpy" or module.startswith("chumpy."):
            return _stub(module, name)
        # scipy moved its sparse classes out of the per-format modules.
        if module.startswith("scipy.sparse.") and name.endswith("_matrix"):
            import scipy.sparse

            return getattr(scipy.sparse, name)
        return super().find_class(module, name)


def _value(stub: _ChumpyStub, key: str) -> np.ndarray:
    """Evaluate the two chumpy node types MANO uses; anything else is an error."""
    state = stub.state
    if stub.kind == "chumpy.ch.Ch":
        return np.array(state["x"])
    if stub.kind == "chumpy.reordering.Select":
        source = state["a"]
        array = _value(source, key) if isinstance(source, _ChumpyStub) else np.asarray(source)
        return array.ravel()[np.asarray(state["idxs"])].reshape(state["preferred_shape"])
    raise ValueError(f"{key}: unsupported chumpy node {stub.kind}")


def _plain(value: Any, key: str) -> Any:
    return _value(value, key) if isinstance(value, _ChumpyStub) else value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(source: Path, target: Path) -> dict[str, Any]:
    with source.open("rb") as handle:
        model = _Unpickler(handle, encoding="latin1").load()
    if not isinstance(model, dict):
        raise ValueError(f"{source}: expected a dict, got {type(model).__name__}")
    plain = {key: _plain(value, key) for key, value in model.items()}
    with target.open("wb") as handle:
        pickle.dump(plain, handle, protocol=4)
    arrays = {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in plain.items()
        if isinstance(value, np.ndarray)
    }
    converted = {key: value.kind for key, value in sorted(model.items()) if isinstance(value, _ChumpyStub)}
    return {
        "source": str(source), "source_sha256": sha256(source),
        "output": target.name, "output_sha256": sha256(target),
        "converted_from_chumpy": converted, "array_sha256": arrays,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("target_dir", type=Path)
    args = parser.parse_args(argv)
    args.target_dir.mkdir(parents=True, exist_ok=True)
    existing = [n for n in (*NAMES, "MANIFEST.json") if (args.target_dir / n).exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite {existing} in {args.target_dir}")
    files = [convert(args.source_dir / name, args.target_dir / name) for name in NAMES]
    licence = args.source_dir / "LICENSE.txt"
    if licence.is_file():
        shutil.copyfile(licence, args.target_dir / "LICENSE.txt")
    manifest = {
        "tool": "tools/models/mano_dechumpy.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "numpy": np.__version__, "files": files,
    }
    (args.target_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({f["output"]: f["output_sha256"] for f in files}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
