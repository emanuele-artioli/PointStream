"""Cast Wan2.1-I2V-14B-480P's DiT shards (fp32, 62 GB) to one bf16 safetensors file (H3e).

    python tools/models/wan_dit_bf16.py SHARD_DIR OUT.safetensors

JointControlVideo's pipeline loads the DiT in bf16 (``torch_dtype=torch.bfloat16``), so the cast is the one the
loader would make; it halves what each host stages. Tensor names and shapes are unchanged (DiffSynth identifies
the model by them). Writes OUT and OUT.json (source shard and output SHA256s).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main(shard_dir: str, out: str) -> int:
    src, dst = Path(shard_dir), Path(out)
    index = json.loads((src / "diffusion_pytorch_model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    tensors: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(src / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).to(torch.bfloat16)
    if set(tensors) != set(index["weight_map"]):
        raise SystemExit("shards do not hold exactly the index's tensors")
    save_file(tensors, str(dst), metadata={"source": "Wan-AI/Wan2.1-I2V-14B-480P@6b73f84", "cast": "bfloat16"})
    record = {"source_shards": {s: sha256(src / s) for s in shards}, "tensors": len(tensors), "output": dst.name,
              "output_sha256": sha256(dst)}
    Path(str(dst) + ".json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
