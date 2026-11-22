"""Historical three-base player quality with corrected modeled 2h asset bills."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "experiments/player-rd-historical-20261122"
SPECS = [
    ("al028", "Alcaraz–Hurkacz", 155683, "#24618a", "o", "a284b09c634e268db8efc866756b88af2b9078f874b86958f5983c579b859af9", "1c8ba83552f04700d6ddbaaa0aab6568dd3b59e9b16a30ee85158d63a197318c"),
    ("med000", "Medvedev–Struff", 201292, "#b84d27", "s", "3c31581552a6172438b0588b5d78a1ff1eadc16640381505d95320692c4bf938", "f8a48a2ccab9e37e4c4ea7ad30e19dfaca4c1e5ed2fd565462224f02e9266c1c"),
    ("nad000", "Nadal–Kyrgios", 366161, "#237554", "^", "e4525a0b0c7024db5358734449294ff62279b9b33136378fc7f9d080093b8494", "e9801ec918a9d092788508c392f6820ebf9344f5a3ed76ef0acd6fd929cdee6b"),
]

fig, ax = plt.subplots(figsize=(7.5, 4.1), constrained_layout=True)
summary = []
for name, label, ref_bytes, color, marker, json_hash, npz_hash in SPECS:
    raw = (DATA / f"{name}-window.json").read_bytes()
    scores_path = DATA / f"{name}-scores.npz"
    assert hashlib.sha256(raw).hexdigest() == json_hash
    assert hashlib.sha256(scores_path.read_bytes()).hexdigest() == npz_hash
    row = json.loads(raw)
    with np.load(scores_path, allow_pickle=False) as scores:
        assert np.array_equal(scores["ticks"], np.arange(0, 240, 3))
        assert np.allclose(scores["foreground_dists"].mean(axis=1), row["means"]["foreground_dists"], atol=1e-12)
    assert [v["scale"] for v in row["rates"]] == [8, 4, 2]
    correction = 4.5 * ref_bytes * 8 / 7_200_000
    x = [v["total_kbps"] + correction for v in row["rates"]]
    y = [row["means"]["foreground_dists"][i] for i in (1, 3, 5)]
    ax.plot(x, y, label=f"{label} ({name})", color=color, marker=marker, linewidth=1.5)
    summary.append({"window":name,"sampled_ticks":80,"corrected_modeled_kbps":x,"player_dists":y,"correction_kbps":correction})
ax.set_xlabel("Modeled total rate (kbps; 2-hour asset reuse)")
ax.set_ylabel("Player-region DISTS (lower is better)")
ax.grid(alpha=.22)
ax.legend(frameon=False, fontsize=9)
fig.savefig(HERE / "player-rd-historical.pdf")
fig.savefig(HERE / "player-rd-historical.png", dpi=180)
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
