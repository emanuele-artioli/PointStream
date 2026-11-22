"""Two rate–quality figures for one authenticated set of native tennis ticks."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / "experiments"
NATIVE = "Ordinary AV1"
ROI = "AV1 with ROI"
TILE = "AV1 + ball tiles"
STREAK = "AV1 + rendered trajectory"
COLD = "Joint PointStream · cold start"
WARM = "Joint PointStream · valid-cache scenario"
NEURAL = "DCVC-UF"
STYLE = {
    NATIVE: ("#235589", "o", "-"),
    ROI: ("#27826b", "s", "--"),
    NEURAL: ("#7036a2", "D", "-"),
    TILE: ("#bc7214", "^", "-."),
    STREAK: ("#7b4c06", "*", "--"),
    COLD: ("#ac3c3e", "P", "-"),
    WARM: ("#ac3c3e", "P", ":"),
}


def load(path):
    return json.loads(path.read_text())


def points():
    rows = []

    def add(kind, rate, q, source):
        rows.append(
            dict(
                mode=kind, kbps=rate, players=(q["p1"] + q["p2"]) / 2, ball=q["ball"], source=source
            )
        )

    for r in load(HERE.parent / "selection/data.json")["rows"]:
        arm = r["arm"]
        kind = next(
            (
                k
                for prefix, k in [("plain-", NATIVE), ("roi-", ROI), ("ball", TILE)]
                if arm.startswith(prefix)
            ),
            None,
        )
        if kind:
            add(kind, r["total_kbps"], r["dists"], "selection/data.json#" + arm)
    name = "starved-base-local-20261122"
    q = load(EXP / name / "dists-receipt.json")["mean_dists"]
    for r in load(EXP / name / "base-receipt.json")["outputs"]:
        label = f"{r['arm']}-{r['crf']}"
        if r["arm"] == "original":
            add(NATIVE, r["stream_kbps"], q[label], name + "#" + label)
        else:
            for key, kind in [("cold_kbps", COLD), ("optimistic_no_reference_kbps", WARM)]:
                add(kind, r[key], q[label], name + "#" + label)
    name = "starved-rally-tiles-20261122"
    q = load(EXP / name / "scores.json")["means"]
    for pair in load(EXP / name / "render.json")["pairs"]:
        for arm, kind in [("plain", NATIVE), ("tile", TILE)]:
            label = arm + str(pair["target"])
            add(kind, pair[arm + "_kbps"], q[label], name + "#" + label)
    for n, ratefile, qualityfile in [
        (name, "strong-result.json", "strong-scores.json"),
        ("starved-stroke-20261122", "result.json", "scores.json"),
    ]:
        q = load(EXP / n / qualityfile)["means"]
        for label, r in load(EXP / n / ratefile)["selected"].items():
            add(ROI if label.startswith("roi") else NATIVE, r["rate"], q[label], n + "#" + label)
    name = "starved-tile-budget-20261122"
    d = load(EXP / name / "result.json")
    q = load(EXP / name / "scores.json")["means"]
    for label, r in d["rungs"].items():
        add(COLD, r["cold"], q[label], name + "#" + label)
        add(WARM, r["running"], q[label], name + "#" + label)
    for label, r in d["anchors"].items():
        add(ROI if label.startswith("roi") else NATIVE, r["kbps"], q[label], name + "#" + label)
    name = "starved-ball-twenty-20261122"
    for prefix in ("", "mid-", "upper-"):
        d = load(EXP / name / f"{prefix}result.json")
        q = load(EXP / name / f"{prefix}scores.json")["means"]
        add(STREAK, d["kbps"], q["pointstream"], name + "/" + prefix + "#pointstream")
        for label, r in d["anchors"].items():
            add(
                ROI if label.startswith("roi") else NATIVE,
                r["kbps"],
                q[label],
                name + "/" + prefix + "#" + label,
            )
    joint_name = "joint-trajectory-20261122"
    for r in load(EXP / joint_name / "scores.json")["points"]:
        source = f"{joint_name}#target-{r['cap']}"
        add(WARM, r["kbps"], r, source)
        add(COLD, r["cold_kbps"], r, source)
    for r in load(EXP / joint_name / "control-scores.json")["points"]:
        source = f"{joint_name}#{r['mode']}-{r['width']}-Q{r['crf']}"
        add(ROI if r["mode"] == "roi" else NATIVE, r["kbps"], r, source)
    for r in load(EXP / joint_name / "strict-20-ordinary.json")["points"]:
        add(NATIVE, r["kbps"], r, f"{joint_name}#strict-20-{r['width']}")
    low_neural = load(EXP / joint_name / "dcvc-q6-human-regions.json")
    add(NEURAL, low_neural["kbps"], low_neural["mean"], joint_name + "#DCVC-QP6")
    neural = EXP / "starved-dcvc-regions-20261122/result.json"
    if neural.exists():
        for r in load(neural)["points"]:
            add(NEURAL, r["kbps"], r["mean"], neural.parent.name + "#QP" + str(r["qp"]))
    ext = EXP / "starved-dcvc-regions-20261122/extension-result.json"
    if ext.exists():
        for r in load(ext)["points"]:
            add(NEURAL, r["kbps"], r["mean"], ext.parent.name + "#QP" + str(r["qp"]))
    return rows


def render(rows, metric):
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    handles = []
    for mode, (color, marker, ls) in STYLE.items():
        subset = sorted(
            (r for r in rows if r["mode"] == mode), key=lambda r: (r["kbps"], r[metric])
        )
        if not subset:
            continue
        best = []
        for r in subset:
            if not best or r[metric] < best[-1][metric] - 1.0e-8:
                best.append(r)
        common = dict(
            marker=marker,
            color=color,
            markerfacecolor="white" if mode == WARM else color,
            markeredgewidth=0.9,
            markersize=5 if mode != STREAK else 10,
        )
        ax.plot(
            [r["kbps"] for r in subset], [r[metric] for r in subset], ls="", alpha=0.32, **common
        )
        (h,) = ax.plot(
            [r["kbps"] for r in best],
            [r[metric] for r in best],
            ls=ls,
            linewidth=1.5,
            label=mode,
            **common,
        )
        handles.append(h)
    title = "Player-body fidelity" if metric == "players" else "Ball-crop fidelity"
    ax.set_title(title + " across bandwidth", loc="left", weight="bold", fontsize=16, pad=18)
    ax.text(
        0,
        1.012,
        "One 50-fps rally  ·  240 delivered frames  ·  12 reviewed instants",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.set_xlabel("Total bitrate (kbps, logarithmic scale)")
    ax.set_ylabel(
        ("Mean player-body" if metric == "players" else "Ball-crop") + " DISTS  (lower is better)"
    )
    ax.set_xscale("log")
    ax.set_xticks([20, 30, 50, 75, 125, 250, 500, 1000, 2000])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(17, 2300)
    ax.grid(axis="both", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=handles, frameon=False, loc="best", fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.86)
    fig.text(
        0.12,
        0.025,
        "Lines join improving measured settings. Open red markers use cached player state. Solid red includes its delivery.",
        fontsize=8,
    )
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{metric}-rate-quality.{ext}", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    rows = points()
    (HERE / "points.json").write_text(json.dumps(rows, indent=2) + "\n")
    for metric in ("players", "ball"):
        render(rows, metric)
