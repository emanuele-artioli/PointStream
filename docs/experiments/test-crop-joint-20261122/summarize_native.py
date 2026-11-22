"""Summarize the physically affordable source-mask comparisons and figure inputs."""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def group(name):
    if name.startswith("joint-"):
        return "PointStream"
    if name.startswith("dcvc-"):
        return "DCVC-UF"
    if name.startswith("roi-"):
        return "SVT-AV1 ROI"
    if name.startswith("av1-"):
        return "SVT-AV1"
    raise ValueError(name)


def collect(input_dir):
    source = json.loads((HERE / "match102-result.json").read_text())["source_sha256"]
    bank = []
    hashes = set()
    for p in sorted(input_dir.glob("*.json")):
        a = json.loads(p.read_text())
        assert (
            a["source_sha256"] == source
            and a["native_fps"] == "30000/1001"
            and len(a["rows"]) == 480
        )
        assert {(r["tick"], r["role"]) for r in a["rows"]} == {
            (t, p) for t in range(240) for p in ("near", "far")
        }
        assert a["pooled"]["ticks"] == 480 and a["near"]["ticks"] == a["far"]["ticks"] == 240
        hashes.add(a["support_rle_sha256"])
        bank.append(
            dict(
                name=a["name"],
                codec=group(a["name"]),
                bytes=a["bytes"],
                kbps=a["kbps"],
                width=a["width"],
                height=a["height"],
                dists=a["pooled"]["dists"],
                near=a["near"]["dists"],
                far=a["far"]["dists"],
                body_psnr=a["pooled"]["rgb_psnr"],
                full_rgb_psnr=a["whole_rgb_psnr"],
                body_dists_twelve_on_sam=a["optional_review_ticks_on_source_model"]["dists"],
                decoded_sha256=a["decoded_sha256"],
                source_support_sha256=a["support_rle_sha256"],
            )
        )
    assert len(hashes) == 1 and len({r["name"] for r in bank}) == len(bank)
    return bank


def main(a):
    bank = collect(a.input)
    paid = json.loads((HERE / "match102-sender.json").read_text())
    qty = next(r for r in paid["rates"] if r["crf"] == paid["selected_crf"])
    ceilings = {"warm": qty["joint_bytes"], "cold": qty["joint_bytes"] + qty["init_bytes"]}
    summaries = {}
    for name, ceiling in ceilings.items():
        groups = {}
        for which in ("SVT-AV1", "SVT-AV1 ROI", "DCVC-UF"):
            rs = [r for r in bank if r["codec"] == which and r["bytes"] <= ceiling]
            if rs:
                best = min(rs, key=lambda x: (x["dists"], x["bytes"]))
                closest = min(rs, key=lambda x: (ceiling - x["bytes"], x["dists"]))
                groups[which] = dict(
                    best_quality=best["name"],
                    dists=best["dists"],
                    bytes=best["bytes"],
                    rate_gap=best["bytes"] / ceiling - 1,
                    nearest_affordable=closest["name"],
                    nearest_bytes=closest["bytes"],
                    qualified_count=len(rs),
                )
        summaries[name] = groups
    joint = next((r for r in bank if r["name"] == f"joint-crf{qty['crf']}"), None)
    compare = {}
    if joint:
        for billing, groups in summaries.items():
            compare[billing] = {
                k: (1 - joint["dists"] / v["dists"]) * 100 for k, v in groups.items()
            }
    conditional = []
    if joint:
        for codec in ("SVT-AV1", "SVT-AV1 ROI", "DCVC-UF"):
            rows = [r for r in bank if r["codec"] == codec and r["dists"] <= joint["dists"]]
            if not rows:
                continue
            cheapest = min(rows, key=lambda r: r["bytes"])
            diff = cheapest["kbps"] - joint["kbps"]
            conditional.append(
                dict(
                    codec=codec,
                    cheapest_quality_matched=cheapest["name"],
                    bytes=cheapest["bytes"],
                    break_even_seconds=8 * qty["init_bytes"] / (1000 * diff) if diff > 0 else None,
                )
            )
    result = dict(
        source=paid["source_sha256"],
        native_frames=240,
        native_fps=paid["native_fps"],
        mask_source="Whole source 960x540 SAM31 proposals nearest-neighbor to 1920x1080; individual tight crops + 32 pixels. No human truth.",
        price_unit="complete delivered bytes except preinstalled generic models, cached and cold treated separately",
        ceilings=ceilings,
        strict_affordable=summaries,
        primary_joint=joint,
        percentage_lower_body_dists=compare,
        conditional_cache_crossover=conditional,
        operating_points=bank,
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
