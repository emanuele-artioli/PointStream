"""Describe variation within the fixed held-out Shanghai shot using the same physical anchors."""

import argparse
import json
from pathlib import Path
from statistics import mean

from summarize_native import group

HERE = Path(__file__).resolve().parent


def main(a):
    bank = {p.stem: json.loads(p.read_text()) for p in a.input.glob("*.json")}
    j = bank[a.joint]
    charge = json.loads((HERE / "match102-sender.json").read_text())
    row = next(x for x in charge["rates"] if f"joint-crf{x['crf']}" == a.joint)
    assert row["joint_bytes"] == j["bytes"] and j["frames"] == 240 and len(j["rows"]) == 480
    results = {}
    for cache, limit in (
        ("warm", row["joint_bytes"]),
        ("cold", row["joint_bytes"] + row["init_bytes"]),
    ):
        results[cache] = {}
        for family in ("SVT-AV1", "SVT-AV1 ROI", "DCVC-UF"):
            rs = [b for b in bank.values() if group(b["name"]) == family and b["bytes"] <= limit]
            if not rs:
                continue
            baseline = min(rs, key=lambda b: (b["pooled"]["dists"], b["bytes"]))
            assert (
                baseline["source_sha256"] == j["source_sha256"]
                and baseline["support_rle_sha256"] == j["support_rle_sha256"]
            )
            anchor = {(r["tick"], r["role"]): r for r in baseline["rows"]}
            pairs = []
            for r in j["rows"]:
                b = anchor[r["tick"], r["role"]]
                assert b["area"] == r["area"]
                pairs.append((r["tick"], r["role"], r["body_dists"], b["body_dists"]))
            items = []
            for start in range(0, 240, 60):
                for role in ("near", "far"):
                    qs = [x for x in pairs if start <= x[0] < start + 60 and x[1] == role]
                    assert len(qs) == 60
                    test = mean(t[2] for t in qs)
                    ref = mean(t[3] for t in qs)
                    items.append(
                        dict(
                            first=start,
                            stop=start + 60,
                            role=role,
                            n=60,
                            pointstream=test,
                            anchor=ref,
                            delta=test - ref,
                            relative_percent_lower=100 * (1 - test / ref),
                            lower_frames=sum(t[2] < t[3] for t in qs),
                        )
                    )
            results[cache][family] = dict(
                anchor=baseline["name"],
                anchor_bytes=baseline["bytes"],
                whole_pooled_reference=baseline["pooled"]["dists"],
                joint_lower_person_ticks=sum(x[2] < x[3] for x in pairs),
                joint_lower_role_quarters=sum(x["delta"] < 0 for x in items),
                role_quarters=items,
            )
    a.out.write_text(
        json.dumps(
            dict(
                native_source=j["source_sha256"],
                support=j["support_rle_sha256"],
                native_fps=j["native_fps"],
                joint=a.joint,
                joint_bytes=j["bytes"],
                cache_init_bytes=row["init_bytes"],
                scored_frames=240,
                windows="Four fixed consecutive 60-tick portions of the same shot, both original athletes separately, strict pooled-best affordable comparator selected once across 480 observations.",
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--joint", choices=("joint-crf50", "joint-crf60", "joint-crf70"), required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
