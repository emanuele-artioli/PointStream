"""Summarize direct ball coordinates from the six original scored streams."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARDS = HERE.parent


def main():
    cases = [
        ("Rotterdam", "ball-machine-anchors-20261122/209-result.json", "tracked"),
        ("Washington", "ball-machine-anchors-20261122/269-result.json", "tracked"),
        ("Indian Wells", "ball-machine-anchors-20261122/235-result.json", "tracked"),
        ("Basel", "ball-basel-confirmation-20261122/machine-result.json", "tile"),
        ("Adelaide", "ball-adelaide-prospective-20261122/machine-result.json", "metadata"),
        ("Shanghai", "shanghai-prospective-20261122/primary-score-result.json", "candidate"),
    ]
    results = []
    for event, path, mode in cases:
        raw = (CARDS / path).read_bytes()
        report = json.loads(raw)
        (arm,) = [item for item in report["results"] if item["mode"] == mode]
        rows = arm["rows"]
        visible = [r for r in rows if r["visibility"] == 1]
        hidden = [r for r in rows if r["visibility"] == 0]
        assert len(set(r["tick"] for r in rows)) == len(rows)
        assert len(visible) + len(hidden) == len(rows)

        def hit(row):
            if "direct_hit_10px" in row:
                return row["direct_hit_10px"]
            error = row.get("direct_native_error", row.get("direct_error_native_px"))
            return row["direct_present"] and error is not None and error <= 10

        rate = arm.get("wire_kbps")
        if event == "Shanghai":
            display = json.loads(
                (CARDS / "shanghai-prospective-20261122/primary-display-result.json").read_text()
            )
            assert report["rgb_sha256"] == display["rgb_sha256"]
            rate = display["candidate"]["bytes"] * 8 * 25 / (240 * 1000)
        assert rate > 0
        results.append(
            dict(
                event=event,
                receipt=path,
                receipt_sha256=hashlib.sha256(raw).hexdigest(),
                mode=mode,
                rate_kbps=rate,
                visible=len(visible),
                hits=sum(hit(r) for r in visible),
                not_visible=len(hidden),
                emissions=sum(r["direct_present"] for r in hidden),
            )
        )
    out = dict(
        sources=results,
        totals={
            k: sum(r[k] for r in results) for k in ("hits", "visible", "emissions", "not_visible")
        },
    )
    (HERE / "result.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out["totals"])


if __name__ == "__main__":
    main()
