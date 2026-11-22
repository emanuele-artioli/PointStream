"""Print a complete, portable event-by-event report from verified evaluation receipts."""

from argparse import ArgumentParser
import json
from pathlib import Path


def text(summary):
    if summary["scored"] != summary["count"]:
        raise ValueError("All twelve original sources must be accounted for.")
    lines = [
        "# Official tennis TEST results",
        "",
        "The fixed panels contain one 240-native-frame passage from each match. The sender received only the original video. Source-only event identification and marks from the test release are documented in the [cohort card](README.md) and [event audit](test-events.md). Lower DISTS indicates a closer ball crop. Both families pay for the delivered stream.",
        "",
        "| Match | Selection | PS / AV1 kbps | PS / AV1 DISTS | Best affordable AV1 kbps / DISTS | Visible hits within 10 px | Emitted / absent marks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary["rows"]:
        a = r["direct"]
        best = (
            f"{r['strongest_kbps']:.2f} / {r['strongest_dists']:.4f}"
            if r["strongest_dists"] is not None
            else "—"
        )
        scores = (
            f"{r['candidate_dists']:.4f} / {r['ordinary_dists']:.4f}"
            if r["candidate_dists"] is not None
            else "—"
        )
        lines.append(
            f"| {r['event']}, {r['players']} | {r['selection']} | {r['candidate_kbps']:.2f} / {r['ordinary_kbps']:.2f} | {scores} | {best} | {a['hit10']} / {a['visible']} | {a['emitted_absent']} / {a['marked_absent']} |"
        )
    lines.extend(
        [
            "",
            "The target classification requires a 0.5% match to the nominal budget as well as the receiver-to-receiver match. The other-matched classification compares the two delivered streams within 0.5% at another observed price. The sensitivity band permits 0.5–2%. The strongest affordable ordinary anchor is the measured 320- or 640-wide AV1 stream with lowest ball-crop DISTS under the larger cost of each paired comparison. The same-resolution anchor is independently byte-calibrated. The cross-resolution selection uses the held-out outcome and measures an achievable envelope.",
            "",
            "| Paired comparison | Eligible matches | DISTS, PS / anchor | Relative decrease | Match wins | 95% interval on decrease |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("target", "other-matched", "matched-actual", "sensitivity", "unmatched"):
        for comp, name in (
            ("same_geometry", "same resolution"),
            ("strongest_affordable", "across resolutions"),
        ):
            g = summary["groups"][key][comp]
            if not g:
                continue
            ci = g["gain_percent_95"]
            interval = f"{ci[0]:+.2f}% to {ci[1]:+.2f}%" if ci else "single match"
            lines.append(
                f"| {key}, {name} | {g['n']} | {g['candidate']:.4f} / {g['ordinary']:.4f} | {g['gain_percent']:+.2f}% | {g['wins']} / {g['n']} | {interval} |"
            )
    route = summary["routing"]
    lines.extend(
        [
            "",
            f"The source-side selector admitted all 240 ticks on {len(route['all_court'])} matches, none on {len(route['all_other'])}, and a mixture on {len(route['mixed'])}. Full non-court routing occurred on {', '.join(route['all_other']) or 'no match'}. The unconditional object-mode table preserves all frozen sources, including shots the router would send via conventional video.",
        ]
    )
    c = summary["combined_direct"]
    lines += [
        "",
        f"Direct positions from the paid stream hit {c['hit10']} of {c['visible']} independently marked visible balls within 10 original pixels. Equal-match mean recall was {100 * summary['macro_hit10']:.1f}%. The receiver emitted a location at {c['emitted_absent']} of {c['marked_absent']} explicitly absent marks. The remaining native frames with no annotation make no contribution to either denominator. On these explicitly annotated ticks, localization precision was {100 * c['annotated_precision10']:.1f}% and F1 was {c['annotated_f1_10']:.3f}. An emitted location outside the 10-pixel gate counts as a false positive and misses the labeled ball. The 95% intervals resample complete matches together with their paired outcomes using 10,000 draws.",
        "",
        "## Neural video control and receiver-side detector",
        "",
        "| Match | DCVC-UF kbps / DISTS | Affordable at paired ceiling | Original detector hits / visible | Source qualified | AV1 / PS / best spatial AV1 detector hits |",
        "|---|---:|---|---:|---|---:|",
    ]
    for r in summary["rows"]:
        neural = (
            f"{r['neural_kbps']:.2f} / {r['neural_dists']:.4f}"
            if r.get("neural_dists") is not None
            else "—"
        )
        afford = {True: "yes", False: "no", None: "pending"}.get(r.get("neural_affordable"))
        q = r.get("source_qualified")
        source = r.get("rgb", {}).get("source", {})
        src = f"{source['hit10']} / {source['visible']}" if source else "—"
        rgb = (
            f"{r['rgb']['ordinary']['hit10']} / {r['rgb']['candidate']['hit10']} / {r['rgb'].get('strongest', {}).get('hit10', 'pending')}"
            if q
            else "—"
        )
        lines.append(
            f"| {r['event']} | {neural} | {afford} | {src} | { {True: 'yes', False: 'no', None: 'pending'}[q] } | {rgb} |"
        )
    lines.extend(
        [
            "",
            "Neural prices refer to the complete fixed 480-wide Q24 bitstream at each native cadence. Source qualification requires a hit within 10 pixels for at least 80% of visible marks on that original source. Detector comparisons on receiver video use only qualifying source matches. Every prematched decoded input passed pixel-hash verification across the scoring and detector hosts. The paid-coordinate outcome uses the full twelve.",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "### Localization on identical qualified sources",
            "",
            "| Receiver readout | Matches | Hits / visible | Emits / explicitly absent | Marked-tick F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    comparisons = [
        (
            "Original video + WASB",
            summary["source_qualified"],
            summary["rgb_source_qualified"]["source"],
        ),
        (
            "Matched AV1 + WASB",
            summary["source_qualified"],
            summary["rgb_source_qualified"]["ordinary"],
        ),
        (
            "PointStream display + WASB",
            summary["source_qualified"],
            summary["rgb_source_qualified"]["candidate"],
        ),
        (
            "Direct PointStream coordinates",
            summary["source_qualified"],
            summary["direct_on_qualified_sources"],
        ),
        (
            "Best affordable spatial AV1 + WASB",
            summary["strongest_judge_count"],
            summary["rgb_strongest_qualified_scored"],
        ),
    ]
    for name, n, c in comparisons:
        f1 = f"{c['annotated_f1_10']:.3f}" if c["annotated_f1_10"] is not None else "—"
        lines.append(
            f"| {name} | {n} | {c['hit10']} / {c['visible']} | {c['emitted_absent']} / {c['marked_absent']} | {f1} |"
        )
    if summary["strongest_judge_count"] != summary["source_qualified"]:
        lines.append(
            "The strongest spatial row has partial coverage. Its exact matching direct and decoded-video counterparts are recorded in the numerical companion."
        )
    paired = summary.get("paired_localization")
    if paired:
        gain = paired["gains"]["candidate_minus_strongest"]
        lo, hi = gain["interval95"]
        lines += [
            "",
            f"Across the {len(paired['matches'])} source-qualified matches with both spatial controls, the PointStream display changed localization by {gain['percentage_points']:+.1f} percentage points relative to the best affordable spatial AV1 display (95% match-bootstrap interval {lo:+.1f} to {hi:+.1f}). Each match has equal weight in this estimate. The counts above pool individual labels.",
        ]
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    args.destination.write_text(text(json.loads(args.summary.read_text())))
