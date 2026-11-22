"""Score a source-blind choice of family, preferred width and affordable native file."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent / "cohort-expansion-20261122"
POLICIES = [
    ("ordinary", 320),
    ("ordinary", 640),
    ("candidate", 320),
    ("candidate", 640),
    ("neural", None),
]


def evaluate():
    hashes = {}

    def read(path):
        contents = path.read_bytes()
        hashes[str(path.relative_to(HERE.parent))] = sha256(contents).hexdigest()
        return json.loads(contents)

    three = read(COHORT / "three-family-all-verified.json")
    route = {
        r["match"]: r["ball_selected"]
        for r in read(HERE.parent / "official-test-route-20261122/result.json")["rows"]
    }
    banks = {}
    scores = {}
    for match, neural in three["native_neural"].items():
        report = read(COHORT / "test-physical" / f"{match}-curve.json")
        assert report["frames"] == 240 and report["match"] == match
        values = report["bank"] + [dict(arm="neural", width=None, **r) for r in neural]
        bank = []
        for r in values:
            packet = r["sha256"]
            assert len(packet) == 64 and 0 < r["bytes"] and 0 < r["kbps"]
            assert packet not in scores or scores[packet] == r["dists"]
            scores[packet] = r["dists"]
            bank.append({k: r[k] for k in ("arm", "width", "bytes", "kbps", "sha256")})
        banks[match] = bank

    def priced(match, cap, rule):
        family, width = rule
        if family == "candidate" and not route[match]:
            family = "ordinary"
        eligible = [r for r in banks[match] if r["kbps"] <= cap and r["arm"] == family]
        preferred = [r for r in eligible if r["width"] == width]
        return min(preferred or eligible, key=lambda r: (-r["bytes"], r["sha256"]))

    outcomes = []
    for scope, cap in [("starved_cohort", 15), ("starved_cohort", 20.4), ("pooled", 35)]:
        group = next(row for row in three["results"][scope] if row["available_kbps"] == cap)
        matches = group["fixed_population"]
        assert len(matches) == (12 if scope == "pooled" else 7)
        measured = {m: {p: priced(m, cap, p) for p in POLICIES} for m in matches}
        trials = []
        for match in matches:
            training = [m for m in matches if m != match]

            def train(rule):
                return mean(scores[measured[m][rule]["sha256"]] for m in training)

            choice = min(POLICIES, key=train)
            ordinary = min(POLICIES[:2], key=train)
            oracle = min(
                (
                    r
                    for r in banks[match]
                    if r["kbps"] <= cap and (r["arm"] != "candidate" or route[match])
                ),
                key=lambda r: scores[r["sha256"]],
            )
            selected = dict(
                chosen=measured[match][choice],
                ordinary=measured[match][ordinary],
                neural=measured[match][POLICIES[-1]],
                oracle=oracle,
            )
            assert all(r["kbps"] <= cap for r in selected.values())
            trials.append(
                dict(
                    match=match,
                    route_to_ball=route[match],
                    trained_on=training,
                    choice=choice,
                    chosen_ordinary=ordinary,
                    streams={k: dict(**r, dists=scores[r["sha256"]]) for k, r in selected.items()},
                )
            )
        summary = {
            k: dict(
                kbps=mean(t["streams"][k]["kbps"] for t in trials),
                dists=mean(t["streams"][k]["dists"] for t in trials),
            )
            for k in trials[0]["streams"]
        }
        assert summary["oracle"]["dists"] <= summary["chosen"]["dists"]
        counts = dict(Counter("/".join(map(str, t["choice"])) for t in trials))
        outcomes.append(
            dict(
                ceiling=cap, matches=len(matches), choice_counts=counts, means=summary, folds=trials
            )
        )
    return dict(
        input_sha256=hashes,
        scorer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        outcomes=outcomes,
    )


if __name__ == "__main__":
    result = evaluate()
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in result["outcomes"]:
        print(
            row["ceiling"],
            row["matches"],
            row["choice_counts"],
            {k: (round(v["kbps"], 2), round(v["dists"], 4)) for k, v in row["means"].items()},
        )
