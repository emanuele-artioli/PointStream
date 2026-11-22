"""Score an attained physical external control after deterministic byte selection."""

import argparse
import json

from receiver_spec import HERE, INBOX, save


def main(a):
    assert len(a.report_sha) == len(a.archive_sha) == 64
    assert all(ch in "0123456789abcdef" for ch in a.report_sha + a.archive_sha)
    base = json.loads((HERE / (a.event + "-score-dcvc-480-q24.json")).read_text())
    i = next(x for x in base["staged_inputs"] if x["name"] == "data")
    i["path"] = str(INBOX / a.job / "full/published.tar")
    i["sha256"] = a.archive_sha
    base["inputs"] = [{k: r[k] for k in ("path", "sha256")} for r in base["staged_inputs"]]
    args = base["arguments"]
    args[args.index("--arm") + 1] = a.arm
    args[args.index("--producer") + 1] = a.job
    args += ["--report-sha256", a.report_sha]
    if a.motion:
        args.append("--with-motion")
    save(HERE / (a.event + "-score-" + a.arm + ".json"), base)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", action="store_true")
    for k in ("event", "arm", "job", "archive-sha", "report-sha"):
        p.add_argument("--" + k, required=True)
    main(p.parse_args())
