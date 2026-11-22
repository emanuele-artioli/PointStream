"""Recover stage timings from the same producers used in the near-22 comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ACCEPTED = HERE.parent / "receiver-replay-20261118" / "independent-idle-20261118"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(args):
    records = []
    for player in (1, 2):
        accepted = json.loads((ACCEPTED / f"player{player}-accepted.json").read_text())
        path = args.archives / f"player{player}-new-full.tar"
        assert sha(path) == accepted["archive_sha256"] and accepted["passed"]
        with tarfile.open(path) as tar:
            generation = json.load(tar.extractfile("publish/generation.json"))
        (out,) = generation["outputs"]
        assert generation["carrier_sha256"] == accepted["carrier_sha256"] and out["shape"][0] == 240
        records.append(
            dict(
                player=player,
                unit=out["unit"],
                job=accepted["job_id"],
                archive_sha256=accepted["archive_sha256"],
                device=generation["device"],
                model_load_seconds=generation["model_load_seconds"],
                render_and_save_seconds=out["render_and_save_seconds"],
            )
        )
    audit = json.loads((ACCEPTED / "finish-full-audit.json").read_text())
    finish = json.loads(args.finish.read_text())
    assert (
        sha(args.finish) == "645d719da006b9df133849753ed45b2da00a821958b716c1b4debf5e625ce131"
        and audit["all_frames_exact"]
    )
    neural = json.loads(args.neural.read_text())
    ns = json.loads((HERE / "dcvc-q6-human-regions.json").read_text())
    assert ns["producer_report_sha256"] == sha(args.neural)
    times = [p["decode_seconds"] for r in neural["decode"].values() for p in r["passes"]]
    assert len(times) == 4
    data = dict(
        playback_seconds=4.8,
        frames=240,
        players=records,
        finish=dict(
            job=audit["job_id"],
            sha256=sha(args.finish),
            total_seconds=finish["elapsed_seconds"],
            loop_seconds=finish["loop_seconds"],
            initialization_seconds=finish["initialization_seconds"],
        ),
        neural=dict(
            qp=6,
            device=neural["encode"]["environment"]["device_name"],
            producer_sha256=sha(args.neural),
            four_decode_pass_seconds=times,
        ),
        scope="Recorded stage intervals on the original allocations. Player saving, model initialization and control decoder scopes differ. No timing for an integrated live application.",
        script_sha256=sha(Path(__file__)),
    )
    with args.out.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(records, times)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("archives", "finish", "neural", "out"):
        cli.add_argument(field, type=Path)
    main(cli.parse_args())
