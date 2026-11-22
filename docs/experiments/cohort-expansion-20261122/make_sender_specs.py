"""Generate field-specific sender metadata and fleet specs without test labels."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "/home/itec/emanuele/Datasets/pointstream-data/tennis/cohort-expansion-20261122"


def main(output):
    output.mkdir()
    template = json.loads(
        (HERE.parent / "shanghai-prospective-20261122/chunk-0-qualified-host.json").read_text()
    )
    records = json.loads((HERE / "staged.json").read_text())["selection"]
    records.append(
        dict(
            match="med000",
            rally="000",
            source=dict(
                path="/home/itec/emanuele/Datasets/pointstream-data/tennis/w7-tier1-20261030/sources/medvedev_struff__000.mp4",
                sha256="3bf7ddaae0db5c48b1072fccb15f9f162e41fa5999886b9a5bde58dc0495c630",
            ),
            video=dict(avg_frame_rate="50/1", width=1920, height=1080, nb_frames=1914),
        )
    )
    for rec in records:
        name = rec["match"]
        meta = {k: rec[k] for k in ("match", "rally", "source", "video")}
        path = output / f"{name}.json"
        path.write_text(json.dumps(meta, indent=2) + "\n")
        md = dict(
            name="metadata",
            path=f"{ROOT}/{name}.json",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            extract=False,
        )
        staged = [
            dict(
                name="source",
                path=rec["source"]["path"],
                sha256=rec["source"]["sha256"],
                extract=False,
            ),
            md,
        ] + template["staged_inputs"][1:]
        module = "experiments.tennis.cohort_ball_sender"
        spec = dict(
            template,
            entrypoint=["-m", module],
            arguments=[
                "--source",
                "{staged:source}",
                "--metadata",
                "{staged:metadata}",
                "--archive",
                "{staged:archive}",
                "--weights",
                "{staged:weights}",
                "--frames",
                "{frames}",
            ],
            staged_inputs=staged,
            inputs=[{k: v for k, v in i.items() if k in ("path", "sha256")} for i in staged],
            validator=["{python}", "-m", module, "--validate"],
            hosts=["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"],
            deadline="2026-11-22T21:01:00Z",
            scale={"frames": {"smoke": 24, "full": 240}},
            smoke={
                "seconds": 500,
                "representative_basis": "Real 24-frame source, unchanged pretrained model with original vs split chunk equivalence, source-fitted rendered trajectory, sealed position stream.",
            },
            full={"seconds": 2450},
            budget_seconds=3200,
            stall_seconds=2500,
        )
        (output / f"{name}-sender-spec.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    main(p.parse_args().output)
