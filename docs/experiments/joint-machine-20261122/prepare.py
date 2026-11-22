"""Stage the full native-sized joint display and the paid ordinary control."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "joint-trajectory-20261122"))
import motion  # noqa: E402


def build(args):
    stage = args.output
    stage.mkdir()
    full, prefix = hashlib.sha256(), hashlib.sha256()
    video = stage / "joint.mkv"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        "1920x1080",
        "-r",
        "50",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "bgr0",
        "-threads",
        "2",
        str(video),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as sink:

        def observe(t, _source, _plain, joint):
            full.update(joint)
            if t < 24:
                prefix.update(joint)
            sink.stdin.write(joint.tobytes())

        args.output = stage / "authenticated-display"
        motion.main(args, observe)
        sink.stdin.close()
        assert sink.wait(timeout=60) == 0
    old = json.loads((HERE.parent / "ball-machine-twenty-20261122/result.json").read_text())
    earlier = json.loads((args.controls / "20-240/result.json").read_text())
    row = next(r for r in earlier["selected"].values() if r["mode"] == "plain" and r["q"] == 63)
    assert (
        row["stream_sha256"] == "40ac346602fc01a02433a2ff0c90e32a3448fad25c6fbe436d2dd9b606d8c757"
    )
    native = stage / "ordinary.ivf"
    shutil.copyfile(row["stream"], native)
    assert motion.crops.file_sha(native) == row["stream_sha256"]
    joint = json.loads((motion.HERE / "result.json").read_text())["streams"]["20"]
    assert (
        full.hexdigest()
        == joint["all_240_rgb_sha256"]
        == json.loads((args.output / "motion-receipt.json").read_text())[
            "original_joint_240_rgb_sha256"
        ]
    )
    marks = json.loads(motion.crops.LABELS.read_text())["frames"]
    assert motion.crops.file_sha(motion.crops.LABELS) == old["labels"] and all(
        r["ball"]["status"] == "visible" for r in marks
    )
    binding = dict(
        frames=240,
        labels_sha256=old["labels"],
        marks={r["tick"]: r["ball"]["center_xy"] for r in marks},
        joint_display=dict(
            full=full.hexdigest(), smoke=prefix.hexdigest(), file=motion.crops.file_sha(video)
        ),
        ordinary=dict(
            stream=row["stream_sha256"],
            yuv=row["decoded_sha256"],
            prior_judge_input=old["results"]["plain240Q63"]["input_rgb_sha256"],
        ),
        ordinary_prior=old["results"]["plain240Q63"]["rows"],
        joint_wire_sha256=joint["wire_sha256"],
        native_neural_sha256="1d5c6f5e19e360b038e73ad8018075854ecea98c0a9ed05d3801f492bf02a30b",
        costs_bytes={"ordinary": 12974, "joint": 12927, "joint_cold": 257266, "neural": 12270},
    )
    mb = stage / "binding.json"
    mb.write_text(json.dumps(binding, indent=2) + "\n")
    with tarfile.open(stage / "inputs.tar", "x") as f:
        for p in (mb, native, video):
            f.add(p, arcname=p.name)
    print(motion.crops.file_sha(stage / "inputs.tar"), (stage / "inputs.tar").stat().st_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "controls", "output"):
        parser.add_argument(name, type=Path)
    build(parser.parse_args())
