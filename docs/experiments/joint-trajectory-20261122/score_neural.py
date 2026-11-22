"""Score authenticated DCVC producer frames on the joint receiver's 12 human regions."""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("joint_score", HERE / "score.py")
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)
q = joint.quarter


sys.path.insert(0, str(HERE.parent / "starved-base-local-20261122"))
import crops  # noqa: E402
from src.pointstream.tracked_patch import yuv_to_rgb  # noqa: E402


def main(args):
    q.cv2.setNumThreads(1)
    q.torch.set_num_threads(2)
    q.torch.set_num_interop_threads(1)
    assert q.sha(q.WEIGHTS) == q.EXPECTED
    net = q.quality.load_dists(q.WEIGHTS, "cpu")
    meta = json.loads((args.data / "review12.json").read_text())
    reports = json.loads((args.data / "reports.json").read_text())
    digest = q.sha(args.data / "native.psdc")
    size = (args.data / "native.psdc").stat().st_size
    assert reports["encode"]["container_sha256"] == digest == meta["container_sha256"]
    arrays = [
        reports["decode"][tag]["passes"][i]["frame_sha256"]
        for tag in ("decode-a", "decode-b")
        for i in (0, 1)
    ]
    assert len(arrays[0]) == 240 and all(v == arrays[0] for v in arrays) and meta["frames"] == 240
    assert [x["tick"] for x in meta["ticks"]] == crops.TICKS
    native = q.np.memmap(args.data / "review12.yuv", q.np.uint8, "r", shape=(12, 816, 960))
    assert (args.data / "review12.yuv").stat().st_size == native.nbytes
    labels = json.loads(crops.LABELS.read_text())["frames"]
    assert q.sha(crops.LABELS) == "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"
    assert q.sha(crops.BUNDLE) == "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb"
    full_sse = 0
    with (
        zipfile.ZipFile(crops.BUNDLE) as source,
        zipfile.ZipFile(args.data / "crops.zip", "x") as archive,
    ):
        for raw, ident, label in zip(native, meta["ticks"], labels, strict=True):
            t = ident["tick"]
            assert (
                t == label["tick"]
                and hashlib.sha256(raw).hexdigest() == arrays[0][t] == ident["sha256"]
            )
            canvas = q.cv2.resize(
                yuv_to_rgb(raw, 960, 544, False), (1920, 1080), interpolation=q.cv2.INTER_CUBIC
            )
            b = source.read(f"frame-{t:03d}/source.png")
            assert hashlib.sha256(b).hexdigest() == label["source_png_sha256"]
            ref = q.cv2.imdecode(q.np.frombuffer(b, q.np.uint8), q.cv2.IMREAD_COLOR)[:, :, ::-1]
            delta = ref.astype(q.np.int32) - canvas
            full_sse += int(q.np.sum(delta * delta, dtype=q.np.int64))
            for region, y, x, mask in crops.supports(label):
                pre = f"{t:03d}/{region}/"
                archive.writestr(pre + "mask.png", crops.png(mask))
                archive.writestr(pre + "source.png", crops.png(ref[y, x, ::-1]))
                archive.writestr(pre + "neural.png", crops.png(canvas[y, x, ::-1]))
    with zipfile.ZipFile(args.data / "crops.zip") as z:
        q.scorer.score_tick(net, z, 10, ["source", "neural"])
        rows = [r for t in crops.TICKS for r in q.scorer.score_tick(net, z, t, ["neural"])]
    means = {
        region: sum(r["dists"] for r in rows if r["region"] == region) / 12
        for region in ("ball", "p1", "p2")
    }
    result = dict(
        qp=reports["encode"]["qp"],
        kbps=size / 600,
        bytes=size,
        mean=means,
        players=(means["p1"] + means["p2"]) / 2,
        packet_sha256=digest,
        producer_report_sha256=q.sha(args.data / "reports.json"),
        frame_slice_sha256=q.sha(args.data / "review12.yuv"),
        four_passes_match=True,
        full_rgb_sse=full_sse,
        reviewed_full_rgb_psnr=10 * math.log10(255**2 * len(labels) * 1920 * 1080 * 3 / full_sse),
        rows=rows,
        metric=q.quality.dists_record(q.WEIGHTS),
        script_sha256=q.sha(Path(__file__)),
        producer_twelve=meta,
        ticks=crops.TICKS,
    )
    (args.data / "scored.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: result[k] for k in ("qp", "kbps", "mean", "players")})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("data", type=Path)
    main(cli.parse_args())
