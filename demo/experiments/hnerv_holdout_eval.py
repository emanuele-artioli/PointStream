"""Score a trained HNeRV on hold-out segments after one checkpoint load.

HNeRV's own ``train_nerv_all.py --eval_only`` rebuilds the network from the
frame count of the folder it is given, and it reloads the checkpoint every
invocation. This runner builds the training architecture once, then quantizes
each hold-out segment on its own so the embedding and the decoder bits stop
at the segment cut.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from demo.experiments.factory_bg_rd import (
    HNERV_MODELSIZE,
    HNERV_ROOT,
    SEGMENT_FRAMES,
    WORK_ROOT,
    hnerv_checkpoint,
    hnerv_import_stub,
    holdout_stems,
    kept_frames,
    segment_starts,
)


def _architecture(train_frames: int):
    """Match train_nerv_all's embedding and decoder-size calculation."""
    import torch
    from model_all import HNeRV

    enc_strds = [5, 3, 2, 2, 2]
    dec_strds = [5, 3, 2, 2, 2]
    final_size = 1080 * 1920
    total_enc = int(np.prod(enc_strds))
    embed_hw = final_size / total_enc**2
    enc_dim1, embed_ratio = 64.0, 0.5
    embed_dim = int(embed_ratio * HNERV_MODELSIZE * 1e6 / train_frames / embed_hw)
    embed_param = float(embed_dim) / total_enc**2 * final_size * train_frames
    decoder_size = HNERV_MODELSIZE * 1e6 - embed_param
    reduce = 1.2
    ch_reduce = 1.0 / reduce
    dec_ks1, dec_ks2 = 1, 5
    lower_width = 12
    a = ch_reduce * sum(
        ch_reduce ** (2 * i) * s**2 * min(2 * i + dec_ks1, dec_ks2) ** 2
        for i, s in enumerate(dec_strds)
    )
    b = embed_dim * 9
    c = 0.0
    fc_dim = int(np.roots([a, b, c - decoder_size]).max())
    args = argparse.Namespace(
        embed="",
        ks="0_1_5",
        num_blks="1_1",
        enc_strds=enc_strds,
        dec_strds=dec_strds,
        enc_dim=f"{int(enc_dim1)}_{embed_dim}",
        conv_type=["convnext", "pshuffel"],
        norm="none",
        act="gelu",
        reduce=reduce,
        lower_width=lower_width,
        fc_dim=fc_dim,
        fc_hw="9_16",
        out_bias="tanh",
        quant_model_bit=8,
        quant_embed_bit=6,
        vid="holdout",
    )
    model = HNeRV(args).cuda()
    return model, args, embed_dim


def _load(model, checkpoint: Path) -> None:
    import torch

    saved = torch.load(checkpoint, map_location="cpu")
    state = saved["state_dict"]
    renamed = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(renamed, strict=True)


def _bits(values: list[int], overhead_elems: int) -> int:
    from dahuffman import HuffmanCodec

    if not values:
        return overhead_elems * 16
    codec = HuffmanCodec.from_data(values)
    table = {symbol: length for symbol, (length, _code) in codec.get_code_table().items()}
    unique, counts = np.unique(values, return_counts=True)
    payload = sum(int(count) * table[int(symbol)] for symbol, count in zip(unique, counts))
    return payload + overhead_elems * 16


def _score_segment(model, quant_model, quant_ckt, frames: list[Path], args) -> dict:
    import torch
    from hnerv_utils import quant_tensor
    from model_all import TransformInput
    from torchvision.io import read_image

    transform = TransformInput(args)
    embeds = []
    pred_psnr = []
    gts = []
    for path in frames:
        image = read_image(str(path)).float().cuda() / 255.0
        image = image.unsqueeze(0)
        model_in, gt, _mask = transform(image)
        with torch.inference_mode():
            pred, embed_list, _dec_time = model(model_in)
        embeds.append(embed_list[0].detach())
        gts.append(gt)
        pred_psnr.append(float(_psnr(pred, gt)))
    vid_embed = torch.cat(embeds, 0)
    quant_embed, dequant = quant_tensor(vid_embed, args.quant_embed_bit)
    quant_psnr = []
    for index, gt in enumerate(gts):
        with torch.inference_mode():
            quant_out, _embed_list, _dec_time = quant_model(gt, dequant[index:index + 1])
        quant_psnr.append(float(_psnr(quant_out, gt)))
    embed_values = quant_embed["quant"].flatten().tolist()
    embed_overhead = quant_embed["min"].nelement() + quant_embed["scale"].nelement()
    model_values: list[int] = []
    model_overhead = 0
    for layer in quant_ckt.values():
        model_values.extend(layer["quant"].flatten().tolist())
        model_overhead += layer["min"].nelement() + layer["scale"].nelement()
    embed_bits = _bits(embed_values, embed_overhead)
    total_bits = _bits(embed_values + model_values, embed_overhead + model_overhead)
    pixels = 1080 * 1920 * len(frames)
    return {
        "frames": len(frames),
        "pred_psnr": sum(pred_psnr) / len(pred_psnr),
        "quant_psnr": sum(quant_psnr) / len(quant_psnr),
        "embed_bits": embed_bits,
        "total_bits": total_bits,
        "embed_bpp": embed_bits / pixels,
        "total_bpp": total_bits / pixels,
        "quant_embed_bit": args.quant_embed_bit,
        "quant_model_bit": args.quant_model_bit,
    }


def _psnr(output, gt) -> float:
    import torch
    import torch.nn.functional as F

    loss = F.mse_loss(output.detach(), gt.detach())
    return float(-10 * torch.log10(loss + 1e-9))


def score_factory(work: Path, factory: str) -> None:
    import torch
    from train_nerv_all import quant_model as quantize_model

    train_frames = len(list((work / "datasets" / f"{factory}-flat").iterdir()))
    checkpoint = hnerv_checkpoint(work, factory)
    model, args, embed_dim = _architecture(train_frames)
    _load(model, checkpoint)
    model.eval()
    quantized, quant_ckt = quantize_model(model, args)
    quant_net = quantized[-1].cuda().eval()
    for stem in holdout_stems(factory):
        frames = kept_frames(work, stem)
        if not frames:
            raise FileNotFoundError(f"no filled frames for {stem}")
        for length in SEGMENT_FRAMES:
            for start in segment_starts(len(frames), length):
                record = work / "scores" / "hnerv" / factory / stem / f"{start:05d}_{length}.json"
                if record.is_file():
                    existing = json.loads(record.read_text())
                    if "total_bits" in existing:
                        print(f"SCORE hnerv exists {factory} {stem} {start} {length}", flush=True)
                        continue
                segment = frames[start:start + length]
                payload = _score_segment(model, quant_net, quant_ckt, segment, args)
                payload.update({
                    "codec": "hnerv",
                    "factory": factory,
                    "stem": stem,
                    "start": start,
                    "embed_dim": embed_dim,
                    "checkpoint": str(checkpoint),
                })
                record.parent.mkdir(parents=True, exist_ok=True)
                record.write_text(json.dumps(payload) + "\n")
                print("SCORE", json.dumps({k: payload[k] for k in ("stem", "start", "frames", "quant_psnr", "total_bpp")}), flush=True)
    del model, quant_net
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, nargs="+", choices=("factory001", "factory002"))
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    args = parser.parse_args(argv)
    stub = hnerv_import_stub(args.work_root / "hnerv-stub")
    sys.path[:0] = [str(stub), str(HNERV_ROOT)]
    for factory in args.factory:
        score_factory(args.work_root, factory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
