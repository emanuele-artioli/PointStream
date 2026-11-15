"""python -m src.pointstream {encode,decode,inspect}; native fallback profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pointstream import codec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    enc = commands.add_parser("encode")
    enc.add_argument("source", type=Path)
    enc.add_argument("output", type=Path)
    enc.add_argument("--config", type=Path, required=True)
    dec = commands.add_parser("decode")
    dec.add_argument("packet", type=Path)
    dec.add_argument("output", type=Path)
    for sub in (enc, dec):
        sub.add_argument("--scratch", type=Path, required=True)
        sub.add_argument("--decoder", default="dav1d")
        sub.add_argument("--ffprobe", default="ffprobe")
        sub.add_argument("--record", type=Path, required=True)
    enc.add_argument("--encoder", default="SvtAv1EncApp")
    enc.add_argument("--ffmpeg", default="ffmpeg")
    show = commands.add_parser("inspect")
    show.add_argument("packet", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "inspect":
            data = codec.read_packet(args.packet)
            if data.startswith(b'PTP1'):
                from src.pointstream import tracked_patch
                report = tracked_patch.inspect(data)
            else:
                report = codec.inspect(data)
            print(json.dumps(report, indent=2))
            return 0
        paths = [args.output.resolve(), args.record.resolve()]
        source_path = args.source if args.operation == "encode" else args.packet
        if len(set(paths)) != 2 or source_path.resolve() in paths:
            raise ValueError("source, output and record paths must be distinct")
        for path in paths:
            if not path.parent.is_dir():
                raise ValueError("output/record parent directory does not exist")
        if args.record.exists():
            raise FileExistsError(args.record)
        if args.operation == "encode":
            config = json.loads(args.config.read_text())
            if config.get('profile') == 'tracked-pixel-v1':
                from src.pointstream import tracked_patch
                record = tracked_patch.encode(args.source, args.output, config, scratch=args.scratch, encoder=args.encoder, decoder=args.decoder)
            else:
                record = codec.encode(args.source, args.output, config, scratch=args.scratch, encoder=args.encoder, decoder=args.decoder, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
        else:
            with args.packet.open('rb') as handle:
                magic = handle.read(4)
            if magic == b'PTP1':
                from src.pointstream import tracked_patch
                record = tracked_patch.decode(args.packet, args.output, scratch=args.scratch, decoder=args.decoder)
            else:
                record = codec.decode(args.packet, args.output, scratch=args.scratch, decoder=args.decoder, ffprobe=args.ffprobe)
        codec.write_new(args.record, (json.dumps(record, indent=2) + "\n").encode())
        print(json.dumps({"operation": args.operation, "packet_bytes": record["packet_bytes"], "record": str(args.record)}))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"pointstream: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
