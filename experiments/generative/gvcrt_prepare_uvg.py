"""Bounded CPU preparation of the registered first 16 native UVG Jockey frames.

Cadence is declared from the filename for infrastructure smoke, not certified
source timing. BT.709 limited-range YUV to full-range RGB is an explicit policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

WIDTH, HEIGHT, SOURCE_FRAMES, SMOKE_FRAMES = 1920, 1080, 600, 16
FRAME_BYTES = WIDTH * HEIGHT * 3 // 2


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if source.name != 'Jockey_1920x1080_120fps_420_8bit_YUV.yuv':
        raise ValueError('Source differs from registered Jockey input')
    if source.stat().st_size != FRAME_BYTES * SOURCE_FRAMES:
        raise ValueError('Native source physical length differs from 600 registered frames')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    full_hash, frame_hashes = hashlib.sha256(), []
    with source.open('rb') as handle:
        for index in range(SOURCE_FRAMES):
            raw = handle.read(FRAME_BYTES)
            if len(raw) != FRAME_BYTES:
                raise ValueError('Truncated native input')
            full_hash.update(raw)
            if index < SMOKE_FRAMES:
                frame_hashes.append(hashlib.sha256(raw).hexdigest())
        if handle.read(1):
            raise ValueError('Trailing source bytes')
    converter = args.ffmpeg.resolve(strict=True)
    command = [str(converter), '-nostdin', '-hide_banner', '-loglevel', 'error',
               '-f', 'rawvideo', '-pixel_format', 'yuv420p', '-video_size', '1920x1080',
               '-framerate', '120', '-i', str(source), '-frames:v', '16',
               '-vf', 'scale=in_range=limited:out_range=full:in_color_matrix=bt709:out_color_matrix=bt709:flags=accurate_rnd+bitexact,format=rgb24',
               '-sws_flags', 'accurate_rnd+bitexact', '-fps_mode', 'passthrough',
               '-threads', '2', '-filter_threads', '1', str(output / 'frame_%06d.png')]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    from PIL import Image
    frames = []
    for index in range(SMOKE_FRAMES):
        target = output / f'frame_{index + 1:06d}.png'
        with Image.open(target) as image:
            if image.mode != 'RGB' or image.size != (WIDTH, HEIGHT):
                raise ValueError('Converted RGB geometry/mode mismatch')
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
        frames.append({'path': str(target), 'sha256': digest(target), 'rgb_pixel_sha256': pixel_hash,
                       'native_frame_index': index, 'native_byte_offset': index * FRAME_BYTES,
                       'native_frame_sha256': frame_hashes[index]})
    (output / 'frames.json').write_text(json.dumps(frames, sort_keys=True, indent=2) + '\n')
    receipt = {'purpose': 'infrastructure_smoke_only', 'source': str(source),
               'source_sha256': full_hash.hexdigest(), 'source_bytes': source.stat().st_size,
               'source_frames_from_physical_length': SOURCE_FRAMES, 'registered_indices': [0, 15],
               'nominal_fps': 120, 'cadence_status': 'declared_filename_cadence_not_certified_timebase',
               'color_policy': 'planar8bit420 BT709 limited YUV to full RGB; original matrix/range provenance unverified',
               'ffmpeg': str(converter), 'ffmpeg_sha256': digest(converter),
               'ffmpeg_version': subprocess.check_output([str(converter), '-version'], text=True),
               'command': command, 'converter_stderr': completed.stderr,
               'prepare_argv': sys.argv, 'prepare_helper_sha256': digest(__file__),
               'frames_json_sha256': digest(output / 'frames.json'), 'elapsed_seconds': time.monotonic() - start}
    (output / 'preparation_receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'frames_json': str(output / 'frames.json'), 'source_sha256': full_hash.hexdigest(),
                      'receipt': str(output / 'preparation_receipt.json')}, indent=2))


if __name__ == '__main__':
    main()
