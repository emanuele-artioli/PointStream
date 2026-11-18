"""Replay the three real ball-only packets and their unenhanced native controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from experiments.tennis import common_player_ball as ball, common_player_packet as player, k4b_features
from src.codecs import svtav1
from src.pointstream import tracked_patch


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'carrier', 'sidecar', 'anchors', 'witness', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    started = time.monotonic()
    w = json.loads(a.witness.read_text())
    check(sha(a.source) == w['source_sha256'] and sha(a.carrier) == w['carrier_sha256'] and
          sha(a.sidecar) == w['sidecar_sha256'], 'Input identity')
    dec = svtav1.tool('dav1d')
    ffmpeg = svtav1.tool('ffmpeg')
    check(ffmpeg['sha256'] == w['ffmpeg']['sha256'] and dec['sha256'] == w['decoder']['sha256'], 'Native identities')
    origins, tile, size = ball._open(a.carrier.read_bytes(), a.sidecar.read_bytes(), dec['sha256'])
    check(size == (1920,1080), 'Original canvas size')
    full = a.output/'source.yuv'
    subprocess.run(k4b_features.convert_command(ffmpeg['path'], a.source, full, 0, 240), check=True, timeout=45)
    check(sha(full) == w['source_yuv_sha256'], 'Native source pixels')
    source = np.memmap(full, np.uint8, 'r', shape=(240,1620,1920))
    tile_raw = subprocess.check_output([ffmpeg['path'], '-v','error','-threads','1','-c:v','libdav1d','-f','ivf',
                                        '-i','pipe:0','-pix_fmt','yuv420p','-f','rawvideo','pipe:1'], input=tile, timeout=30)
    tile_yuv = np.frombuffer(tile_raw, np.uint8).reshape((240,96,64))
    profile = {'mode':'native_only','recipe':player.NATIVE_RECIPE,'resources':{'native_decoder':ffmpeg['sha256']},
               'coded_size':[960,544],'canvas_size':[1920,1080],'cuts':[],'runs':[],'order':[]}
    rows = []
    for arm in json.loads(a.anchors.read_text())['rows']:
        if arm['mode'] != 'plain':
            continue
        begin = time.monotonic()
        base_path = Path(arm['stream'])
        check(sha(base_path) == arm['stream_sha256'], 'Anchor stream')
        native = player.build(profile, base_path.read_bytes(), None, {})
        subdir = a.output/str(arm['crf'])
        subdir.mkdir()
        (subdir/'native.pscp').write_bytes(native)
        payload = ball.pack(native, origins, tile, decoder_sha256=dec['sha256'])
        (subdir/'ball.sidecar').write_bytes(payload)
        native_yuv = subdir/'decoded.yuv'
        native_cmd = [ffmpeg['path'],'-v','error','-nostdin','-threads','1','-c:v','libdav1d','-f','ivf','-i','pipe:0',
                      '-pix_fmt','yuv420p','-f','rawvideo',str(native_yuv)]
        sections, _, _ = player.unpack(native)
        subprocess.run(native_cmd, input=sections['base'], check=True, timeout=35)
        check(sha(native_yuv) == arm['decoded_sha256'], 'Independent full native decode')
        planes = np.memmap(native_yuv,np.uint8,'r',shape=(240,816,960))
        latest = []
        def canvases():
            for t, frame in enumerate(planes):
                rgb = cv2.resize(tracked_patch.yuv_to_rgb(frame,960,544,False),size,interpolation=cv2.INTER_CUBIC)
                latest[:] = [rgb]
                yield t,rgb
        err = [0.,0.]
        present = count = 0
        whole = hashlib.sha256()
        for t, recon in ball.render(native,payload,canvases(),scratch=subdir,decoder=dec['path']):
            base = latest[0]
            expected = base.copy()
            if origins[t] is not None:
                x,y = origins[t]
                rect = np.s_[y:y+32,x:x+32]
                patch = tracked_patch.yuv_to_rgb(tile_yuv[t],64,64,False)[16:48,16:48]
                expected[rect] = patch
                truth = tracked_patch.yuv_to_rgb(source[t],1920,1080,False)[rect]
                for i, canvas in enumerate((base,recon)):
                    err[i] += np.square(canvas[rect].astype(float)-truth).sum()
                present += 1
            check(np.array_equal(expected,recon),f'Canvas mismatch at {t}')
            whole.update(recon)
            count += 1
        check(count == 240 and present == 195,'Timestamp or track coverage')
        rows.append(dict(crf=arm['crf'],base_ivf_bytes=len(sections['base']),native_bytes=len(native),
                         sidecar_bytes=len(payload),total_bytes=len(native)+len(payload),
                         plain_ivf_kbps=arm['complete_kbps'],native_packet_kbps=len(native)/600,
                         ball_packet_kbps=(len(native)+len(payload))/600,
                         predicted_patch_rgb_psnr=[10*math.log10(255**2*present*32*32*3/e) for e in err],
                         native_sha256=sha(subdir/'native.pscp'),sidecar_sha256=sha(subdir/'ball.sidecar'),
                         all_frame_rgb_sha256=whole.hexdigest(),frames=count,predicted_patch_frames=present,
                         elapsed_s=time.monotonic()-begin))
    check(len(rows)==3, 'Missing rate')
    report = dict(rows=rows,source_sha256=sha(a.source),sender_witness_sha256=sha(a.witness),anchor_report_sha256=sha(a.anchors),
                  decoder=dec,native_base_decoder=ffmpeg,script_sha256=sha(Path(__file__)),elapsed_s=time.monotonic()-started)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:
        print(row['crf'],row['native_packet_kbps'],row['ball_packet_kbps'],row['predicted_patch_rgb_psnr'])


if __name__ == '__main__':
    main()
