"""CPU regression test of report identity; mocks are never codec evidence."""
from contextlib import nullcontext
import hashlib
import sys
from types import SimpleNamespace

from src.codecs import dcvc_uf_worker as worker


def test_yuv_report_preserves_container_identity_after_writing_pixels(tmp_path, monkeypatch):
    container = worker.pack_container(b'native-test-payload', structure='htl', frame_count=1)
    source = tmp_path/'stream.psdc'
    source.write_bytes(container)
    output = tmp_path/'decoded.yuv'
    pixels = b'pixel-bytes-distinct-from-container'
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        inference_mode=nullcontext, cuda=SimpleNamespace(synchronize=lambda: None)))
    helper = SimpleNamespace(NalType=SimpleNamespace(NAL_SPS=1, NAL_I=0),
        SPSHelper=lambda: SimpleNamespace(get_sps_by_id=lambda _: {'height': 4, 'width': 4}),
        read_header=lambda _: {'nal_type': 0, 'sps_id': 0},
        read_ip_remaining=lambda stream: (0, 0, 0, stream.read()),
        read_sps_remaining=lambda *_: None)
    monkeypatch.setitem(sys.modules, 'src.utils.stream_helper', helper)
    i_net = SimpleNamespace(decompress=lambda *_: {'x_hat': object()})
    p_net = SimpleNamespace(clear_dpb=lambda: None, add_ref_feature_from_frame=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker, '_build', lambda _: (i_net, p_net, 8, {}))
    monkeypatch.setattr(worker, '_require_claimed_device', lambda: None)
    monkeypatch.setattr(worker, '_import_extension', lambda: {})
    monkeypatch.setattr(worker, '_ensure_image_proxy', lambda _: None)
    monkeypatch.setattr(worker, '_to_yuv420_bytes', lambda *_: pixels)
    monkeypatch.setattr(worker, '_environment', lambda _: {})
    report = worker.decode({'container': str(source), 'structure': 'htl',
                            'src_type': 'yuv420', 'out_file': str(output)})
    assert report['container_sha256'] == hashlib.sha256(container).hexdigest()
    assert report['deterministic']
    assert output.read_bytes() == pixels
    assert report['passes'][0]['frame_sha256'] == [hashlib.sha256(pixels).hexdigest()]
