"""Canonical paid companion manifest for the campus common-player receiver."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

GEN_ENV = 'bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e'
SAM_ENV = '32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661'
SAM_WEIGHT = '0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6'
NATIVE = '7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2'


def encode(carrier_sha256: str) -> bytes:
    if not re.fullmatch('[a-f0-9]{64}', carrier_sha256):
        raise ValueError('invalid carrier identity')
    data = {'profile': 'common-tennis-linux-receiver-v1', 'carrier_sha256': carrier_sha256,
            'generator_env': GEN_ENV, 'sam_env': SAM_ENV, 'sam_weight': SAM_WEIGHT,
            'sam_commit': '2345a4ad109ac29c569da749c91d84f10dc08c40', 'native_ffmpeg': NATIVE,
            'sam_policy': 'independent-frame-pose-prompts', 'composition': 'decoded-base,farthest-to-nearest-players,optional-ball'}
    return json.dumps(data, sort_keys=True, separators=(',', ':')).encode('ascii')


def verify(path: Path, carrier_sha256: str) -> dict:
    expected = encode(carrier_sha256)
    if path.stat().st_size != len(expected) or path.read_bytes() != expected:
        raise ValueError('receiver manifest or carrier identity mismatch')
    return {'manifest_bytes': len(expected), 'manifest_sha256': hashlib.sha256(expected).hexdigest(), **json.loads(expected)}
