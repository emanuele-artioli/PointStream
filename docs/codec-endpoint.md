# Complete native fallback endpoint

The prospective contribution ladder starts with a functioning native fallback.
This is infrastructure, not evidence of a semantic compression benefit. PSC1 is
a new profile; historical PSTM/PSBL packets and measurements are unchanged.

Use the project's pinned Python environment and declared native tools. Supply
scratch on host-local disk (on GPU hosts, `/local/users/emanuele`, never `/tmp`).
All output and record paths must be new. Input can be video or raw planar 8-bit
YUV420. Geometry must be even; variable-rate/interlaced video is unsupported in
this initial constant-frame-rate, BT.709 profile. Selected video frame timestamps
are checked with ffprobe; unknown/HDR colour metadata is rejected. Raw inputs
are explicitly interpreted as BT.709. Each decoded window is capped at 512 MiB. Video colour conversion is explicit,
recorded, and must also be applied to scientific comparison sources.

Example configuration (`config.json`):

```json
{"width":1920,"height":1080,"fps":[25,1],"frames":240,"full_range":false,"crf":40,"preset":4,"input_format":"video"}
```

```bash
python -m src.pointstream encode input.mp4 output.psc --config config.json --scratch /absolute/local/scratch --record encode.json
python -m src.pointstream inspect output.psc
python -m src.pointstream decode output.psc decoded.yuv --scratch /absolute/local/scratch --record decode.json
```

The decoder receives only packet bytes and a native dav1d executable whose
SHA256 matches the packet's declared installed decoder. ffprobe verifies actual
coded geometry, pixel format and colour metadata before decode. Dynamic library
identities still require the frozen runtime manifest for scientific replication. Output is raw planar
YUV420 at the packet's geometry, cadence and range. The packet retains the
complete IVF stream, a canonical versioned JSON header and SHA256 checksum.
Every packet byte counts. The checksum detects corruption; it is not a signature.
Installed generic software is an explicit external resource. This profile has
no content-specific models, adaptation, masks, reference bank or enhancement.

`inspect` reports exact native/framing bytes, rate, geometry and resource ID.
Encode/decode records separately retain tool identities, commands and measured
wall/child-CPU timings; timing is not inferred from serialized video. Native
process separation is implemented, but OS-enforced source denial must be
qualified separately before a scientific receiver claim. Removing the source
before a test decode checks functional source independence only.

Semantic ROI, pixel-patch and ball profiles must pass their design and budget
gates before adding their complete decoder contracts. A functioning fallback
cannot count as a contribution-ladder win. Real-time operation, streaming
recovery and arbitrary corrupted-stream recovery are not qualified here.

The source-only annotation reviewer is `python -m tools.datasets.annotation_review`;
its module documentation defines the hash-bound manifest, prepare and import
commands. Serve a prepared review with the existing loopback K8 review server.
Masks approved by a human have model-assisted provenance; blind reviewed frames
retain separate provenance. Rejected and uncertain frames remain in the ledger.
