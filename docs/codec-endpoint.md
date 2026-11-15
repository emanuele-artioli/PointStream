# Complete codec endpoints

The prospective contribution ladder starts with a functioning native fallback.
This is infrastructure, not evidence of a semantic compression benefit. PSC1 is
a new profile; historical PSTM/PSBL packets and measurements are unchanged.

## PSC1 native fallback

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

## PTP1 tracked pixel patches

The same commands also support the `tracked-pixel-v1` profile. Its initial
input is **raw planar 8-bit YUV420 only**, with explicit geometry, frame rate,
range and frame count. The `positions` array supplies one encoder-side centre
hint `[x, y]` or `null` for every frame. Coordinates are converted to even,
clamped 32×32 origins; a 64×64 coded tile contains each source crop. Each
frame without a patch retains the decoded base. A `null` hint means no patch
is sent; it does not assert that the object or foreground is physically absent.
If all positions are `null`, the packet contains only the native stream. The
egocentric correctness test explicitly selects this fallback even when hands
and objects are present.

For example, a two-frame configuration for a 64×64 input is:

```json
{"profile":"tracked-pixel-v1","width":64,"height":64,"fps":[30,1],"frames":2,"full_range":false,"crf":48,"preset":4,"input_format":"yuv420p","positions":[[30,30],null],"position_provenance":{"kind":"caller-supplied"}}
```

```bash
python -m src.pointstream encode input.yuv output.ptp --config config.json --scratch /absolute/local/scratch --record encode.json
python -m src.pointstream inspect output.ptp
python -m src.pointstream decode output.ptp decoded.rgb --scratch /absolute/local/scratch --record decode.json
```

PTP1 decode produces **raw RGB24**, in frame order at the declared geometry,
not PSC1's planar YUV420. It requires the packet-declared dav1d executable and
NumPy version; retain the packed runtime manifest to identify the rest of the
installed software. The packet contains the complete base and optional patch
IVF streams, all per-frame placements, canonical metadata and a checksum;
`inspect` counts every byte. The decoder receives no source frames, hints,
labels or content-specific model. Source-denial claims require a separate
OS-enforced qualification.

Hints can come from cached automatic BALL positions. Record their source in
`position_provenance`; the encoder receipt hashes the supplied hints, but does
not measure their generation cost or establish fresh tracker inference. PTP1
limits decoded RGB output to 512 MiB; this is not a process-memory limit.
Neither endpoint establishes streaming recovery or real-time throughput. A
functioning endpoint alone does not establish a compression benefit.

The source-only annotation reviewer is `python -m tools.datasets.annotation_review`;
its module documentation defines the hash-bound manifest, prepare and import
commands. Serve a prepared review with the existing loopback K8 review server.
Masks approved by a human have model-assisted provenance; blind reviewed frames
retain separate provenance. Rejected and uncertain frames remain in the ledger.
