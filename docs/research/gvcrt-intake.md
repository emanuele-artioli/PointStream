# GVC-RT read-only intake (2026-10-01)

No installation, checkpoint download, inference, training or GPU reservation was performed for this intake. This is a prerequisite audit, not a codec result. Ordered execution remains gated by the coordinator after packet coding and matched anchors.

## Frozen source and checkpoint prerequisites

Official repository: https://github.com/semcomm/GVC-RT, pinned clean source `d0e32bfa3e8e282f9a77437c223c605858eea637`. Isolated clone: `/private/tmp/gvcrt-intake-official-h1-20261001`. Initial HTTP/2 clone and ls-remote failed; HTTP/1.1 shallow clone succeeded. No source import was executed.

Official checkpoint folder: https://drive.google.com/drive/folders/1cI1KWy-sfujSc2YsUxVWfhX7qcOJIxhS. Expected filenames are `GVC_RT-I.pt` and `GVC_RT-P.pt`. Public web rendering did not expose file download IDs or hashes; the clone contains only a checkpoint placeholder. Exact checkpoint IDs, bytes, SHA256, provenance and training/test exposure are **unverified**. A bounded depth-two GPU1 search under shared `third_party` and `weights` found no GVC/MAGVIT artifacts; this does not establish global absence.

The I loader in `src/models/image_model_gvcrt.py` tolerates missing checkpoints and selects only matching tensor names/shapes before `strict=False`. Its duplicate `student` branch and misleading EMA message require recording the actual selected dictionary. Qualification must fail closed on missing files and require complete model tensor coverage, matching shapes and explicit dictionary selection. The P loader is stricter, but its selection order must also be recorded. The accompanying stdlib helper verifies source and checkpoint identities only; it does not certify tensors or decode.

## Build and existing environment

Official reference is Python 3.12.12, torch 2.6.0+cu124; requirements additionally pin numpy 2.4.4, Pillow 12.0.0, scipy 1.17.0, lpips 0.1.4, DISTS-pytorch 0.1 and pybind11 3.0.1. Native entropy module `MLCodec_extensions_cpp` requires pybind11, setuptools, Python >=3.12 and C++17; Linux flags include `-Werror`. Optional `inference_extensions_cuda` uses torch CUDAExtension, nvcc and `-arch=native`; absent fused kernels fall back to PyTorch CUDA, not a demonstrated CPU codec.

Bounded GPU1 inventory found `pointstream-dcvc` Python 3.12.14, torch 2.5.1+cu121, numpy 2.5.2, pybind11 and an existing `MLCodec_extensions_cpp.cpython-312-x86_64-linux-gnu.so`. ABI/API compatibility is untested. Environment-local nvcc is CUDA 12.1.105 and ninja 1.13.2; system g++ is Ubuntu 11.4.0. LPIPS and DISTS were not discoverable in that environment. `pointstream-neural` has torch 2.5.1+cu121/numpy1.26.4; `pointstream` has torch2.5.1/numpy1.26.4. These are inventory observations, not failed algorithm evaluations. Do not modify shared environments; any installation needs the later gate and an isolated environment.

## Supported-domain source inventory

GPU1 has `/home/itec/emanuele/Datasets/UVG/1920x1080/{Jockey,ReadySteadyGo}_1920x1080_120fps_420_8bit_YUV.yuv`, each 1,866,240,000 bytes, consistent with 600 planar 8-bit 4:2:0 frames. Full file hashes, provenance, color matrix/range and timebase remain unverified; filename cadence is not sufficient temporal certification. No UVG/HEVC/MCL names appeared in the separate bounded shared-assets search.

Register Jockey frames 0–15 for a bounded smoke only after verifying source identity and conversion. Convert to exact RGB PNG inputs using explicit matrix/range and preserve decoder/tool version, command and per-frame hashes. Prefer RGB mode because the released YUV distortion branch appears to return LPIPS/DISTS variables that it never assigns. This is an integration defect to verify, not evidence against compression quality.

## Frozen proposed smoke and fresh decoder contract

Use 16 consecutive native 1920x1080 RGB frames, one registered QP, `reset_interval=8`, no forced additional I frames and `check_existing=false`. Register QP before outcomes once checkpoint-supported QP range is confirmed. The released reset condition is frame index modulo interval equal to one, so frames 1 and 9 exercise the feature-adaptor reset; preserve this exact policy. Encode and persist the real entropy stream, not estimated likelihood bits. Preserve code/patch hashes, model hashes, source/conversion identities, command, GPU UUID, environment and native extension identities. Smoke is infrastructure evidence only.

The script pads to at least 1088x1920 with replicated right/bottom pixels. Official RTX4090 peak-memory table reports 1202 MB at 1080p, but this does not guarantee our peak. Use a conservative 4 GiB initial workload estimate plus fleet-required 4 GiB free reserve; reinspect all six hosts and claim via fleet immediately before any launch. No host is selected or reserved by this document.

Implement a separate decoder process before scientific execution. Its inputs are only persisted stream, charged manifest and verified model artifacts; no source paths, source reader or metric models. The manifest must specify original crop geometry, coded canvas, frame count, cadence, model deployment identities and reset/access policy; charge its actual serialized bytes. Decode SPS and I/P packets, initialize entropy coders, reset DPB on I, reproduce feature-adaptor resets from SPS, reject truncated/trailing data and verify exact frame count. Deny source access in the receiver test. Persist reconstructed pixels and compare their hashes with the official decoder before independent scoring. This receiver is not yet implemented or qualified.

A separate scorer may read the exact original RGB targets. Declare float-vs-uint8 reconstruction convention and crop, independently score all-frame PSNR and LPIPS with correct [-1,1] input or `normalize=True`, and verify DISTS input convention. Released LPIPS receives [0,1] without normalization, so do not inherit its metric labels as calibrated results. Preserve per-frame scores and distinguish mean per-frame PSNR from pooled-MSE PSNR.

## Rate and claim constraints

Official kbps assumes 30 fps. Recompute `8*(stream_bytes+envelope_bytes)/(N/fps)` from measured physical files and registered cadence: never copy its kbps for 12 fps or filename-120fps UVG. Report original-crop and padded-canvas bpp separately. Native bytes include SPS/packet headers; model deployment is a separately disclosed shared-state assumption, not silently free transmission. Match anchors on exact RGB targets, crop, cadence, access/reset horizon and full charged payload. Do not infer a matched-quality advantage from same-QP rows.

Before a full run: checkpoint identities and complete tensor coverage; native entropy ABI; verified conversion/timebase; fresh source-denied receiver parity; correct independent metrics; bounded smoke; fleet claim/monitor receipts. License terms for original code/weights and checkpoint exposure also remain unverified (third-party notices alone do not establish them). A blocked prerequisite or installation failure is not a negative scientific result for GVC-RT.
