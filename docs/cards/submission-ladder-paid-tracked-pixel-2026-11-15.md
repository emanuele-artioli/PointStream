# Card: pay and independently reconstruct tracked pixel patches

ROOT releases a **new 7,500 CPU-second / zero GPU** stage under the user's twelve-hour authority ending **2026-11-15T11:07:34Z**. It follows accepted coded-pixel oracle `20261115T020807Z-26806765`. Existing closed allocations remain closed. No training, provider change, paper submission or automatic confirmation-outcome access is authorized by this card.

## Scientific decision

The finite oracle saves 48.29% crop-DISTS BD-rate versus native and 31.53% versus paid streak. Its worst crop-LPIPS value (0.0151399) is better than streak's best (0.0156854) and native's best (0.030699); sampled rate ranges overlap. There is **no common-quality interval for LPIPS BD-rate**. Preserve the literal implementation's `native_gate_passed=null`; strict quality-range separation supplies the registered improving-LPIPS corroboration for a paid test, without inventing a BD-rate. Full-frame rate costs of roughly 10–12% remain visible.

Question: does the reconstruction retain a scoped benefit after paying all placement/framing/resource costs and denying the receiver access to source and encoder caches? Use the exact three 720p/30fps base points (CRF 48/55/62), CRF20/preset4 padded 64×64 tiles containing 32×32 source crops, and the actual quantized PSBL trajectories. No tuning or ground-truth placement. Compare the paid candidate with native, streak, the matched GTROI control and the pooled native/neural/ROI reference. A negative result closes this construction rather than prompting hidden retuning.

## Complete contract and correctness gate

PTP1 provides public `encode`, `decode` and `inspect` entrypoints while preserving PSC1 and historical formats. Initial input is raw YUV420 plus explicit configuration and content-specific position hints. Receipts bind their source and hashes. These are retained genuine automatic encoder outputs, not fresh inference or an end-to-end timing qualification. The packet pays both complete IVF streams and every per-frame placement, geometry, timing, mode, checksum and resource declaration.

Decode takes only the packet and declared generic dav1d/NumPy resources, producing raw RGB24 under the specified BT.709 nearest-chroma profile. It requires no content-specific source, model, hints or cache. Missing positions retain the base; an all-null trajectory omits the patch stream. Essential decoder failure fails closed. The 512 MiB decoded-output cap is not a process-memory cap.

Tennis smoke (16 frames), followed by the full 48-frame stage, authenticates original JPEG/YUV/native-62 controls, runs core encoding and fresh public decoding for all three points, and verifies complete byte accounting. Every full-stage RGB frame hash, aggregate hash, base stream and patch stream must match the accepted oracle before reusing its authenticated metric arrays. Recompute rates from actual complete PTP1 bytes. If parity fails, preserve the failure and make no cached-score quality claim; any rescoring needs a separate budget. Zero new GPU is released here.

Egocentric correctness uses accepted P09_103 source `15cc7ab1`, 1080p/50fps/full-range, with a 16→64-frame native-fallback case. It establishes functional decoding only. Existing quality scores use another RGB conversion and cannot be reused for this output. Tests cover mixed missing positions, bounds/corruption, colour/cadence/count and native fallback; no arbitrary-cut, streaming or real-time claim follows.

## Independent receiver and stopping conditions

Reuse the existing tested Landlock/fresh-exec mechanism through the reviewed narrow adapter. Authenticate actual original/staged archives, source JPEG/YUV, labels, encoder state and private detector weights before restriction. After fresh execution, require EACCES/EPERM read failures and exact reconstruction from packet plus installed resources. ENOENT, a fresh process alone or removed source files do not establish denial.

The proof claims inherited filesystem restrictions. It does not claim network/MCP isolation or independently observed native-child births. Missing runtime inventory, unavailable isolation, failed reconstruction, failed gates or exhausted budgets stop the stage; do not build another qualification stack automatically. Preserve functional evidence if OS qualification remains unresolved.

| Part | CPU seconds | GPU seconds |
|---|---:|---:|
| Tennis three paid points | 1,530 | 0 |
| Egocentric native fallback | 1,530 | 0 |
| Independent receiver | 1,530 | 0 |
| Implementation, review, publication and failure allowance | 2,910 | 0 |

Each fleet job uses smoke≤450s, full≤900s, validator≤60s and lifecycle allowance≤120s, with a canonical wait≤1,530s. Freeze source/spec/input hashes, substantive validators and measured cost before scaling. Reuse valid unchanged qualifications. Never overwrite sources, replay unknown jobs or create recurring model polling.

Confirmation stays sealed until final-candidate freeze and corrected human-reviewed references. Reserved VISOR windows have annotation gaps; never turn missing masks into empty foreground or substitute an easier window after outcomes. Both domains retain their separately achieved level. A successful paid tennis pilot would remain a development finding, not submission readiness.
