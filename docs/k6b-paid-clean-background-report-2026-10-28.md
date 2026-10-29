# Paid clean-background evaluation — two-metric gate failed

All80Al031@129 ticks completed in job `20261028T100153Z-27f39d64`. The
independent pixel/rate/decision audit passes. The formal result is
**MIXED/UNRESOLVED**: the source-pixel oracle wins DISTS, but both new policies
lose LPIPS robustly. Neither meets the registered requirement that ONE global
policy beat BOTH metrics by more than0.001. Close this fixed construction.
This is an accepted negative decision, not an aborted or incomplete full run.

## Result at the new total rate

Same80saved ticks0:240:3, full1920x1080, one familiar development window.
R is the frozen foreground region from the original player study. Lower
distances are better. Native whole-frame SVT provides the achievable
three-point upper concave anchor hull; interpolation stays inside its measured
range. One method point does not establish BD-rate or a two-window result.

| Policy at21.989629kbps | Whole-frame LPIPS | Whole-frame DISTS |
|---|---:|---:|
| Achievable SVT anchor | **0.240532** | 0.168336 |
| Frozen actual overlay on paid cleaned base | 0.274039 | 0.168824 |
| Source pixels inside fixed R on paid cleaned base | 0.268848 | **0.166890** |

The source-pixel policy loses LPIPS by0.028316784 and wins DISTS by0.001445586.
The actual policy loses LPIPS by0.033507470; its DISTS is centrally worse by
0.000487678, within the fixed0.001 sensitivity, so that cell is unresolved.
Do not combine different policies across metrics or relabel this as a joint win.
The source-pixel row grants original foreground pixels and R to the oracle;
it is not a delivered codec. Including the actual arm avoids assuming perceptual
metrics are mathematically monotone under source-pixel replacement. This finite
result does not bound arbitrary renderers, fills or background representations.

## What changed and what it cost in bits

Before any new encoding or scoring, the accepted card froze Telea radius3 on
the native Y/U/V planes independently, with original R projected nearest to
each plane, then the original area scale4 downsample. Every unmasked source
sample stays identical before compression. The unchanged native SVT path is
480x272, CRF63, preset4, one-CPU encoder affinity, all240frames at25fps.
The decoder receives the resulting paid base; source R is an encoder-side
oracle grant and is not secretly a new decoder input to the actual arm.

Complete IVF bytes fall from20557 to19659: **898bytes saved, 4.37%** of this
base stream. This is the difference between two complete encodes, not an
intrinsic allocation of bits to players. New base16.382500kbps plus unchanged
optimistic side5.607129kbps =21.989629kbps. The side retains KPS, amortized
reference bank/adapter and bootstrap. Bootstrap stays444bytes, with only the
base hash changed. No cost was removed to promote the result. Unmeasured final
wire/model-installation costs remain limitations of this optimistic diagnostic.

CPU-only preparation job `20261028T091736Z-bf9f97d2` completed all240native
frames and prepared all80scoring inputs. Original source YUV matches its frozen
hash, both native decodes agree, and all original source/base/actual/R replays
pass. Full preparation command225.812976s; no GPU claim or GPU time. Its full
summary's `projected_full_seconds` reuses the smoke expansion formula and is
not a full-run cost estimate; actual native time is the applicable record.
The repository now scales future metadata by240/frames, preserving the smoke
formula and correcting the full-stage label. Frozen evidence was not altered.

## Correctness, execution and independent audit

The fresh GPU smoke passes all19retained checks. Variable timing is explicitly
secondary: base warm ratio1.6043 is recorded, not vetoed. Slower warm samples,
setup once,35% safety and measured publication project239.300933s including
publication, below the350s full allowance. Full command actually completes in
157.656945s, internal worker154.863009s. These are shared-GPU cost records,
not performance benchmark claims. Peak allocated3082.722168MiB/reserved3744MiB
is below8000. No training, new generation or SAM inference occurred.

All80historical base/actual LPIPS/DISTS score controls reproduce exactly,
maximum error0. Original CUDA source RGB hashes also match all80ticks; repeated
new-base GPU conversions agree. The independently published new decoded-base
RGB array retains all80frames. A NumPy-only audit checks240original golden RGB
identities,80old overlay/fallback replays,80originalR oracle replays,160new
actual/oracle pixel reconstructions, all raw score arrays/means, source samples,
240-frame IVF framing, all84prepared member identities, complete byte accounting,
anchor interpolation, sensitivity and the formal decision. It passes. Its
`quality_passed` field means the quality-result integrity audit passed; the
scientific two-metric gate still failed.

## Request recovery and GPU ledger

The single allocated request's source extraction acknowledgement timed out90s
before publication. Every one of its214source files then matched the original
archive, and the extractor was confirmed gone on the actual transfer hostgpu2.
No readiness or admission existed. The usage-limit approval-review failure
executed no recovery action. After human resume, fresh inspection again proved
the request unpublished. The FIRST publication was completed under the SAME
ID, with a native source-identity guard before readiness; no new submission,
extraction replay or failed-gate retry. Only the expired absolute queue deadline
was refreshed before admission. All inputs, scientific gates and800s allocation
were unchanged. Original spec58c71c85 is preserved beside published spece1e7b877.

gpu3 A6000 index0, UUID `GPU-6f70a27e-4171-22d7-3ffd-647e7f186a49`.
Native admitted interval223.966597s; through canonical closure240.366170s.
Charge **250GPU-s conservatively**. Global ledger **2980/3600GPU-s**, **620s
unused**. The pre-publication CPU/usage interruption is not charged as GPU time.
Supervisor2778255/campaign2778260 are gone; foreign2655895 was untouched.
Completion22561a234c465516ba9fd1c5b131d872 and sharing
0e9405897c745595a8060553ab639f62 were delivered/acknowledged. Native waits ended;
no recurring model monitor, replacement request or further GPU authority.

## Provenance and preserved evidence

- GPU code `10a1ae76b57d1007b3959286791e2f8e223889d9`, source archive
  `7bc3be4243afee27ce19e1462165dd8e52ceeb8e4a3556a10a9587679dbab2a9`,
  native tree `82eee1d08d6212e59830fc018f9aba3087622daf15c8cc4c66c45472eab9ca6c`.
- CPU code `e542dfedd3a51118aba5cb3e7471305b7fc5c642`, source archive
  `6a5cf4377738569d007097f7f684314675e2802ca23b4837aac73c3147599ce0`.
- Published GPU spec `e1e7b8772abd141d717a338db219519e37a7249e144469390dcc86dfd40388eb`.
  Full output `eea7147698b036c9182cee3ec635a57361854b049967febc753b03e979fc1ab5`,
  497725440bytes; contains scores, summary and all decoded-base GPU RGB.
- Prepared outer input `56f54ccdc9db75ce98796597d2f2de0c849ad8041ab57c2def28dc5a4cd25ae0`,
  660490240bytes. Clean IVF `9d7a96db9a9a6f2f7a50911bac11865247531ecb2fa29c65fe832b94d0f8d065`.
- Packed runtime `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`;
  PyTorch2.10.0+cu128, cuDNN91002, OpenCV4.10.0. Original LPIPS AlexNet
  `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`,
  DISTS VGG `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`.
  Exact native encoder/decoder/FFmpeg paths, versions and hashes are in the
  prepared manifest and native dispatch; no alternative tools were substituted.

Read-only evidence archive:
`/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-paid-clean-background-20261028-27f39d64/evidence.tar`,
SHA `3da5b858f279f411938607bd92fd44fee76452473c8ffe4c0d40f9266e53af91`,
8611840bytes/49members. The manifest identifies every member and the two large
canonical raw publications retained separately under their job IDs. The archive
contains raw scores, original controls, exact213/214-file source archives,
native metadata/stream/bootstrap, audit code/results and recovery receipts.

Run `audit/audit.py` after extracting the bundle, passing the two raw publications
as `--prepared`/`--result`, their recorded `--prepared-sha`/`--result-sha`,
`control/golden-manifest.json` as `--golden`, `control/prior-real-player.tar`
as `--prior`, and a new `--out` path. No GPU is needed for this audit.

The original actual-player failure remains unchanged. Fair pixel ROI, the
required neural-codec anchor, complete serialization, remaining160generated
ticks, untouched evaluation and experience-track judgments are not established.
Neither this point nor its DISTS oracle win establishes TOMM readiness. A new
background/rate representation needs its own research decision and dominating
feasibility test before additional component compute; the unused620s is not
an automatic grant to another recipe.


## Repository verification

Ruff and full configured mypy pass (255source files);32focused checks pass.
Full local macOS suite:989passed/25skipped, with only the two previously
recorded platform limitations (sandbox-deniedps and server-path symlink
assertion). The process test passes outside the sandbox. Linux CI remains the
merge gate. Implementation, native color/bit accounting and raw artifact
reconstructions were reviewed; passing unit tests alone was not treated as
scientific evidence. Archive SHA and all49member identities were verified
on the destination, along with0444protection of archive and large payloads.
