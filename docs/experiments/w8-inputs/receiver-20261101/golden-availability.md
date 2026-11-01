# W8 nad000 80-tick golden feasibility — local metadata inspection

The historical judge published **per-frame SHA256 identities for both native
base RGB and composed RGB at every saved tick**, plus the four per-tick LPIPS
and DISTS arrays. The original diagnostic code also published actual 80-frame
native RGB arrays, and the original SAM receiver code published actual projected
mask arrays. Those two large historical result archives are named and were
consumed by the original judge, but they are **not present in the inspected
local recovery**. Their current remote existence, exact archive bytes and
members still require bounded read-only recovery before any GPU build or run.
The original composite RGB files were not published in the inspected judge
archive; exact reconstruction from recovered components must be demonstrated
against every contemporaneous judge hash before they can act as pixel goldens.

- Locally recovered generated-crop archive
  `/private/tmp/w8-input-recovery-20261101/golden-generated.tar`, previously
  verified SHA256 `cce89567a57bd202db8dbd6a39599688bce8a52e048d53f2b38fd7fda7da4985`,
  contains the two `@match5.npz` crop arrays and sidecar metadata. Each sidecar
  records 240 frames generated and 80 retained; its recorded NPZ SHA256s are
  `dad82d637a445cc1c1b92907c3a274e485f26e33cd88a3206770f3efb71e80ff`
  and `bc5a67fb5a81578f605131151c6dd68a98612374e1564fd8e0dab5515a718fde`.
  NPZ metadata lists `t` and `crops` arrays. These are original saved crop
  outputs, not original full-frame composites.
- Original projected SAM arrays are expected in receiver job
  `20261031T173041Z-afee870c` archive SHA256
  `a60cd08a254945a7f0f180b7ba783dda1b127814ba9f4c0ac04d1333758d4ba7`.
  `k6b_receiver_sam.py` publishes per-unit `*-masks.npz` and records per-frame
  and whole-mask hashes; the judge requires saved receiver tick and output
  identities before scoring. The archive itself has not been inspected here;
  existence, names, tick coverage and hashes remain unverified now. Do not
  conflate these projected prediction masks with validator-only target masks.
- Original native RGB arrays are expected in RGB diagnostic job
  `20261031T172705Z-92d5a784` archive SHA256
  `91a1426879ec85bfaadd9e01b07086654267efeccf38d192853729ccf27b1332`.
  Its producing code writes `publish/<window>-s4-gpu.npy`, containing 80
  `(1080,1920,3)` uint8 frames, plus `diagnostic.json` with per-frame SHA256s.
  The judge verified that file's SHA256 as
  `f5795d8a30277d6624b327cffae6fd0500ab8bcb196e1248b37df5a610dab0db`.
  This is a contemporaneous recorded **NPY-file hash**, not a newly checked
  archive or pixel array. The original paid IVF is retained locally at
  `/private/tmp/w8-native-correct-layout-20261101/nad000-s4.ivf` and embedded
  in the paid packet; its SHA256 is
  `f9622b6af4cddea3d70226c0b22e12374f4c77aadf2120e54d5d21ba55bfbbb8`.
  Re-decoding IVF is not the same as retrieving the saved original RGB, and
  would additionally need the recorded native/colour path and exact hashes.
- The locally recovered original judge archive
  `/private/tmp/w9-inbox/20261031T180054Z-b9927a25/full/published.tar`, SHA256
  `bf05e2630d898a2ec68452f81407e063b0f0ad0490365c41d53a3988f5adb299`,
  contains `window.json` and `scores.npz`, not frame files. Its `ticks` cover
  0 through 237 in steps of 3, with six per-frame hashes ordered base/composite
  for scales 8, 4 and 2; indices 2 and 3 are the required scale-4 pair. For
  tick 0 they are respectively
  `61745226096def914d67b7d3714356ec3cee1a777682cb67913825a296d89e50` and
  `4798ceec4370ab853ff02aa76435fdec90151c19c7199d555e28b9782ee712f5`.
  All 80 pairs are in the archived JSON. Score NPZ arrays for full/foreground
  LPIPS and DISTS each have shape `(6,80)` with the same ordering. These
  hashes establish exact identity but alone cannot measure a per-channel ≤1
  fallback against a non-identical new frame.

Cheapest next step, subject to root's bounded recovery approval and remaining
original W8 CPU budget: inspect canonical metadata and current availability of
those exact receiver and RGB diagnostic jobs, recover their declared archives
read-only into new local paths under an explicit resource bound, and verify
archive hashes, native NPY file SHA, 80 individual base-frame hashes, original
mask coverage and recorded mask hashes. Then, using only recovered original
base frames, the two saved generated-crop arrays, saved predicted masks and the
original exact boxes/order, recompose offline with the frozen implementation
and compare all 80 resulting frame hashes to the judge JSON. Recomposition is a
new derivation, not a saved composite. Only a complete hash match makes its
pixels suitable as the original comparison reference for the card's exact or
≤1-channel fallback; saved mask pixels enable its bit-exact mask gate. The
same original score arrays support the 0.001 per-tick/mean metric gate. If any
archive, input, frame, exact reconstruction or provenance is missing, the
required pixel goldens are unresolved and the GPU witness must not proceed.
No model run, GPU execution or decoding was performed for this inventory.
