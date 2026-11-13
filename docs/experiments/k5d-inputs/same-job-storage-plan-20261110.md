# Same-job receiver storage review — 2026-11-10

Review only; no run, admission, throughput measurement or cost-fit claim. Applies to the reviewed `0bb51211` source and six 1280×720 uint8 points. The 292/586 gates and old FFE split remain held. Paths and exact archive membership must be checked against the eventual frozen spec and actual outputs.

## Raw byte arithmetic

Let P = frames × 1280 × 720. Per point, YUV420 is 1.5P and rendered RGB is 3P bytes. Six receiver pairs are R = 27P; six RGB arrays are A = 18P, excluding `.npy` headers. The original receiver work and retained checkpoint are two separate copies of R. Sizes below are bytes, not GiB; files, headers and archive padding add bytes.

| Frames | Six YUV | Six rendered RGB | R per receiver copy | Six RGB arrays A |
|---:|---:|---:|---:|---:|
| 16 | 132,710,400 | 265,420,800 | 398,131,200 | 265,420,800 |
| 48 | 398,131,200 | 796,262,400 | 1,194,393,600 | 796,262,400 |
| 292 | 2,421,964,800 | 4,843,929,600 | 7,265,894,400 | 4,843,929,600 |
| 586 | 4,860,518,400 | 9,721,036,800 | 14,581,555,200 | 9,721,036,800 |

## Paths, lifetimes and I/O

- `k5d_encoder.run` creates source YUV, `converted.npy`, six decoder `decoded.npy` arrays and the six actual receiver work directories under local `PS_SCRATCH_DIR`; native encoded and wire files also exist. `retain` copies each actual receiver's YUV, RGB and receipt into `publish/<point>/receiver-evidence`, flushes/fsyncs each and verifies hashes. The work copies remain until scratch release. Sealing re-reads content for hashes; it is additional local I/O, not another persistent payload. In-memory source, fits, models, conversion and OS cache are RAM costs separate from disk; their peak is unknown.
- Normal fleet publication archives `scratch/publish` directly to NFS `published.tar`, then hashes that NFS archive and releases successful local scratch. Thus the archive contains one R, not the work copy. Tar headers, padding and other checkpoint files make its exact size larger than R. This path does **not** produce a nested `checkpoint.tar`: fleet's separate `checkpoint.tar` is salvage-only, made from `scratch/checkpoint` if nonempty. Salvage can also publish `partial.tar`; preserve both and failed local scratch. A spec placing another archive inside `publish` would change this conclusion and must be audited.
- Scoring stages the NFS `published.tar`: cache miss reads NFS while copying and hashing to a local cache archive; cache hit hashes that archive locally. Extraction adds a separate local tree (read-only) and retains the archive. It then verifies checkpoint manifest/content and uses retained YUV/RGB directly; no later native decode in the same-job branch. `rgb_frames` creates one `native-rgb.npy` per point under scratch, and the six files remain through assembly. Their total payload is A plus headers. Native inputs and other model/environment cache files are additional; cache hits still occupy host disk but are already reflected in free space, not necessarily new allocations.
- `assemble` verifies the checkpoint again and `copytree` skips every named `receiver-evidence` directory and the marker when the marker exists. Therefore its assembly copy and normal score `published.tar` omit R; it still copies other checkpoint contents. This is a name-based ignore, not a general bulk filter. Other large files, tar padding, metadata, validation/extraction and any explicitly nested archive remain to be measured. `publish_scratch` hashes its resulting NFS tar; validator paths can add further archive reads/extractions depending on the selected spec.
- A physical disk local root uses the fleet's **50 GiB free reserve** in addition to declared `local_storage_gib`; the declaration is new available space, not the total filesystem size. A RAM-backed root instead uses the separate dynamic memory margin and competes with process RAM. No automatic deletion of existing cached assets or results is assumed.

## Lower bounds, not declarations

Prepare necessarily retains 2R + A + 4.5P bytes for receiver copies, six `decoded.npy` payloads, source YUV and `converted.npy`, respectively; this excludes decoder YUV, source/encoded/wire files, headers, caches and other work. Score needs at least 2R + A for staged archive + extracted checkpoint + six RGB arrays, excluding archive overhead and all other inputs. These are per-stage bounds and do not add prepare scratch to later score scratch after successful release. If cache coexistence or concurrent jobs changes, recalculate.

| Frames | Prepare lower bound bytes | Score lower bound bytes | Necessary declared GiB, prepare / score |
|---:|---:|---:|---:|
| 16 | 1,128,038,400 | 1,061,683,200 | at least 2 / 1 |
| 48 | 3,384,115,200 | 3,185,049,600 | at least 4 / 3 |
| 292 | 20,586,700,800 | 19,375,718,400 | at least 20 / 19 |
| 586 | 41,314,406,400 | 38,884,147,200 | at least 39 / 37 |

Actual declaration must exceed these floors by the measured remaining high-water, any new input/environment cache and concurrent claims; no sufficient value can yet be established. In particular, 32 GiB is below the 586 prepare and score floors, before the separate 50 GiB disk reserve. NFS needs space for each retained successful publication and any partial/salvage archives, plus existing evidence; no deletion or overwrite is assumed.

## Smallest useful next measurement

Prefer waiting for the actual independently accepted 16/48 same-job outputs: inspect real file sizes and use the existing staging, seal/verify, archive and hash paths to measure local peak, NFS publication/read, CPU and wall by phase on an independently bounded CPU allocation. A 16-frame infrastructure-only generated-byte test of these existing paths is an alternative only if root separately finds it useful and approves a concrete ceiling; generated bytes establish transport mechanics, not codec speed, GPU costs, compression, quality or a 292/586 bound. Measure 48-frame scaling before extrapolation, track warm/cold cache and shared NFS contention, and remeasure on the actual host. The prior 3,350-second pair estimate and 32 GiB declaration are not qualified by this worksheet.
