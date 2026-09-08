# Pull Request and Direct Commit Index

This index records all 72 pull requests in the repository's history (through PR #72), fetched directly from the GitHub API. Titles are exact GitHub API titles.

---

## 1. Codec

| PR | Exact Title | State | Merge Commit |
|---|---|---|---|
| [#1](https://github.com/emanuele-artioli/PointStream/pull/1) | fix(pipeline): Refactor metadata muxing and demuxing pipeline | MERGED | [`a3e4fc0103`](https://github.com/emanuele-artioli/PointStream/commit/a3e4fc0103bb656309de1862a346464ec7d8e380) |
| [#3](https://github.com/emanuele-artioli/PointStream/pull/3) | feat(server): Enrich scene metadata with video properties | MERGED | [`07703d461e`](https://github.com/emanuele-artioli/PointStream/commit/07703d461ed3d6ffbba0dcf5998195d17f387546) |
| [#4](https://github.com/emanuele-artioli/PointStream/pull/4) | docs: Add detailed comments to configuration files | MERGED | [`9a1a316947`](https://github.com/emanuele-artioli/PointStream/commit/9a1a316947dc5dd4987610b51c167eb5a8c6aa08) |
| [#7](https://github.com/emanuele-artioli/PointStream/pull/7) | Fix duplicate sections in config.ini | MERGED | [`48c321ba47`](https://github.com/emanuele-artioli/PointStream/commit/48c321ba474606390799b1913f0e3cd77a65937b) |
| [#8](https://github.com/emanuele-artioli/PointStream/pull/8) | feat: Preserve original video FPS and resolution in reconstruction | MERGED | [`4977473cd3`](https://github.com/emanuele-artioli/PointStream/commit/4977473cd37184fb8d29c83e387b8df2f1971860) |
| [#9](https://github.com/emanuele-artioli/PointStream/pull/9) | feat: Implement segmentation parameter logic | MERGED | [`ebc7ad07a5`](https://github.com/emanuele-artioli/PointStream/commit/ebc7ad07a574af18487a1c64ff484ef9262f51ba) |
| [#10](https://github.com/emanuele-artioli/PointStream/pull/10) | feat: Implement temporal smoothing for homography matrices | MERGED | [`93b57bf625`](https://github.com/emanuele-artioli/PointStream/commit/93b57bf625deb7ce257f7c94bbd0855b08c54ac4) |
| [#13](https://github.com/emanuele-artioli/PointStream/pull/13) | refactor: update GenAI compositor to use residual deltas and implemen… | MERGED | [`669345ca42`](https://github.com/emanuele-artioli/PointStream/commit/669345ca427246d659c0e0e8368072a78b18a37a) |
| [#22](https://github.com/emanuele-artioli/PointStream/pull/22) | BP23: a tier config runs end to end and is scored (P0 item 1) | MERGED | [`7cf8e896e5`](https://github.com/emanuele-artioli/PointStream/commit/7cf8e896e5675c65894202488ee1f07805b3c234) |
| [#26](https://github.com/emanuele-artioli/PointStream/pull/26) | fix(bp26): wire ablation-lattice config names into the runner | MERGED | [`ac0ca6df84`](https://github.com/emanuele-artioli/PointStream/commit/ac0ca6df8486bb5bac645247db73cfad655cabae) |
| [#29](https://github.com/emanuele-artioli/PointStream/pull/29) | BP24: a real rate axis — coded plate, coded residual, and a ledger that refuses to lie | MERGED | [`824adc2cec`](https://github.com/emanuele-artioli/PointStream/commit/824adc2cec1e7923bef64c1df201e07497557070) |
| [#34](https://github.com/emanuele-artioli/PointStream/pull/34) | BP29 §1.1: what the plate's codec knob costs, and what it buys | MERGED | [`00ee988d34`](https://github.com/emanuele-artioli/PointStream/commit/00ee988d341a8a0926edd7c322f2ef27dcb48396) |
| [#35](https://github.com/emanuele-artioli/PointStream/pull/35) | fix(paths): weights live under the data root, not the checkout | MERGED | [`420c3bec4a`](https://github.com/emanuele-artioli/PointStream/commit/420c3bec4a2be281475b15605ccdc598b9645a7a) |
| [#36](https://github.com/emanuele-artioli/PointStream/pull/36) | feat(background): av1 and vvc intra sidecars for the plate | MERGED | [`09c92880c4`](https://github.com/emanuele-artioli/PointStream/commit/09c92880c453d032249e2669b36e637e7d315586) |
| [#38](https://github.com/emanuele-artioli/PointStream/pull/38) | Wire build_plate into the runner: the panorama the runner never called | MERGED | [`2c9622871f`](https://github.com/emanuele-artioli/PointStream/commit/2c9622871f785c550cb4ae616ea29f574baece65) |
| [#39](https://github.com/emanuele-artioli/PointStream/pull/39) | BP30: the background as a stream — it amortises, reference selection does not | MERGED | [`2aa5d8461f`](https://github.com/emanuele-artioli/PointStream/commit/2aa5d8461f8f8bf89797435e0a5ebf6e1a9cbd7b) |
| [#41](https://github.com/emanuele-artioli/PointStream/pull/41) | feat(background): wire the cross-scene stream into the runner as `panorama-stream` | MERGED | [`65c7540002`](https://github.com/emanuele-artioli/PointStream/commit/65c75400029b5dd94fd6b4c374d78d399f0b7d62) |
| [#45](https://github.com/emanuele-artioli/PointStream/pull/45) | BP31: the cross-scene stream had never run, and the plate lever is 1.45x where the ladder operates | MERGED | [`ecebd9b97d`](https://github.com/emanuele-artioli/PointStream/commit/ecebd9b97df362c8447e73c3dda6149cd7601625) |
| [#48](https://github.com/emanuele-artioli/PointStream/pull/48) | Retract BP43's client-side plate — it was circular | MERGED | [`a66995c9a6`](https://github.com/emanuele-artioli/PointStream/commit/a66995c9a6627b9e4f94dfa79fed89ee39afeab8) |
| [#50](https://github.com/emanuele-artioli/PointStream/pull/50) | BP44: offline canonical canvas for long compatible scenes | MERGED | [`bb5d17445f`](https://github.com/emanuele-artioli/PointStream/commit/bb5d17445ffa786c20384aab02a346b3f42840bd) |
| [#52](https://github.com/emanuele-artioli/PointStream/pull/52) | Integrate BP44–BP46 and validate low-rate search plumbing | MERGED | [`68a03dc542`](https://github.com/emanuele-artioli/PointStream/commit/68a03dc5429bcc388e1213a4377ec357c9815b7f) |
| [#61](https://github.com/emanuele-artioli/PointStream/pull/61) | BP53: bounded background transport scaling | MERGED | [`f63019c15f`](https://github.com/emanuele-artioli/PointStream/commit/f63019c15fa8834119e42c3e8f04c07388c7b6dd) |
| [#64](https://github.com/emanuele-artioli/PointStream/pull/64) | BP56: bounded background encoder-effort pilot | MERGED | [`60a18f725e`](https://github.com/emanuele-artioli/PointStream/commit/60a18f725e2673a2a191acf559739cb15cfeaa81) |

---

## 2. Evaluation

| PR | Exact Title | State | Merge Commit |
|---|---|---|---|
| [#14](https://github.com/emanuele-artioli/PointStream/pull/14) | Phase 0: make evaluation run the decoder's path, then rank on residual bytes | MERGED | [`67c5bec213`](https://github.com/emanuele-artioli/PointStream/commit/67c5bec2139501cb34759b75f55db2e39bbfd93f) |
| [#18](https://github.com/emanuele-artioli/PointStream/pull/18) | BP21: widen headroom selection, plate-NaN fill, common-QP helpers | MERGED | [`52a250393f`](https://github.com/emanuele-artioli/PointStream/commit/52a250393ffd1c498138f561492b77b1cff9920f) |
| [#24](https://github.com/emanuele-artioli/PointStream/pull/24) | BP27: pin VMAF/LPIPS instrument limits as invariants | MERGED | [`0d1ade6962`](https://github.com/emanuele-artioli/PointStream/commit/0d1ade696225762b579ade1b4bffe13b0580de24) |
| [#31](https://github.com/emanuele-artioli/PointStream/pull/31) | BP24: run the paired ladder — PointStream loses to the codec it is built on | MERGED | [`764e9d9b29`](https://github.com/emanuele-artioli/PointStream/commit/764e9d9b2926ea9699409fa7b14bcd80dd14cebc) |
| [#37](https://github.com/emanuele-artioli/PointStream/pull/37) | BP29 §2: no low-rate crossover, and the anchor has a 44.44 dB ceiling | MERGED | [`a00c6fb0db`](https://github.com/emanuele-artioli/PointStream/commit/a00c6fb0dbe902ca0f9802be11e03a789d957438) |
| [#51](https://github.com/emanuele-artioli/PointStream/pull/51) | BP45 M1: metric-axis typing and AV1/VVC ultra-low-rate floors | MERGED | [`ecaf8de097`](https://github.com/emanuele-artioli/PointStream/commit/ecaf8de0970296f10d2dd9977e104ca6d7a02de7) |
| [#55](https://github.com/emanuele-artioli/PointStream/pull/55) | Allow legal neighbour QPs and same-dir reference resume | MERGED | [`35a57163bd`](https://github.com/emanuele-artioli/PointStream/commit/35a57163bd80fb259f68aae2241e495b27c6cf6b) |
| [#58](https://github.com/emanuele-artioli/PointStream/pull/58) | Run bounded BP52 background CRF search | MERGED | [`bcb3a63b37`](https://github.com/emanuele-artioli/PointStream/commit/bcb3a63b371b52e352bf67bc1aa243209735f05a) |
| [#65](https://github.com/emanuele-artioli/PointStream/pull/65) | feat(gate-a): implement Gates 0-2 infrastructure, disjoint timing, and rate ladder | MERGED | [`91b33e623f`](https://github.com/emanuele-artioli/PointStream/commit/91b33e623fba2b71fe7899260a8f106be22214db) |
| [#66](https://github.com/emanuele-artioli/PointStream/pull/66) | Fix Gate A prelaunch contract enforcement | MERGED | [`606cf53893`](https://github.com/emanuele-artioli/PointStream/commit/606cf5389344449a1b1572dfcc3440cd9ae23aae) |
| [#69](https://github.com/emanuele-artioli/PointStream/pull/69) | docs: report and adjudicate Gate A 48-frame native run | MERGED | [`648325b113`](https://github.com/emanuele-artioli/PointStream/commit/648325b11333c602e83944bd32dad48cac159cf4) |
| [#70](https://github.com/emanuele-artioli/PointStream/pull/70) | docs(plan): add strategic plan for competitive operating regime and anchor analysis | MERGED | [`1713883455`](https://github.com/emanuele-artioli/PointStream/commit/1713883455b42df0600a6257810ff8d89179a7a6) |
| [#71](https://github.com/emanuele-artioli/PointStream/pull/71) | docs(plan): revise competitive regime plan with empirical VVC background benchmark and fast evaluation | MERGED | [`3267a91861`](https://github.com/emanuele-artioli/PointStream/commit/3267a91861fbe5e99f50014d1655f1f04cf04b2c) |
| [#72](https://github.com/emanuele-artioli/PointStream/pull/72) | docs(plan): refine Gate A competitive regime plan | MERGED | [`bc09184d87`](https://github.com/emanuele-artioli/PointStream/commit/bc09184d8707529f924a41d0dccc2b6f20e88d3e) |

---

## 3. Data

| PR | Exact Title | State | Merge Commit |
|---|---|---|---|
| [#40](https://github.com/emanuele-artioli/PointStream/pull/40) | fix(probe-set): repair the view's links, and stop resolving the manifest at the cwd | MERGED | [`e70f273836`](https://github.com/emanuele-artioli/PointStream/commit/e70f273836a0dd7de9915bb2b4b59dcff7702842) |
| [#56](https://github.com/emanuele-artioli/PointStream/pull/56) | docs(bp46): add match-by-match prior use and contamination audit | MERGED | [`09727a4f16`](https://github.com/emanuele-artioli/PointStream/commit/09727a4f161ed37b811eb4067226579ca5c16fb6) |
| [#57](https://github.com/emanuele-artioli/PointStream/pull/57) | BP51: Confirmation Split & Contamination Audit | MERGED | [`77f30ecbe0`](https://github.com/emanuele-artioli/PointStream/commit/77f30ecbe0fdc1a12a829ee26169aba43427c9c4) |
| [#60](https://github.com/emanuele-artioli/PointStream/pull/60) | BP54: Fresh confirmation source shortlist and candidate manifest | MERGED | [`f6f4f72cd9`](https://github.com/emanuele-artioli/PointStream/commit/f6f4f72cd96ab07d1fd23fe573a9ec65f5113c6b) |
| [#63](https://github.com/emanuele-artioli/PointStream/pull/63) | BP57: confirmation acquisition pilot report and candidate manifest | MERGED | [`5ca87b2ee4`](https://github.com/emanuele-artioli/PointStream/commit/5ca87b2ee48cc8808bada2b72d42daf6b1849a48) |

---

## 4. Generation

| PR | Exact Title | State | Merge Commit |
|---|---|---|---|
| [#2](https://github.com/emanuele-artioli/PointStream/pull/2) | Implement reference-based training and add orchestration script | MERGED | [`6dd763a4fa`](https://github.com/emanuele-artioli/PointStream/commit/6dd763a4faa3c640ec9c82006c788ec651e7ad39) |
| [#5](https://github.com/emanuele-artioli/PointStream/pull/5) | refactor(pipeline): Implement vector-based input for generative models | CLOSED | *(not merged)* |
| [#6](https://github.com/emanuele-artioli/PointStream/pull/6) | Implement reference-based training and add orchestration script | MERGED | [`d1fb82cba6`](https://github.com/emanuele-artioli/PointStream/commit/d1fb82cba6cbcbace8ce3a1bb522c0d46bb1ddd3) |
| [#17](https://github.com/emanuele-artioli/PointStream/pull/17) | fix(decoder): restore build_genai_strategy and _resolve_strategy_weight | MERGED | [`178ec8c9a8`](https://github.com/emanuele-artioli/PointStream/commit/178ec8c9a83296e543b1c46d2910642801cdf0c1) |
| [#20](https://github.com/emanuele-artioli/PointStream/pull/20) | feat(training): stop on the coding-task bar (BP14) | MERGED | [`fd36b4b338`](https://github.com/emanuele-artioli/PointStream/commit/fd36b4b338778c90fd33959da6e10507463b667d) |
| [#27](https://github.com/emanuele-artioli/PointStream/pull/27) | BP25: re-score IP-Adapter and close P0 item 5 | MERGED | [`8d9a4b1669`](https://github.com/emanuele-artioli/PointStream/commit/8d9a4b166981c1ad4d41707459445c7f62236db5) |
| [#28](https://github.com/emanuele-artioli/PointStream/pull/28) | plans: engine roster, BP28 brief, and the prompts Cursor and Antigravity need | MERGED | [`66da545dcf`](https://github.com/emanuele-artioli/PointStream/commit/66da545dcf8a9d3f2a9a895a871fa82654f3a66b) |

---

## 5. Paper

| PR | Exact Title | State | Merge Commit |
|---|---|---|---|
| [#46](https://github.com/emanuele-artioli/PointStream/pull/46) | Roadmap for the remaining work, the three papers written in advance, and two gates that were not closing | MERGED | [`29868945e0`](https://github.com/emanuele-artioli/PointStream/commit/29868945e02456cbe954a0463a85a010792ad5dc) |
| [#47](https://github.com/emanuele-artioli/PointStream/pull/47) | The search is the method, speed is a dimension, and the plate has two untried levers | MERGED | [`7aa9a8c4d0`](https://github.com/emanuele-artioli/PointStream/commit/7aa9a8c4d0ff3a76eff2cf2ffce21d770e604190) |
| [#49](https://github.com/emanuele-artioli/PointStream/pull/49) | docs: submission roadmap and fair codec protocol | MERGED | [`bb5d17445f`](https://github.com/emanuele-artioli/PointStream/commit/bb5d17445ffa786c20384aab02a346b3f42840bd) |

---

## 6. Infrastructure

| PR | Exact Title | State | Merge Commit |
|---|---|---|---|
| [#11](https://github.com/emanuele-artioli/PointStream/pull/11) | March26version | MERGED | [`32c4c1cc27`](https://github.com/emanuele-artioli/PointStream/commit/32c4c1cc278fa7eaceae99e1fbdd6ede4cd5f949) |
| [#12](https://github.com/emanuele-artioli/PointStream/pull/12) | Backup/main before fix 20260422 115306 | MERGED | [`dda976aa18`](https://github.com/emanuele-artioli/PointStream/commit/dda976aa180c39790b502256444bbe65bf09ccc3) |
| [#15](https://github.com/emanuele-artioli/PointStream/pull/15) | Agent-native repo setup: generated rules, run invariants, architecture map | MERGED | [`44226cf5e4`](https://github.com/emanuele-artioli/PointStream/commit/44226cf5e4b90f3610ddbca41fa69d6172475469) |
| [#16](https://github.com/emanuele-artioli/PointStream/pull/16) | Centralize shared hooks (session-status, guard-rm, paper-sync-reminde… | MERGED | [`8c3a720391`](https://github.com/emanuele-artioli/PointStream/commit/8c3a7203913f3856038baabb2e17baa2cd0d9fd8) |
| [#19](https://github.com/emanuele-artioli/PointStream/pull/19) | refactor(bp15): cull leftover decoder/shared (wave 3 stream B) | MERGED | [`4afe36808a`](https://github.com/emanuele-artioli/PointStream/commit/4afe36808ac920edac3cb40099926289e3b7413f) |
| [#21](https://github.com/emanuele-artioli/PointStream/pull/21) | fix(lint): exempt train_controlnet's path bootstrap from E402 — main is red | MERGED | [`51139a0c2a`](https://github.com/emanuele-artioli/PointStream/commit/51139a0c2af092e3d8a3bbc6dfdef633b17d609c) |
| [#23](https://github.com/emanuele-artioli/PointStream/pull/23) | plans: wave 5 — status update and three-platform schedule | MERGED | [`ca0f75af30`](https://github.com/emanuele-artioli/PointStream/commit/ca0f75af30365e554f24f8278d60caad7fbc3dd8) |
| [#25](https://github.com/emanuele-artioli/PointStream/pull/25) | BP22: finish the cull; src.shared stays condemned | MERGED | [`8262562cb8`](https://github.com/emanuele-artioli/PointStream/commit/8262562cb8096715abf46ac5c166cd771905ccbc) |
| [#30](https://github.com/emanuele-artioli/PointStream/pull/30) | plans: record BP24's rate axis, and the next-session handoff | MERGED | [`52750a639b`](https://github.com/emanuele-artioli/PointStream/commit/52750a639b475b8cb95c60f551daea2c135508ea) |
| [#32](https://github.com/emanuele-artioli/PointStream/pull/32) | BP24 follow-up: retract §17, move the data out of the tree, and two design briefs | MERGED | [`6575451e1a`](https://github.com/emanuele-artioli/PointStream/commit/6575451e1a68e06839399145f68d65b00c19042d) |
| [#33](https://github.com/emanuele-artioli/PointStream/pull/33) | wave 8: coordination docs, and the BP30 causality gate cleared | MERGED | [`fefb59f21f`](https://github.com/emanuele-artioli/PointStream/commit/fefb59f21f86cc8447e43f9e8d215d7397147367) |
| [#42](https://github.com/emanuele-artioli/PointStream/pull/42) | chore(mypy): put experiments/ inside the gate, and land BP31 + the wave-8 resume note | MERGED | [`ddbef2476c`](https://github.com/emanuele-artioli/PointStream/commit/ddbef2476c7d9493ff0f68095715179ed3bc9302) |
| [#43](https://github.com/emanuele-artioli/PointStream/pull/43) | chore(mypy): widen the gate over experiments/, and the NFS/editor brief | MERGED | [`fa64c7848e`](https://github.com/emanuele-artioli/PointStream/commit/fa64c7848ef2cc93f72b2a70fc5f890ccbdf6314) |
| [#44](https://github.com/emanuele-artioli/PointStream/pull/44) | plans: widen BP31 to every plate lever, and the prompt to run it | MERGED | [`68cf1c939a`](https://github.com/emanuele-artioli/PointStream/commit/68cf1c939a0111efb9e49227236742cb269e3327) |
| [#53](https://github.com/emanuele-artioli/PointStream/pull/53) | Make scene recovery identity-safe and preserve measurement records | MERGED | [`ec581e957d`](https://github.com/emanuele-artioli/PointStream/commit/ec581e957dffaf1f4a35ec0043d799a41fda9224) |
| [#54](https://github.com/emanuele-artioli/PointStream/pull/54) | Organize completed briefs and wave reports to plans/done | MERGED | [`3ba0e0bc38`](https://github.com/emanuele-artioli/PointStream/commit/3ba0e0bc38da33605d6ef5d3b4ab42630a71e739) |
| [#59](https://github.com/emanuele-artioli/PointStream/pull/59) | Integrate BP51/BP52 findings and scope BP53/BP54 | MERGED | [`6b2f6c4be3`](https://github.com/emanuele-artioli/PointStream/commit/6b2f6c4be3e226d18779a261dc2921323866fc33) |
| [#62](https://github.com/emanuele-artioli/PointStream/pull/62) | Retire BP53/BP54 and dispatch BP56/BP57 | MERGED | [`a59934a707`](https://github.com/emanuele-artioli/PointStream/commit/a59934a70736f26e8e5d063083e0b3b7b2581142) |
| [#67](https://github.com/emanuele-artioli/PointStream/pull/67) | docs: clean up completed briefs and add Gate-A 48-frame dispatch prompt | MERGED | [`b62ab3a56d`](https://github.com/emanuele-artioli/PointStream/commit/b62ab3a56d0dc1e81c6cd26c97dd5f077e80c66d) |
| [#68](https://github.com/emanuele-artioli/PointStream/pull/68) | feat(git): add cleanup script for safely removing merged worktrees | MERGED | [`956ad3c277`](https://github.com/emanuele-artioli/PointStream/commit/956ad3c277bc58a35ee9832249a3b4aada5b26a1) |

---

## 7. Direct Commits to Main

| Commit | Description / Subject | Area |
|---|---|---|
| [`ee17a8a`](https://github.com/emanuele-artioli/PointStream/commit/ee17a8a) | Keep background recovery state free of wall-clock totals | Infrastructure / Codec |
| [`cdd3e95`](https://github.com/emanuele-artioli/PointStream/commit/cdd3e95) | BP57: manifest and report terminology, overlap, and totals corrections | Data / Evaluation |
| [`2da1dbc`](https://github.com/emanuele-artioli/PointStream/commit/2da1dbc) | docs: close BP53 BP54 and dispatch bounded encoder and acquisition pilots | Infrastructure / Data |
