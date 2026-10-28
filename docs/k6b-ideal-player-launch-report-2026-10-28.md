# Ideal-player oracle: launch failure, no scientific result

Historical first attempt. The separately approved repaired attempt also stopped,
at its timing gate; read the [final closure](k6b-ideal-player-report-2026-10-28.md).

The user approved the [finite appearance bound](cards/k6b-ideal-player-bound.md)
on2026-10-28. Exactly one request was submitted:
`20261028T071835Z-2964e9ef`. It failed before the campaign and before the smoke.
No LPIPS/DISTS arrays were produced, no cost gate passed, and full evaluation
never launched. This establishes neither headroom nor a negative appearance
bound. The earlier actual-player recipe's scientific failure is unchanged.

## Cause and repair

The selected34-file snapshot contained the oracle and its metric dependencies,
but omitted `experiments.jobs.inbox`. The native supervisor launches that module
from the workload snapshot for both campaign and salvage. Its recorded command
log says `No module named experiments.jobs.inbox`; the source-only metric import
preflight therefore checked an incomplete execution path. This was an agent
snapshot-selection error, not a model failure, memory failure or measured cost
failure. Foreign GPU activity was reported under the approved continue policy;
it did not cause this import failure and its process was left untouched.

The repair makes `ps-fleet submit --snapshot-path ...` automatically include
the complete campaign runtime (`experiments/jobs`, namespace initializers and
`src/contracts` when present). Default whole-checkout snapshots retain their
existing behavior. A regression test checks the selected submission path.
The repaired40-file snapshot is checked on both eligible hosts with CUDA hidden:
native campaign and supervisor `--help`, plus packed-runtime oracle `--help`.
Both hosts passed all three commands:3.684005s gpu2 /5.896582s gpu3,
archiveSHA33205d53c83a89045f861bb648d9a9be41f5e14a319a47684924ff94612854bb.
Those CPU checks do not qualify GPU metrics or authorize another request.
Lint and mypy pass; the complete local suite has970passed/24skipped and only
the previously documented macOS protected-symlink assertion fails. The process
state test passes outside the sandbox. PR262 merged with Linux CI passing; the launcher repair is checked in a
separate follow-up PR.

## Frozen request and evidence

- Code: `2e84c530512616b109b13a47327e53c42b63f0e3`; selected archive
  `4aba7d36eff2532e490cc79bb50a17ac03145e86011be6225808b31f40c1c935`.
- Spec: `84b59f64180c3dd91302d88e17b2c8e217dcdd67f624fec6f9a1e684a7c88dca`;
  deadline2026-10-28T07:57:41.004164+00:00; smoke200/full550/validator100,
  total1100GPU-s;8000MiB allocation and8192MiB sharing margin.
- gpu3 A6000 index0 UUID `GPU-6f70a27e-4171-22d7-3ffd-647e7f186a49`.
- New read-only input pack:
  `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-ideal-player-input-20261028/input.tar`,
  SHA256 `468250a72a1ce764e9451fc7b4cd51bd29372d3e688b832e1e9c1f002e11bc6c`,
  594739200bytes. Its manifest is
  `85f6ae06d641e8be73804d9d6428ee8f7b386cc64326082fb5f0e9b0f7f04568`.
- Packed environment `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`;
  frozen LPIPS/DISTS weights remain those of the preceding audited study.
- Request evidence remains under
  `/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261028T071835Z-2964e9ef/`.
  No original data, source, failed request or previous evidence was overwritten.

CPU preparation verified all80 original source/base/actual frame identities,
receiver-mask identities, deterministic overlays and unchanged pixels outside
support. It took108.997904wall/116.270602CPU-s with CUDA hidden. Those inputs
remain reusable after new authority; they are not quality evidence.

## Resource closure and next decision

Native run interval1793171999.384285–1793171999.6753113 was0.291026s;
canonical terminal reconciliation at1793172058.468792 was59.084507s after
run start. Charge **60GPU-s conservatively**, despite no metric inference.
Cumulative study charge is **2410/3600GPU-s** (0.669444/1GPU-hour), leaving
**1190GPU-s**. No scientific outcome is accepted and no retry has been submitted.
Both terminal and foreign-workload events were delivered and acknowledged.
The bounded wait ended; no recurring model monitor was created.

The accepted card permits exactly one request, now consumed. A fresh request
requires an explicit resource-contract decision. The concrete repair can be
reviewed without funding another renderer or changing the scientific question.
A proposed recovery retains the same1100s cap, controls, smoke/full gates and
stop conditions, leaving90s unallocated. See the
[separate recovery proposal](cards/k6b-ideal-launch-recovery.md).

Failure evidence is published read-only at
`/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-ideal-launch-20261028-2964e9ef/evidence.tar`,
SHA256 `c24e904e6cdbc6226f6f51891dae192a30b9a723b2993be2fb7d0e4866af023a`,
1116160bytes/20files. Whole-file and all member hashes verified after transfer.
Includes native state/log/spec/ready, CPU input manifest and exact preparer,
repaired source archive, both host launch checks and event acknowledgments.
