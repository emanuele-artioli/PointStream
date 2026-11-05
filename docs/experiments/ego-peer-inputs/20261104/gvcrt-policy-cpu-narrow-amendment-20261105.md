# GVC CPU oracle: narrow source-package amendment

The original CPU request `20261105T073455Z-b4be8cbf` remains unpublished after its source upload timed out; its partial archive is preserved and the original phase closes with the root-recorded 60 CPU-second external charge. This prospective amendment requests a **new**, separately reviewed phase capped at 180 CPU / 0 GPU seconds (120 campaign plus 60 external). It does not authorize replay or submission.

From unchanged frozen commit `3d38e706c5ee0184ec0a4333c18bbe0f89dab851`, the existing fleet `--snapshot-path` implementation constructs a 399,360-byte, 17-file source tar, compared with the original 17,827,840-byte full-checkout snapshot. Canonical snapshot SHA is `bde697a465aabc71e059b4b55524af89935b43b19b6a250632b600fc148436a8`; extracted source identity is `520786b6402f0cb24ce66aede4383b170d458ad7e9a997e1c46602e74be9110c`. The [manifest and local import proof](gvcrt-policy-cpu-narrow-source-proof-20261105.json) preserve every member SHA. All three policy/worker/supervisor modules are byte-identical to the frozen source. This changes only which tracked files are packaged; it does not change policy, numerical code, uploader, manager or old failed evidence.

A Python isolated-mode check imported the actual oracle, candidate worker, peer supervisor, entropy helper and fleet manager from the extracted package only, with no workspace fallback, Torch or NumPy. It exercised the entrypoint parser, validated the existing draft fleet spec and verified the validator rejects a missing publication. `experiments.generative` is a namespace package at this revision: it has no `__init__.py`. These are import checks only, not Linux isolation or model evidence. The existing CLI automatically adds `experiments/__init__.py`, `experiments/jobs` and `src/__init__.py` to a narrowed snapshot; no tracked `src/contracts` directory exists in this frozen checkout.

The proposed submission would use the existing draft spec after replacing its expired deadline, the same frozen source checkout and exactly these explicit options:

```text
--snapshot-path experiments/generative/gvcrt_policy_cpu_oracle.py
--snapshot-path experiments/generative/gvcrt_receiver_worker_w8leaves.py
--snapshot-path experiments/generative/gvcrt_peer_job_w8leaves.py
--snapshot-path experiments/visor/__init__.py
--snapshot-path experiments/visor/gvcrt_entropy_pilot.py
```

Before dispatch, root must approve the selected package, reserve the new phase, verify the canonical snapshot and source SHA against the eventual submission, and repeat host/job/doctor/path checks and set a fresh bounded deadline. Retain the current host-contention caveat; this is an actual-policy-helper correctness oracle, not a performance qualification. No new uploader, manager, GPU work or automatic replay is included.
