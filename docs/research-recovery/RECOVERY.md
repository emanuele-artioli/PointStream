# Retention and verification instructions

The committed dossier is the reading copy. The recovered Git histories are retained outside Git at:

`gpu3:/home/itec/emanuele/pointstream-data/audits/research-recovery-20260930-01a0f179`

The final small selected-source archive is retained locally outside the repository at:

`/Users/manu/.codex/visualizations/2026/09/30/01a0f179-134b-7892-a32c-ddee662b0988/recovery-artifacts/selected-source-recovery-20260930.tar.gz`

[Retention receipts](records/retention.json) provide filenames, locations, sizes and SHA-256 values. Shared NFS aliases do not create independent copies. The earlier three-domain archive is a superseded draft; use the committed branch and final selected-source archive.

To verify documentation alone, run `python3 docs/research-recovery/verify.py` from a checkout. To verify source identities too, copy the three final recovery files named in the receipts into a separate empty recovery directory. Check their SHA-256 values before extraction. The archive contains relative paths and small retained sources; it excludes media, checkpoints, unrelated chats and hidden reasoning. Restore the recovered Git mirrors alongside those paths:

```sh
mkdir -p recovered/source
tar -xzf selected-source-recovery-20260930.tar.gz -C recovered
git clone --mirror pointstream-recovery-complete-20260930.bundle recovered/source/code.git
git clone --mirror pointstream-paper-recovery-20260930.bundle recovered/source/paper.git
python3 docs/research-recovery/verify.py --source-root recovered
```

The validator reconciles the recovered commit/PR union and manuscript union, checks dispositions and local links, and verifies retained file hashes. The manifest embedded in the archive covers every selected source file. Committed copies of small reports are checked against their source identities. A hash match verifies retention, not scientific validity; detailed cards preserve the latter's missing fields and corrections.

The recovered code mirror's bare `main` is the GPU3 historical state. Use `refs/heads/github/main` for the public main pinned at collection. Do not sync any source checkout from the recovery mirror or replay dirty captures; these are evidence. CPU reanalysis scripts are retained under `workers/coordinator/` and `rawworkers/experiments/` with their source captures. They reproduce stored arithmetic and interpolation, and do not re-score pixels, run codecs or launch GPU work.

The demo is actively evolving. Findings refer to the hashes in the initial/addendum records; [close-out state](records/final-source-state.json) records one changed preparation script and three later pending sources. Later implementations need their own evidence record.
