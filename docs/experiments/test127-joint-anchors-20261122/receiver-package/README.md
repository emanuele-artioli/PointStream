# Sydney receiver source preflight

The two paid Sydney player streams are bound to a common appearance-and-pose carrier from sender job `20261122T134208Z-0e12634b`. The first generator attempts failed while importing the court-analysis helper `experiments.visor.b1`. Their source selection included only the tennis and fleet experiment directories. The complete tracked `experiments`, `src`, `scripts` and `tools` directories supply the receiver and fleet dependencies. The isolated receiver projects only authenticated Python code from this archive before decoding.

On source revision `b44691bba163f9778b93dfce29f2725ba5802b5b`, the corrected 9,349,120-byte fleet archive has SHA-256 `32f19b8a4dfe3aaf1c209f8e4f202bd7bf8a35ec12ea7128972cb7fede242517`. Two independent builds gave the same archive and player conditioning hashes. The original campus archive reproduced the missing import. The corrected archive and the receiver's separate 554-file Python tree each loaded the real generator entry point and the paired 24-frame entry point. The checks decoded the original carrier into the reference, poses, boxes and timestamps for both roles at 60 fps. The 24- and 64-tick arrays were identical to the prefixes of their full 240-tick arrays. The original sender digest, registered role specs, fixed producer implementations, generator-environment binding and local interpreter versions are checked or recorded. [Validation receipt](preflight-b44691bb.json) · [Original archive and extracted-tree identities](original-source.json)

This is a CPU-side source and conditioning check. CUDA generation, the Linux source-denial runtime, full-stage validation, and composition still require their respective fleet stages. The matched Sydney 24-frame pilot has its own spec and timing gate. The independent role specs retain their 64-tick pilots and 240-tick full stages.

## Rebuild on a clean checkout

Use the project Python environment with NumPy and OpenCV. Set `source_checkout` to a clean checkout at the recorded revision, `sender_full` to the verified sender job's published full-stage directory, `original_source` to `source.tar` from near generator job `20261122T134804Z-ff2bc372`, and `output_directory` to a **new directory outside the source checkout**. Run the following recorder from the working repository.

```sh
python docs/experiments/test127-joint-anchors-20261122/receiver-package/prepare.py \
  --source-tree "$source_checkout" \
  --carrier "$sender_full/crf70-players.pscp" \
  --manifest "$sender_full/crf70-players.manifest" \
  --failed-archive "$original_source" \
  --pair --output "$output_directory"
```

The recorder writes `source.tar` and `preflight.json`. The isolated checks disable automatic site-directory initialization, including editable installs which can otherwise import missing code from a second checkout. The host runtime supplies NumPy and OpenCV. The carrier and manifest are the two sender artifacts used for the local check.

For a subsequently approved full-replay window, adding `--deadline <UTC-deadline>` writes both validated fleet specs to the output directory. Their sole change from the registered role specifications is the deadline. The executable fleet snapshot selection is `--source-worktree "$source_checkout" --snapshot-path experiments --snapshot-path src --snapshot-path scripts --snapshot-path tools` in this order. Compare the submitted archive digest against the receipt for that revision. Preparation performs zero fleet registrations.
