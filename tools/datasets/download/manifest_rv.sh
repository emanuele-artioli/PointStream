# RacketVision manifest: tar hashes from host-local copies, plus a sidecar of member hashes (args: tool git revision).
REV=$1
P=/tmp/ps-dl/venv/bin/python; T=/tmp/ps-dl/tools/dataset_manifest.py; D=/home/itec/emanuele/Datasets; M=$D/manifests
K=/tmp/ps-dl/known; S=/dev/shm/ps-dl/inspect/racketvision/tars
(cd $S && sha256sum *.tar | awk '{printf "%s\t%s\tcomputed_from_host_local_copy\n", $2, $1}') > $K/racketvision.tsv
cp /dev/shm/ps-dl/rv_members.sha256 $M/RacketVision.members.sha256 && chmod 0444 $M/RacketVision.members.sha256
$P - <<PY
import json
f = "/tmp/ps-dl/facts/racketvision.json"
d = json.load(open(f))
d["hf_revision"] = "85157ca21faa2abca96d837dd2b963738029bcc8"
d["archive_members"] = {
    "sidecar": "$M/RacketVision.members.sha256",
    "note": "sha256 of all 33,042 files as downloaded, before packing into annotations/info/data_traj/badminton/tabletennis/tennis tars; 1,682 LFS files matched the Hugging Face LFS sha256, 0 mismatches, 0 missing",
}
d["label_density_note"] = "ball rows on 20.1% (badminton), 11.5% (table tennis), 14.3% (tennis) of frames, i.e. sampled every few frames; racket box+keypoints on 9.2%, 3.9%, 4.9%, usually one racket per labelled frame"
json.dump(d, open(f, "w"), indent=1)
PY
export PS_TOOL_REVISION=$REV
cd /tmp && $P $T manifest racketvision --root $D/RacketVision --facts /tmp/ps-dl/facts/racketvision.json --known-hashes $K/racketvision.tsv --out $M/RacketVision.json
