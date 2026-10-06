# Build manifests for OpenTTGames, TrackNet, EgoHOS and HOT3D (args: tool git revision).
REV=$1
P=/tmp/ps-dl/venv/bin/python; T=/tmp/ps-dl/tools/dataset_manifest.py; D=/home/itec/emanuele/Datasets
S=/dev/shm/ps-dl/inspect; K=/tmp/ps-dl/known; M=$D/manifests
mkdir -p $K $M
cd /tmp
$P $T inspect openttgames --root $D/OpenTTGames --scratch $S --out /tmp/ps-dl/facts/openttgames.json
# Hashes of archives already staged host-locally (same bytes as on NFS; stage() checks size).
(cd $S/openttgames/zips && sha256sum *.zip | awk '{printf "raw/%s\t%s\tcomputed_from_host_local_copy\n", $2, $1}') > $K/openttgames.tsv
(cd $S/tracknet/zips && sha256sum Dataset.zip | awk '{printf "%s\t%s\tcomputed_from_host_local_copy\n", $2, $1}') > $K/tracknet.tsv
(cd $S/egohos/zips && sha256sum data.zip | awk '{printf "%s\t%s\tcomputed_from_host_local_copy\n", $2, $1}') > $K/egohos.tsv
awk -F'\t' '$2 != "-" {printf "%s\t%s\tverified_against_hf_lfs_sha256\n", $1, $2}' /tmp/ps-dl/hot3d_files.tsv > $K/hot3d.tsv
export PS_TOOL_REVISION=$REV
$P $T manifest openttgames --root $D/OpenTTGames --facts /tmp/ps-dl/facts/openttgames.json --known-hashes $K/openttgames.tsv --out $M/OpenTTGames.json
$P $T manifest tracknet --root $D/TrackNet --facts /tmp/ps-dl/facts/tracknet.json --known-hashes $K/tracknet.tsv --out $M/TrackNet.json
$P $T manifest egohos --root $D/EgoHOS --facts /tmp/ps-dl/facts/egohos.json --known-hashes $K/egohos.tsv --out $M/EgoHOS.json
$P $T manifest hot3d --root $D/HOT3D --facts /tmp/ps-dl/facts/hot3d.json --known-hashes $K/hot3d.tsv --out $M/HOT3D.json
echo DONE-MANIFESTS1
