# Rewrite the first four manifests so smoke overlays live in manifests/smoke (args: tool git revision).
REV=$1
P=/tmp/ps-dl/venv/bin/python; T=/tmp/ps-dl/tools/dataset_manifest.py; D=/home/itec/emanuele/Datasets; M=$D/manifests
OLD=/dev/shm/ps-dl/superseded-manifests; mkdir -p $OLD
export PS_TOOL_REVISION=$REV
cd /tmp
for pair in openttgames:OpenTTGames tracknet:TrackNet egohos:EgoHOS hot3d:HOT3D; do
  n=${pair%%:*}; d=${pair#*:}
  mv $M/$d.json $OLD/$d.json
  $P $T manifest $n --root $D/$d --facts /tmp/ps-dl/facts/$n.json --known-hashes $OLD/$d.json --out $M/$d.json || echo "FAIL $n"
done
echo DONE-MANIFESTS1B
