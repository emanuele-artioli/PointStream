D=/home/itec/emanuele/Datasets; S=/dev/shm/ps-dl/inspect; P=/tmp/ps-dl/venv/bin/python; T=/tmp/ps-dl/tools/dataset_manifest.py
mkdir -p /tmp/ps-dl/facts
for pair in openttgames:OpenTTGames tracknet:TrackNet egohos:EgoHOS racketvision:RacketVision; do
  n=${pair%%:*}; d=${pair#*:}
  echo "== $n"; $P $T inspect $n --root $D/$d --scratch $S --out /tmp/ps-dl/facts/$n.json || echo "FAIL-INSPECT $n"
done
echo DONE-INSPECT4
