# Finish staging the VISOR zip in RAM, then verify it and list it.
Z=/dev/shm/ps-dl/inspect/visor/zips/2v6cgv1x04ol22qp9rm9x2j6a7.zip
SRC=/home/itec/emanuele/Datasets/EPIC-KITCHENS-VISOR/2v6cgv1x04ol22qp9rm9x2j6a7.zip
while pgrep -u emanuele -x cp >/dev/null; do sleep 10; done
[ "$(stat -c %s $Z)" = "$(stat -c %s $SRC)" ] || { echo "SIZE-MISMATCH"; exit 1; }
sha256sum $Z | awk '{print "SHA256", $1}'
unzip -tq $Z && echo "ZIP-CRC-OK" || echo "ZIP-CRC-FAIL"
unzip -l $Z > /dev/shm/ps-dl/inspect/visor/listing.txt
echo DONE-VISOR-STAGE
