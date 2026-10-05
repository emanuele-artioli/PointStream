v=$1; m=$2; u=$3
o=/home/itec/emanuele/Datasets/EPIC-KITCHENS-VISOR/epic_kitchens_videos/$v.MP4
for t in 1 2 3; do
  got=$(wget -q -O - "$u" | tee "$o.part" >(sha256sum | cut -d" " -f1 > /tmp/ps-dl/sha/$v.sha256) | md5sum | cut -d" " -f1)
  sleep 1
  if [ "$got" = "$m" ]; then mv "$o.part" "$o"; echo "OK $v md5=$got sha256=$(cat /tmp/ps-dl/sha/$v.sha256)"; exit 0; fi
  echo "RETRY $v $t got=$got"
done
echo "FAIL $v"
