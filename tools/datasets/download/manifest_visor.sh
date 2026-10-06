# VISOR manifest once all EPIC-KITCHENS videos are in (args: tool git revision).
REV=$1
P=/tmp/ps-dl/venv/bin/python; T=/tmp/ps-dl/tools/dataset_manifest.py; D=/home/itec/emanuele/Datasets; M=$D/manifests
R=$D/EPIC-KITCHENS-VISOR; K=/tmp/ps-dl/known; mkdir -p $K
until grep -q DONE-EK /tmp/ps-dl/logs/ek.log; do sleep 30; done
echo "ek OK $(grep -c '^OK' /tmp/ps-dl/logs/ek.log) FAIL $(grep -c '^FAIL' /tmp/ps-dl/logs/ek.log)"
# Videos hashed while streaming (md5 matched the EPIC-KITCHENS md5.csv).
grep '^OK .* sha256=' /tmp/ps-dl/logs/ek.log | awk '{split($4,s,"="); printf "epic_kitchens_videos/%s.MP4\t%s\tstreamed_during_download_md5_verified\n", $2, s[2]}' > $K/visor.tsv
# Videos from the first script version: md5 was verified by read-back; sha256 needs one more read.
grep -E '^OK [A-Z0-9_]+$' /tmp/ps-dl/logs/ek.log | awk '{print $2}' > $K/visor_readback.txt
echo "readback $(wc -l < $K/visor_readback.txt) videos"
(cd $R/epic_kitchens_videos && sed 's/$/.MP4/' $K/visor_readback.txt | xargs -P4 -n1 sha256sum) \
  | awk '{printf "epic_kitchens_videos/%s\t%s\tcomputed_from_stored_file_md5_verified\n", $2, $1}' >> $K/visor.tsv
sha256sum /dev/shm/ps-dl/inspect/visor/zips/2v6cgv1x04ol22qp9rm9x2j6a7.zip \
  | awk '{printf "2v6cgv1x04ol22qp9rm9x2j6a7.zip\t%s\tcomputed_from_host_local_copy_zip_crc_ok\n", $1}' >> $K/visor.tsv
cp /tmp/ps-dl/ek_videos.tsv $M/EPIC-KITCHENS-VISOR.videos_md5.tsv && chmod 0444 $M/EPIC-KITCHENS-VISOR.videos_md5.tsv
export PS_TOOL_REVISION=$REV
cd /tmp && $P $T manifest visor --root $R --facts /tmp/ps-dl/facts/visor.json --known-hashes $K/visor.tsv --out $M/EPIC-KITCHENS-VISOR.json
echo DONE-MANIFEST-VISOR
