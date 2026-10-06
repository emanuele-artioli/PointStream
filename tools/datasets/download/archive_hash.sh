cd /home/itec/emanuele/Datasets/archive/pre-reset-2026-10-05
awk -F'\t' '$1=="f"{print $4}' /dev/shm/ps-dl/archive/listing.tsv | tr '\n' '\0' | xargs -0 -P16 -n200 sha256sum > /dev/shm/ps-dl/archive/sha256.txt 2>/dev/shm/ps-dl/archive/sha256.err
echo "HASHED $(wc -l < /dev/shm/ps-dl/archive/sha256.txt) err=$(wc -l < /dev/shm/ps-dl/archive/sha256.err)"
echo DONE-ARCHIVE-HASH
