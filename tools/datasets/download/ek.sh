awk -F'\t' '{print $1" "$2" "$3}' /tmp/ps-dl/ek_videos.tsv | xargs -P6 -n3 bash /tmp/ps-dl/ek_one.sh
echo DONE-EK
