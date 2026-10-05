rev=$(cat /tmp/ps-dl/hot3d_rev.txt)
awk -F'\t' '{print $1" "$2}' /tmp/ps-dl/hot3d_files.tsv | xargs -P6 -n2 bash -c 'bash /tmp/ps-dl/hf_one.sh bop-benchmark/hot3d '$rev' /home/itec/emanuele/Datasets/HOT3D "$0" "$1"'
echo DONE-HOT3D
