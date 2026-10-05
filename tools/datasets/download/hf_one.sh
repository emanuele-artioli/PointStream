repo=$1; rev=$2; root=$3; p=$4; sha=$5
o="$root/$p"; mkdir -p "$(dirname "$o")"
u="https://huggingface.co/datasets/$repo/resolve/$rev/$p"
for t in 1 2 3; do
  got=$(wget -q --header="Authorization: Bearer $(cat /home/itec/emanuele/.cache/huggingface/token)" -O - "$u" | tee "$o.part" | sha256sum | cut -d" " -f1)
  if [ "$sha" = "-" ] || [ "$got" = "$sha" ]; then mv "$o.part" "$o"; echo "OK $p $got"; exit 0; fi
  echo "RETRY $p sha"; sleep 5
done
echo "FAIL $p"
