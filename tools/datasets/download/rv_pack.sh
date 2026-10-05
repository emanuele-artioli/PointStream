set -e
cd /dev/shm/ps-dl/RacketVision
rm -rf .cache
find . -type f ! -path "./.cache/*" -printf "%P\0" | sort -z | xargs -0 -P32 -n500 sha256sum > /dev/shm/ps-dl/rv_members.sha256
/tmp/ps-dl/venv/bin/python - <<P
import json
api={s["rfilename"]:s for s in json.load(open("/tmp/ps-dl/rv_api.json"))["siblings"]}
loc={}
for l in open("/dev/shm/ps-dl/rv_members.sha256"):
    h,p=l.rstrip("\n").split("  ",1); loc[p]=h
miss=[p for p in api if p not in loc]; bad=[p for p,s in api.items() if p in loc and s.get("lfs") and s["lfs"]["sha256"]!=loc[p]]
nl=sum(1 for s in api.values() if s.get("lfs"))
print("api files",len(api),"local",len(loc),"missing",len(miss),miss[:5],"lfs checked",nl,"lfs mismatches",len(bad),bad[:5])
P
O=/home/itec/emanuele/Datasets/RacketVision
for p in annotations info data_traj badminton tabletennis tennis; do tar --sort=name --owner=0 --group=0 --numeric-owner -cf /dev/shm/ps-dl/$p.tar $p; cp /dev/shm/ps-dl/$p.tar $O/$p.tar.part && mv $O/$p.tar.part $O/$p.tar; rm /dev/shm/ps-dl/$p.tar; echo packed $p; done
cp README.md .gitattributes $O/
echo DONE-RVPACK
