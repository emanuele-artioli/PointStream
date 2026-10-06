# Labelled dataset acquisition

`download/` holds the scripts that fetched OpenTTGames, RacketVision, TrackNet,
EgoHOS, EPIC-KITCHENS VISOR (with the 179 EPIC-KITCHENS videos it annotates)
and HOT3D-Clips into `/home/itec/emanuele/Datasets/<name>` on 5–6 October 2026,
and HInt (`download/hint.sh`, TLS verification off; see the script) on 6 October.
They ran on gpu1 from `/tmp/ps-dl` in tmux, with logs in `/tmp/ps-dl/logs`.
Each script verifies against the provider's checksum where one exists: HF LFS
sha256, or EPIC-KITCHENS md5. The later versions hash while streaming, so
nothing is read back from NFS.

`dataset_manifest.py` stages archives to `/dev/shm`, inspects labels,
smoke-reads one labelled sample (writing an overlay), and writes the immutable
manifests in `/home/itec/emanuele/Datasets/manifests/`. Never extract these
archives onto the NFS home.
