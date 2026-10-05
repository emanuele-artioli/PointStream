export HF_HOME=/dev/shm/ps-dl/hfhome
mkdir -p /dev/shm/ps-dl/RacketVision
/tmp/ps-dl/venv/bin/hf download linfeng302/RacketVision --repo-type dataset --local-dir /dev/shm/ps-dl/RacketVision || echo FAIL-RV
echo DONE-RV
