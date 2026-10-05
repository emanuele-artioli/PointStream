G=/tmp/ps-dl/venv/bin/gdown
cd /home/itec/emanuele/Datasets/TrackNet && $G 1GzJZeEPEi8lJjEAAtVnHhRdX8TVR14yK || echo FAIL-TRACKNET
cd /home/itec/emanuele/Datasets/EgoHOS && $G 1sk0TVEhZESNF67OW3fz9D5coqpIWkwuK || echo FAIL-EGOHOS
ls -la /home/itec/emanuele/Datasets/TrackNet /home/itec/emanuele/Datasets/EgoHOS
echo DONE-GD
