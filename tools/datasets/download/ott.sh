cd /home/itec/emanuele/Datasets/OpenTTGames/raw
B=https://lab.osai.ai/datasets/openttgames/data
for n in game_1 game_2 game_3 game_4 game_5 test_1 test_2 test_3 test_4 test_5 test_6 test_7; do
  for e in zip mp4; do wget -c -nv "$B/$n.$e" || echo "FAIL $n.$e"; done
done
echo DONE-OTT
