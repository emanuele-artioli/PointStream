# PointStream
 
## Run in local machine
1. conda create -n pointstream
2. conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics opencv
3. VIDEO_FILE="path/to/your/video.mp4"
4. cd PointStream
5. python pointstream.py $VIDEO_FILE

## Run in Docker
1. Build pointstream:latest image.
2. Modify Dockerfile based on where it will run from.
2.1. If you run from CLI, add ENV VIDEO_FILE="path/to/your/video.mp4"
2.2. If you run from VS CODE, leave Dockerfile as is and pass VIDEO_FILE from tasks.json
2.3. If you have issues with implementation, please open a Github issue.
5. Run container.
