# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app

# Add the pretrained models
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt /app/yolo11n-seg.pt
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt /app/yolo11n-pose.pt
ADD https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt /app/sam2.1_b.pt

COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "pointstream.py"]
