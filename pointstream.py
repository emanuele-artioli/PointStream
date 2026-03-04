from utils import load_video_frames, detect_with_yolo

def main():
    video_path = "/home/itec/emanuele/Datasets/federer_djokovic/libsvtav1_crf35_pre5/scene_004.mp4"
    exp_folder = "/home/itec/emanuele/pointstream/experiments/sample_exp"
    
    frames = load_video_frames(video_path)
    
    # detect_with_yolo yields detection results as dictionaries
    for detection in detect_with_yolo(frames, exp_folder):
        print(detection)

if __name__ == "__main__":
    main()