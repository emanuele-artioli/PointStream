import os
import time
import subprocess

def main():
    video_file = os.environ["VIDEO_FILE"]

    # Crop video to isolate playing field
    cropped_video_file = '/app/cropped_video.mp4'
    if not os.path.isfile(cropped_video_file):
        subprocess.call([
            'python', 'crop_video.py',
            '--video_file', video_file,
            '--output_file', cropped_video_file,
            '--top_crop', '100',
            '--side_crop', '50',
            '--triangle_offset', '50'
        ])

    # Perform object detection on the frames if not already done
    detected_folder = '/app/detected_objects'
    os.makedirs(detected_folder, exist_ok=True)
    if not os.listdir(detected_folder):
        # Calculate the elapsed time of this task
        start_time = time.time()
        subprocess.call([
            'python', 'object_detection.py',
            '--video_file', video_file,
            '--detected_folder', detected_folder,
        ])
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for object detection: {elapsed_time}")

    # # Perform instance segmentation on the frames if not already done
    # segmented_folder = '/app/segmented_objects'
    # os.makedirs(segmented_folder, exist_ok=True)
    # if not os.listdir(segmented_folder):
    #     # Calculate the elapsed time of this task
    #     start_time = time.time()
    #     subprocess.call([
    #         'python', 'instance_segmentation.py',
    #         '--video_file', video_file,
    #         '--segmented_folder', segmented_folder
    #     ])
    #     elapsed_time = time.time() - start_time
    #     print(f"Elapsed time for instance segmentation: {elapsed_time}")

if __name__ == "__main__":
    main()

# video_file = os.environ["VIDEO_FILE"]
# # extract video file resolution
# width, height = subprocess.check_output([
#     'ffprobe', '-v', 'error', 
#     '-select_streams', 'v:0', 
#     '-show_entries', 'stream=width,height', 
#     '-of', 'csv=s=x:p=0', 
#     video_file]).decode('utf-8').strip().split('x')

# # Perform object detection on the frames if not already done
# detected_folder = '/app/detected_objects'
# os.makedirs(detected_folder, exist_ok=True)
# if not os.listdir(detected_folder):
#     # Calculate the elapsed time of this task
#     start_time = time.time()
#     subprocess.call([
#         'python', 'object_detection.py',
#         '--input_file', video_file,
#         '--detected_folder', detected_folder,
#     ])
#     elapsed_time = time.time() - start_time
#     print(f"Elapsed time for object detection: {elapsed_time}")

# # Perform instance segmentation on the frames if not already done
# segmented_folder = '/app/segmented_objects'
# os.makedirs(segmented_folder, exist_ok=True)
# if not os.listdir(segmented_folder):
#     # Calculate the elapsed time of this task
#     start_time = time.time()
#     subprocess.call([
#         'python', 'instance_segmentation.py',
#         '--video_file', video_file,
#         '--segmented_folder', segmented_folder
#     ])
#     elapsed_time = time.time() - start_time
#     print(f"Elapsed time for instance segmentation: {elapsed_time}")

# # Perform pose estimation on the segmented objects if not already done
# pose_csv = "/app/poses.csv"
# if not os.path.isfile(pose_csv):
#     start_time = time.time()
#     subprocess.call([
#         'python', 'pose_estimation.py',
#         '--segmented_folder', segmented_folder,
#         '--pose_csv', pose_csv
#     ])
#     elapsed_time = time.time() - start_time
#     print(f"Elapsed time for pose estimation: {elapsed_time}")

# # Perform jersey number recognition on segmented objects if not already done
# jersey_csv = "/app/jersey_numbers.csv"
# if not os.path.isfile(jersey_csv):
#     start_time = time.time()
#     method = os.environ.get("JERSEY_RECOGNITION_METHOD", "easyocr")  # Default to 'easyocr' if not specified
#     subprocess.call([
#         'python', 'jersey_recognition.py',
#         '--segmented_objects_folder', segmented_folder,
#         '--jersey_csv', jersey_csv,
#         '--method', method
#     ])
#     elapsed_time = time.time() - start_time
#     print(f"Elapsed time for jersey recognition: {elapsed_time}")