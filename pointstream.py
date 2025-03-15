import os
import time
import subprocess
import cv2
import shutil
import concurrent.futures

def stitch_background_images(background_folder, stride=5):
    """Combines periodic background frames into a single stitched image."""
    all_images = []
    for i, name in enumerate(sorted(os.listdir(background_folder))):
        if name.endswith(".png") and i % stride == 0:
            img = cv2.imread(os.path.join(background_folder, name))
            if img is not None:
                all_images.append(img)
    if not all_images:
        print("No background images found.")
        return None
    stitched = all_images[0].copy()
    for img in all_images[1:]:
        mask = (stitched == 0)
        stitched[mask] = img[mask]
    return stitched

def segment_scene(video_file, working_dir, device, experiment_folder):
    subprocess.call([
        'python', f'{working_dir}/instance_segmentation.py',
        '--video_file', video_file,
        '--experiment_folder', experiment_folder,
        '--device', device
    ])

def postprocess_scene(experiment_folder):
    '''Delete people folders that are missing too many frames (i.e., everyone besides players), rename the ones that are not, based on their class, then zip the experiment.'''
    # Get maximum number of frames from the frame id in the last row of the CSV file
    with open(os.path.join(experiment_folder, 'bounding_boxes.csv')) as f:
        frame_id = int(f.readlines()[-1].split(',')[0])
    min_frames = frame_id * 0.9
    for obj_folder in os.listdir(experiment_folder):
        if obj_folder.startswith('0_'):
            num_frames = len(os.listdir(os.path.join(experiment_folder, obj_folder)))
            if num_frames < min_frames:
                shutil.rmtree(os.path.join(experiment_folder, obj_folder))
            else:
                os.rename(os.path.join(experiment_folder, obj_folder), os.path.join(experiment_folder, 'person_' + obj_folder[2:]))
        elif obj_folder.startswith('32_'):
            os.rename(os.path.join(experiment_folder, obj_folder), os.path.join(experiment_folder, 'ball_' + obj_folder[3:]))
        elif obj_folder.startswith('38_'):
            os.rename(os.path.join(experiment_folder, obj_folder), os.path.join(experiment_folder, 'racket_' + obj_folder[3:]))

    # Stitch background images
    background_folder = os.path.join(experiment_folder, 'background')
    stitched = stitch_background_images(background_folder)
    if stitched is not None:
        cv2.imwrite(os.path.join(experiment_folder, 'background.png'), stitched)
    shutil.rmtree(background_folder)
    
    # Zip the experiment folder
    shutil.make_archive(experiment_folder, 'zip', experiment_folder)
    shutil.rmtree(experiment_folder)

def main():
    # get current working directory
    working_dir = os.environ.get("WORKING_DIR", "/PointStream")
    video_folder = os.environ.get("VIDEO_FOLDER", "/scenes")
    video_file = os.environ.get("VIDEO_FILE")
    device = os.environ.get("DEVICE", "cpu")
    parallel = int(os.environ.get("PARALLEL", 0))

    video_folder = os.path.join(working_dir, video_folder)

     # If no video_file is provided, use every video in the folder
    if not video_file:
        all_videos = [v for v in os.listdir(video_folder) if v.endswith(('.mp4','.mov','.avi'))]
    else:
        all_videos = [video_file]

    # Whether to process each video in parallel or sequentially
    for vid in all_videos:
        experiment_folder = f'{working_dir}/experiments/{os.path.basename(vid).split(".")[0]}'
        os.makedirs(experiment_folder, exist_ok=True)
        video_file = os.path.join(video_folder, vid)
        if parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                seg_future = executor.submit(segment_scene, video_file, working_dir, device, experiment_folder)
                seg_future.result()  # Wait for segmentation to finish
            postprocess_scene(experiment_folder)
        else:
            segment_scene(video_file, working_dir, device, experiment_folder)
            postprocess_scene(experiment_folder)


if __name__ == "__main__":
    main()