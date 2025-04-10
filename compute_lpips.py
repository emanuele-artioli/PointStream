import cv2
import pandas as pd
import torch
import lpips

def compute_lpips(source_video_path, generated_video_path):
    # Initialize the LPIPS model (e.g., using AlexNet)
    lpips_fn = lpips.LPIPS(net='alex').cuda()  # or .cpu() if no GPU

    cap_source = cv2.VideoCapture(source_video_path)
    cap_gen = cv2.VideoCapture(generated_video_path)

    scores = []

    while True:
        # only compute one frame every X
        if cap_source.get(cv2.CAP_PROP_POS_FRAMES) % 50 != 0:
            cap_source.grab()
            cap_gen.grab()
            continue
        ret_s, frame_s = cap_source.read()
        ret_g, frame_g = cap_gen.read()
        if not ret_s or not ret_g:
            break

        # Convert BGR to RGB
        frame_s = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)
        frame_g = cv2.cvtColor(frame_g, cv2.COLOR_BGR2RGB)

        # Ensure both frames have the same size
        if frame_s.shape != frame_g.shape:
            frame_g = cv2.resize(frame_g, (frame_s.shape[1], frame_s.shape[0]))

        # Convert to tensors in the range [-1, 1]
        t_s = torch.from_numpy(frame_s).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0
        t_g = torch.from_numpy(frame_g).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0

        t_s = t_s.cuda()  # or .cpu() if needed
        t_g = t_g.cuda()

        # Compute LPIPS distance
        distance = lpips_fn(t_s, t_g)
        scores.append(distance.item())
        print(f"LPIPS score for frame: {distance.item()}")

    cap_source.release()
    cap_gen.release()
    return scores

if __name__ == "__main__":
    source_video_path = 'PointStream/djokovic_federer/source/024.mp4'
    generated_video_paths = [
        'PointStream/djokovic_federer/encoded/024/2s/hevc/640x360_145k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/768x432_300k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/960x540_600k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/960x540_900k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/960x540_1600k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/1280x720_2400k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/1280x720_3400k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/1920x1080_4500k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/1920x1080_5800k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/2560x1440_8100k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/3840x2160_11600k.mp4',
        'PointStream/djokovic_federer/encoded/024/2s/hevc/3840x2160_16800k.mp4',
        'PointStream/djokovic_federer/generated/canny_qualty_background_10_scene_024.mp4',
        'PointStream/djokovic_federer/generated/canny_qualty_background_20_scene_024.mp4',
        'PointStream/djokovic_federer/generated/canny_qualty_background_40_scene_024.mp4',
        'PointStream/djokovic_federer/generated/canny_qualty_background_70_scene_024.mp4', 
        'PointStream/djokovic_federer/generated/canny_qualty_background_100_scene_024.mp4',
        'PointStream/djokovic_federer/generated/grid3_qualty_background_10_scene_024.mp4',
        'PointStream/djokovic_federer/generated/grid3_qualty_background_20_scene_024.mp4',
        'PointStream/djokovic_federer/generated/grid3_qualty_background_40_scene_024.mp4',
        'PointStream/djokovic_federer/generated/grid3_qualty_background_70_scene_024.mp4',
        'PointStream/djokovic_federer/generated/grid3_qualty_background_100_scene_024.mp4',
        'PointStream/djokovic_federer/generated/grid6_qualty_background_10_scene_024.mp4',
        'PointStream/djokovic_federer/generated/grid6_qualty_background_20_scene_024.mp4',
        'PointStream/djokovic_federer/generated/grid6_qualty_background_40_scene_024.mp4',
        'PointStream/djokovic_federer/generated/grid6_qualty_background_70_scene_024.mp4',
        'PointStream/djokovic_federer/generated/grid6_qualty_background_100_scene_024.mp4',
        'PointStream/djokovic_federer/generated/skeleton_qualty_background_10_scene_024.mp4',
        'PointStream/djokovic_federer/generated/skeleton_qualty_background_20_scene_024.mp4',
        'PointStream/djokovic_federer/generated/skeleton_qualty_background_40_scene_024.mp4',
        'PointStream/djokovic_federer/generated/skeleton_qualty_background_70_scene_024.mp4',
        'PointStream/djokovic_federer/generated/skeleton_qualty_background_100_scene_024.mp4',
    ]
    # Create a DataFrame to store the LPIPS scores
    df = pd.DataFrame(columns=['Generated Video', 'LPIPS'])
    # Iterate through each generated video
    for generated_video_path in generated_video_paths:
        # Compute LPIPS for each generated video
        lpips_scores = compute_lpips(source_video_path, generated_video_path)
        # Save the average scores to a pandas DataFrame
        avg_score = sum(lpips_scores) / len(lpips_scores)
        # Append the results to the DataFrame
        df = df.append({'Generated Video': generated_video_path, 'LPIPS': avg_score}, ignore_index=True)
        # Print the average LPIPS score for each generated video
        print(f"Average LPIPS score for {generated_video_path}: {avg_score:.4f}")
    
    # Save the dataframe to CSV
    df.to_csv('PointStream/djokovic_federer/lpips_scores.csv', index=False)
    print("LPIPS scores saved to PointStream/djokovic_federer/lpips_scores.csv")