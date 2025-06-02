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
        ret_s, frame_s = cap_source.read()
        ret_g, frame_g = cap_gen.read()
        # Break the loop if either video ends or upon reaching frame 170
        if not ret_s or not ret_g or cap_source.get(cv2.CAP_PROP_POS_FRAMES) > 171:
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
        # print(f"LPIPS score for frame: {distance.item()}")

    cap_source.release()
    cap_gen.release()
    return scores

if __name__ == "__main__":
    source_video_path = 'PointStream/djokovic_federer/source/024.mp4'
    generated_video_paths = [
        'PointStream/djokovic_federer/generated video/Res_128/skeleton_quality_background_10_scene_024.mp4',
        'PointStream/djokovic_federer/generated video/Res_128/skeleton_quality_background_20_scene_024.mp4',
        'PointStream/djokovic_federer/generated video/Res_128/skeleton_quality_background_40_scene_024.mp4',
        'PointStream/djokovic_federer/generated video/Res_128/skeleton_quality_background_70_scene_024.mp4',
        'PointStream/djokovic_federer/generated video/Res_128/skeleton_quality_background_100_scene_024.mp4',
    ]
    # Create a DataFrame to store the LPIPS scores (it needs 170 columns for lpips scores)
    df = pd.DataFrame(columns=['Generated Video'] + [f'LPIPS_{i}' for i in range(171)])
    # Iterate through each generated video
    for generated_video_path in generated_video_paths:
        # Compute LPIPS for each generated video
        lpips_scores = compute_lpips(source_video_path, generated_video_path)
        # Ensure the LPIPS scores are in the correct format
        lpips_scores = lpips_scores[:171]
        # Create a row with the generated video path and LPIPS scores
        row = [generated_video_path] + lpips_scores
        # Append the row to the DataFrame
        df.loc[len(df)] = row
        print(f"LPIPS scores for {generated_video_path} computed.")
    
    # Save the dataframe to CSV
    df.to_csv('PointStream/djokovic_federer/lpips_scores.csv', index=False)
    print("LPIPS scores saved to PointStream/djokovic_federer/lpips_scores.csv")