import cv2
import torch
import lpips

def compute_lpips(source_video_path, generated_video_path):
    # Initialize the LPIPS model (e.g., using AlexNet)
    lpips_fn = lpips.LPIPS(net='alex').cpu()  # or .cpu() if no GPU

    cap_source = cv2.VideoCapture(source_video_path)
    cap_gen = cv2.VideoCapture(generated_video_path)

    scores = []

    while True:
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

        t_s = t_s.cpu()  # or .cpu() if needed
        t_g = t_g.cpu()

        # Compute LPIPS distance
        distance = lpips_fn(t_s, t_g)
        scores.append(distance.item())

    cap_source.release()
    cap_gen.release()
    return scores

if __name__ == "__main__":
    source_video_path = "scenes_encoded/djokovic_federer/source/024.mp4"
    generated_video_path = "scenes_encoded/djokovic_federer/generated/skeleton_024.mp4"

    lpips_scores = compute_lpips(source_video_path, generated_video_path)
    print("LPIPS scores:", lpips_scores)