import cv2
import numpy as np
import json
import os
import argparse

# TODO: Can we speed up by not calculating homographies for every frame? Maybe every N frames and interpolate? Maybe calculate residuals and only recalc if above threshold?
# TODO: Can we improve blending? Maybe use multi-band blending or seam finding?
# TODO: Can we use GPU acceleration for feature detection/matching?
# TODO: Can we use other feature detectors (SIFT, AKAZE) for better results?
# TODO: Can we use optical flow to improve alignment between frames?
# TODO: Can we handle dynamic scenes better? Maybe use RANSAC to filter out moving objects? Or use semantic segmentation to mask out foreground objects?
# TODO: Can we implement exposure compensation to handle varying lighting conditions between frames?
# TODO: Can we implement a GUI to visualize the panorama creation process step-by-step?
# TODO: Can we add support for different video formats and codecs? FFmpeg instead of OpenCV?
# TODO: We should compare quality of recontruction against av1 at the same size. There is probably a movement threshold above which av1 is better, we should determine it and suggest when to use which method.

class VideoPanorama:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_homography(self, img1, img2):
        """Finds the perspective transformation (homography) between two frames."""
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None: return None
        
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Need at least 4 matches to find a homography
        if len(matches) < 4: return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Calculate Homography: Transform from img1 to img2
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def create_panorama(self, video_path, output_image="panorama.png", output_meta="meta.json", skip_frames=5):
        print(f"Processing {video_path}...")
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        homographies = [] # Stores H relative to the previous processed frame
        global_homographies = [] # Stores H relative to the FIRST frame (Frame 0)

        # 1. READ FRAMES
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if count % skip_frames == 0:
                frames.append(frame)
            count += 1
        cap.release()

        if not frames:
            print("No frames extracted.")
            return

        print(f"Extracted {len(frames)} frames. Computing alignment...")

        # 2. COMPUTE ALIGNMENT
        # Initialize identity matrix for the first frame
        current_H = np.eye(3) 
        global_homographies.append(current_H) # Frame 0 is the anchor
        
        for i in range(len(frames) - 1):
            # Find transform from Frame[i+1] -> Frame[i]
            # We want to map everything BACK to the coordinate system of Frame 0
            H_rel = self.get_homography(frames[i+1], frames[i])
            
            if H_rel is None:
                print(f"Warning: Could not align frame {i+1}. Using Identity.")
                H_rel = np.eye(3)
            
            # Chain the transforms: H_global_next = H_global_prev @ H_rel
            current_H = current_H @ H_rel 
            global_homographies.append(current_H)

        # 3. CALCULATE CANVAS SIZE
        # We need to project the corners of every frame to find the min/max X and Y
        h, w = frames[0].shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        
        all_corners = []
        for H in global_homographies:
            warped_corners = cv2.perspectiveTransform(corners, H)
            all_corners.append(warped_corners)
        
        all_corners = np.concatenate(all_corners, axis=0)
        [xmin, ymin] = all_corners.min(axis=0).ravel()
        [xmax, ymax] = all_corners.max(axis=0).ravel()

        # Translation to shift negative coordinates into positive space
        translation_dist = [-xmin, -ymin]
        H_translation = np.array([[1, 0, translation_dist[0]], 
                                  [0, 1, translation_dist[1]], 
                                  [0, 0, 1]])

        canvas_w = int(round(xmax - xmin))
        canvas_h = int(round(ymax - ymin))
        
        print(f"Canvas size: {canvas_w}x{canvas_h}")
        
        # 4. WARP AND STITCH
        panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # We draw frames from first to last. 
        # Note: Simple overlay (Painter's Algorithm). 
        # For better results, you'd use blending, but this is faster.
        for idx, frame in enumerate(frames):
            # Combine the translation adjustment with the frame's global homography
            H_final = H_translation @ global_homographies[idx]
            
            warped_frame = cv2.warpPerspective(frame, H_final, (canvas_w, canvas_h))
            
            # Create a mask to overlay only non-black pixels (simple masking)
            mask = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            
            # Copy non-black pixels from warped_frame to panorama
            # This overwrites previous frames. To preserve earlier frames, reverse this loop.
            panorama = cv2.bitwise_and(panorama, panorama, mask=cv2.bitwise_not(mask))
            panorama = cv2.add(panorama, warped_frame)

        cv2.imwrite(output_image, panorama)
        
        # 5. SAVE METADATA
        # We save the H_final matrices so we can extract the "views" later
        # We also need the original frame size
        meta_data = {
            "frame_size": [int(w), int(h)],
            "homographies": [H.tolist() for H in global_homographies],
            "translation": [float(translation_dist[0]), float(translation_dist[1])],
            "skip_frames": skip_frames,
            "canvas_size": [canvas_w, canvas_h]
        }
        
        with open(output_meta, "w") as f:
            json.dump(meta_data, f)
            
        print(f"Saved panorama to {output_image} and metadata to {output_meta}")

    def reconstruct_video(self, panorama_path, meta_path, output_video_path):
        print("Reconstructing video...")
        
        if not os.path.exists(panorama_path) or not os.path.exists(meta_path):
            print("Files not found.")
            return

        panorama = cv2.imread(panorama_path)
        with open(meta_path, "r") as f:
            data = json.load(f)

        frame_w, frame_h = data["frame_size"]
        homographies = np.array(data["homographies"])
        tx, ty = data["translation"]
        
        # Matrix to shift panorama back to 0,0 before applying inverse frame homography
        H_translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

        # Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (frame_w, frame_h))

        for H_raw in homographies:
            # We used H_final = H_trans @ H_global to put frame ONTO panorama
            # Now we want to extract the frame FROM panorama.
            # We need the inverse of that operation.
            
            H_global = np.array(H_raw)
            H_composite = H_translation @ H_global
            
            # Invert to map from [Output Frame] -> [Panorama Source]
            H_inv = np.linalg.inv(H_composite)
            
            # Warp the specific region of the panorama back into a frame
            frame = cv2.warpPerspective(panorama, H_inv, (frame_w, frame_h))
            out.write(frame)

        out.release()
        print(f"Reconstruction saved to {output_video_path}")

# --- USAGE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and reconstruct video panoramas")
    parser.add_argument("--video_path", type=str, default="/home/itec/emanuele/Datasets/djokovic_federer/015.mp4", help="Path to the input video")
    args = parser.parse_args()

    input_video = args.video_path
    processor = VideoPanorama()

    # 0. Create experiment folder as timestamped directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"/home/itec/emanuele/pointstream/experiments/{timestamp}_pano"
    os.makedirs(experiment_dir, exist_ok=True)

    # Start timing execution
    import time
    start_time = time.time()
    
    # 1. Create Panorama
    processor.create_panorama(input_video, f"{experiment_dir}/pano.png", f"{experiment_dir}/meta.json", skip_frames=2)
    
    # 2. Reconstruct Video
    processor.reconstruct_video(f"{experiment_dir}/pano.png", f"{experiment_dir}/meta.json", f"{experiment_dir}/reconstructed.mp4")

    # End timing execution
    end_time = time.time()
    print(f"Execution speed: { (len(os.listdir(experiment_dir)) / (end_time - start_time)) :.2f} fps.")

    # 3. evaluate reconstruction quality with SSIM
    from skimage.metrics import structural_similarity as ssim
    import cv2
    cap_orig = cv2.VideoCapture(input_video)
    cap_recon = cv2.VideoCapture(f"{experiment_dir}/reconstructed.mp4")
    ssim_values = []
    while True:
        ret1, frame1 = cap_orig.read()
        ret2, frame2 = cap_recon.read()
        if not ret1 or not ret2:
            break
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        ssim_index = ssim(frame1_gray, frame2_gray)
        ssim_values.append(ssim_index)
    cap_orig.release()
    cap_recon.release()
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
    print(f"Average SSIM between original and reconstructed video: {avg_ssim:.4f}")