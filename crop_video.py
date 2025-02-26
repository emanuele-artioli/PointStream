import cv2
import numpy as np
import argparse

def detect_court_lines(frame):
    """
    Detects tennis court lines using edge detection and Hough Transform.
    
    :param frame: Input image (video frame).
    :return: Estimated left and right court line x-coordinates.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

    left_lines = []
    right_lines = []
    mid_x = frame.shape[1] // 2  # Middle of the frame

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) > 50:  # Ensure we only consider vertical(ish) lines
                if x1 < mid_x and x2 < mid_x:
                    left_lines.append((x1 + x2) // 2)
                elif x1 > mid_x and x2 > mid_x:
                    right_lines.append((x1 + x2) // 2)

    if left_lines:
        left_line_x = int(np.median(left_lines))
    else:
        left_line_x = frame.shape[1] // 4  # Fallback estimate

    if right_lines:
        right_line_x = int(np.median(right_lines))
    else:
        right_line_x = 3 * frame.shape[1] // 4  # Fallback estimate

    return left_line_x, right_line_x

def preprocess_tennis_video(input_path, output_path, crop_top, crop_sides, triangle_offset):
    """
    Preprocesses a tennis video by detecting court lines, cropping top and sides, 
    and adding black triangles to remove referees and ball catchers.
    
    :param input_path: Path to the input video.
    :param output_path: Path to save the processed video.
    :param crop_top: Number of pixels to crop from the top.
    :param crop_sides: Number of pixels to crop from both left and right sides.
    :param triangle_offset: Distance (in pixels) to place triangles slightly outside court lines.
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width - 2 * crop_sides, frame_height - crop_top))

    first_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Store the first frame for court detection
        if first_frame is None:
            first_frame = frame.copy()
            left_line_x, right_line_x = detect_court_lines(first_frame)

        # Crop frame
        frame = frame[crop_top:, crop_sides:-crop_sides]

        # Define triangle coordinates based on detected court lines
        height, width, _ = frame.shape
        triangle_height = height // 3  # Approximate height of triangle masks

        # Adjust court lines for cropped frame
        left_line_x -= crop_sides
        right_line_x -= crop_sides

        # Left triangle
        left_triangle = np.array([
            [max(0, left_line_x - triangle_offset), height],  # Bottom
            [max(0, left_line_x), height - triangle_height],  # Top near court
            [max(0, left_line_x - triangle_offset * 1.2), height - triangle_height]  # Further out
        ], np.int32)

        # Right triangle
        right_triangle = np.array([
            [min(width, right_line_x + triangle_offset), height],  # Bottom
            [min(width, right_line_x), height - triangle_height],  # Top near court
            [min(width, right_line_x + triangle_offset * 1.2), height - triangle_height]  # Further out
        ], np.int32)

        # Draw black triangles
        cv2.fillPoly(frame, [left_triangle], (0, 0, 0))
        cv2.fillPoly(frame, [right_triangle], (0, 0, 0))

        # Write frame
        out.write(frame)

    # Cleanup
    cap.release()
    out.release()
    print(f"Processing complete. Saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop video to isolate playing field with inclined side bars")
    parser.add_argument("--video_file", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the cropped video")
    parser.add_argument("--top_crop", type=int, default=100, help="Number of pixels to crop from the top")
    parser.add_argument("--side_crop", type=int, default=50, help="Number of pixels to crop from both sides")
    parser.add_argument("--triangle_offset", type=int, default=50, help="Inclination angle of side bars in degrees")
    args = parser.parse_args()

    preprocess_tennis_video(args.video_file, args.output_file, args.top_crop, args.side_crop, args.triangle_offset)