# import cv2
import os
os.system("ffmpeg -i /home/farzad/Desktop/onGithub/CGAN/video/4K.mp4 -vf fps=50 /home/farzad/Desktop/onGithub/CGAN/video/Frames/frame_%04d.jpg")

# # Define the video file path
# video_path = '/home/farzad/Desktop/onGithub/CGAN/video/8K.mp4'  # Change this to your video file path
#
# output_dir = '/home/farzad/Desktop/onGithub/CGAN/video/Frames'  # Directory to save the frames
#
# # Create the output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # Open the video file
# video = cv2.VideoCapture(video_path)
#
# # Check if the video opened successfully
# if not video.isOpened():
#     print("Error: Could not open video.")
#     exit()
#
# frame_count = 0
#
# # Loop through each frame in the video
# while True:
#     ret, frame = video.read()
#
#     # If we've reached the end of the video, break the loop
#     if not ret:
#         break
#
#     # Save the current frame to the output directory
#     frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
#     cv2.imwrite(frame_filename, frame)
#
#     frame_count += 1
#     # if frame_count==100:
#     #     break
#
# # Release the video capture object
# video.release()
# print(f"Extracted {frame_count} frames.")
#
