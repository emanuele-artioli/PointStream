from PIL import Image
import os

# Define the folder path
folder_path = '/home/farzad/Desktop/onGithub/CGAN/video/player10/concatenated'

# Define the new size
new_size = (512, 256)

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Filter for image files
        image_path = os.path.join(folder_path, filename)

        # Open and resize the image
        with Image.open(image_path) as img:
            img_resized = img.resize(new_size)

            # Save the resized image with the same name in the same folder
            img_resized.save(image_path)

print("All images have been resized to 512x256 and saved.")
