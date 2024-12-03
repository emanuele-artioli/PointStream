import os
from PIL import Image

# Define the folder paths
folder1 = '/home/farzad/Desktop/onGithub/CGAN/video/player33/player_33_blue'  # Path to the first folder of images
folder2 = '/home/farzad/Desktop/onGithub/CGAN/video/player33/ske_33_blue'  # Path to the second folder of images
output_folder = '/home/farzad/Desktop/onGithub/CGAN/video/player33/concatenated'  # Folder to save concatenated images

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all files in folder1 and folder2 (assuming the same filenames in both)

for filename in os.listdir(folder1):
    # cmd='mv /home/farzad/Desktop/onGithub/DGAN/concatenated_images/'+filename+' /home/farzad/Desktop/onGithub/DGAN/concatenated_images/'+filename.replace('concatenated_image_','')
    # print(cmd)
    # os.system(cmd)
    # continue
    # Open images from both folders
    img1_path = os.path.join(folder1, filename)
    img2_path = os.path.join(folder2, filename.split('.')[0]+'_rendered.png')

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)


    # Concatenate images side by side (horizontally)
    concatenated_img = Image.new('RGB', (img1.width + img2.width, img1.height))
    concatenated_img.paste(img1, (0, 0))
    concatenated_img.paste(img2, (img1.width, 0))

    # Save the concatenated image in the output folder
    output_path = os.path.join(output_folder, f'concatenated_{filename}')
    concatenated_img.save(output_path)

print(f"All images have been concatenated and saved to the '{output_folder}' folder.")
