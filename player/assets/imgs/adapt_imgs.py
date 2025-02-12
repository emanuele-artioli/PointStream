import os
import argparse
from PIL import Image

def center_images_in_canvas(input_folder, output_folder, canvas_size):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files from the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]
    
    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = Image.open(img_path)
        
        # Create a transparent canvas with the specified size
        canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        
        # Calculate the position to center the image on the canvas
        img_width, img_height = img.size
        canvas_width, canvas_height = canvas_size
        x_offset = (canvas_width - img_width) // 2
        y_offset = (canvas_height - img_height) // 2
        
        # Paste the image onto the canvas at the calculated position
        canvas.paste(img, (x_offset, y_offset), img.convert("RGBA").getchannel('A'))

        # Save the final image
        output_path = os.path.join(output_folder, f"centered_{image_file}")
        canvas.save(output_path)
        print(f"Saved centered image: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Center images in a transparent canvas.")
    parser.add_argument('input_folder', type=str, help="Path to the folder containing input images")
    parser.add_argument('output_folder', type=str, help="Path to the folder to save centered images")
    parser.add_argument('canvas_width', type=int, help="Width of the transparent canvas")
    parser.add_argument('canvas_height', type=int, help="Height of the transparent canvas")
    
    args = parser.parse_args()

    # Call the function with the parsed arguments
    canvas_size = (args.canvas_width, args.canvas_height)
    center_images_in_canvas(args.input_folder, args.output_folder, canvas_size)

if __name__ == "__main__":
    main()
