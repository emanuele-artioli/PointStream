import os
import sys
import cv2

def stitch_background_images(background_folder, stride=50):
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

if __name__ == "__main__":
    background_folder = sys.argv[1]
    stitched = stitch_background_images(background_folder)
    if stitched is not None:
        cv2.imwrite("full_background.png", stitched)