import os
import random
from PIL import Image

def create_random_grid(image_dir, output_path="grid_output.png"):
    # Get all image files from directory
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)]

    if len(images) < 4:
        print("Need at least 4 images in the directory.")
        return

    # Randomly pick 4 images
    selected = random.sample(images, 4)

    loaded_images = []

    for img_name in selected:
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        loaded_images.append(img)

    # Resize all images to same size (based on first image)
    base_width, base_height = loaded_images[0].size

    resized = [img.resize((base_width, base_height)) for img in loaded_images]

    # Create blank canvas for 2x2 grid
    grid = Image.new("RGB", (base_width * 2, base_height * 2))

    # Paste images into grid
    grid.paste(resized[0], (0, 0))
    grid.paste(resized[1], (base_width, 0))
    grid.paste(resized[2], (0, base_height))
    grid.paste(resized[3], (base_width, base_height))

    # Save result
    grid.save(output_path)
    print(f"Saved grid image to: {output_path}")


if __name__ == "__main__":
    directory = 'work\clusters_kmeans\cluster_2\zoom_out'
    create_random_grid(directory)
