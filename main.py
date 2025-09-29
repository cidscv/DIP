from dip import analysis as aly
from dip import baseline_proc as baseline
import os
import cv2

INPUT_DIR = "input_images"

if __name__ == "__main__":

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [
        f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        exit()

    print(f"Found {len(image_files)} images to process\n")

    # Process each image
    for filename in image_files:

        filename = os.path.splitext(filename)[0]

        image = cv2.imread(f"input_images/{filename}.JPG")

        if image is None:
            print("Error: Could not load image")
            continue

        print(f"Original image shape: {image.shape}")
        aly.generate_histogram_from_image(image, name=f"{filename}_original")

        print("\n=== Running Baseline Pipeline ===\n")
        processed = baseline.run_pipeline(image, name=f"{filename}")
        print("\n=== Finished Baseline Pipeline ===\n")

        aly.generate_histogram_from_image(processed, name=f"{filename}_processed")
