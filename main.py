from dip import analysis as aly
from dip import baseline_proc as baseline
from dip import custom_proc as custom
from dip import util
import argparse
import os
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="DIP",
        description="Digital Image Processing pipeline - processes images using different pipeline methods",
    )

    parser.add_argument(
        "--image_dir",
        metavar="imagedir",
        required=True,
        help="Directory containing input images to process",
    )

    parser.add_argument(
        "--out",
        metavar="output_dir",
        required=True,
        help="Directory where processed images will be saved",
    )

    parser.add_argument(
        "--pipeline",
        metavar="pipeline",
        required=True,
        choices=["A", "B"],
        help="Pipeline to use for processing (choices: A, B)",
    )

    args = parser.parse_args()

    # Use the parsed arguments
    INPUT_DIR = args.image_dir
    OUTPUT_DIR = args.out

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_files = [
        f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        exit()

    print(f"Found {len(image_files)} images to process\n")

    # Process each image
    for file_path in image_files:

        print(file_path)

        filename = os.path.splitext(file_path)[0]
        image = cv2.imread(f"{INPUT_DIR}/{filename}.JPG")

        if image is None:
            print(f"Error: Could not load image {filename}")
            continue

        print(f"Original image shape: {image.shape}")
        aly.generate_histogram_from_image(image, name=f"{filename}_original")

        print(f"Generating EXIF Data")
        util.extract_exif(f"{INPUT_DIR}/{file_path}", OUTPUT_DIR, name=f"{filename}")

        # Run the appropriate pipeline based on the argument
        if args.pipeline == "A":
            print("\n=== Running Pipeline A ===\n")
            processed = baseline.run_pipeline(image, OUTPUT_DIR, name=f"{filename}")
            print("\n=== Finished Pipeline A ===\n")

        elif args.pipeline == "B":
            print("\n=== Running Pipeline B ===\n")
            processed = custom.run_custom_pipeline(
                image, OUTPUT_DIR, name=f"{filename}"
            )
            print("\n=== Finished Pipeline B ===\n")

        aly.generate_histogram_from_image(processed, name=f"{filename}_processed")
