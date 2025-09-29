from dip import analysis as aly
from dip import baseline_proc as baseline
import cv2

if __name__ == "__main__":
    # Load the image first
    image = cv2.imread("input_images/pic01_iso100.JPG")

    if image is None:
        print("Error: Could not load image")
        exit()

    print(f"Original image shape: {image.shape}")
    aly.generate_histogram_from_image(image, name="pic01_original")

    print("\n=== Running Baseline Pipeline ===\n")
    processed = baseline.run_pipeline(image)
    print("\n=== Finished Baseline Pipeline ===\n")

    aly.generate_histogram_from_image(processed, name="pic01_processed")
