from dip import util
import numpy as np
import cv2
import json
import os


def run_custom_pipeline(image, name=""):

    clipping_output = {}
    clipping_output["original"] = util.detect_clipping(image)
    clipping_output["processed"] = {}

    # TODO
    processed = None

    # Step 1: Color balance correction

    # Step 2: CLAHE for local contrast enhancement
    processed = apply_clahe(processed)
    clipping_output["processed"]["After_CLAHE"] = util.detect_clipping(processed)

    # Step 3: Denoising

    # Step 4: Targeted sharpening
    # Restore detail lost during denoising

    with open(f"analysis_output/{name}_custom_clipping_detection.json", "w") as f:
        json.dump(clipping_output, f, indent=4)

    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Output path should be '<image_name>_<pipeline>_processed.jpg'
    output_path = os.path.join(output_dir, f"{name}_custom_processed.jpg")
    cv2.imwrite(output_path, processed)
    print(f"Processed image saved to: {output_path}")

    return processed


def apply_clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    equalized = clahe.apply(gray)
    return equalized
