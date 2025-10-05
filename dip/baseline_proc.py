from dip import filters as fil
from dip import util
import os
import json
import cv2


def run_pipeline(image, output_dir, name="unknown"):

    # Want to analyze the amount of clipping occuring after each step and compare with original image
    clipping_output = {}

    clipping_output["original"] = util.detect_clipping(image)
    clipping_output["processed"] = {}

    # Levels and curves adjustment
    # Applied first to establish proper tonal foundation before other enhancements
    processed = fil.adjust_levels_and_curves(image)
    clipping_output["processed"]["after_levelscurves"] = util.detect_clipping(processed)

    # Gamma adjustment
    # Applied second to fine-tune midtone brightness after initial tonal corrections
    processed = fil.apply_gamma_transform(processed)
    clipping_output["processed"]["after_gamma"] = util.detect_clipping(processed)

    # Light noise reduction
    # Applied after tonal adjustments to avoid amplifying noise during contrast changes
    processed = fil.apply_gaussian_filter(processed)
    clipping_output["processed"]["after_noise"] = util.detect_clipping(processed)

    # Unsharp mask
    # Applied last to restore sharpness without enhancing noise or processing artifacts
    processed = fil.unsharp_mask(processed)
    clipping_output["processed"]["after_unsharp"] = util.detect_clipping(processed)

    analysis_dir = "analysis_output"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    with open(f"{analysis_dir}/{name}_baseline_clipping_detection.json", "w") as f:
        json.dump(clipping_output, f, indent=4)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Output path should be '<image_name>_<pipeline>_processed.jpg'
    output_path = os.path.join(output_dir, f"{name}_baseline_processed.jpg")
    cv2.imwrite(output_path, processed)
    print(f"Processed image saved to: {output_path}")

    return processed
