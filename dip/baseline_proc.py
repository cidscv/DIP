import numpy as np
import os
import json
import cv2


def run_pipeline(image):

    # Want to analyze the amount of clipping occuring after each step and compare with original image
    clipping_output = {}

    clipping_output["original"] = detect_clipping(image)
    clipping_output["processed"] = {}

    # Levels and curves adjustment
    # Applied first to establish proper tonal foundation before other enhancements
    processed = adjust_levels_and_curves(image)
    clipping_output["processed"]["after_levelscurves"] = detect_clipping(processed)

    # Gamma adjustment
    # Applied second to fine-tune midtone brightness after initial tonal corrections
    processed = apply_gamma_transform(processed)
    clipping_output["processed"]["after_gamma"] = detect_clipping(processed)

    # Light noise reduction
    # Applied after tonal adjustments to avoid amplifying noise during contrast changes
    processed = apply_median_filter(processed)
    clipping_output["processed"]["after_noise"] = detect_clipping(processed)

    # Unsharp mask
    # Applied last to restore sharpness without enhancing noise or processing artifacts
    processed = unsharp_mask(processed)
    clipping_output["processed"]["after_unsharp"] = detect_clipping(processed)

    with open("analysis_output/clipping_detection.json", "w") as f:
        json.dump(clipping_output, f, indent=4)

    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Output path should be '<image_name>_processed.jpg'
    output_path = os.path.join(output_dir, "testimage_processed.jpg")
    cv2.imwrite(output_path, processed)
    print(f"Processed image saved to: {output_path}")

    return processed


def adjust_levels_and_curves(image, black_point=0, white_point=255, gamma=1.2):
    """
    black_point=0: Preserves existing shadow detail without crushing blacks
    white_point=255: Uses full dynamic range without clipping highlights
    gamma=1.2: Brightens midtones for better visibility in low-light scenes
               Values > 1.0 lift midtones while keeping black/white points intact
    """

    img_float = (
        image.astype(np.float32) / 255.0
    )  # Convert to float for precise calculations

    # Adjust levels - stretches tonal range between black and white points
    img_float = np.clip(
        (img_float - black_point / 255.0) / ((white_point - black_point) / 255.0), 0, 1
    )

    # Gamma correction - power curve brightens midtones
    img_float = np.power(img_float, 1.0 / gamma)

    return (img_float * 255).astype(np.uint8)


def apply_gamma_transform(image, gamma=1.1):
    """
    gamma=1.1: Subtle additional brightening after initial levels adjustment
               Small increment provides gentle fine-tuning of midtones
               Applied as secondary correction for optimal low-light enhancement
    """

    inv_gamma = 1.0 / gamma

    # Lookup table method is more efficient than per-pixel power calculations
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    return cv2.LUT(image, table)


def apply_median_filter(image):
    """
    kernel_size=3: Smallest effective size for noise reduction
                   Larger kernels would blur important details
                   3x3 balances noise suppression with detail preservation
                   Median filter chosen over Gaussian to preserve edges
    """
    return cv2.medianBlur(image, 3)


def unsharp_mask(image, radius=1.0, amount=0.5, threshold=0):
    """
    radius=1.0: Small radius targets fine details without creating halos
                Focuses on restoring detail lost during previous processing steps

    amount=0.5: Moderate sharpening strength (50% effect)
                Conservative value prevents over-sharpening artifacts
                Suitable for baseline enhancement without dramatic changes

    threshold=0: No threshold means all edge differences are enhanced
                 Ensures consistent processing across all image content
                 Could be increased to avoid sharpening noise in noisy images
    """
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.where(np.abs(image - blurred) > threshold, sharpened, image)


def detect_clipping(image, threshold=0.01):
    """
    Detect clipping in shadows and highlights
    threshold=0.01: 1% of pixels at extremes indicates clipping
    """

    total_pixels = image.shape[0] * image.shape[1]

    if len(image.shape) == 3:
        clipping_info = {}
        channel_names = ["Blue", "Green", "Red"]

        for i, channel in enumerate(channel_names):
            # Count pixels at extreme values (0 = pure black, 255 = pure white)
            shadows = np.sum(image[:, :, i] == 0) / total_pixels
            highlights = np.sum(image[:, :, i] == 255) / total_pixels

            clipping_info[channel] = {
                "shadows_clipped": bool(shadows > threshold),
                "highlights_clipped": bool(highlights > threshold),
                "shadow_percent": float(shadows * 100),
                "highlight_percent": float(highlights * 100),
            }
        return clipping_info
    else:
        shadows = np.sum(image == 0) / total_pixels
        highlights = np.sum(image == 255) / total_pixels
        return {
            "shadows_clipped": bool(shadows > threshold),
            "highlights_clipped": bool(highlights > threshold),
            "shadow_percent": float(shadows * 100),
            "highlight_percent": float(highlights * 100),
        }
