import numpy as np
import os
import json
from PIL import Image
from PIL.ExifTags import TAGS


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


def extract_exif(image_path, image_output_dir, name=""):
    """
    Extract EXIF data from an image
    Args:
        image_path: path to image file
    Returns:
        dict: EXIF data with human-readable tags
    """
    image = Image.open(image_path)
    exif_data = image.getexif()

    if exif_data is None:
        return {"error": "No EXIF data found"}

    # Convert EXIF data to readable format
    exif_dict = {}
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        exif_dict[tag] = str(value)

    # Get ISO
    exif_ifd = exif_data.get_ifd(0x8769)

    if exif_ifd:
        for tag_id, value in exif_ifd.items():
            tag = TAGS.get(tag_id, tag_id)
            exif_dict[tag] = str(value)

    # Save to JSON file
    output_dir = f"{image_output_dir}/exif_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_exif_data.json")

    with open(output_path, "w") as f:
        json.dump(exif_dict, f, indent=4)

    print(f"\nEXIF data saved to: {output_path}")

    return exif_dict
