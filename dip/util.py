import numpy as np


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
