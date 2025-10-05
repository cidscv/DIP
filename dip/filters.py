import numpy as np
import cv2


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


def apply_gaussian_filter(image, radius=3.0):
    """
    Soft blur to reduce noise while better preserving edges
    radius=3.0: Small radius to smooth image without over-blurring
    """
    return cv2.GaussianBlur(image, (0, 0), radius)


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


def apply_clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge([l_clahe, a, b])

    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return result


def high_boost_filter(image):

    img_float = image.astype(np.float64)
    img_blur = cv2.GaussianBlur(img_float, (15, 15), 0, 0)

    k = 2.5

    high_boost_float = (1 + k) * img_float - k * img_blur
    high_boost_img = cv2.convertScaleAbs(high_boost_float)

    return high_boost_img


def apply_median_filter(image, kernel_size=3):
    """
    kernel_size=3: Smallest effective size for noise reduction
                   Larger kernels would blur important details
                   3x3 balances noise suppression with detail preservation
                   Median filter chosen over Gaussian to preserve edges
    """
    return cv2.medianBlur(image, kernel_size)


def white_balance_correction(image):
    result = image.copy().astype(np.float32)

    # Calculate average for each channel
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    # Find the overall average
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Scale each channel
    result[:, :, 0] = result[:, :, 0] * (avg_gray / avg_b)
    result[:, :, 1] = result[:, :, 1] * (avg_gray / avg_g)
    result[:, :, 2] = result[:, :, 2] * (avg_gray / avg_r)

    return np.clip(result, 0, 255).astype(np.uint8)
