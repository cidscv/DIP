import numpy as np
import cv2
from sklearn.cluster import KMeans


def manual_threshold(image, threshold_value, invert=False):
    """
    Apply manual global thresholding.
    
    Args:
        image: Input image (will be converted to grayscale if color)
        threshold_value: Threshold value (0-255)
        invert: If True, invert the binary mask
        
    Returns:
        binary: Binary segmentation mask
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, threshold_value, 255, thresh_type)
    
    return binary


def otsu_threshold(image, invert=False):
    """
    Apply Otsu's automatic thresholding.
    Automatically computes optimal threshold by minimizing within-class variance.
    
    Args:
        image: Input image (will be converted to grayscale if color)
        invert: If True, invert the binary mask
        
    Returns:
        binary: Binary segmentation mask
        threshold_value: Computed threshold value
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Otsu thresholding
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    threshold_value, binary = cv2.threshold(
        gray, 0, 255, thresh_type + cv2.THRESH_OTSU
    )
    
    return binary, threshold_value


def adaptive_threshold_mean(image, block_size=11, C=2, invert=False):
    """
    Apply adaptive thresholding using mean of neighborhood.
    Handles varying lighting conditions by computing local thresholds.
    
    Args:
        image: Input image (will be converted to grayscale if color)
        block_size: Size of neighborhood (must be odd, e.g., 11, 15, 21)
        C: Constant subtracted from mean (fine-tunes threshold)
        invert: If True, invert the binary mask
        
    Returns:
        binary: Binary segmentation mask
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Apply adaptive thresholding
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh_type, block_size, C
    )
    
    return binary


def adaptive_threshold_gaussian(image, block_size=11, C=2, invert=False):
    """
    Apply adaptive thresholding using Gaussian-weighted mean.
    More robust to noise than simple mean.
    
    Args:
        image: Input image (will be converted to grayscale if color)
        block_size: Size of neighborhood (must be odd, e.g., 11, 15, 21)
        C: Constant subtracted from weighted mean
        invert: If True, invert the binary mask
        
    Returns:
        binary: Binary segmentation mask
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Apply adaptive thresholding
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block_size, C
    )
    
    return binary


def kmeans_segmentation(image, k=3, attempts=10, return_labels=False):
    """
    Segment image using K-means clustering in color space.
    Groups pixels by color/intensity similarity.
    
    Args:
        image: Input image (color or grayscale)
        k: Number of clusters (2-8 typical)
        attempts: Number of k-means runs with different initializations
        return_labels: If True, return label map instead of colored segmentation
        
    Returns:
        segmented: Segmented image with cluster colors
        (optional) labels: Label map if return_labels=True
        centers: Cluster centers
    """
    # Reshape image to 2D array of pixels
    if len(image.shape) == 3:
        pixel_values = image.reshape((-1, 3))
    else:
        pixel_values = image.reshape((-1, 1))
    
    # Convert to float32
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Convert back to uint8
    centers = np.uint8(centers)
    
    if return_labels:
        # Return label map
        labels_map = labels.reshape(image.shape[:2])
        return labels_map, centers
    else:
        # Create segmented image by replacing pixels with cluster centers
        segmented = centers[labels.flatten()]
        segmented = segmented.reshape(image.shape)
        return segmented, centers


def region_growing_floodfill(image, seed_point, tolerance=5, connectivity=4):
    """
    Segment region using flood fill (region growing) from a seed point.
    Grows region by adding neighboring pixels with similar intensity.
    
    Args:
        image: Input image (grayscale or color)
        seed_point: (x, y) coordinates of seed point
        tolerance: Max intensity difference to include in region
        connectivity: 4 or 8 (neighbor connectivity)
        
    Returns:
        mask: Binary mask of segmented region
        seed_fill: Image with filled region highlighted
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create mask (must be 2 pixels larger in each dimension)
    h, w = gray.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Make a copy for flood fill
    seed_fill = gray.copy()
    
    # Set up flood fill parameters
    flags = connectivity
    if connectivity == 4:
        flags = 4
    elif connectivity == 8:
        flags = 8
    
    flags |= cv2.FLOODFILL_MASK_ONLY
    flags |= (255 << 8)  # Fill with 255 in mask
    
    # Perform flood fill
    lo_diff = (tolerance,)
    up_diff = (tolerance,)
    
    cv2.floodFill(
        seed_fill, mask, seed_point, 255, lo_diff, up_diff, flags
    )
    
    # Extract the mask (remove the 2-pixel border)
    mask = mask[1:-1, 1:-1]
    
    return mask, seed_fill


def region_growing_custom(image, seed_point, threshold=10, max_iterations=10000):
    """
    Custom implementation of region growing algorithm.
    More flexible than floodFill, allows custom similarity criteria.
    
    Args:
        image: Input grayscale image
        seed_point: (x, y) coordinates of seed point
        threshold: Max intensity difference for region membership
        max_iterations: Safety limit on iterations
        
    Returns:
        mask: Binary mask of segmented region
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Initialize with seed point
    seed_x, seed_y = seed_point
    seed_value = gray[seed_y, seed_x]
    
    # List of pixels to check
    to_check = [(seed_x, seed_y)]
    checked = set()
    
    iterations = 0
    while to_check and iterations < max_iterations:
        x, y = to_check.pop(0)
        
        # Skip if already checked
        if (x, y) in checked:
            continue
        
        checked.add((x, y))
        
        # Check if pixel is similar enough to seed
        if abs(int(gray[y, x]) - int(seed_value)) <= threshold:
            mask[y, x] = 255
            
            # Add neighbors (4-connectivity)
            neighbors = [
                (x+1, y), (x-1, y), (x, y+1), (x, y-1)
            ]
            
            for nx, ny in neighbors:
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in checked:
                    to_check.append((nx, ny))
        
        iterations += 1
    
    return mask


def color_based_threshold(image, lower_bound, upper_bound, color_space='BGR'):
    """
    Threshold image based on color range.
    Useful for isolating specific colors (e.g., green foliage, red objects).
    
    Args:
        image: Input color image
        lower_bound: Lower color bound (tuple of 3 values)
        upper_bound: Upper color bound (tuple of 3 values)
        color_space: 'BGR', 'HSV', or 'LAB'
        
    Returns:
        mask: Binary mask where pixels fall within color range
    """
    # Convert color space if needed
    if color_space == 'HSV':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'LAB':
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif color_space == 'BGR':
        converted = image.copy()
    else:
        raise ValueError(f"Unknown color space: {color_space}")
    
    # Create mask
    lower = np.array(lower_bound, dtype=np.uint8)
    upper = np.array(upper_bound, dtype=np.uint8)
    mask = cv2.inRange(converted, lower, upper)
    
    return mask


def multi_otsu_threshold(image, n_classes=3):
    """
    Multi-level Otsu thresholding for segmenting into multiple classes.
    
    Args:
        image: Input grayscale image
        n_classes: Number of classes (2-5 typical)
        
    Returns:
        segmented: Labeled image with class assignments
        thresholds: List of threshold values
    """
    from skimage.filters import threshold_multiotsu
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Compute multi-level thresholds
    thresholds = threshold_multiotsu(gray, classes=n_classes)
    
    # Apply thresholds to create regions
    segmented = np.digitize(gray, bins=thresholds)
    
    # Scale to 0-255 range for visualization
    segmented = np.uint8(255 * segmented / (n_classes - 1))
    
    return segmented, thresholds


def evaluate_segmentation_histogram(image, plot=False):
    """
    Evaluate if image is suitable for global thresholding by checking histogram.
    Bimodal histograms work best with global thresholding.
    
    Args:
        image: Input grayscale image
        plot: If True, return histogram data for plotting
        
    Returns:
        is_bimodal: Boolean indicating if histogram appears bimodal
        histogram: Histogram data if plot=True
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    
    # Simple bimodal detection: look for two peaks
    # Smooth histogram first
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist, sigma=2)
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hist_smooth, distance=20, prominence=100)
    
    is_bimodal = len(peaks) >= 2
    
    if plot:
        return is_bimodal, hist
    else:
        return is_bimodal


def auto_select_threshold_method(image):
    """
    Automatically recommend thresholding method based on image characteristics.
    
    Args:
        image: Input image
        
    Returns:
        recommendation: String describing recommended method
        params: Dict of suggested parameters
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Check if bimodal
    is_bimodal = evaluate_segmentation_histogram(gray)
    
    # Compute local variance to detect illumination variations
    mean_filter = cv2.blur(gray, (15, 15))
    variance = np.var(mean_filter)
    
    if is_bimodal and variance < 1000:
        return "Otsu's thresholding", {"method": "otsu"}
    elif variance > 2000:
        return "Adaptive thresholding (Gaussian)", {
            "method": "adaptive_gaussian",
            "block_size": 11,
            "C": 2
        }
    else:
        return "K-means clustering", {"method": "kmeans", "k": 3}
