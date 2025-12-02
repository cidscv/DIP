"""
Morphological Operations Module
Based on Week 7 & Week 9 lecture content - CP 8315 Digital Image Processing

Implements morphological operations for image post-processing:
- Basic operations: dilation, erosion, opening, closing
- Advanced operations: top-hat, bottom-hat, gradient
- Reconstruction operations
- Connected component analysis
"""

import numpy as np
import cv2


def create_structuring_element(shape='rect', size=(5, 5)):
    """
    Create a structuring element for morphological operations.
    
    Args:
        shape: 'rect', 'ellipse', or 'cross'
        size: (width, height) of the structuring element
        
    Returns:
        kernel: Structuring element
    """
    if shape == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    else:
        raise ValueError(f"Unknown shape: {shape}. Use 'rect', 'ellipse', or 'cross'")
    
    return kernel


def dilate(image, kernel_size=(5, 5), kernel_shape='rect', iterations=1):
    """
    Dilate image to expand bright regions.
    Adds pixels to boundaries of objects.
    
    Args:
        image: Input binary or grayscale image
        kernel_size: Size of structuring element
        kernel_shape: 'rect', 'ellipse', or 'cross'
        iterations: Number of times to apply dilation
        
    Returns:
        dilated: Dilated image
    """
    kernel = create_structuring_element(kernel_shape, kernel_size)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    return dilated


def erode(image, kernel_size=(5, 5), kernel_shape='rect', iterations=1):
    """
    Erode image to shrink bright regions.
    Removes pixels from boundaries of objects.
    
    Args:
        image: Input binary or grayscale image
        kernel_size: Size of structuring element
        kernel_shape: 'rect', 'ellipse', or 'cross'
        iterations: Number of times to apply erosion
        
    Returns:
        eroded: Eroded image
    """
    kernel = create_structuring_element(kernel_shape, kernel_size)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    return eroded


def opening(image, kernel_size=(5, 5), kernel_shape='rect', iterations=1):
    """
    Morphological opening: erosion followed by dilation.
    Removes small bright noise/speckles while preserving object shapes.
    Separates connected objects.
    
    Args:
        image: Input binary or grayscale image
        kernel_size: Size of structuring element
        kernel_shape: 'rect', 'ellipse', or 'cross'
        iterations: Number of times to apply operation
        
    Returns:
        opened: Opened image
    """
    kernel = create_structuring_element(kernel_shape, kernel_size)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opened


def closing(image, kernel_size=(5, 5), kernel_shape='rect', iterations=1):
    """
    Morphological closing: dilation followed by erosion.
    Fills small holes and gaps in objects.
    Merges nearby regions.
    
    Args:
        image: Input binary or grayscale image
        kernel_size: Size of structuring element
        kernel_shape: 'rect', 'ellipse', or 'cross'
        iterations: Number of times to apply operation
        
    Returns:
        closed: Closed image
    """
    kernel = create_structuring_element(kernel_shape, kernel_size)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def morphological_gradient(image, kernel_size=(5, 5), kernel_shape='rect'):
    """
    Morphological gradient: dilation - erosion.
    Highlights boundaries/edges of objects.
    
    Args:
        image: Input binary or grayscale image
        kernel_size: Size of structuring element
        kernel_shape: 'rect', 'ellipse', or 'cross'
        
    Returns:
        gradient: Morphological gradient image
    """
    kernel = create_structuring_element(kernel_shape, kernel_size)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return gradient


def top_hat(image, kernel_size=(9, 9), kernel_shape='rect'):
    """
    Top-hat transform: image - opening(image).
    Extracts small bright features/objects on dark background.
    Useful for extracting bright details.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of structuring element (should be larger than features)
        kernel_shape: 'rect', 'ellipse', or 'cross'
        
    Returns:
        tophat: Top-hat transformed image
    """
    kernel = create_structuring_element(kernel_shape, kernel_size)
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return tophat


def bottom_hat(image, kernel_size=(9, 9), kernel_shape='rect'):
    """
    Bottom-hat (black-hat) transform: closing(image) - image.
    Extracts small dark features/objects on bright background.
    Useful for extracting dark details.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of structuring element (should be larger than features)
        kernel_shape: 'rect', 'ellipse', or 'cross'
        
    Returns:
        blackhat: Bottom-hat transformed image
    """
    kernel = create_structuring_element(kernel_shape, kernel_size)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return blackhat


def opening_by_reconstruction(image, kernel_size=(5, 5), kernel_shape='rect'):
    """
    Opening by reconstruction: better preserves shape than standard opening.
    Performs erosion followed by morphological reconstruction.
    
    Args:
        image: Input binary or grayscale image
        kernel_size: Size of structuring element
        kernel_shape: 'rect', 'ellipse', or 'cross'
        
    Returns:
        reconstructed: Opened image with better shape preservation
    """
    kernel = create_structuring_element(kernel_shape, kernel_size)
    
    # Erode to create marker
    eroded = cv2.erode(image, kernel)
    
    # Reconstruct by dilation
    reconstructed = reconstruct_dilation(eroded, image)
    
    return reconstructed


def closing_by_reconstruction(image, kernel_size=(5, 5), kernel_shape='rect'):
    """
    Closing by reconstruction: better preserves shape than standard closing.
    Performs dilation followed by morphological reconstruction.
    
    Args:
        image: Input binary or grayscale image
        kernel_size: Size of structuring element
        kernel_shape: 'rect', 'ellipse', or 'cross'
        
    Returns:
        reconstructed: Closed image with better shape preservation
    """
    kernel = create_structuring_element(kernel_shape, kernel_size)
    
    # Dilate to create marker
    dilated = cv2.dilate(image, kernel)
    
    # Reconstruct by erosion
    reconstructed = reconstruct_erosion(dilated, image)
    
    return reconstructed


def reconstruct_dilation(marker, mask, kernel_size=(3, 3), max_iterations=100):
    """
    Morphological reconstruction by dilation.
    Geodesic dilation: iteratively dilate marker constrained by mask.
    
    Args:
        marker: Marker image (starting point)
        mask: Mask image (constraint)
        kernel_size: Size of structuring element
        max_iterations: Maximum number of iterations
        
    Returns:
        reconstructed: Reconstructed image
    """
    kernel = create_structuring_element('rect', kernel_size)
    
    reconstructed = marker.copy()
    for _ in range(max_iterations):
        previous = reconstructed.copy()
        
        # Dilate marker
        dilated = cv2.dilate(reconstructed, kernel)
        
        # Take minimum with mask (geodesic constraint)
        reconstructed = cv2.min(dilated, mask)
        
        # Check for convergence
        if np.array_equal(reconstructed, previous):
            break
    
    return reconstructed


def reconstruct_erosion(marker, mask, kernel_size=(3, 3), max_iterations=100):
    """
    Morphological reconstruction by erosion.
    Geodesic erosion: iteratively erode marker constrained by mask.
    
    Args:
        marker: Marker image (starting point)
        mask: Mask image (constraint)
        kernel_size: Size of structuring element
        max_iterations: Maximum number of iterations
        
    Returns:
        reconstructed: Reconstructed image
    """
    kernel = create_structuring_element('rect', kernel_size)
    
    reconstructed = marker.copy()
    for _ in range(max_iterations):
        previous = reconstructed.copy()
        
        # Erode marker
        eroded = cv2.erode(reconstructed, kernel)
        
        # Take maximum with mask (geodesic constraint)
        reconstructed = cv2.max(eroded, mask)
        
        # Check for convergence
        if np.array_equal(reconstructed, previous):
            break
    
    return reconstructed


def remove_small_objects(binary_image, min_size=50):
    """
    Remove small connected components (noise removal).
    
    Args:
        binary_image: Binary input image
        min_size: Minimum object size in pixels
        
    Returns:
        cleaned: Image with small objects removed
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    
    # Create output image
    cleaned = np.zeros_like(binary_image)
    
    # Keep only large enough components (skip background label 0)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    
    return cleaned


def fill_holes(binary_image, max_hole_size=None):
    """
    Fill holes in binary objects.
    
    Args:
        binary_image: Binary input image
        max_hole_size: Maximum hole size to fill (None = fill all)
        
    Returns:
        filled: Image with holes filled
    """
    # Invert image
    inverted = cv2.bitwise_not(binary_image)
    
    # Find connected components (holes are now objects)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    
    # Create output image
    filled = binary_image.copy()
    
    # Fill holes (skip background label 0 which is the actual objects)
    for i in range(1, num_labels):
        hole_size = stats[i, cv2.CC_STAT_AREA]
        if max_hole_size is None or hole_size <= max_hole_size:
            filled[labels == i] = 255
    
    return filled


def connected_components_analysis(binary_image, min_area=0):
    """
    Analyze connected components in binary image.
    
    Args:
        binary_image: Binary input image
        min_area: Minimum component area to include
        
    Returns:
        num_objects: Number of objects found
        labeled_image: Image with each object labeled uniquely
        stats: Statistics for each component (area, bbox, etc.)
        centroids: Centroids of each component
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    
    # Filter by minimum area
    if min_area > 0:
        valid_labels = [0]  # Keep background
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                valid_labels.append(i)
        
        # Create filtered label image
        mask = np.isin(labels, valid_labels)
        labels_filtered = labels * mask
        
        # Remap labels to be consecutive
        unique_labels = np.unique(labels_filtered)
        labels_remapped = np.zeros_like(labels)
        for new_label, old_label in enumerate(unique_labels):
            labels_remapped[labels_filtered == old_label] = new_label
        
        num_objects = len(unique_labels) - 1  # Exclude background
        return num_objects, labels_remapped, stats[valid_labels], centroids[valid_labels]
    else:
        num_objects = num_labels - 1  # Exclude background
        return num_objects, labels, stats, centroids


def border_clearing(binary_image, border_size=1):
    """
    Remove objects touching image borders.
    
    Args:
        binary_image: Binary input image
        border_size: Size of border region to check
        
    Returns:
        cleared: Image with border objects removed
    """
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_image, connectivity=8)
    
    # Find labels touching borders
    h, w = binary_image.shape
    border_labels = set()
    
    # Top and bottom borders
    border_labels.update(labels[:border_size, :].flatten())
    border_labels.update(labels[-border_size:, :].flatten())
    
    # Left and right borders
    border_labels.update(labels[:, :border_size].flatten())
    border_labels.update(labels[:, -border_size:].flatten())
    
    # Remove background label
    border_labels.discard(0)
    
    # Create output image
    cleared = binary_image.copy()
    for label in border_labels:
        cleared[labels == label] = 0
    
    return cleared


def skeletonize(binary_image, method='thinning'):
    """
    Compute morphological skeleton of binary objects.
    Reduces objects to 1-pixel-wide representations.
    
    Args:
        binary_image: Binary input image
        method: 'thinning' or 'medial_axis'
        
    Returns:
        skeleton: Skeletonized image
    """
    if method == 'thinning':
        # Use OpenCV thinning
        skeleton = cv2.ximgproc.thinning(binary_image)
        return skeleton
    elif method == 'medial_axis':
        # Use scikit-image medial axis
        from skimage.morphology import medial_axis
        skeleton = medial_axis(binary_image > 0)
        return (skeleton * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")


def cleanup_segmentation(binary_image, 
                         remove_small=True, min_size=50,
                         fill_holes_flag=True, max_hole_size=None,
                         opening_size=None, closing_size=None,
                         clear_border=False):
    """
    Complete cleanup pipeline for segmentation masks.
    Combines multiple morphological operations.
    
    Args:
        binary_image: Binary segmentation mask
        remove_small: Remove small objects
        min_size: Minimum object size to keep
        fill_holes_flag: Fill holes in objects
        max_hole_size: Maximum hole size to fill
        opening_size: Kernel size for opening (None to skip)
        closing_size: Kernel size for closing (None to skip)
        clear_border: Remove objects touching borders
        
    Returns:
        cleaned: Cleaned segmentation mask
    """
    cleaned = binary_image.copy()
    
    # Apply opening if specified
    if opening_size is not None:
        cleaned = opening(cleaned, kernel_size=(opening_size, opening_size))
    
    # Apply closing if specified
    if closing_size is not None:
        cleaned = closing(cleaned, kernel_size=(closing_size, closing_size))
    
    # Remove small objects
    if remove_small:
        cleaned = remove_small_objects(cleaned, min_size=min_size)
    
    # Fill holes
    if fill_holes_flag:
        cleaned = fill_holes(cleaned, max_hole_size=max_hole_size)
    
    # Clear border objects
    if clear_border:
        cleaned = border_clearing(cleaned)
    
    return cleaned
