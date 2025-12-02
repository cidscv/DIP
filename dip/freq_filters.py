"""
Frequency Domain Filtering Module
Based on Week 8 lecture content - CP 8315 Digital Image Processing

Implements frequency-domain filters including:
- Ideal, Butterworth, and Gaussian filters (low-pass, high-pass)
- Band-pass and band-reject filters
- Spectrum visualization utilities
"""

import numpy as np
import cv2


def compute_dft(image):
    """
    Compute 2D DFT and shift zero frequency to center.
    
    Args:
        image: Input image (grayscale or will be converted)
        
    Returns:
        dft_shift: Centered frequency domain representation
        magnitude_spectrum: Log-scaled magnitude spectrum for visualization
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Compute DFT and shift to center
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Compute magnitude spectrum for visualization
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_spectrum = 20 * np.log(magnitude + 1)  # Log scale
    
    # Normalize for display
    magnitude_spectrum = np.uint8(255 * magnitude_spectrum / magnitude_spectrum.max())
    
    return dft_shift, magnitude_spectrum


def compute_idft(dft_shift):
    """
    Inverse DFT to convert back to spatial domain.
    
    Args:
        dft_shift: Centered frequency domain representation
        
    Returns:
        image: Reconstructed spatial domain image
    """
    # Inverse shift
    f_ishift = np.fft.ifftshift(dft_shift)
    
    # Inverse DFT
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    return np.clip(img_back, 0, 255).astype(np.uint8)


def create_ideal_lpf_mask(shape, cutoff_radius):
    """
    Create ideal low-pass filter mask with sharp cutoff.
    Note: Can cause ringing artifacts due to abrupt frequency transition.
    
    Args:
        shape: (rows, cols) of the image
        cutoff_radius: Cutoff frequency radius
        
    Returns:
        mask: Binary mask for frequency domain
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create coordinate grid
    y, x = np.ogrid[:rows, :cols]
    
    # Compute distance from center
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Ideal filter: 1 inside radius, 0 outside
    mask = (distance <= cutoff_radius).astype(np.float32)
    
    return mask


def create_butterworth_lpf_mask(shape, cutoff_radius, order=2):
    """
    Create Butterworth low-pass filter mask with smooth transition.
    Smoother than ideal filter, reduces ringing artifacts.
    
    Args:
        shape: (rows, cols) of the image
        cutoff_radius: Cutoff frequency radius
        order: Filter order (higher = sharper transition). Default=2
        
    Returns:
        mask: Butterworth filter mask
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create coordinate grid
    y, x = np.ogrid[:rows, :cols]
    
    # Compute distance from center
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Butterworth formula: H = 1 / (1 + (D/D0)^(2n))
    # Avoid division by zero at DC
    distance[distance == 0] = 1e-10
    mask = 1 / (1 + (distance / cutoff_radius)**(2 * order))
    
    return mask.astype(np.float32)


def create_gaussian_lpf_mask(shape, cutoff_radius):
    """
    Create Gaussian low-pass filter mask.
    Smooth exponential roll-off, no ringing artifacts.
    
    Args:
        shape: (rows, cols) of the image
        cutoff_radius: Cutoff frequency radius (standard deviation)
        
    Returns:
        mask: Gaussian filter mask
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create coordinate grid
    y, x = np.ogrid[:rows, :cols]
    
    # Compute distance from center
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Gaussian formula: H = exp(-(D^2)/(2*D0^2))
    mask = np.exp(-(distance**2) / (2 * cutoff_radius**2))
    
    return mask.astype(np.float32)


def create_highpass_mask(lpf_mask):
    """
    Create high-pass filter from low-pass filter.
    H_HP = 1 - H_LP
    
    Args:
        lpf_mask: Low-pass filter mask
        
    Returns:
        hpf_mask: High-pass filter mask
    """
    return 1 - lpf_mask


def create_bandpass_mask(shape, inner_radius, outer_radius, filter_type='butterworth', order=2):
    """
    Create band-pass filter mask to pass frequencies in a range.
    
    Args:
        shape: (rows, cols) of the image
        inner_radius: Inner cutoff radius (D1)
        outer_radius: Outer cutoff radius (D2)
        filter_type: 'ideal', 'butterworth', or 'gaussian'
        order: Filter order (for Butterworth only)
        
    Returns:
        mask: Band-pass filter mask
    """
    # Create two low-pass filters and subtract
    if filter_type == 'ideal':
        lpf_outer = create_ideal_lpf_mask(shape, outer_radius)
        lpf_inner = create_ideal_lpf_mask(shape, inner_radius)
    elif filter_type == 'butterworth':
        lpf_outer = create_butterworth_lpf_mask(shape, outer_radius, order)
        lpf_inner = create_butterworth_lpf_mask(shape, inner_radius, order)
    elif filter_type == 'gaussian':
        lpf_outer = create_gaussian_lpf_mask(shape, outer_radius)
        lpf_inner = create_gaussian_lpf_mask(shape, inner_radius)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Band-pass = outer LPF - inner LPF
    bandpass = lpf_outer - lpf_inner
    
    return bandpass


def create_bandreject_mask(shape, inner_radius, outer_radius, filter_type='butterworth', order=2):
    """
    Create band-reject (notch) filter mask to suppress frequencies in a range.
    Useful for removing periodic noise patterns.
    
    Args:
        shape: (rows, cols) of the image
        inner_radius: Inner cutoff radius (D1)
        outer_radius: Outer cutoff radius (D2)
        filter_type: 'ideal', 'butterworth', or 'gaussian'
        order: Filter order (for Butterworth only)
        
    Returns:
        mask: Band-reject filter mask
    """
    # Band-reject = 1 - band-pass
    bandpass = create_bandpass_mask(shape, inner_radius, outer_radius, filter_type, order)
    return 1 - bandpass


def apply_frequency_filter(image, filter_mask, return_spectrum=False):
    """
    Apply frequency domain filter to an image.
    
    Args:
        image: Input image (grayscale or color)
        filter_mask: Frequency domain filter mask
        return_spectrum: If True, also return before/after spectra
        
    Returns:
        filtered_image: Filtered result in spatial domain
        (optional) spectra: Dict with 'before' and 'after' magnitude spectra
    """
    # Handle color images by processing each channel
    if len(image.shape) == 3:
        channels = cv2.split(image)
        filtered_channels = []
        
        for channel in channels:
            # Compute DFT
            dft_shift, _ = compute_dft(channel)
            
            # Apply filter (multiply by mask in both real and imaginary parts)
            filtered_dft = dft_shift.copy()
            filtered_dft[:, :, 0] *= filter_mask
            filtered_dft[:, :, 1] *= filter_mask
            
            # Convert back to spatial domain
            filtered_channel = compute_idft(filtered_dft)
            filtered_channels.append(filtered_channel)
        
        filtered_image = cv2.merge(filtered_channels)
        
        if return_spectrum:
            # Return spectrum of first channel for visualization
            dft_shift_orig, spectrum_before = compute_dft(channels[0])
            filtered_dft = dft_shift_orig.copy()
            filtered_dft[:, :, 0] *= filter_mask
            filtered_dft[:, :, 1] *= filter_mask
            magnitude = cv2.magnitude(filtered_dft[:, :, 0], filtered_dft[:, :, 1])
            spectrum_after = 20 * np.log(magnitude + 1)
            spectrum_after = np.uint8(255 * spectrum_after / spectrum_after.max())
            
            return filtered_image, {
                'before': spectrum_before,
                'after': spectrum_after
            }
    else:
        # Grayscale processing
        dft_shift, spectrum_before = compute_dft(image)
        
        # Apply filter
        filtered_dft = dft_shift.copy()
        filtered_dft[:, :, 0] *= filter_mask
        filtered_dft[:, :, 1] *= filter_mask
        
        # Convert back to spatial domain
        filtered_image = compute_idft(filtered_dft)
        
        if return_spectrum:
            magnitude = cv2.magnitude(filtered_dft[:, :, 0], filtered_dft[:, :, 1])
            spectrum_after = 20 * np.log(magnitude + 1)
            spectrum_after = np.uint8(255 * spectrum_after / spectrum_after.max())
            
            return filtered_image, {
                'before': spectrum_before,
                'after': spectrum_after
            }
    
    return filtered_image


def apply_ideal_lpf(image, cutoff_radius, return_spectrum=False):
    """
    Apply ideal low-pass filter.
    Note: May cause ringing artifacts.
    
    Args:
        image: Input image
        cutoff_radius: Cutoff frequency radius
        return_spectrum: If True, return before/after spectra
        
    Returns:
        filtered_image: Filtered result
        (optional) spectra: Magnitude spectra before/after
    """
    shape = image.shape[:2]
    mask = create_ideal_lpf_mask(shape, cutoff_radius)
    return apply_frequency_filter(image, mask, return_spectrum)


def apply_butterworth_lpf(image, cutoff_radius, order=2, return_spectrum=False):
    """
    Apply Butterworth low-pass filter.
    Smoother transition than ideal, less ringing.
    
    Args:
        image: Input image
        cutoff_radius: Cutoff frequency radius
        order: Filter order (default=2)
        return_spectrum: If True, return before/after spectra
        
    Returns:
        filtered_image: Filtered result
        (optional) spectra: Magnitude spectra before/after
    """
    shape = image.shape[:2]
    mask = create_butterworth_lpf_mask(shape, cutoff_radius, order)
    return apply_frequency_filter(image, mask, return_spectrum)


def apply_gaussian_lpf(image, cutoff_radius, return_spectrum=False):
    """
    Apply Gaussian low-pass filter.
    Smooth exponential roll-off, no ringing.
    
    Args:
        image: Input image
        cutoff_radius: Cutoff frequency radius
        return_spectrum: If True, return before/after spectra
        
    Returns:
        filtered_image: Filtered result
        (optional) spectra: Magnitude spectra before/after
    """
    shape = image.shape[:2]
    mask = create_gaussian_lpf_mask(shape, cutoff_radius)
    return apply_frequency_filter(image, mask, return_spectrum)


def apply_highpass_filter(image, cutoff_radius, filter_type='gaussian', order=2, return_spectrum=False):
    """
    Apply high-pass filter to emphasize edges and details.
    H_HP = 1 - H_LP
    
    Args:
        image: Input image
        cutoff_radius: Cutoff frequency radius
        filter_type: 'ideal', 'butterworth', or 'gaussian'
        order: Filter order (for Butterworth only)
        return_spectrum: If True, return before/after spectra
        
    Returns:
        filtered_image: Filtered result
        (optional) spectra: Magnitude spectra before/after
    """
    shape = image.shape[:2]
    
    # Create low-pass mask then invert for high-pass
    if filter_type == 'ideal':
        lpf_mask = create_ideal_lpf_mask(shape, cutoff_radius)
    elif filter_type == 'butterworth':
        lpf_mask = create_butterworth_lpf_mask(shape, cutoff_radius, order)
    elif filter_type == 'gaussian':
        lpf_mask = create_gaussian_lpf_mask(shape, cutoff_radius)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    hpf_mask = create_highpass_mask(lpf_mask)
    return apply_frequency_filter(image, hpf_mask, return_spectrum)


def apply_bandpass_filter(image, inner_radius, outer_radius, 
                         filter_type='butterworth', order=2, return_spectrum=False):
    """
    Apply band-pass filter to isolate mid-frequency range.
    
    Args:
        image: Input image
        inner_radius: Inner cutoff radius (D1)
        outer_radius: Outer cutoff radius (D2)
        filter_type: 'ideal', 'butterworth', or 'gaussian'
        order: Filter order (for Butterworth only)
        return_spectrum: If True, return before/after spectra
        
    Returns:
        filtered_image: Filtered result
        (optional) spectra: Magnitude spectra before/after
    """
    shape = image.shape[:2]
    mask = create_bandpass_mask(shape, inner_radius, outer_radius, filter_type, order)
    return apply_frequency_filter(image, mask, return_spectrum)


def apply_bandreject_filter(image, inner_radius, outer_radius,
                           filter_type='butterworth', order=2, return_spectrum=False):
    """
    Apply band-reject filter to suppress periodic noise patterns.
    
    Args:
        image: Input image
        inner_radius: Inner cutoff radius (D1)
        outer_radius: Outer cutoff radius (D2)
        filter_type: 'ideal', 'butterworth', or 'gaussian'
        order: Filter order (for Butterworth only)
        return_spectrum: If True, return before/after spectra
        
    Returns:
        filtered_image: Filtered result
        (optional) spectra: Magnitude spectra before/after
    """
    shape = image.shape[:2]
    mask = create_bandreject_mask(shape, inner_radius, outer_radius, filter_type, order)
    return apply_frequency_filter(image, mask, return_spectrum)


def visualize_filter_mask(mask, title="Filter Mask"):
    """
    Visualize a frequency domain filter mask.
    
    Args:
        mask: Filter mask to visualize
        title: Plot title
        
    Returns:
        mask_vis: Uint8 image for display
    """
    mask_vis = np.uint8(255 * mask)
    return mask_vis
