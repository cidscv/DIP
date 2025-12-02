# Quick Reference Guide - Assignment 2 Code Examples

## 🚀 Getting Started

```python
import cv2
from dip import freq_filters, segmentation, morphology, metrics
```

## 📊 Frequency Domain Filtering

### Basic Low-Pass Filtering
```python
# Read image
image = cv2.imread('input.jpg')

# Option 1: Gaussian (recommended - no ringing)
filtered = freq_filters.apply_gaussian_lpf(image, cutoff_radius=30)

# Option 2: Butterworth (smooth transition)
filtered = freq_filters.apply_butterworth_lpf(image, cutoff_radius=30, order=2)

# Option 3: Ideal (sharp cutoff - may cause ringing)
filtered = freq_filters.apply_ideal_lpf(image, cutoff_radius=30)
```

### High-Pass Filtering (Edge Enhancement)
```python
# Emphasize edges and details
edges = freq_filters.apply_highpass_filter(
    image, 
    cutoff_radius=30,
    filter_type='gaussian'
)
```

### Band-Pass/Band-Reject
```python
# Isolate mid-frequency textures
bandpass = freq_filters.apply_bandpass_filter(
    image,
    inner_radius=20,
    outer_radius=60,
    filter_type='butterworth',
    order=2
)

# Suppress periodic noise patterns
bandreject = freq_filters.apply_bandreject_filter(
    image,
    inner_radius=20,
    outer_radius=60,
    filter_type='butterworth',
    order=2
)
```

### Getting Spectrum Visualization
```python
# Get before/after spectra
filtered, spectra = freq_filters.apply_gaussian_lpf(
    image, 
    cutoff_radius=30,
    return_spectrum=True
)

# Save spectrum images
cv2.imwrite('spectrum_before.jpg', spectra['before'])
cv2.imwrite('spectrum_after.jpg', spectra['after'])
```

## 🎯 Segmentation Methods

### Otsu Thresholding (Automatic)
```python
# Simple global thresholding
binary, threshold_value = segmentation.otsu_threshold(image)
print(f"Computed threshold: {threshold_value}")

# Inverted (bright background, dark objects)
binary_inv, threshold = segmentation.otsu_threshold(image, invert=True)
```

### Adaptive Thresholding (Uneven Lighting)
```python
# Mean adaptive
binary = segmentation.adaptive_threshold_mean(
    image,
    block_size=11,  # Must be odd (11, 15, 21, etc.)
    C=2             # Fine-tuning constant
)

# Gaussian adaptive (more robust to noise)
binary = segmentation.adaptive_threshold_gaussian(
    image,
    block_size=11,
    C=2
)
```

### K-Means Clustering (Color-Based)
```python
# Segment into K color clusters
segmented, centers = segmentation.kmeans_segmentation(
    image,
    k=3,            # Number of clusters (try 2-5)
    attempts=10
)

# Get label map instead
labels, centers = segmentation.kmeans_segmentation(
    image,
    k=3,
    return_labels=True
)
```

### Region Growing
```python
# Flood fill from seed point
mask, filled = segmentation.region_growing_floodfill(
    image,
    seed_point=(100, 150),  # (x, y) coordinates
    tolerance=10,            # Intensity difference threshold
    connectivity=4           # 4 or 8
)

# Custom region growing
mask = segmentation.region_growing_custom(
    image,
    seed_point=(100, 150),
    threshold=15
)
```

## 🔧 Morphological Operations

### Basic Operations
```python
# Dilate (expand bright regions)
dilated = morphology.dilate(
    binary_image,
    kernel_size=(5, 5),
    kernel_shape='rect'  # 'rect', 'ellipse', or 'cross'
)

# Erode (shrink bright regions)
eroded = morphology.erode(binary_image, kernel_size=(5, 5))

# Opening (remove noise)
opened = morphology.opening(binary_image, kernel_size=(5, 5))

# Closing (fill holes)
closed = morphology.closing(binary_image, kernel_size=(5, 5))
```

### Advanced Operations
```python
# Top-hat (extract bright features)
tophat = morphology.top_hat(
    grayscale_image,
    kernel_size=(15, 15)  # Larger than features
)

# Bottom-hat (extract dark features)
bottomhat = morphology.bottom_hat(
    grayscale_image,
    kernel_size=(15, 15)
)

# Morphological gradient (boundaries)
gradient = morphology.morphological_gradient(
    image,
    kernel_size=(3, 3)
)
```

### Complete Cleanup Pipeline
```python
# One function to clean up segmentation
cleaned = morphology.cleanup_segmentation(
    binary_image,
    remove_small=True,
    min_size=100,
    fill_holes_flag=True,
    max_hole_size=500,
    opening_size=3,
    closing_size=5,
    clear_border=False
)
```

### Connected Components Analysis
```python
# Analyze objects in binary image
num_objects, labels, stats, centroids = morphology.connected_components_analysis(
    binary_image,
    min_area=50  # Minimum object size
)

print(f"Found {num_objects} objects")

# stats contains for each object:
# - Area (stats[i, cv2.CC_STAT_AREA])
# - Bounding box (stats[i, cv2.CC_STAT_LEFT/TOP/WIDTH/HEIGHT])
```

## 📏 Evaluation Metrics

### Quick Evaluation
```python
# Compute all metrics at once
all_metrics = metrics.compute_all_metrics(
    predicted_mask,
    ground_truth_mask
)

# Print nicely formatted results
metrics.print_metrics(all_metrics, title="My Segmentation Results")
```

### Individual Metrics
```python
# Dice coefficient
dice = metrics.dice_coefficient(pred_mask, gt_mask)

# IoU (Jaccard index)
iou = metrics.iou_score(pred_mask, gt_mask)

# Pixel accuracy
accuracy = metrics.pixel_accuracy(pred_mask, gt_mask)

# Precision and recall
precision, recall = metrics.precision_recall(pred_mask, gt_mask)

# Confusion matrix
components = metrics.confusion_matrix_components(pred_mask, gt_mask)
print(f"TP: {components['tp']}, FP: {components['fp']}")
print(f"FN: {components['fn']}, TN: {components['tn']}")
```

### Visualization
```python
# Create overlay visualization
# Green = correct, Red = false positives, Blue = false negatives
overlay = metrics.overlay_segmentation(
    original_image,
    predicted_mask,
    ground_truth_mask,
    alpha=0.5
)

cv2.imwrite('segmentation_overlay.jpg', overlay)
```

## 🔄 Complete Pipeline Examples

### Pipeline A: Baseline (From Assignment)
```python
# 1. Low-pass pre-processing
filtered = freq_filters.apply_gaussian_lpf(image, cutoff_radius=30)

# 2. Global thresholding
binary, threshold = segmentation.otsu_threshold(filtered)

# 3. Morphological cleanup
opened = morphology.opening(binary, kernel_size=(3, 3))
closed = morphology.closing(opened, kernel_size=(5, 5))
final = morphology.remove_small_objects(closed, min_size=100)

# 4. Evaluate
if ground_truth is not None:
    scores = metrics.compute_all_metrics(final, ground_truth)
    metrics.print_metrics(scores, title="Pipeline A Results")
```

### Pipeline B: Problem-Oriented Examples

#### For Periodic Noise:
```python
# 1. Band-reject filtering
filtered = freq_filters.apply_bandreject_filter(
    image, 
    inner_radius=20, 
    outer_radius=60
)

# 2. Adaptive thresholding
binary = segmentation.adaptive_threshold_gaussian(filtered, block_size=15, C=2)

# 3. Cleanup
final = morphology.cleanup_segmentation(binary)
```

#### For Color-Based Segmentation:
```python
# 1. Light pre-filtering
filtered = freq_filters.apply_gaussian_lpf(image, cutoff_radius=50)

# 2. K-means clustering
segmented, centers = segmentation.kmeans_segmentation(filtered, k=3)

# 3. Convert to binary (select dominant cluster)
labels, _ = segmentation.kmeans_segmentation(filtered, k=3, return_labels=True)
binary = (labels == 1).astype(np.uint8) * 255  # Select cluster 1

# 4. Cleanup
final = morphology.cleanup_segmentation(binary)
```

#### For Low Contrast with Uneven Lighting:
```python
# 1. Top-hat to extract bright features
tophat = morphology.top_hat(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 
                            kernel_size=(15, 15))

# 2. Adaptive thresholding on top-hat result
binary = segmentation.adaptive_threshold_gaussian(tophat, block_size=11, C=2)

# 3. Cleanup
final = morphology.cleanup_segmentation(binary)
```

## 🎨 Ablation Study Examples

### Compare Different Filters
```python
# Test different cutoff radii for Gaussian LPF
results = {}
for cutoff in [20, 30, 40, 50]:
    filtered = freq_filters.apply_gaussian_lpf(image, cutoff_radius=cutoff)
    binary, _ = segmentation.otsu_threshold(filtered)
    cleaned = morphology.cleanup_segmentation(binary)
    
    results[f'cutoff_{cutoff}'] = cleaned
    
    # Evaluate
    if ground_truth is not None:
        scores = metrics.compute_all_metrics(cleaned, ground_truth)
        print(f"\nCutoff Radius = {cutoff}")
        print(f"Dice: {scores['dice']:.4f}, IoU: {scores['iou']:.4f}")
```

### Compare Filter Types
```python
# Compare Ideal vs Butterworth vs Gaussian
filters = {
    'ideal': freq_filters.apply_ideal_lpf(image, 30),
    'butterworth': freq_filters.apply_butterworth_lpf(image, 30, order=2),
    'gaussian': freq_filters.apply_gaussian_lpf(image, 30)
}

for name, filtered in filters.items():
    binary, _ = segmentation.otsu_threshold(filtered)
    cleaned = morphology.cleanup_segmentation(binary)
    
    if ground_truth is not None:
        scores = metrics.compute_all_metrics(cleaned, ground_truth)
        print(f"\n{name.capitalize()} Filter:")
        print(f"Dice: {scores['dice']:.4f}")
```

### Compare Thresholding Methods
```python
# Compare Otsu vs Adaptive
methods = {
    'otsu': segmentation.otsu_threshold(image)[0],
    'adaptive_mean': segmentation.adaptive_threshold_mean(image, block_size=11),
    'adaptive_gaussian': segmentation.adaptive_threshold_gaussian(image, block_size=11)
}

for name, binary in methods.items():
    cleaned = morphology.cleanup_segmentation(binary)
    scores = metrics.compute_all_metrics(cleaned, ground_truth)
    print(f"\n{name}: Dice={scores['dice']:.4f}, IoU={scores['iou']:.4f}")
```

## 💾 Saving Results

```python
# Save processed image
cv2.imwrite('output/filtered.jpg', filtered)

# Save binary mask
cv2.imwrite('output/segmentation_mask.png', binary_mask)

# Save visualization overlay
overlay = metrics.overlay_segmentation(original, predicted, ground_truth)
cv2.imwrite('output/overlay.jpg', overlay)

# Save spectrum
_, spectrum = freq_filters.compute_dft(image)
cv2.imwrite('output/magnitude_spectrum.jpg', spectrum)
```

## 🔍 Extracting Crops for Report

```python
# Extract 200-400 pixel crops of interesting regions
h, w = image.shape[:2]
crop_size = 300

# Top-left crop
crop = image[0:crop_size, 0:crop_size]
cv2.imwrite('output/crop_topleft.jpg', crop)

# Center crop showing boundary detail
center_y, center_x = h // 2, w // 2
half_crop = crop_size // 2
crop = image[center_y-half_crop:center_y+half_crop, 
             center_x-half_crop:center_x+half_crop]
cv2.imwrite('output/crop_detail.jpg', crop)
```

## 📊 Logging Decision Process

```python
# Document all parameters for reproducibility
log = {
    'image': 'photo_001.jpg',
    'pipeline': 'A',
    'steps': [
        {'operation': 'gaussian_lpf', 'cutoff_radius': 30},
        {'operation': 'otsu_threshold', 'threshold_computed': 127},
        {'operation': 'morphology_opening', 'kernel_size': (3, 3)},
        {'operation': 'morphology_closing', 'kernel_size': (5, 5)},
        {'operation': 'remove_small_objects', 'min_size': 100}
    ],
    'metrics': {
        'dice': 0.8542,
        'iou': 0.7456,
        'pixel_accuracy': 0.9123
    }
}

import json
with open('output/decision_log.json', 'w') as f:
    json.dump(log, f, indent=2)
```

---

## 🎓 Assignment-Specific Notes

### For Part C (Evidence):
- Always save spectrum images when doing frequency filtering
- Save both original and processed for side-by-side comparison
- Extract crops showing important boundaries/textures
- Document ALL parameter values in decision log

### For Part D (Literature Review):
- Look for papers on:
  - Frequency domain filtering techniques
  - Otsu thresholding and adaptive methods
  - K-means for image segmentation
  - Morphological operations for post-processing

### Recommended Metrics to Report:
- **Primary:** Dice coefficient and IoU
- **Secondary:** Precision and Recall
- **Qualitative:** Visual overlay showing FP/FN regions

---

**Created:** December 2024  
**Course:** CP 8315 - Digital Image Processing  
**Assignment:** Take-Home Project 2
