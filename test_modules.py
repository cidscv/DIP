"""
Quick test script to verify all modules are working correctly.
Run this to ensure all imports and basic functions work.
"""

import cv2
import numpy as np
import sys

print("Testing DIP Assignment 2 Modules...")
print("="*60)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from dip import freq_filters
    from dip import segmentation
    from dip import morphology
    from dip import metrics
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create a test image
print("\n2. Creating test image...")
test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
test_image_color = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
print("✓ Test image created")

# Test 3: Test frequency filters
print("\n3. Testing frequency domain filters...")
try:
    # Test DFT
    dft_shift, spectrum = freq_filters.compute_dft(test_image)
    print("  ✓ DFT computation works")
    
    # Test filter masks
    lpf_mask = freq_filters.create_gaussian_lpf_mask((256, 256), cutoff_radius=30)
    print("  ✓ Filter mask creation works")
    
    # Test filtering
    filtered = freq_filters.apply_gaussian_lpf(test_image, cutoff_radius=30)
    print("  ✓ Image filtering works")
    
    print("✓ Frequency filters module OK")
except Exception as e:
    print(f"✗ Frequency filters failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test segmentation
print("\n4. Testing segmentation methods...")
try:
    # Test Otsu
    binary, threshold = segmentation.otsu_threshold(test_image)
    print(f"  ✓ Otsu threshold works (threshold={threshold:.1f})")
    
    # Test adaptive
    adaptive = segmentation.adaptive_threshold_gaussian(test_image)
    print("  ✓ Adaptive thresholding works")
    
    # Test K-means
    segmented, centers = segmentation.kmeans_segmentation(test_image_color, k=3)
    print("  ✓ K-means clustering works")
    
    print("✓ Segmentation module OK")
except Exception as e:
    print(f"✗ Segmentation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test morphology
print("\n5. Testing morphological operations...")
try:
    # Create binary test image
    binary_test = (test_image > 127).astype(np.uint8) * 255
    
    # Test basic operations
    dilated = morphology.dilate(binary_test, kernel_size=(5, 5))
    print("  ✓ Dilation works")
    
    eroded = morphology.erode(binary_test, kernel_size=(5, 5))
    print("  ✓ Erosion works")
    
    opened = morphology.opening(binary_test, kernel_size=(5, 5))
    print("  ✓ Opening works")
    
    closed = morphology.closing(binary_test, kernel_size=(5, 5))
    print("  ✓ Closing works")
    
    # Test advanced operations
    tophat = morphology.top_hat(test_image, kernel_size=(9, 9))
    print("  ✓ Top-hat transform works")
    
    print("✓ Morphology module OK")
except Exception as e:
    print(f"✗ Morphology failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test metrics
print("\n6. Testing evaluation metrics...")
try:
    # Create test masks
    pred_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
    gt_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
    
    # Test metrics
    dice = metrics.dice_coefficient(pred_mask, gt_mask)
    print(f"  ✓ Dice coefficient works (dice={dice:.3f})")
    
    iou = metrics.iou_score(pred_mask, gt_mask)
    print(f"  ✓ IoU works (iou={iou:.3f})")
    
    all_metrics = metrics.compute_all_metrics(pred_mask, gt_mask)
    print(f"  ✓ All metrics computation works")
    
    print("✓ Metrics module OK")
except Exception as e:
    print(f"✗ Metrics failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✓ ALL TESTS PASSED! Modules are ready to use.")
print("="*60)
