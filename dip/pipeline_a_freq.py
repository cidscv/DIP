import cv2
import numpy as np
from dip import freq_filters, segmentation, morphology, metrics
from collections import OrderedDict


def run_baseline_pipeline(image, cutoff_radius=30, 
                          opening_size=3, closing_size=5, 
                          min_object_size=100,
                          return_all_stages=False):
    """
    Execute baseline pipeline on an image.
    
    Pipeline Steps:
    1. Gaussian low-pass filter (frequency domain)
    2. Otsu automatic thresholding
    3. Morphological opening (remove noise)
    4. Morphological closing (fill gaps)
    5. Remove small objects
    
    Args:
        image: Input image (color or grayscale)
        cutoff_radius: Cutoff frequency for low-pass filter (default=30)
        opening_size: Kernel size for opening operation (default=3)
        closing_size: Kernel size for closing operation (default=5)
        min_object_size: Minimum object size to keep in pixels (default=100)
        return_all_stages: If True, return dict with all intermediate stages
        
    Returns:
        result: Final segmentation mask
        (optional) stages: OrderedDict of all processing stages
        (optional) params: Dict of parameters used
    """
    stages = OrderedDict()
    
    # Store original
    stages['1_original'] = image.copy()
    
    # Step 1: Gaussian low-pass filtering
    print(f"  Step 1: Applying Gaussian low-pass filter (cutoff={cutoff_radius})...")
    filtered, spectra = freq_filters.apply_gaussian_lpf(
        image, 
        cutoff_radius=cutoff_radius,
        return_spectrum=True
    )
    stages['2_filtered'] = filtered
    
    # Step 2: Otsu thresholding
    print(f"  Step 2: Applying Otsu thresholding...")
    binary, threshold_value = segmentation.otsu_threshold(filtered)
    stages['3_otsu'] = binary
    print(f"    Computed threshold: {threshold_value:.1f}")
    
    # Step 3: Morphological opening (remove noise)
    print(f"  Step 3: Morphological opening (kernel_size={opening_size})...")
    opened = morphology.opening(binary, kernel_size=(opening_size, opening_size))
    stages['4_opened'] = opened
    
    # Step 4: Morphological closing (fill gaps)
    print(f"  Step 4: Morphological closing (kernel_size={closing_size})...")
    closed = morphology.closing(opened, kernel_size=(closing_size, closing_size))
    stages['5_closed'] = closed
    
    # Step 5: Remove small objects
    print(f"  Step 5: Removing small objects (min_size={min_object_size})...")
    result = morphology.remove_small_objects(closed, min_size=min_object_size)
    stages['6_final'] = result
    
    # Store parameters used
    params = {
        'pipeline': 'A (Baseline)',
        'cutoff_radius': cutoff_radius,
        'threshold_method': 'otsu',
        'threshold_value': float(threshold_value),
        'opening_kernel_size': opening_size,
        'closing_kernel_size': closing_size,
        'min_object_size': min_object_size
    }
    
    if return_all_stages:
        return result, stages, params, spectra
    else:
        return result


def run_baseline_with_evaluation(image, ground_truth=None, 
                                 cutoff_radius=30,
                                 opening_size=3, 
                                 closing_size=5,
                                 min_object_size=100):
    """
    Run baseline pipeline and evaluate results if ground truth provided.
    
    Args:
        image: Input image
        ground_truth: Ground truth mask (optional)
        cutoff_radius: Cutoff frequency for filtering
        opening_size: Kernel size for opening
        closing_size: Kernel size for closing
        min_object_size: Minimum object size
        
    Returns:
        result: Final segmentation
        evaluation: Dict with stages, params, spectra, and metrics
    """
    # Run pipeline
    result, stages, params, spectra = run_baseline_pipeline(
        image,
        cutoff_radius=cutoff_radius,
        opening_size=opening_size,
        closing_size=closing_size,
        min_object_size=min_object_size,
        return_all_stages=True
    )
    
    evaluation = {
        'stages': stages,
        'params': params,
        'spectra': spectra
    }
    
    # Evaluate if ground truth provided
    if ground_truth is not None:
        print("\n  Evaluating segmentation...")
        all_metrics = metrics.compute_all_metrics(result, ground_truth)
        evaluation['metrics'] = all_metrics
        
        print(f"    Dice coefficient: {all_metrics['dice']:.4f}")
        print(f"    IoU: {all_metrics['iou']:.4f}")
        print(f"    Pixel accuracy: {all_metrics['pixel_accuracy']:.4f}")
    
    return result, evaluation


def run_baseline_ablation_study(image, ground_truth=None):
    """
    Run ablation study on baseline pipeline parameters.
    Tests different cutoff radii and morphology kernel sizes.
    
    Args:
        image: Input image
        ground_truth: Ground truth mask for evaluation (optional)
        
    Returns:
        results: Dict of {variant_name: (result_mask, metrics_dict, params)}
    """
    print("\n=== Running Baseline Pipeline Ablation Study ===\n")
    
    results = {}
    
    # Test different cutoff radii
    cutoff_radii = [20, 30, 40, 50]
    
    print("Testing different cutoff radii...")
    for cutoff in cutoff_radii:
        print(f"\nVariant: Cutoff Radius = {cutoff}")
        result = run_baseline_pipeline(
            image,
            cutoff_radius=cutoff,
            opening_size=3,
            closing_size=5,
            min_object_size=100
        )
        
        variant_name = f"Cutoff_{cutoff}"
        
        if ground_truth is not None:
            variant_metrics = metrics.compute_all_metrics(result, ground_truth)
            print(f"  Dice: {variant_metrics['dice']:.4f}, IoU: {variant_metrics['iou']:.4f}")
        else:
            variant_metrics = None
        
        results[variant_name] = (result, variant_metrics, {'cutoff_radius': cutoff})
    
    # Test different morphology kernel sizes
    kernel_sizes = [(3, 3), (3, 5), (5, 5), (5, 7)]
    
    print("\n\nTesting different morphology kernel sizes...")
    for opening_size, closing_size in kernel_sizes:
        print(f"\nVariant: Opening={opening_size}x{opening_size}, Closing={closing_size}x{closing_size}")
        result = run_baseline_pipeline(
            image,
            cutoff_radius=30,
            opening_size=opening_size,
            closing_size=closing_size,
            min_object_size=100
        )
        
        variant_name = f"Morph_O{opening_size}_C{closing_size}"
        
        if ground_truth is not None:
            variant_metrics = metrics.compute_all_metrics(result, ground_truth)
            print(f"  Dice: {variant_metrics['dice']:.4f}, IoU: {variant_metrics['iou']:.4f}")
        else:
            variant_metrics = None
        
        results[variant_name] = (result, variant_metrics, {
            'opening_size': opening_size,
            'closing_size': closing_size
        })
    
    print("\n=== Ablation Study Complete ===\n")
    
    return results


def compare_filter_types(image, ground_truth=None, cutoff_radius=30):
    """
    Compare Ideal vs Butterworth vs Gaussian low-pass filters.
    
    Args:
        image: Input image
        ground_truth: Ground truth mask (optional)
        cutoff_radius: Cutoff frequency to use
        
    Returns:
        results: Dict of {filter_type: (result_mask, metrics_dict)}
    """
    print("\n=== Comparing Filter Types ===\n")
    
    results = {}
    filter_functions = {
        'Ideal': lambda img: freq_filters.apply_ideal_lpf(img, cutoff_radius),
        'Butterworth': lambda img: freq_filters.apply_butterworth_lpf(img, cutoff_radius, order=2),
        'Gaussian': lambda img: freq_filters.apply_gaussian_lpf(img, cutoff_radius)
    }
    
    for filter_name, filter_func in filter_functions.items():
        print(f"\nTesting {filter_name} filter...")
        
        # Apply filter
        filtered = filter_func(image)
        
        # Complete pipeline
        binary, _ = segmentation.otsu_threshold(filtered)
        opened = morphology.opening(binary, kernel_size=(3, 3))
        closed = morphology.closing(opened, kernel_size=(5, 5))
        result = morphology.remove_small_objects(closed, min_size=100)
        
        if ground_truth is not None:
            variant_metrics = metrics.compute_all_metrics(result, ground_truth)
            print(f"  Dice: {variant_metrics['dice']:.4f}, IoU: {variant_metrics['iou']:.4f}")
        else:
            variant_metrics = None
        
        results[filter_name] = (result, variant_metrics)
    
    print("\n=== Filter Comparison Complete ===\n")
    
    return results


def save_baseline_results(image, result, evaluation, output_dir, image_name='baseline'):
    """
    Save all baseline pipeline results to output directory.
    
    Args:
        image: Original image
        result: Final segmentation result
        evaluation: Evaluation dict from run_baseline_with_evaluation
        output_dir: Directory to save results
        image_name: Base name for output files
    """
    import os
    from dip import visualization
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving results to {output_dir}...")
    
    # Save final result
    result_path = os.path.join(output_dir, f'{image_name}_final.png')
    cv2.imwrite(result_path, result)
    print(f"  Saved final result: {result_path}")
    
    # Save all stages
    for stage_name, stage_img in evaluation['stages'].items():
        stage_path = os.path.join(output_dir, f'{image_name}_{stage_name}.png')
        cv2.imwrite(stage_path, stage_img)
    print(f"  Saved {len(evaluation['stages'])} processing stages")
    
    # Save spectra
    spectrum_before = evaluation['spectra']['before']
    spectrum_after = evaluation['spectra']['after']
    
    spectrum_path = os.path.join(output_dir, f'{image_name}_spectrum_comparison.png')
    visualization.plot_spectrum_comparison(
        spectrum_before,
        spectrum_after,
        titles=('Original Spectrum', 'After Gaussian LPF'),
        save_path=spectrum_path
    )
    
    # Save pipeline visualization
    pipeline_stages = OrderedDict([
        ('Original', evaluation['stages']['1_original']),
        ('Filtered', evaluation['stages']['2_filtered']),
        ('Otsu', evaluation['stages']['3_otsu']),
        ('Final', evaluation['stages']['6_final'])
    ])
    
    pipeline_path = os.path.join(output_dir, f'{image_name}_pipeline.png')
    visualization.create_processing_pipeline_figure(
        pipeline_stages,
        title=f"Baseline Pipeline - {image_name}",
        save_path=pipeline_path
    )
    
    # Save parameters
    import json
    params_path = os.path.join(output_dir, f'{image_name}_params.json')
    with open(params_path, 'w') as f:
        json.dump(evaluation['params'], f, indent=2)
    print(f"  Saved parameters: {params_path}")
    
    # Save metrics if available
    if 'metrics' in evaluation:
        metrics_path = os.path.join(output_dir, f'{image_name}_metrics.json')
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v)
                       for k, v in evaluation['metrics'].items()}
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"  Saved metrics: {metrics_path}")
        
        # Save overlay visualization
        if 'ground_truth' in evaluation:
            overlay = metrics.overlay_segmentation(
                image, result, evaluation['ground_truth'], alpha=0.5
            )
            overlay_path = os.path.join(output_dir, f'{image_name}_overlay.png')
            cv2.imwrite(overlay_path, overlay)
            print(f"  Saved overlay: {overlay_path}")
    
    print("All results saved successfully!\n")


# Example usage
if __name__ == "__main__":
    # Load test image
    test_image = cv2.imread('input_images/test.jpg')
    
    if test_image is not None:
        print("Running baseline pipeline on test image...")
        
        # Run pipeline
        result, evaluation = run_baseline_with_evaluation(test_image)
        
        # Save results
        save_baseline_results(test_image, result, evaluation, 
                            'output_images/baseline', 
                            image_name='test')
        
        print("\nBaseline pipeline complete!")
    else:
        print("Test image not found. Please update the path.")
