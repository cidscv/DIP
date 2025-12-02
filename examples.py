"""
Example Scripts for Assignment 2
Demonstrates complete workflows for common tasks
"""

import cv2
import os
from dip import freq_filters, segmentation, morphology, metrics, visualization
from dip import pipeline_a_freq, pipeline_b_custom


def example_1_basic_pipeline_a():
    """
    Example 1: Run baseline Pipeline A on a single image
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Pipeline A")
    print("="*70 + "\n")
    
    # Load image
    image = cv2.imread('input_images/test.jpg')
    if image is None:
        print("Error: Could not load test.jpg")
        return
    
    # Run Pipeline A
    result, evaluation = pipeline_a_freq.run_baseline_with_evaluation(
        image,
        cutoff_radius=30
    )
    
    # Save results
    output_dir = 'output_images/example1'
    pipeline_a_freq.save_baseline_results(
        image, result, evaluation, output_dir, image_name='test'
    )
    
    print(f"\n✓ Results saved to {output_dir}")


def example_2_pipeline_a_with_ground_truth():
    """
    Example 2: Pipeline A with ground truth evaluation
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Pipeline A with Ground Truth")
    print("="*70 + "\n")
    
    # Load image and ground truth
    image = cv2.imread('input_images/test.jpg')
    ground_truth = cv2.imread('ground_truth/test_gt.png', cv2.IMREAD_GRAYSCALE)
    
    if image is None or ground_truth is None:
        print("Error: Could not load image or ground truth")
        return
    
    # Run Pipeline A
    result, evaluation = pipeline_a_freq.run_baseline_with_evaluation(
        image,
        ground_truth=ground_truth,
        cutoff_radius=30
    )
    
    # Print metrics
    if 'metrics' in evaluation:
        metrics.print_metrics(evaluation['metrics'], title="Pipeline A Results")
        
        # Create overlay visualization
        overlay = metrics.overlay_segmentation(image, result, ground_truth, alpha=0.5)
        cv2.imwrite('output_images/example2/overlay.jpg', overlay)
    
    print("\n✓ Evaluation complete")


def example_3_ablation_study():
    """
    Example 3: Ablation study on Pipeline A
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Ablation Study")
    print("="*70 + "\n")
    
    # Load image and optional ground truth
    image = cv2.imread('input_images/test.jpg')
    ground_truth = cv2.imread('ground_truth/test_gt.png', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load image")
        return
    
    # Run ablation study
    results = pipeline_a_freq.run_baseline_ablation_study(
        image,
        ground_truth=ground_truth
    )
    
    # Save all variants
    output_dir = 'output_images/example3/ablation'
    os.makedirs(output_dir, exist_ok=True)
    
    for variant_name, (result_mask, variant_metrics, params) in results.items():
        cv2.imwrite(f'{output_dir}/{variant_name}.png', result_mask)
        
        if variant_metrics:
            print(f"\n{variant_name}:")
            print(f"  Dice: {variant_metrics['dice']:.4f}")
            print(f"  IoU:  {variant_metrics['iou']:.4f}")
    
    # Create comparison visualization
    if ground_truth is not None:
        ablation_dict = {name: (mask, m) for name, (mask, m, _) in results.items()}
        visualization.create_ablation_comparison(
            image, ablation_dict, ground_truth,
            save_path=f'{output_dir}/comparison.png'
        )
    
    print(f"\n✓ Ablation results saved to {output_dir}")


def example_4_problem_oriented_pipeline():
    """
    Example 4: Problem-oriented Pipeline B with different strategies
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Problem-Oriented Pipeline B")
    print("="*70 + "\n")
    
    # Load image
    image = cv2.imread('input_images/test.jpg')
    if image is None:
        print("Error: Could not load image")
        return
    
    # Try different strategies
    strategies = {
        'periodic_noise': {'inner_radius': 20, 'outer_radius': 60},
        'uneven_lighting': {'block_size': 15, 'C': 5},
        'color_based': {'k': 3, 'use_lpf': True},
    }
    
    for strategy_name, params in strategies.items():
        print(f"\nTesting {strategy_name} strategy...")
        
        result = pipeline_b_custom.run_problem_oriented_pipeline(
            image,
            strategy=strategy_name,
            params=params
        )
        
        # Save result
        output_path = f'output_images/example4/{strategy_name}.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        
        print(f"  Saved: {output_path}")
    
    print("\n✓ All strategies tested")


def example_5_custom_frequency_filtering():
    """
    Example 5: Custom frequency domain filtering
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Frequency Filtering")
    print("="*70 + "\n")
    
    # Load image
    image = cv2.imread('input_images/test.jpg')
    if image is None:
        print("Error: Could not load image")
        return
    
    output_dir = 'output_images/example5'
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: Low-pass filters
    print("Testing low-pass filters...")
    for cutoff in [20, 30, 50]:
        filtered = freq_filters.apply_gaussian_lpf(image, cutoff_radius=cutoff)
        cv2.imwrite(f'{output_dir}/lpf_cutoff_{cutoff}.jpg', filtered)
    
    # Test 2: High-pass filter
    print("Testing high-pass filter...")
    edges = freq_filters.apply_highpass_filter(image, cutoff_radius=30, filter_type='gaussian')
    cv2.imwrite(f'{output_dir}/highpass.jpg', edges)
    
    # Test 3: Band-reject (for periodic noise)
    print("Testing band-reject filter...")
    bandreject = freq_filters.apply_bandreject_filter(
        image, inner_radius=20, outer_radius=60, filter_type='butterworth'
    )
    cv2.imwrite(f'{output_dir}/bandreject.jpg', bandreject)
    
    # Test 4: Get spectra
    print("Generating spectrum visualization...")
    filtered, spectra = freq_filters.apply_gaussian_lpf(
        image, cutoff_radius=30, return_spectrum=True
    )
    
    visualization.plot_spectrum_comparison(
        spectra['before'], spectra['after'],
        save_path=f'{output_dir}/spectrum_comparison.png'
    )
    
    print(f"\n✓ Frequency filtering examples saved to {output_dir}")


def example_6_segmentation_methods():
    """
    Example 6: Compare different segmentation methods
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Segmentation Methods")
    print("="*70 + "\n")
    
    # Load image
    image = cv2.imread('input_images/test.jpg')
    if image is None:
        print("Error: Could not load image")
        return
    
    output_dir = 'output_images/example6'
    os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Otsu
    print("Testing Otsu thresholding...")
    binary_otsu, threshold = segmentation.otsu_threshold(image)
    cv2.imwrite(f'{output_dir}/otsu.png', binary_otsu)
    print(f"  Threshold: {threshold:.1f}")
    
    # Method 2: Adaptive (Gaussian)
    print("Testing adaptive thresholding...")
    binary_adaptive = segmentation.adaptive_threshold_gaussian(
        image, block_size=15, C=2
    )
    cv2.imwrite(f'{output_dir}/adaptive.png', binary_adaptive)
    
    # Method 3: K-means
    print("Testing K-means clustering...")
    segmented, centers = segmentation.kmeans_segmentation(image, k=3)
    cv2.imwrite(f'{output_dir}/kmeans.jpg', segmented)
    
    # Create comparison
    results = {
        'Otsu': binary_otsu,
        'Adaptive': binary_adaptive,
        'K-means': segmented
    }
    
    visualization.create_comparison_grid(
        results,
        save_path=f'{output_dir}/segmentation_comparison.png'
    )
    
    print(f"\n✓ Segmentation examples saved to {output_dir}")


def example_7_morphology_cleanup():
    """
    Example 7: Morphological operations for cleanup
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Morphological Cleanup")
    print("="*70 + "\n")
    
    # Load and segment image
    image = cv2.imread('input_images/test.jpg')
    if image is None:
        print("Error: Could not load image")
        return
    
    binary, _ = segmentation.otsu_threshold(image)
    
    output_dir = 'output_images/example7'
    os.makedirs(output_dir, exist_ok=True)
    
    # Original
    cv2.imwrite(f'{output_dir}/1_original_binary.png', binary)
    
    # Opening (remove noise)
    opened = morphology.opening(binary, kernel_size=(5, 5))
    cv2.imwrite(f'{output_dir}/2_opened.png', opened)
    
    # Closing (fill gaps)
    closed = morphology.closing(opened, kernel_size=(5, 5))
    cv2.imwrite(f'{output_dir}/3_closed.png', closed)
    
    # Remove small objects
    cleaned = morphology.remove_small_objects(closed, min_size=100)
    cv2.imwrite(f'{output_dir}/4_small_removed.png', cleaned)
    
    # Fill holes
    filled = morphology.fill_holes(cleaned)
    cv2.imwrite(f'{output_dir}/5_holes_filled.png', filled)
    
    # All-in-one cleanup
    final = morphology.cleanup_segmentation(
        binary,
        opening_size=5,
        closing_size=5,
        remove_small=True,
        fill_holes_flag=True
    )
    cv2.imwrite(f'{output_dir}/6_complete_cleanup.png', final)
    
    # Create pipeline visualization
    from collections import OrderedDict
    stages = OrderedDict([
        ('Original Binary', binary),
        ('After Opening', opened),
        ('After Closing', closed),
        ('Complete Cleanup', final)
    ])
    
    visualization.create_processing_pipeline_figure(
        stages,
        title="Morphological Cleanup Pipeline",
        save_path=f'{output_dir}/cleanup_pipeline.png'
    )
    
    print(f"\n✓ Morphology examples saved to {output_dir}")


def example_8_complete_workflow_with_evaluation():
    """
    Example 8: Complete workflow from filtering to evaluation
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Complete Workflow")
    print("="*70 + "\n")
    
    # Load image and ground truth
    image = cv2.imread('input_images/test.jpg')
    ground_truth = cv2.imread('ground_truth/test_gt.png', cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load image")
        return
    
    output_dir = 'output_images/example8'
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Frequency filtering
    print("Step 1: Frequency filtering...")
    filtered, spectra = freq_filters.apply_gaussian_lpf(
        image, cutoff_radius=30, return_spectrum=True
    )
    
    # Step 2: Segmentation
    print("Step 2: Segmentation...")
    binary, threshold = segmentation.otsu_threshold(filtered)
    
    # Step 3: Morphological cleanup
    print("Step 3: Morphological cleanup...")
    cleaned = morphology.cleanup_segmentation(
        binary,
        opening_size=3,
        closing_size=5,
        remove_small=True,
        min_size=100
    )
    
    # Step 4: Evaluation
    if ground_truth is not None:
        print("\nStep 4: Evaluation...")
        all_metrics = metrics.compute_all_metrics(cleaned, ground_truth)
        metrics.print_metrics(all_metrics, title="Final Results")
        
        # Save metrics
        import json
        with open(f'{output_dir}/metrics.json', 'w') as f:
            json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                      for k, v in all_metrics.items()}, f, indent=2)
        
        # Create overlay
        overlay = metrics.overlay_segmentation(image, cleaned, ground_truth, alpha=0.5)
        cv2.imwrite(f'{output_dir}/overlay.jpg', overlay)
    
    # Save all stages
    from collections import OrderedDict
    stages = OrderedDict([
        ('Original', image),
        ('Filtered', filtered),
        ('Segmented', binary),
        ('Final', cleaned)
    ])
    
    for name, img in stages.items():
        cv2.imwrite(f'{output_dir}/{name.lower()}.jpg', img)
    
    # Create figure
    visualization.create_processing_pipeline_figure(
        stages,
        title="Complete Processing Workflow",
        save_path=f'{output_dir}/complete_workflow.png'
    )
    
    # Save spectra
    visualization.plot_spectrum_comparison(
        spectra['before'], spectra['after'],
        save_path=f'{output_dir}/spectrum.png'
    )
    
    print(f"\n✓ Complete workflow results saved to {output_dir}")


def run_all_examples():
    """
    Run all examples in sequence
    """
    print("\n" + "="*70)
    print("RUNNING ALL EXAMPLES")
    print("="*70)
    
    examples = [
        ("Basic Pipeline A", example_1_basic_pipeline_a),
        ("Pipeline A with Ground Truth", example_2_pipeline_a_with_ground_truth),
        ("Ablation Study", example_3_ablation_study),
        ("Problem-Oriented Pipeline", example_4_problem_oriented_pipeline),
        ("Frequency Filtering", example_5_custom_frequency_filtering),
        ("Segmentation Methods", example_6_segmentation_methods),
        ("Morphology Cleanup", example_7_morphology_cleanup),
        ("Complete Workflow", example_8_complete_workflow_with_evaluation),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    import numpy as np
    
    # Uncomment to run individual examples:
    # example_1_basic_pipeline_a()
    # example_2_pipeline_a_with_ground_truth()
    # example_3_ablation_study()
    # example_4_problem_oriented_pipeline()
    # example_5_custom_frequency_filtering()
    # example_6_segmentation_methods()
    # example_7_morphology_cleanup()
    # example_8_complete_workflow_with_evaluation()
    
    # Or run all:
    # run_all_examples()
    
    print("\nTo run examples, uncomment the desired function call at the bottom of this script.")
    print("Available examples:")
    print("  - example_1_basic_pipeline_a()")
    print("  - example_2_pipeline_a_with_ground_truth()")
    print("  - example_3_ablation_study()")
    print("  - example_4_problem_oriented_pipeline()")
    print("  - example_5_custom_frequency_filtering()")
    print("  - example_6_segmentation_methods()")
    print("  - example_7_morphology_cleanup()")
    print("  - example_8_complete_workflow_with_evaluation()")
    print("  - run_all_examples()")
