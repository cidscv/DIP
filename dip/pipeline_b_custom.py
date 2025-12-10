import cv2
import numpy as np
from dip import freq_filters, segmentation, morphology, metrics
from collections import OrderedDict


class ProblemOrientedPipeline:
    """
    Flexible pipeline that adapts to different image challenges.
    """
    
    def __init__(self):
        self.strategies = {
            'periodic_noise': self.strategy_periodic_noise,
            'uneven_lighting': self.strategy_uneven_lighting,
            'color_based': self.strategy_color_based,
            'low_contrast': self.strategy_low_contrast,
            'texture_emphasis': self.strategy_texture_emphasis,
            'custom': self.strategy_custom
        }
    
    def run(self, image, strategy='periodic_noise', params=None, 
            return_all_stages=False):
        """
        Run problem-oriented pipeline with specified strategy.
        
        Args:
            image: Input image
            strategy: Strategy name (see self.strategies)
            params: Dict of strategy-specific parameters
            return_all_stages: Return all intermediate stages
            
        Returns:
            result: Final segmentation
            (optional) stages, params, extra_info
        """
        if params is None:
            params = {}
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Available: {list(self.strategies.keys())}")
        
        print(f"\n=== Running Problem-Oriented Pipeline ===")
        print(f"Strategy: {strategy}")
        print(f"Parameters: {params}\n")
        
        # Execute strategy
        result, stages, used_params, extra_info = self.strategies[strategy](
            image, params
        )
        
        if return_all_stages:
            return result, stages, used_params, extra_info
        else:
            return result
    
    def strategy_periodic_noise(self, image, params):
        """
        Strategy for images with periodic noise patterns.
        Uses band-reject filtering to suppress periodic frequencies.
        """
        stages = OrderedDict()
        stages['1_original'] = image.copy()
        
        # Parameters
        inner_radius = params.get('inner_radius', 20)
        outer_radius = params.get('outer_radius', 60)
        filter_type = params.get('filter_type', 'butterworth')
        order = params.get('order', 2)
        
        # Step 1: Band-reject filtering
        print(f"  Step 1: Band-reject filtering ({inner_radius}-{outer_radius})...")
        filtered, spectra = freq_filters.apply_bandreject_filter(
            image,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            filter_type=filter_type,
            order=order,
            return_spectrum=True
        )
        stages['2_bandreject'] = filtered
        
        # Step 2: Adaptive thresholding
        print(f"  Step 2: Adaptive thresholding...")
        block_size = params.get('block_size', 15)
        C = params.get('C', 2)
        binary = segmentation.adaptive_threshold_gaussian(
            filtered, block_size=block_size, C=C
        )
        stages['3_adaptive_threshold'] = binary
        
        # Step 3: Morphological cleanup
        print(f"  Step 3: Morphological cleanup...")
        result = morphology.cleanup_segmentation(
            binary,
            opening_size=3,
            closing_size=5,
            remove_small=True,
            min_size=100,
            fill_holes_flag=False
        )
        stages['4_final'] = result
        
        used_params = {
            'strategy': 'periodic_noise',
            'inner_radius': inner_radius,
            'outer_radius': outer_radius,
            'filter_type': filter_type,
            'order': order,
            'block_size': block_size,
            'C': C
        }
        
        return result, stages, used_params, {'spectra': spectra}
    
    def strategy_uneven_lighting(self, image, params):
        """
        Strategy for images with uneven lighting.
        Uses adaptive thresholding with optional preprocessing.
        """
        stages = OrderedDict()
        stages['1_original'] = image.copy()
        
        # Parameters
        use_clahe = params.get('use_clahe', True)
        block_size = params.get('block_size', 15)
        C = params.get('C', 5)
        
        # Step 1: Optional CLAHE preprocessing
        if use_clahe:
            print(f"  Step 1: CLAHE preprocessing...")
            if len(image.shape) == 3:
                # Apply CLAHE to L channel in LAB space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                processed = cv2.merge([l, a, b])
                processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(image)
            stages['2_clahe'] = processed
        else:
            processed = image.copy()
            stages['2_no_preprocessing'] = processed
        
        # Step 2: Adaptive thresholding
        print(f"  Step 2: Adaptive thresholding (block_size={block_size})...")
        binary = segmentation.adaptive_threshold_gaussian(
            processed, block_size=block_size, C=C
        )
        stages['3_adaptive_threshold'] = binary
        
        # Step 3: Morphological cleanup
        print(f"  Step 3: Morphological cleanup...")
        result = morphology.cleanup_segmentation(
            binary,
            opening_size=3,
            closing_size=5,
            remove_small=True,
            fill_holes_flag=False
        )
        stages['4_final'] = result
        
        used_params = {
            'strategy': 'uneven_lighting',
            'use_clahe': use_clahe,
            'block_size': block_size,
            'C': C
        }
        
        return result, stages, used_params, {}
    
    def strategy_color_based(self, image, params):
        """
        Strategy for color-based segmentation.
        Uses K-means clustering in color space.
        """
        stages = OrderedDict()
        stages['1_original'] = image.copy()
        
        # Parameters
        k = params.get('k', 3)
        target_cluster = params.get('target_cluster', None)
        use_lpf = params.get('use_lpf', True)
        cutoff_radius = params.get('cutoff_radius', 150)
        
        filtered = image.copy()
        stages['2_no_filtering'] = filtered
        
        # Step 2: K-means clustering
        print(f"  Step 2: K-means clustering (k={k})...")
        labels, centers = segmentation.kmeans_segmentation(
            filtered, k=k, return_labels=True
        )
        stages['3_kmeans_labels'] = (labels * (255 // (k-1))).astype(np.uint8)
        
        # Step 3: Select target cluster
        if target_cluster is None:
            # Automatically select largest non-background cluster
            print("  Step 3: Auto-selecting target cluster...")
            # Find cluster with MEDIUM area (not largest background, not smallest noise)
            cluster_sizes = [np.sum(labels == i) for i in range(k)]
            sorted_clusters = np.argsort(cluster_sizes)  # Sort from smallest to largest

            # Select middle cluster for k=3, smaller for k=2
            if k == 3:
                sorted_clusters = np.argsort(cluster_sizes)
                # Largest is usually background - pick second largest
                target_cluster = sorted_clusters[-2]  # Second largest cluster
            elif k == 2:
                target_cluster = sorted_clusters[0]
            else:
                target_cluster = sorted_clusters[k//2]
            
            print(f"    Selected cluster {target_cluster}")
            print(f"    DEBUG - Cluster percentages: {[f'{s/labels.size*100:.1f}%' for s in cluster_sizes]}")
            print(f"    DEBUG - Sorted order: {sorted_clusters}")

            # Show cluster sizes for debugging
            for i in range(k):
                pct = (cluster_sizes[i] / labels.size) * 100
                print(f"    Cluster {i}: {cluster_sizes[i]:,} pixels ({pct:.1f}%)")
            print(f"    Selected cluster {target_cluster}")

            # Create binary mask for selected cluster
            # Cluster pixels = 255 (white = foreground), others = 0 (black = background)
            mask = (labels == target_cluster).astype(np.uint8) * 255
            
            # SAVE THE MASK BEFORE MORPHOLOGY FOR INSPECTION
            cv2.imwrite('debug_mask_before_morphology.png', mask)
            print(f"    DEBUG - Saved debug_mask_before_morphology.png")

            binary = (labels == target_cluster).astype(np.uint8) * 255
            print(f"    DEBUG - Binary mask: pixels={np.sum(binary==255):,} ({np.sum(binary==255)/binary.size*100:.1f}%)")
            cv2.imwrite('debug_binary_before_cleanup.png', binary)

            # NO inversion needed - cluster already has object as white
            stages['3_clustered'] = mask
        
        binary = (labels == target_cluster).astype(np.uint8) * 255
        stages['4_selected_cluster'] = binary
        
        # Step 4: Morphological cleanup
        print(f"  Step 4: Morphological cleanup...")
        # TEMP: Skip all morphology to see raw K-means output
        result = binary.copy()
        # result = morphology.cleanup_segmentation(...)
        stages['5_final'] = result

        used_params = {
            'strategy': 'color_based',
            'k': k,
            'target_cluster': int(target_cluster),
            'use_lpf': use_lpf,
            'cutoff_radius': cutoff_radius
        }
        
        return result, stages, used_params, {'centers': centers}
    
    def strategy_low_contrast(self, image, params):
        """
        Strategy for low contrast images.
        Uses top-hat transform to extract bright features.
        """
        stages = OrderedDict()
        stages['1_original'] = image.copy()
        
        # Parameters
        kernel_size = params.get('kernel_size', 15)
        use_bottomhat = params.get('use_bottomhat', False)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        stages['2_grayscale'] = gray
        
        # Step 1: Top-hat transform
        print(f"  Step 1: Top-hat transform (kernel_size={kernel_size})...")
        tophat = morphology.top_hat(gray, kernel_size=(kernel_size, kernel_size))
        stages['3_tophat'] = tophat
        
        # Optional: Bottom-hat for dark features
        if use_bottomhat:
            print(f"  Step 2: Bottom-hat transform...")
            bottomhat = morphology.bottom_hat(gray, kernel_size=(kernel_size, kernel_size))
            stages['4_bottomhat'] = bottomhat
            
            # Combine top-hat and bottom-hat
            combined = cv2.add(tophat, bottomhat)
            stages['5_combined'] = combined
            to_threshold = combined
        else:
            to_threshold = tophat
        
        # Step 2/3: Adaptive thresholding
        print(f"  Step {3 if not use_bottomhat else 6}: Adaptive thresholding...")
        binary = segmentation.adaptive_threshold_gaussian(
            to_threshold, block_size=11, C=2
        )
        stages[f"{4 if not use_bottomhat else 7}_threshold"] = binary
        
        # Step 3/4: Morphological cleanup
        print(f"  Step {4 if not use_bottomhat else 7}: Morphological cleanup...")
        result = morphology.cleanup_segmentation(
            binary,
            opening_size=3,
            closing_size=5,
            remove_small=True
        )
        stages[f"{5 if not use_bottomhat else 8}_final"] = result
        
        used_params = {
            'strategy': 'low_contrast',
            'kernel_size': kernel_size,
            'use_bottomhat': use_bottomhat
        }
        
        return result, stages, used_params, {}
    
    def strategy_texture_emphasis(self, image, params):
        """
        Strategy emphasizing texture using band-pass filtering.
        """
        stages = OrderedDict()
        stages['1_original'] = image.copy()
        
        # Parameters
        inner_radius = params.get('inner_radius', 10)
        outer_radius = params.get('outer_radius', 50)
        
        # Step 1: Band-pass filtering
        print(f"  Step 1: Band-pass filtering...")
        filtered, spectra = freq_filters.apply_bandpass_filter(
            image,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            filter_type='butterworth',
            return_spectrum=True
        )
        stages['2_bandpass'] = filtered
        
        # Step 2: Otsu thresholding
        print(f"  Step 2: Otsu thresholding...")
        binary, threshold = segmentation.otsu_threshold(filtered)
        stages['3_otsu'] = binary
        print(f"    Threshold: {threshold:.1f}")
        
        # Step 3: Morphological cleanup
        print(f"  Step 3: Morphological cleanup...")
        result = morphology.cleanup_segmentation(
            binary,
            opening_size=3,
            closing_size=5,
            remove_small=True
        )
        stages['4_final'] = result
        
        used_params = {
            'strategy': 'texture_emphasis',
            'inner_radius': inner_radius,
            'outer_radius': outer_radius
        }
        
        return result, stages, used_params, {'spectra': spectra}
    
    def strategy_custom(self, image, params):
        """
        Fully custom strategy - user provides all processing steps.
        """
        stages = OrderedDict()
        stages['1_original'] = image.copy()
        
        # This is a template - users can customize as needed
        steps = params.get('steps', [])
        
        current = image.copy()
        for i, step in enumerate(steps, 2):
            operation = step['operation']
            step_params = step.get('params', {})
            
            print(f"  Step {i}: {operation}...")
            
            # Execute operation
            if operation == 'gaussian_lpf':
                current = freq_filters.apply_gaussian_lpf(current, **step_params)
            elif operation == 'otsu':
                current, _ = segmentation.otsu_threshold(current, **step_params)
            elif operation == 'adaptive':
                current = segmentation.adaptive_threshold_gaussian(current, **step_params)
            elif operation == 'kmeans':
                current, _ = segmentation.kmeans_segmentation(current, **step_params)
            elif operation == 'opening':
                current = morphology.opening(current, **step_params)
            elif operation == 'closing':
                current = morphology.closing(current, **step_params)
            elif operation == 'cleanup':
                current = morphology.cleanup_segmentation(current, **step_params)
            
            stages[f"{i}_{operation}"] = current.copy()
        
        result = current
        stages[f"{len(steps)+2}_final"] = result
        
        used_params = {
            'strategy': 'custom',
            'steps': steps
        }
        
        return result, stages, used_params, {}


def run_problem_oriented_pipeline(image, strategy='periodic_noise', 
                                  params=None, ground_truth=None,
                                  return_all_stages=False):
    """
    Convenient wrapper for running problem-oriented pipeline.
    
    Args:
        image: Input image
        strategy: Strategy name
        params: Strategy parameters
        ground_truth: Ground truth for evaluation (optional)
        return_all_stages: Return all processing stages
        
    Returns:
        result: Final segmentation
        evaluation: Dict with stages, params, and metrics
    """
    pipeline = ProblemOrientedPipeline()
    
    result, stages, used_params, extra_info = pipeline.run(
        image, strategy=strategy, params=params, return_all_stages=True
    )
    
    evaluation = {
        'stages': stages,
        'params': used_params,
        'extra_info': extra_info
    }
    
    # Evaluate if ground truth provided
    if ground_truth is not None:
        print("\n  Evaluating segmentation...")
        all_metrics = metrics.compute_all_metrics(result, ground_truth)
        evaluation['metrics'] = all_metrics
        
        print(f"    Dice coefficient: {all_metrics['dice']:.4f}")
        print(f"    IoU: {all_metrics['iou']:.4f}")
        print(f"    Pixel accuracy: {all_metrics['pixel_accuracy']:.4f}")
    
    print("\n=== Pipeline Complete ===\n")
    
    if return_all_stages:
        return result, evaluation
    else:
        return result


def compare_strategies(image, ground_truth=None):
    """
    Compare different problem-oriented strategies.
    
    Args:
        image: Input image
        ground_truth: Ground truth mask (optional)
        
    Returns:
        results: Dict of {strategy_name: (result, metrics)}
    """
    print("\n=== Comparing Problem-Oriented Strategies ===\n")
    
    pipeline = ProblemOrientedPipeline()
    results = {}
    
    strategies_to_test = [
        'periodic_noise',
        'uneven_lighting',
        'color_based',
        'low_contrast',
        'texture_emphasis'
    ]
    
    for strategy in strategies_to_test:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy}")
        print(f"{'='*60}")
        
        result = pipeline.run(image, strategy=strategy)
        
        if ground_truth is not None:
            strategy_metrics = metrics.compute_all_metrics(result, ground_truth)
            print(f"\nMetrics:")
            print(f"  Dice: {strategy_metrics['dice']:.4f}")
            print(f"  IoU: {strategy_metrics['iou']:.4f}")
        else:
            strategy_metrics = None
        
        results[strategy] = (result, strategy_metrics)
    
    print("\n=== Strategy Comparison Complete ===\n")
    
    return results


# Example usage
if __name__ == "__main__":
    # Load test image
    test_image = cv2.imread('input_images/test.jpg')
    
    if test_image is not None:
        print("Running problem-oriented pipeline on test image...")
        
        # Test periodic noise strategy
        result, evaluation = run_problem_oriented_pipeline(
            test_image,
            strategy='periodic_noise',
            params={
                'inner_radius': 20,
                'outer_radius': 60,
                'block_size': 15
            }
        )
        
        # Save result
        cv2.imwrite('output_images/problem_oriented_result.png', result)
        
        print("\nProblem-oriented pipeline complete!")
    else:
        print("Test image not found. Please update the path.")
