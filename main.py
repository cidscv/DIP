"""
Digital Image Processing - Assignment 2
Frequency-Domain Filtering & Segmentation Pipelines

Updated to support:
- Pipeline A: Baseline frequency + segmentation
- Pipeline B: Problem-oriented strategies
- Ablation studies and comparisons
"""

from dip import analysis as aly
from dip import baseline_proc as baseline_old
from dip import custom_proc as custom_old
from dip import util
from dip import pipeline_a_freq
from dip import pipeline_b_custom
from dip import visualization
from dip import metrics
import argparse
import os
import cv2
import json


def main():
    parser = argparse.ArgumentParser(
        prog="DIP Assignment 2",
        description="Frequency-domain filtering and segmentation pipelines",
    )

    parser.add_argument(
        "--image_dir",
        metavar="imagedir",
        required=True,
        help="Directory containing input images to process",
    )

    parser.add_argument(
        "--out",
        metavar="output_dir",
        required=True,
        help="Directory where processed images will be saved",
    )

    parser.add_argument(
        "--pipeline",
        metavar="pipeline",
        required=False,
        choices=["A", "B", "A_old", "B_old"],
        help="Pipeline: A (freq baseline), B (problem-oriented), A_old/B_old (original)",
    )

    parser.add_argument(
        "--strategy",
        metavar="strategy",
        required=False,
        default="periodic_noise",
        help="Strategy for Pipeline B (periodic_noise, uneven_lighting, color_based, etc.)",
    )

    parser.add_argument(
        "--gen_hists",
        action="store_true",
        help="Generate histograms only",
    )

    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study for Pipeline A",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare different strategies/filters",
    )

    parser.add_argument(
        "--ground_truth",
        metavar="gt_dir",
        required=False,
        help="Directory containing ground truth masks for evaluation",
    )

    parser.add_argument(
        "--cutoff",
        type=int,
        default=30,
        help="Cutoff radius for frequency filters (default: 30)",
    )

    parser.add_argument(
        "--save_all_stages",
        action="store_true",
        help="Save all intermediate processing stages",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Find images
    INPUT_DIR = args.image_dir
    OUTPUT_DIR = args.out
    GT_DIR = args.ground_truth

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG")
    image_files = [
        f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"\n{'='*70}")
    print(f"Digital Image Processing - Assignment 2")
    print(f"Found {len(image_files)} images to process")
    print(f"{'='*70}\n")

    # Process each image
    for file_path in image_files:
        print(f"\n{'='*70}")
        print(f"Processing: {file_path}")
        print(f"{'='*70}\n")

        filename = os.path.splitext(file_path)[0]
        
        # Try different extensions
        image = None
        for ext in image_extensions:
            try:
                test_path = os.path.join(INPUT_DIR, filename + ext)
                if os.path.exists(test_path):
                    image = cv2.imread(test_path)
                    if image is not None:
                        break
            except:
                continue

        if image is None:
            print(f"Error: Could not load image {filename}")
            continue

        print(f"Original image shape: {image.shape}")

        # Load ground truth if available
        ground_truth = None
        if GT_DIR:
            gt_path = os.path.join(GT_DIR, f"{filename}_gt.png")
            if os.path.exists(gt_path):
                ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                print(f"Loaded ground truth: {gt_path}")
            else:
                print(f"No ground truth found for {filename}")

        # Generate histogram
        if args.gen_hists or not args.pipeline:
            print(f"\nGenerating histogram...")
            aly.generate_histogram_from_image(image, name=f"{filename}_original")

        # Extract EXIF data
        print(f"\nExtracting EXIF data...")
        for ext in image_extensions:
            test_path = os.path.join(INPUT_DIR, filename + ext)
            if os.path.exists(test_path):
                util.extract_exif(test_path, OUTPUT_DIR, name=f"{filename}")
                break

        # Run pipelines
        if args.pipeline == "A":
            print("\n" + "="*70)
            print("PIPELINE A: Baseline Frequency + Segmentation")
            print("="*70)
            
            if args.ablation:
                # Run ablation study
                results = pipeline_a_freq.run_baseline_ablation_study(
                    image, ground_truth=ground_truth
                )
                
                # Save ablation results
                ablation_dir = os.path.join(OUTPUT_DIR, 'ablation', filename)
                os.makedirs(ablation_dir, exist_ok=True)
                
                for variant_name, (result, variant_metrics, variant_params) in results.items():
                    variant_path = os.path.join(ablation_dir, f'{variant_name}.png')
                    cv2.imwrite(variant_path, result)
                    
                    # Save metrics
                    if variant_metrics:
                        metrics_path = os.path.join(ablation_dir, f'{variant_name}_metrics.json')
                        with open(metrics_path, 'w') as f:
                            json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                     for k, v in variant_metrics.items()}, f, indent=2)
                
                # Create ablation comparison figure
                ablation_results = {name: (mask, m) for name, (mask, m, _) in results.items()}
                fig_path = os.path.join(ablation_dir, 'comparison.png')
                visualization.create_ablation_comparison(
                    image, ablation_results, ground_truth, save_path=fig_path
                )
                
                print(f"\nAblation study results saved to {ablation_dir}")
            
            elif args.compare:
                # Compare filter types
                results = pipeline_a_freq.compare_filter_types(
                    image, ground_truth=ground_truth, cutoff_radius=args.cutoff
                )
                
                compare_dir = os.path.join(OUTPUT_DIR, 'comparison', filename)
                os.makedirs(compare_dir, exist_ok=True)
                
                for filter_name, (result, variant_metrics) in results.items():
                    result_path = os.path.join(compare_dir, f'{filter_name}.png')
                    cv2.imwrite(result_path, result)
                
                # Create comparison figure
                results_dict = {name: result for name, (result, _) in results.items()}
                fig_path = os.path.join(compare_dir, 'filter_comparison.png')
                visualization.create_comparison_grid(
                    results_dict, save_path=fig_path, figsize=(15, 10)
                )
                
                print(f"\nFilter comparison saved to {compare_dir}")
            
            else:
                # Standard pipeline run
                result, evaluation = pipeline_a_freq.run_baseline_with_evaluation(
                    image,
                    ground_truth=ground_truth,
                    cutoff_radius=args.cutoff
                )
                
                # Save results
                result_dir = os.path.join(OUTPUT_DIR, 'pipeline_a', filename)
                pipeline_a_freq.save_baseline_results(
                    image, result, evaluation, result_dir, image_name=filename
                )
                
                print(f"\nPipeline A results saved to {result_dir}")

        elif args.pipeline == "B":
            print("\n" + "="*70)
            print(f"PIPELINE B: Problem-Oriented ({args.strategy})")
            print("="*70)
            
            if args.compare:
                # Compare different strategies
                results = pipeline_b_custom.compare_strategies(
                    image, ground_truth=ground_truth
                )
                
                compare_dir = os.path.join(OUTPUT_DIR, 'strategies', filename)
                os.makedirs(compare_dir, exist_ok=True)
                
                for strategy_name, (result, strategy_metrics) in results.items():
                    result_path = os.path.join(compare_dir, f'{strategy_name}.png')
                    cv2.imwrite(result_path, result)
                    
                    if strategy_metrics:
                        metrics_path = os.path.join(compare_dir, f'{strategy_name}_metrics.json')
                        with open(metrics_path, 'w') as f:
                            json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                     for k, v in strategy_metrics.items()}, f, indent=2)
                
                # Create comparison visualization
                if ground_truth is not None:
                    metrics_dict = {name: m for name, (_, m) in results.items() if m}
                    fig_path = os.path.join(compare_dir, 'metrics_comparison.png')
                    visualization.plot_metrics_comparison(metrics_dict, save_path=fig_path)
                
                print(f"\nStrategy comparison saved to {compare_dir}")
            
            else:
                # Standard pipeline run with specified strategy
                result, evaluation = pipeline_b_custom.run_problem_oriented_pipeline(
                    image,
                    strategy=args.strategy,
                    ground_truth=ground_truth
                )
                
                # Save results
                result_dir = os.path.join(OUTPUT_DIR, 'pipeline_b', args.strategy, filename)
                os.makedirs(result_dir, exist_ok=True)
                
                # Save final result
                result_path = os.path.join(result_dir, f'{filename}_final.png')
                cv2.imwrite(result_path, result)
                
                # Save all stages if requested
                if args.save_all_stages:
                    for stage_name, stage_img in evaluation['stages'].items():
                        stage_path = os.path.join(result_dir, f'{filename}_{stage_name}.png')
                        cv2.imwrite(stage_path, stage_img)
                
                # Save parameters
                params_path = os.path.join(result_dir, f'{filename}_params.json')
                with open(params_path, 'w') as f:
                    json.dump(evaluation['params'], f, indent=2)
                
                # Save metrics if available
                if 'metrics' in evaluation:
                    metrics_path = os.path.join(result_dir, f'{filename}_metrics.json')
                    with open(metrics_path, 'w') as f:
                        json.dump({k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                 for k, v in evaluation['metrics'].items()}, f, indent=2)
                    
                    metrics.print_metrics(evaluation['metrics'], title=f"{filename} Results")
                
                # Save spectra if available
                if 'spectra' in evaluation.get('extra_info', {}):
                    spectra = evaluation['extra_info']['spectra']
                    spectrum_path = os.path.join(result_dir, f'{filename}_spectrum.png')
                    visualization.plot_spectrum_comparison(
                        spectra['before'], spectra['after'], save_path=spectrum_path
                    )
                
                print(f"\nPipeline B results saved to {result_dir}")

        elif args.pipeline == "A_old":
            # Original Assignment 1 baseline pipeline
            print("\n=== Running Original Pipeline A (Assignment 1) ===\n")
            processed = baseline_old.run_pipeline(image, OUTPUT_DIR, name=f"{filename}")
            aly.generate_histogram_from_image(processed, name=f"{filename}_processed")

        elif args.pipeline == "B_old":
            # Original Assignment 1 custom pipeline
            print("\n=== Running Original Pipeline B (Assignment 1) ===\n")
            processed = custom_old.run_custom_pipeline(
                image, OUTPUT_DIR, name=f"{filename}"
            )
            aly.generate_histogram_from_image(processed, name=f"{filename}_processed")

    print("\n" + "="*70)
    print("All images processed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
