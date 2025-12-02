"""
Visualization Module for Assignment 2
Generates figures, plots, and visual comparisons for the report

Functions for:
- Frequency spectrum visualization (before/after)
- Side-by-side image comparisons
- Segmentation overlays with color coding
- Multi-panel figure generation
- Crop extraction for detailed views
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def plot_spectrum_comparison(original_spectrum, filtered_spectrum, 
                             titles=None, save_path=None, figsize=(12, 5)):
    """
    Create side-by-side comparison of frequency spectra.
    
    Args:
        original_spectrum: Magnitude spectrum before filtering
        filtered_spectrum: Magnitude spectrum after filtering
        titles: Tuple of (original_title, filtered_title)
        save_path: Path to save figure (None to display)
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure object
    """
    if titles is None:
        titles = ('Original Spectrum', 'Filtered Spectrum')
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Original spectrum
    axes[0].imshow(original_spectrum, cmap='gray')
    axes[0].set_title(titles[0], fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Filtered spectrum
    axes[1].imshow(filtered_spectrum, cmap='gray')
    axes[1].set_title(titles[1], fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectrum comparison to {save_path}")
    
    return fig


def create_processing_pipeline_figure(image_dict, title="Processing Pipeline", 
                                      save_path=None, figsize=(16, 4)):
    """
    Create figure showing multiple processing stages in a row.
    
    Args:
        image_dict: OrderedDict or dict of {step_name: image}
        title: Overall figure title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure object
    """
    n_images = len(image_dict)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for ax, (step_name, img) in zip(axes, image_dict.items()):
        # Handle BGR to RGB conversion for color images
        if len(img.shape) == 3:
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_display = img
        
        ax.imshow(img_display, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(step_name, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved pipeline figure to {save_path}")
    
    return fig


def create_comparison_grid(images_dict, titles=None, save_path=None, 
                           figsize=(12, 8), cmap='gray'):
    """
    Create grid comparison of multiple images (2x2, 2x3, etc.).
    
    Args:
        images_dict: Dict of {name: image} or list of images
        titles: List of titles (optional)
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap for grayscale images
        
    Returns:
        fig: Matplotlib figure object
    """
    if isinstance(images_dict, dict):
        images = list(images_dict.values())
        if titles is None:
            titles = list(images_dict.keys())
    else:
        images = images_dict
        if titles is None:
            titles = [f"Image {i+1}" for i in range(len(images))]
    
    n_images = len(images)
    
    # Determine grid size
    if n_images <= 2:
        rows, cols = 1, n_images
    elif n_images <= 4:
        rows, cols = 2, 2
    elif n_images <= 6:
        rows, cols = 2, 3
    elif n_images <= 9:
        rows, cols = 3, 3
    else:
        rows = (n_images + 3) // 4
        cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        if idx >= len(axes):
            break
            
        # Handle BGR to RGB conversion
        if len(img.shape) == 3:
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img_display)
        else:
            axes[idx].imshow(img, cmap=cmap)
        
        axes[idx].set_title(title, fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison grid to {save_path}")
    
    return fig


def create_ablation_comparison(original, results_dict, gt_mask=None,
                               save_path=None, figsize=(16, 10)):
    """
    Create comprehensive ablation study visualization.
    Shows original + all variants + metrics.
    
    Args:
        original: Original image
        results_dict: Dict of {variant_name: (result_mask, metrics_dict)}
        gt_mask: Ground truth mask (optional)
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure object
    """
    n_variants = len(results_dict)
    n_cols = min(3, n_variants + 1)  # +1 for original
    n_rows = (n_variants + 1 + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)
    
    # Plot original
    ax = fig.add_subplot(gs[0, 0])
    if len(original.shape) == 3:
        ax.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(original, cmap='gray')
    ax.set_title('Original', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Plot each variant
    for idx, (variant_name, (result_mask, metrics)) in enumerate(results_dict.items(), 1):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Show result mask
        ax.imshow(result_mask, cmap='gray')
        
        # Create title with metrics
        if metrics:
            title = f"{variant_name}\n"
            title += f"Dice: {metrics.get('dice', 0):.3f} | "
            title += f"IoU: {metrics.get('iou', 0):.3f}"
            ax.set_title(title, fontsize=9, fontweight='bold')
        else:
            ax.set_title(variant_name, fontsize=9, fontweight='bold')
        
        ax.axis('off')
    
    plt.suptitle('Ablation Study Comparison', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ablation comparison to {save_path}")
    
    return fig


def create_detailed_result_figure(original, filtered, segmented, cleaned,
                                  crop_coords=None, save_path=None):
    """
    Create comprehensive result figure with original, processing steps, and crop.
    
    Args:
        original: Original image
        filtered: Frequency-filtered image
        segmented: Segmentation result
        cleaned: Final cleaned result
        crop_coords: (x, y, width, height) for crop region
        save_path: Path to save figure
        
    Returns:
        fig: Matplotlib figure object
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.25)
    
    # Row 1: Processing pipeline
    images_row1 = [
        (original, 'Original', None),
        (filtered, 'Frequency Filtered', None),
        (segmented, 'Segmented', 'gray'),
        (cleaned, 'Morphology Cleaned', 'gray')
    ]
    
    for idx, (img, title, cmap) in enumerate(images_row1):
        ax = fig.add_subplot(gs[0, idx])
        if len(img.shape) == 3 and cmap is None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Draw crop rectangle if specified
        if crop_coords and idx == 0:
            x, y, w, h = crop_coords
            rect = plt.Rectangle((x, y), w, h, fill=False, 
                                edgecolor='red', linewidth=2)
            ax.add_patch(rect)
    
    # Row 2: Detailed crops
    if crop_coords:
        x, y, w, h = crop_coords
        crops = [
            (original[y:y+h, x:x+w], 'Original Crop'),
            (filtered[y:y+h, x:x+w], 'Filtered Crop'),
            (segmented[y:y+h, x:x+w], 'Segmented Crop'),
            (cleaned[y:y+h, x:x+w], 'Cleaned Crop')
        ]
        
        for idx, (crop, title) in enumerate(crops):
            ax = fig.add_subplot(gs[1, idx])
            if len(crop.shape) == 3:
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(crop, cmap='gray')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle('Detailed Processing Results', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved detailed result figure to {save_path}")
    
    return fig


def extract_crops(image, crop_size=300, num_crops=4, strategy='grid'):
    """
    Extract multiple crops from image for detailed viewing.
    
    Args:
        image: Input image
        crop_size: Size of square crops
        num_crops: Number of crops to extract
        strategy: 'grid', 'corners', or 'interest'
        
    Returns:
        crops: List of (crop_image, (x, y)) tuples
    """
    h, w = image.shape[:2]
    crops = []
    
    if strategy == 'grid':
        # Evenly spaced grid
        rows = int(np.sqrt(num_crops))
        cols = (num_crops + rows - 1) // rows
        
        for i in range(rows):
            for j in range(cols):
                if len(crops) >= num_crops:
                    break
                
                x = int(j * (w - crop_size) / max(1, cols - 1))
                y = int(i * (h - crop_size) / max(1, rows - 1))
                
                x = min(x, w - crop_size)
                y = min(y, h - crop_size)
                
                crop = image[y:y+crop_size, x:x+crop_size]
                crops.append((crop, (x, y)))
    
    elif strategy == 'corners':
        # Four corners + center
        positions = [
            (0, 0),  # Top-left
            (w - crop_size, 0),  # Top-right
            (0, h - crop_size),  # Bottom-left
            (w - crop_size, h - crop_size),  # Bottom-right
            ((w - crop_size) // 2, (h - crop_size) // 2)  # Center
        ]
        
        for x, y in positions[:num_crops]:
            crop = image[y:y+crop_size, x:x+crop_size]
            crops.append((crop, (x, y)))
    
    elif strategy == 'interest':
        # Find regions with high edge content
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute edge strength
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide into grid and find high-edge regions
        grid_size = crop_size // 2
        interest_scores = []
        
        for y in range(0, h - crop_size, grid_size):
            for x in range(0, w - crop_size, grid_size):
                region = edges[y:y+crop_size, x:x+crop_size]
                score = np.sum(region)
                interest_scores.append((score, x, y))
        
        # Sort by interest and take top N
        interest_scores.sort(reverse=True)
        
        for _, x, y in interest_scores[:num_crops]:
            crop = image[y:y+crop_size, x:x+crop_size]
            crops.append((crop, (x, y)))
    
    return crops


def save_crops_grid(crops, save_path, titles=None):
    """
    Save multiple crops as a single grid image.
    
    Args:
        crops: List of crop images or list of (crop, coords) tuples
        save_path: Path to save grid image
        titles: Optional titles for each crop
    """
    # Extract just images if tuples provided
    if isinstance(crops[0], tuple):
        crop_images = [crop[0] for crop in crops]
    else:
        crop_images = crops
    
    n_crops = len(crop_images)
    cols = min(4, n_crops)
    rows = (n_crops + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if n_crops > 1 else [axes]
    
    for idx, crop in enumerate(crop_images):
        if len(crop.shape) == 3:
            axes[idx].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        else:
            axes[idx].imshow(crop, cmap='gray')
        
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx], fontsize=10, fontweight='bold')
        else:
            axes[idx].set_title(f'Crop {idx+1}', fontsize=10, fontweight='bold')
        
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(crop_images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved crops grid to {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path=None, figsize=(10, 6)):
    """
    Create bar plot comparing metrics across different methods.
    
    Args:
        metrics_dict: Dict of {method_name: metrics_dict}
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure object
    """
    methods = list(metrics_dict.keys())
    metric_names = ['dice', 'iou', 'precision', 'recall', 'f1_score']
    
    # Extract values
    data = {metric: [] for metric in metric_names}
    for method in methods:
        for metric in metric_names:
            data[metric].append(metrics_dict[method].get(metric, 0))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(methods))
    width = 0.15
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (metric, color) in enumerate(zip(metric_names, colors)):
        offset = width * (idx - len(metric_names) / 2)
        ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title(), 
               color=color, alpha=0.8)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Segmentation Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    
    return fig


def create_report_figure_set(image_path, results, output_dir, 
                             image_name='image'):
    """
    Generate complete set of figures for report from a single image.
    
    Args:
        image_path: Path to original image
        results: Dict containing all processing results
        output_dir: Directory to save figures
        image_name: Base name for output files
        
    Returns:
        figure_paths: Dict of generated figure paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    original = cv2.imread(image_path)
    figure_paths = {}
    
    # 1. Spectrum comparison
    if 'spectra' in results:
        path = os.path.join(output_dir, f'{image_name}_spectrum.png')
        plot_spectrum_comparison(
            results['spectra']['before'],
            results['spectra']['after'],
            save_path=path
        )
        figure_paths['spectrum'] = path
        plt.close()
    
    # 2. Processing pipeline
    if 'pipeline_stages' in results:
        path = os.path.join(output_dir, f'{image_name}_pipeline.png')
        create_processing_pipeline_figure(
            results['pipeline_stages'],
            title=f"Processing Pipeline - {image_name}",
            save_path=path
        )
        figure_paths['pipeline'] = path
        plt.close()
    
    # 3. Detailed results with crops
    if all(k in results for k in ['filtered', 'segmented', 'cleaned']):
        path = os.path.join(output_dir, f'{image_name}_detailed.png')
        
        # Extract interesting crop
        crops = extract_crops(original, crop_size=300, num_crops=1, strategy='interest')
        crop_coords = (crops[0][1][0], crops[0][1][1], 300, 300)
        
        create_detailed_result_figure(
            original,
            results['filtered'],
            results['segmented'],
            results['cleaned'],
            crop_coords=crop_coords,
            save_path=path
        )
        figure_paths['detailed'] = path
        plt.close()
    
    # 4. Metrics visualization
    if 'metrics' in results:
        from dip.metrics import overlay_segmentation
        
        overlay = overlay_segmentation(
            original,
            results['cleaned'],
            results.get('ground_truth'),
            alpha=0.5
        )
        
        path = os.path.join(output_dir, f'{image_name}_overlay.png')
        cv2.imwrite(path, overlay)
        figure_paths['overlay'] = path
    
    print(f"\nGenerated {len(figure_paths)} figures for {image_name}")
    return figure_paths
