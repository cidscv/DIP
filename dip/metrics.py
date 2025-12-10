import numpy as np
import cv2


def dice_coefficient(pred_mask, gt_mask):
    """
    Compute Dice coefficient (F1-score) between predicted and ground truth masks.
    Dice = 2*TP / (2*TP + FP + FN)
    Range: [0, 1], where 1 is perfect overlap.
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        dice: Dice coefficient
    """
    # Ensure binary masks
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Compute intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    
    # Handle edge case where both masks are empty
    if pred_sum + gt_sum == 0:
        return 1.0
    
    # Compute Dice coefficient
    dice = (2.0 * intersection) / (pred_sum + gt_sum)
    
    return dice


def iou_score(pred_mask, gt_mask):
    """
    Compute Intersection over Union (IoU / Jaccard index).
    IoU = TP / (TP + FP + FN) = Intersection / Union
    Range: [0, 1], where 1 is perfect overlap.
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        iou: Intersection over Union score
    """
    # Ensure binary masks
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Compute intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # Handle edge case where both masks are empty
    if union == 0:
        return 1.0
    
    # Compute IoU
    iou = intersection / union
    
    return iou


def pixel_accuracy(pred_mask, gt_mask):
    """
    Compute pixel-wise accuracy.
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        accuracy: Pixel accuracy
    """
    # Ensure binary masks
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Compute accuracy
    correct = (pred_binary == gt_binary).sum()
    total = pred_binary.size
    
    accuracy = correct / total
    
    return accuracy


def precision_recall(pred_mask, gt_mask):
    """
    Compute precision and recall.
    Precision = TP / (TP + FP) - how many predicted positives are correct
    Recall = TP / (TP + FN) - how many actual positives were found
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        precision: Precision score
        recall: Recall score
    """
    # Ensure binary masks
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Compute confusion matrix components
    tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
    fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
    fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()
    
    # Compute precision
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    # Compute recall
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    return precision, recall


def confusion_matrix_components(pred_mask, gt_mask):
    """
    Compute all confusion matrix components: TP, TN, FP, FN.
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        components: Dict with 'tp', 'tn', 'fp', 'fn'
    """
    # Ensure binary masks
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # Compute components
    tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
    tn = np.logical_and(pred_binary == 0, gt_binary == 0).sum()
    fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
    fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()
    
    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def f1_score(pred_mask, gt_mask):
    """
    Compute F1-score (same as Dice coefficient).
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        f1: F1-score
    """
    precision, recall = precision_recall(pred_mask, gt_mask)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def specificity(pred_mask, gt_mask):
    """
    Compute specificity (true negative rate).
    Specificity = TN / (TN + FP)
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        spec: Specificity score
    """
    components = confusion_matrix_components(pred_mask, gt_mask)
    tn = components['tn']
    fp = components['fp']
    
    if tn + fp == 0:
        return 0.0
    
    spec = tn / (tn + fp)
    
    return spec


def compute_all_metrics(pred_mask, gt_mask):
    """
    Compute all segmentation metrics at once.
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        metrics: Dict containing all metric values
    """
    dice = dice_coefficient(pred_mask, gt_mask)
    iou = iou_score(pred_mask, gt_mask)
    accuracy = pixel_accuracy(pred_mask, gt_mask)
    precision, recall = precision_recall(pred_mask, gt_mask)
    f1 = f1_score(pred_mask, gt_mask)
    spec = specificity(pred_mask, gt_mask)
    components = confusion_matrix_components(pred_mask, gt_mask)
    
    metrics = {
        'dice': dice,
        'iou': iou,
        'pixel_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': spec,
        'tp': components['tp'],
        'tn': components['tn'],
        'fp': components['fp'],
        'fn': components['fn']
    }
    
    return metrics


def print_metrics(metrics, title="Segmentation Metrics"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dict of metric values (from compute_all_metrics)
        title: Title for the printout
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"Dice Coefficient:    {metrics['dice']:.4f}")
    print(f"IoU (Jaccard):       {metrics['iou']:.4f}")
    print(f"Pixel Accuracy:      {metrics['pixel_accuracy']:.4f}")
    print(f"Precision:           {metrics['precision']:.4f}")
    print(f"Recall:              {metrics['recall']:.4f}")
    print(f"F1-Score:            {metrics['f1_score']:.4f}")
    print(f"Specificity:         {metrics['specificity']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:    {metrics['tp']:,}")
    print(f"  True Negatives:    {metrics['tn']:,}")
    print(f"  False Positives:   {metrics['fp']:,}")
    print(f"  False Negatives:   {metrics['fn']:,}")
    print(f"{'='*50}\n")


def overlay_segmentation(image, pred_mask, gt_mask=None, alpha=0.5):
    """
    Create visualization overlay of segmentation results.
    
    Args:
        image: Original image (BGR)
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth mask (optional)
        alpha: Transparency of overlay
        
    Returns:
        overlay: Visualization image
    """
    # Ensure image is color
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    
    # Create colored overlay
    overlay = image_color.copy()
    
    # Predicted mask in green
    pred_binary = (pred_mask > 0).astype(np.uint8)
    overlay[pred_binary == 1] = [0, 255, 0]  # Green
    
    # If ground truth provided, show differences
    if gt_mask is not None:
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # False positives in red
        fp_mask = np.logical_and(pred_binary == 1, gt_binary == 0)
        overlay[fp_mask] = [0, 0, 255]  # Red
        
        # False negatives in blue
        fn_mask = np.logical_and(pred_binary == 0, gt_binary == 1)
        overlay[fn_mask] = [255, 0, 0]  # Blue
        
        # True positives in green (already set above)
    
    # Blend with original image
    result = cv2.addWeighted(image_color, 1 - alpha, overlay, alpha, 0)
    
    return result


def compare_segmentations(image, masks_dict, gt_mask=None):
    """
    Compare multiple segmentation results side by side.
    
    Args:
        image: Original image
        masks_dict: Dict of {name: mask} for different methods
        gt_mask: Ground truth mask (optional)
        
    Returns:
        comparison_results: Dict with metrics for each method
    """
    results = {}
    
    for name, mask in masks_dict.items():
        if gt_mask is not None:
            metrics = compute_all_metrics(mask, gt_mask)
            results[name] = metrics
        else:
            results[name] = {'mask': mask}
    
    return results


def boundary_error(pred_mask, gt_mask, distance_threshold=2):
    """
    Compute boundary-based error metric.
    Measures distance between predicted and ground truth boundaries.
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        distance_threshold: Distance threshold for considering boundary match
        
    Returns:
        boundary_f1: F1-score based on boundary matches
        avg_distance: Average distance between boundaries
    """
    # Find contours (boundaries)
    pred_contours, _ = cv2.findContours(
        (pred_mask > 0).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE
    )
    gt_contours, _ = cv2.findContours(
        (gt_mask > 0).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE
    )
    
    # Create boundary images
    pred_boundary = np.zeros_like(pred_mask)
    gt_boundary = np.zeros_like(gt_mask)
    cv2.drawContours(pred_boundary, pred_contours, -1, 255, 1)
    cv2.drawContours(gt_boundary, gt_contours, -1, 255, 1)
    
    # Compute distance transform
    pred_dist = cv2.distanceTransform(
        255 - pred_boundary, cv2.DIST_L2, 3
    )
    gt_dist = cv2.distanceTransform(
        255 - gt_boundary, cv2.DIST_L2, 3
    )
    
    # Find boundary matches
    pred_coords = np.argwhere(pred_boundary > 0)
    gt_coords = np.argwhere(gt_boundary > 0)
    
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return 0.0, float('inf')
    
    # Compute average distance
    pred_to_gt_dist = np.mean([gt_dist[y, x] for y, x in pred_coords])
    gt_to_pred_dist = np.mean([pred_dist[y, x] for y, x in gt_coords])
    avg_distance = (pred_to_gt_dist + gt_to_pred_dist) / 2
    
    # Compute boundary F1
    pred_matches = np.sum([gt_dist[y, x] <= distance_threshold for y, x in pred_coords])
    gt_matches = np.sum([pred_dist[y, x] <= distance_threshold for y, x in gt_coords])
    
    precision = pred_matches / len(pred_coords) if len(pred_coords) > 0 else 0
    recall = gt_matches / len(gt_coords) if len(gt_coords) > 0 else 0
    
    if precision + recall == 0:
        boundary_f1 = 0.0
    else:
        boundary_f1 = 2 * (precision * recall) / (precision + recall)
    
    return boundary_f1, avg_distance


def volume_similarity(pred_mask, gt_mask):
    """
    Compute volume similarity coefficient.
    VS = 1 - |FP - FN| / (2*TP + FP + FN)
    
    Args:
        pred_mask: Predicted binary segmentation mask
        gt_mask: Ground truth binary segmentation mask
        
    Returns:
        vs: Volume similarity score
    """
    components = confusion_matrix_components(pred_mask, gt_mask)
    tp = components['tp']
    fp = components['fp']
    fn = components['fn']
    
    denominator = 2 * tp + fp + fn
    
    if denominator == 0:
        return 1.0
    
    vs = 1 - abs(fp - fn) / denominator
    
    return vs
