import numpy as np
import torch
import cv2
import json
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional, List, Union
from termcolor import cprint

def evaluate_rgb_consistency(trajectory_maps: np.ndarray, rgb_images: List[np.ndarray], output_dir: str = None, visibility_maps: np.ndarray = None, visibility_threshold: float = 0.5) -> dict:
    """
    Transfer first-frame RGB values to later frames with trajectory_maps and compare them to real RGB values.
    
    Args:
        trajectory_maps: (T, 2, H, W) trajectory maps.
        rgb_images: List of (H, W, 3) RGB images.
        output_dir: Output directory.
        visibility_maps: Optional (T, H, W) visibility maps.
        visibility_threshold: Visibility threshold; only pixels above this threshold are evaluated.
    
    Returns:
        dict: RGB error statistics.
    """
    # Handle different input shapes.
    if len(trajectory_maps.shape) == 5:
        # (1, T, 2, H, W) -> (T, 2, H, W)
        trajectory_maps = trajectory_maps[0]
    elif len(trajectory_maps.shape) == 3:
        # (T, H, W) needs reorganization.
        print(f"Warning: trajectory_maps shape is {trajectory_maps.shape}, expected (T, 2, H, W)")
        return {'mean_rgb_error': float('inf'), 'max_rgb_error': float('inf'), 'min_rgb_error': float('inf'), 'errors_per_frame': []}
    
    if len(trajectory_maps.shape) != 4:
        print(f"Error: trajectory_maps shape is {trajectory_maps.shape}, expected (T, 2, H, W)")
        return {'mean_rgb_error': float('inf'), 'max_rgb_error': float('inf'), 'min_rgb_error': float('inf'), 'errors_per_frame': []}
    
    T, _, H, W = trajectory_maps.shape
    first_rgb = rgb_images[0]  # First frame is the reference.
    
    # Handle visibility_maps.
    if visibility_maps is not None:
        if hasattr(visibility_maps, 'cpu'):
            visibility_maps = visibility_maps.cpu().numpy()
        if len(visibility_maps.shape) == 5:  # (1, T, H, W) -> (T, H, W)
            visibility_maps = visibility_maps[0]
    
    rgb_errors = []
    valid_pixel_counts = []  # Number of valid pixels per frame.
    
    # Images used for visualization.
    vis_images = []
    
    for t in range(1, T):  # Start from the second frame.
        current_rgb = rgb_images[t]
        tx_map = trajectory_maps[t, 0]  # x coordinate.
        ty_map = trajectory_maps[t, 1]  # y coordinate.
        
        # Create transferred RGB image.
        transferred_rgb = np.zeros_like(current_rgb)
        
        # Compute RGB error for each pixel.
        pixel_errors = []
        valid_pixels = 0
        
        for i in range(H):
            for j in range(W):
                # Check visibility if visibility_maps are provided.
                if visibility_maps is not None:
                    visibility = visibility_maps[t, 0, i, j]
                    if visibility < visibility_threshold:
                        continue  # Skip low-visibility pixels.
                
                target_x = int(round(tx_map[i, j]))
                target_y = int(round(ty_map[i, j]))
                
                # Check whether the target position is inside the image.
                if 0 <= target_x < W and 0 <= target_y < H:
                    ref_rgb = first_rgb[i, j]  # RGB in the first frame.
                    cur_rgb = current_rgb[target_y, target_x]  # RGB at the corresponding current-frame position.
                    
                    # Transfer first-frame RGB into the transferred image.
                    transferred_rgb[target_y, target_x] = ref_rgb
                    
                    # Compute RGB difference.
                    rgb_diff = np.abs(ref_rgb.astype(float) - cur_rgb.astype(float))
                    pixel_error = np.mean(rgb_diff)  # Mean RGB error.
                    pixel_errors.append(pixel_error)
                    valid_pixels += 1
        
        # Save six comparison panels (plus visibility/confidence visualizations) in a 2x3 layout.
        if output_dir:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # First frame (reference), top-left.
            axes[0, 0].imshow(first_rgb)
            axes[0, 0].set_title('Reference Frame')
            axes[0, 0].axis('off')
            
            # Transferred RGB, top-center.
            axes[0, 1].imshow(transferred_rgb)
            axes[0, 1].set_title(f'Transferred RGB (Frame {t})')
            axes[0, 1].axis('off')
            
            # Actual current-frame RGB, top-right.
            axes[0, 2].imshow(current_rgb)
            axes[0, 2].set_title(f'Actual Frame {t}')
            axes[0, 2].axis('off')
            
            # Visibility map, bottom-left.
            if visibility_maps is not None:
                vis_map = visibility_maps[t, 0] if t < visibility_maps.shape[0] else np.zeros((H, W))
                conf_map = visibility_maps[t, 1] if visibility_maps.shape[1] > 1 else None
                
                im1 = axes[1, 0].imshow(vis_map, cmap='viridis', vmin=0, vmax=1)
                axes[1, 0].set_title(f'Visibility (threshold={visibility_threshold})')
                axes[1, 0].axis('off')
                
                # Confidence map, bottom-center.
                if conf_map is not None:
                    im2 = axes[1, 1].imshow(conf_map, cmap='RdYlGn', vmin=0, vmax=1)  # Red=low confidence, green=high confidence.
                    axes[1, 1].set_title('Confidence Map (Green=High, Red=Low)')
                    axes[1, 1].axis('off')
                else:
                    # Show a blank image when confidence data is unavailable.
                    dummy_img = np.zeros((H, W))
                    axes[1, 1].imshow(dummy_img, cmap='gray', vmin=0, vmax=1)
                    axes[1, 1].set_title('No Confidence Data')
                    axes[1, 1].axis('off')
                
                # RGB-difference heatmap, bottom-right.
                rgb_diff_map = np.zeros((H, W))
                for i in range(H):
                    for j in range(W):
                        target_x = int(round(tx_map[i, j]))
                        target_y = int(round(ty_map[i, j]))
                        if 0 <= target_x < W and 0 <= target_y < H:
                            ref_rgb = first_rgb[i, j]
                            cur_rgb = current_rgb[target_y, target_x]
                            rgb_diff = np.mean(np.abs(ref_rgb.astype(float) - cur_rgb.astype(float)))
                            rgb_diff_map[target_y, target_x] = rgb_diff
                
                im3 = axes[1, 2].imshow(rgb_diff_map, cmap='hot', vmin=0, vmax=np.percentile(rgb_diff_map[rgb_diff_map > 0], 95))
                axes[1, 2].set_title('RGB Error Map')
                axes[1, 2].axis('off')
                
            else:
                # Show blank images to keep the layout consistent when visibility data is unavailable.
                for i in range(3):
                    dummy_img = np.zeros((H, W))
                    axes[1, i].imshow(dummy_img, cmap='gray', vmin=0, vmax=1)
                    titles = ['No Visibility Data', 'No Confidence Data', 'No Error Map']
                    axes[1, i].set_title(titles[i])
                    axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'rgb_comparison_frame_{t:03d}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        valid_pixel_counts.append(valid_pixels)
        if pixel_errors:
            frame_error = np.mean(pixel_errors)
            rgb_errors.append(frame_error)
        else:
            rgb_errors.append(float('inf'))
    
    # Return simple statistics.
    valid_errors = [e for e in rgb_errors if e != float('inf')]
    
    return {
        'mean_rgb_error': np.mean(valid_errors) if valid_errors else float('inf'),
        'max_rgb_error': np.max(valid_errors) if valid_errors else float('inf'),
        'min_rgb_error': np.min(valid_errors) if valid_errors else float('inf'),
        'errors_per_frame': rgb_errors,
        'valid_pixel_counts': valid_pixel_counts,
        'visibility_threshold': visibility_threshold,
        'total_valid_pixels': sum(valid_pixel_counts)
    }





def evaluate_trajectory_maps(trajectory_maps, visible_maps, window_images_dict, output_dir, visibility_threshold=0.5):
    """
    Simple RGB consistency evaluation with visibility filtering.
    
    Args:
        trajectory_maps: Trajectory maps.
        visible_maps: Visibility maps.
        window_images_dict: Image dictionary.
        output_dir: Output directory.
        visibility_threshold: Visibility threshold.
    """
    # Convert to numpy if needed.
    if hasattr(trajectory_maps, 'cpu'):
        trajectory_maps_np = trajectory_maps.cpu().numpy()
    else:
        trajectory_maps_np = trajectory_maps
    
    # Run RGB consistency evaluation.
    cprint(f"  RGB Consistency Evaluation (visibility_threshold={visibility_threshold}):", "blue")
    rgb_images = window_images_dict["rgbs"]
    
    rgb_results = evaluate_rgb_consistency(
        trajectory_maps_np, 
        rgb_images, 
        output_dir,
        visibility_maps=visible_maps,
        visibility_threshold=visibility_threshold
    )
    
    # Print results.
    if rgb_results['mean_rgb_error'] != float('inf'):
        cprint(f"    Mean RGB Error: {rgb_results['mean_rgb_error']:.3f}", "blue")
        cprint(f"    Max RGB Error: {rgb_results['max_rgb_error']:.3f}", "blue") 
        cprint(f"    Min RGB Error: {rgb_results['min_rgb_error']:.3f}", "blue")
        cprint(f"    Total Valid Pixels: {rgb_results['total_valid_pixels']}", "blue")
    else:
        cprint(f"    Warning: No valid RGB comparisons", "yellow")
    
    # Save results.
    # results_file = os.path.join(output_dir, "rgb_evaluation.json")
    # with open(results_file, 'w') as f:
    #     json.dump(rgb_results, f, indent=2)

    # cprint(f"    Results saved to: {results_file}", "blue")


def check_segmentation_consistency(final_point_seg_weights, point_indices, segmentation_maps, valid_seg_masks, T1, T2):
    """
    Check segmentation consistency and choose the better segmentation direction.
    
    Args:
        final_point_seg_weights: Optimized segmentation weights, shape (N, 1).
        point_indices: Point indices (y_indices, x_indices).
        segmentation_maps: Segmentation map from the previous window, shape (H, W).
        valid_seg_masks: Valid segmentation mask, shape (H, W).
    
    Returns:
        corrected_weights: Corrected segmentation weights.
        transform_t1, transform_t2: Transform matrices, possibly swapped.
        consistency_info: Consistency-check information.
    """
    if segmentation_maps is None or valid_seg_masks is None:
        cprint("    No reference segmentation available, using original weights", "blue")
        return final_point_seg_weights, T1, T2

    y_indices, x_indices = point_indices
    final_weights = final_point_seg_weights.reshape(-1)
    prev_seg = segmentation_maps[y_indices, x_indices].reshape(-1)
    valid_point_mask = valid_seg_masks[y_indices, x_indices].reshape(-1).bool()

    if not bool(valid_point_mask.any()):
        cprint("    No valid points for consistency check, using original weights", "yellow")
        return final_point_seg_weights, T1, T2

    valid_final = final_weights[valid_point_mask]
    valid_prev = prev_seg[valid_point_mask]
    valid_final_reverse = 1.0 - valid_final

    loss_original = torch.mean((valid_final - valid_prev) ** 2)
    loss_reverse = torch.mean((valid_final_reverse - valid_prev) ** 2)

    loss_original_val = float(loss_original.item())
    loss_reverse_val = float(loss_reverse.item())

    if loss_reverse_val < loss_original_val:
        cprint(
            f"    Reversed segmentation selected (loss_reverse={loss_reverse_val:.4f} < loss_original={loss_original_val:.4f})",
            "yellow",
        )
        return (1.0 - final_weights).reshape(final_point_seg_weights.shape), T2, T1

    cprint(
        f"    Original segmentation retained (loss_original={loss_original_val:.4f} <= loss_reverse={loss_reverse_val:.4f})",
        "green",
    )
    return final_point_seg_weights, T1, T2
