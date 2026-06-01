import os
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
import shutil
import argparse
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import ImageLoader
import numpy as np
from termcolor import cprint
import cv2
from PIL import Image
from preprocessing.grounded_sam_toolkit import propagate_mask_in_video, load_model_once


def load_seg_map(tracking_dir, target_size=None):

    """
    Load segmentation maps from tracking directory
    Args:
        tracking_dir: Directory containing window folders with seg_map files
        target_size: (width, height) tuple for resizing seg_maps, if None then no resize
    """
    window_dirs = []
    for subdir in os.listdir(tracking_dir):
        full_path = os.path.join(tracking_dir, subdir)
        if os.path.isdir(full_path) and subdir.startswith("window_"):
            window_dirs.append(full_path)
    window_dirs.sort()  #
    seg_map_list = None
    key_frame_index = []
    for window_dir in window_dirs:
        file_list = []
        for fname in os.listdir(window_dir):
            if fname.startswith("seg_map_") and fname.endswith(".npy"):
                seg_map_path = os.path.join(window_dir, fname)
                file_list.append(seg_map_path)
        file_list.sort()
        for i, file_path in enumerate(file_list):
            seg_map = np.load(file_path)
            original_shape = seg_map.shape
            # Resize seg_map to target size if specified
            if target_size is not None:
                seg_map = cv2.resize(seg_map, target_size, interpolation=cv2.INTER_LINEAR)
                # if i == 0:  # Only print for first file in each window to avoid spam
                #     cprint(f"Resized seg_map from {original_shape} to {seg_map.shape}", "cyan")
            if i == 0:
                if seg_map_list is None:
                    seg_map_list = [seg_map]
                    key_frame_index.append(0)
                else:
                    seg_map_list[-1]=(seg_map)
                    key_frame_index.append(len(seg_map_list) - 1)
            else:
                seg_map_list.append(seg_map)
    cprint(f"Loaded {len(seg_map_list)} segmentation maps from {len(window_dirs)} windows.", "green")
    return seg_map_list, key_frame_index

def convert_image_to_jpg(window_dir, original_image_path):
    """
    Create an image_ori_jpg folder for window_dir and convert the required images to JPG.
    
    Args:
        window_dir: Path to the window directory.
        original_image_path: Path to the original image directory.
    """
    # Check whether image_ori_jpg already exists.
    image_ori_jpg_dir = os.path.join(window_dir, "image_ori_jpg")
    
    if os.path.exists(image_ori_jpg_dir) and os.listdir(image_ori_jpg_dir):

        return
    
    # Read window_info.json to get the required frame indices.
    window_info_path = os.path.join(window_dir, "window_info.json")
    
    with open(window_info_path, 'r') as f:
        window_info = json.load(f)
    
    frame_indices = window_info.get("frame_indices", [])
    os.makedirs(image_ori_jpg_dir, exist_ok=True)
    
    for frame_idx in frame_indices:

        jpg_img_name = f"{frame_idx:05d}.jpg"
        png_img_name = f"{frame_idx:05d}.png"
        
        original_img_path = None
        original_img_name = None
        
        # Try JPG format first.
        jpg_path = os.path.join(original_image_path, jpg_img_name)
        if os.path.exists(jpg_path):
            original_img_path = jpg_path
            original_img_name = jpg_img_name

        # Output JPG path.
        jpg_img_name = f"{frame_idx:05d}.jpg"
        jpg_img_path = os.path.join(image_ori_jpg_dir, jpg_img_name)
        

        shutil.copy2(original_img_path, jpg_img_path)
        

def batch_convert_windows_to_jpg(tracking_dir, original_image_path):
    """
    Batch-create JPG images for all window directories.
    
    Args:
        tracking_dir: Path to the tracking directory containing window_xxxxx subdirectories.
        original_image_path: Path to the original image directory.
    """
    window_dirs = []
    for subdir in os.listdir(tracking_dir):
        full_path = os.path.join(tracking_dir, subdir)
        if os.path.isdir(full_path) and subdir.startswith("window_"):
            window_dirs.append(full_path)
    
    window_dirs.sort()
    cprint(f"Found {len(window_dirs)} window directories", "blue")
    
    for window_dir in window_dirs:
        convert_image_to_jpg(window_dir, original_image_path)
    
    cprint("Batch conversion completed.", "green")


def split_seg_maps_to_masks(args, seg_map_list, key_frame_index, threshold=0.5):
    """
    Split each seg_map in seg_map_list into two masks.
    
    Args:
        seg_map_list: List containing seg_maps.
        threshold: Segmentation threshold, default 0.5.
    
    Returns:
        mask_pairs: List of (mask_high, mask_low) tuples.
    """
    mask1_list = []
    mask2_list = []

    object_mask_dir = os.path.join(args.data_dir, "masks")
    tracking_dir = os.path.join(args.data_dir, "tracking")
    
    # Get all window directories.
    window_dirs = []
    for subdir in os.listdir(tracking_dir):
        full_path = os.path.join(tracking_dir, subdir)
        if os.path.isdir(full_path) and subdir.startswith("window_"):
            window_dirs.append(full_path)
    window_dirs.sort()
    
    # Create directories for saving masks.
    mask1_output_dir = os.path.join(args.data_dir, "masks1")
    os.makedirs(mask1_output_dir, exist_ok=True)
    mask2_output_dir = os.path.join(args.data_dir, "masks2")
    os.makedirs(mask2_output_dir, exist_ok=True)

    # Only process seg_maps specified by key_frame_index.
    for window_idx, i in enumerate(key_frame_index):
        if i < len(seg_map_list):
            seg_map = seg_map_list[i]
            
            # Create two masks.
            mask1 = (seg_map > threshold).astype(np.uint8)  # Region above threshold.
            mask2 = (seg_map <= threshold).astype(np.uint8)   # Region below or equal to threshold.

            # Use the actual frame index.
            frame_idx = i
            object_mask_path = os.path.join(object_mask_dir, f"{frame_idx:05d}.png")
            object_mask = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
            
            if object_mask is not None:
                object_mask = (object_mask > 127).astype(np.uint8)

                mask1 = mask1 * object_mask
                mask2 = mask2 * object_mask

            # Save masks under the corresponding window directory.
            if window_idx < len(window_dirs):
                window_dir = window_dirs[window_idx]
                
                # Create a masks subdirectory under the window directory.
                window_mask_dir = os.path.join(window_dir, "masks")
                os.makedirs(window_mask_dir, exist_ok=True)
                
                # Save mask files.
                mask1_path = os.path.join(window_mask_dir, f"mask1.png")
                mask2_path = os.path.join(window_mask_dir, f"mask2.png")

                mask1_save = mask1 * 255
                mask2_save = mask2 * 255
                
                cv2.imwrite(mask1_path, mask1_save)
                cv2.imwrite(mask2_path, mask2_save)


            mask1_save = mask1 * 255
            mask2_save = mask2 * 255
            
            cv2.imwrite(mask1_path, mask1_save)
            cv2.imwrite(mask2_path, mask2_save)

            mask1_list.append(mask1)
            mask2_list.append(mask2)
 
    return mask1_list, mask2_list

def visualize_propagated_masks(image_dir, mask_dir, output_dir, alpha=0.5, mask_color=(0, 255, 0)):
    """
    Visualize masks generated by propagate_mask_in_video on original images.
    
    Args:
        image_dir: Original image directory.
        mask_dir: Mask directory (output directory of propagate_mask_in_video).
        output_dir: Directory for visualization results.
        alpha: Opacity in [0, 1].
        mask_color: Mask color in BGR format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files.
    image_files = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(fname)
    image_files.sort()
    
    # Get all mask files.
    mask_files = []
    for fname in os.listdir(mask_dir):
        if fname.lower().endswith('.png'):
            mask_files.append(fname)
    mask_files.sort()
    
    cprint(f"Found {len(image_files)} images and {len(mask_files)} masks", "blue")
    
    for img_file in image_files:
        # Build the corresponding mask filename.
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}.png"
        
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        if not os.path.exists(mask_path):
            cprint(f"Mask not found for {img_file}, skipping", "yellow")
            continue
            
        # Read image and mask.
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            cprint(f"Failed to load {img_file} or {mask_file}", "red")
            continue
            
        # Ensure matching dimensions.
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask.
        mask_binary = (mask > 127).astype(np.uint8)
        colored_mask = np.zeros_like(image)
        colored_mask[mask_binary == 1] = mask_color
        
        # Blend original image and mask.
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        # Draw contours on the mask boundary.
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, mask_color, 2)
        
        # Save result.
        output_path = os.path.join(output_dir, f"overlay_{base_name}.png")
        cv2.imwrite(output_path, overlay)
    
    cprint(f"Visualization saved to {output_dir}", "green")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and resize segmentation maps to match RGB image size")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--bg_color', type=str, default='black', help='Background color for visualization')
    parser.add_argument('--masked_rgb_dir', type=str, default='masked_rgb', help='Directory to save masked RGB images')
    args = parser.parse_args()
    
    imageloader = ImageLoader(args.data_dir)
    
    # Get RGB image size for resizing seg_maps
    rgbs = imageloader.rgbs
    masks = imageloader.masks
    # Get size from first image (height, width, channels)
    rgb_height, rgb_width = rgbs[0].shape[:2]
    target_size = (rgb_width, rgb_height)  # OpenCV uses (width, height)
    cprint(f"RGB image size: {rgb_width}x{rgb_height}, will resize seg_maps to this size", "blue")
    
    seg_map_list, key_frame_index = load_seg_map(os.path.join(args.data_dir, "tracking"), target_size=target_size)

    batch_convert_windows_to_jpg(os.path.join(args.data_dir, "tracking"), os.path.join(args.data_dir, "images_ori_jpg"))

    image_dir_jpg = os.path.join(args.data_dir, "tracking/window_00000/image_ori_jpg")
    image_ori_dir = os.path.join(args.data_dir, "images_ori")

    mask1_list, mask2_list = split_seg_maps_to_masks(args, seg_map_list, key_frame_index, threshold=0.5)

    sam2_model_cfg, predictor, model, sam2_checkpoint = load_model_once()

    propagate_mask_in_video(args, mask1_list[0], image_dir_jpg, image_ori_dir, sam2_model_cfg, predictor, model, sam2_checkpoint)

    # visualize_propagated_masks(
    #     image_dir=image_dir_jpg,
    #     mask_dir=os.path.join(args.output_dir, "sam"),
    #     output_dir=os.path.join(args.output_dir, "visualization_mask1"),
    #     alpha=0.5,
    #     mask_color=(0, 255, 0)  # Green color for mask
    # )
