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
import glob
import torch
from tqdm import tqdm
import tempfile
from sam2.build_sam import build_sam2_video_predictor

def convert_depth_npz_to_png(depth_dir, scale=1000.0):
    """
    Converts .npz files in the depth folder to .png files, while keeping the original .npz files.

    Args:
        depth_dir: The folder containing .npz depth files.
        scale: The scaling factor for converting npz units to PNG, default is 1000.0 (meters -> millimeters).
    """

    out_dir =   depth_dir

    npz_files = sorted(glob.glob(os.path.join(depth_dir, "*.npz")))
    for npz_file in npz_files:
        data = np.load(npz_file)
        depth = data["depth"]
        depth_png = (depth * scale).astype(np.uint16)

        base_name = os.path.splitext(os.path.basename(npz_file))[0]
        png_path = os.path.join(out_dir, f"{base_name}.png")
        cv2.imwrite(png_path, depth_png)

def split_seg_maps_to_masks(args, threshold=0.5):
    """
    Reads seg_map_*.npy from each window_xxxxx directory,
    detects if the sequence was ping-pong concatenated, and if so keeps only the
    latter half and restores original forward order. Then splits into mask_high / mask_low
    and saves to data_dir/mask_1 and data_dir/mask_0.

    Args:
        args.data_dir: The root data directory, which must contain a tracking subdirectory.
        threshold: The threshold for segmentation (default 0.5).
    """
    tracking_dir = os.path.join(args.data_dir, "tracking")
    base_mask_dir = os.path.join(args.data_dir, "masks") 

    mask_high_dir = os.path.join(args.data_dir, "mask_1")
    mask_low_dir = os.path.join(args.data_dir, "mask_0")
    os.makedirs(mask_high_dir, exist_ok=True)
    os.makedirs(mask_low_dir, exist_ok=True)

    # Gather all seg_map paths in order (drop last file per-window to avoid overlap)
    window_dirs = [
        os.path.join(tracking_dir, d)
        for d in sorted(os.listdir(tracking_dir))
        if d.startswith("window_") and os.path.isdir(os.path.join(tracking_dir, d))
    ]
    seg_paths = []
    for idx, window_dir in enumerate(window_dirs):
        seg_files = sorted([
            f for f in os.listdir(window_dir)
            if f.startswith("seg_map_") and f.endswith(".npy")
        ])
        if len(seg_files) > 0 and idx < len(window_dirs) - 1:
            # For non-final windows, drop the last file to avoid duplication across windows
            seg_files = seg_files[:-1]
        # For the final window, keep all seg_maps so we don't lose the last frame
        seg_paths.extend([os.path.join(window_dir, f) for f in seg_files])

    total_seg = len(seg_paths)

    # Try to infer original frame count from images directory
    def _count_images(dir_path):
        if not os.path.isdir(dir_path):
            return 0
        files = [f for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        return len(sorted(files))

    orig_candidates = [
        os.path.join(args.data_dir, "images_ori"),
        os.path.join(args.data_dir, "images"),
        os.path.join(args.data_dir, "rgb"),
        os.path.join(args.data_dir, "images_ori_jpg"),
    ]
    orig_len = 0
    for d in orig_candidates:
        orig_len = _count_images(d)
        if orig_len > 0:
            break

    # Select which seg maps to use and their output order
    selected_paths = seg_paths
    if getattr(args, 'no_pingpong', False):
        cprint(f"no_pingpong=True → keep natural order across windows (total={total_seg}).", "cyan")
    else:
        if orig_len > 0 and total_seg >= max(2 * orig_len - 2, orig_len + 1):
            # Likely ping-pong: keep the latter orig_len seg maps and reverse to forward order
            start_idx = max(0, total_seg - orig_len)
            selected_paths = seg_paths[start_idx:][::-1]
            cprint(f"Detected ping-pong results: total={total_seg}, orig={orig_len}. Keeping latter half and restoring order.", "cyan")
        else:
            cprint(f"No ping-pong detected or unknown original length (total={total_seg}, orig={orig_len}). Keeping current order.", "cyan")

    # If we still have one fewer than the original image count, and we can
    # safely append the very last frame from the last window, do so.
    if orig_len > 0 and len(selected_paths) + 1 == orig_len and len(window_dirs) > 0:
        last_win = window_dirs[-1]
        last_seg_all = sorted([
            f for f in os.listdir(last_win)
            if f.startswith("seg_map_") and f.endswith(".npy")
        ])
        if len(last_seg_all) > 0:
            last_seg_full_path = os.path.join(last_win, last_seg_all[-1])
            # Only append if it is not already in selected_paths.
            if last_seg_full_path not in selected_paths:
                selected_paths.append(last_seg_full_path)
                cprint(f"Appended final seg_map to reach orig_len: {os.path.basename(last_seg_full_path)}", "cyan")

    # Write masks in the decided order, aligning to optional base masks if present
    for out_idx, seg_path in enumerate(selected_paths):
        seg_map = np.load(seg_path)

        mask_high = (seg_map > threshold + 0.01).astype(np.uint8)
        mask_low = ((seg_map < threshold - 0.01) & (seg_map > 0.0000001)).astype(np.uint8)

        frame_name = f"{out_idx:05d}.png"
        base_mask_path = os.path.join(base_mask_dir, frame_name)

        if os.path.exists(base_mask_path):
            base_mask = cv2.imread(base_mask_path, cv2.IMREAD_GRAYSCALE)
            base_mask = (base_mask > 127).astype(np.uint8)
            if mask_high.shape != base_mask.shape:
                mask_high = cv2.resize(mask_high, (base_mask.shape[1], base_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_low = cv2.resize(mask_low, (base_mask.shape[1], base_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_high = (mask_high & base_mask).astype(np.uint8)
            mask_low = (mask_low & base_mask).astype(np.uint8)

        mask_high_path = os.path.join(mask_high_dir, frame_name)
        mask_low_path = os.path.join(mask_low_dir, frame_name)

        cv2.imwrite(mask_high_path, mask_high * 255)
        cv2.imwrite(mask_low_path, mask_low * 255)

    print("BundleSDF preprocess done.")


def smooth_masks_in_directory(mask_dir, image_ori_dir, image_jpg_dir, sam2_model_cfg, sam2_checkpoint):
    """
    Smooth masks in a directory using SAM2 video predictor.
    
    Args:
        mask_dir: Directory containing masks to smooth
        image_ori_dir: Directory containing original images
        image_jpg_dir: Directory containing JPG images for SAM2
        sam2_model_cfg: SAM2 model config
        sam2_checkpoint: SAM2 checkpoint path
    """
    device = "cuda"
    predictor_video = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_frame_dir = os.path.join(temp_dir, "frames")
    os.makedirs(temp_frame_dir, exist_ok=True)
    
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])
    
    cprint(f"Smoothing masks in {mask_dir}...", "green")
    for mask_file in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, mask_file)
        
        # Find corresponding JPG image
        jpg_name = mask_file.replace('.png', '.jpg')
        image_path = os.path.join(image_jpg_dir, jpg_name)
        
        if not os.path.isfile(image_path):
            cprint(f"Warning: JPG image not found for {mask_file}, skipping", "yellow")
            continue
        
        # Clear temp directory
        for f in os.listdir(temp_frame_dir):
            os.remove(os.path.join(temp_frame_dir, f))
        
        # Copy current frame twice
        shutil.copy(image_path, os.path.join(temp_frame_dir, "00000.jpg"))
        shutil.copy(image_path, os.path.join(temp_frame_dir, "00001.jpg"))
        
        # Read mask
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            continue
        if mask_img.ndim == 3:
            mask_img = mask_img[:, :, 0]
        initial_mask = (mask_img > 0).astype(np.uint8)
        
        # Smooth using SAM2 video predictor
        state = predictor_video.init_state(temp_frame_dir)
        predictor_video.add_new_mask(state, 0, 0, initial_mask)
        
        smoothed = None
        for frame_idx, object_ids, smooth_masks in predictor_video.propagate_in_video(state):
            if frame_idx == 1:  # Get smoothed mask from second frame
                smoothed = (smooth_masks[0, 0] > 0).cpu().numpy()
                break
        
        if smoothed is None:
            smoothed = initial_mask
        
        # Save smoothed mask
        sm_uint8 = (smoothed.astype(np.uint8)) * 255
        cv2.imwrite(mask_path, sm_uint8)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    cprint(f"Smoothing completed for {len(mask_files)} masks", "green")


def generate_masked_images(mask_dir, image_ori_dir, output_images_dir, bg_color="white"):
    """
    Generate masked images using masks and original images.
    
    Args:
        mask_dir: Directory containing masks
        image_ori_dir: Directory containing original images
        output_images_dir: Directory to save masked images
        bg_color: Background color ('white' or 'black')
    """
    os.makedirs(output_images_dir, exist_ok=True)
    
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])
    
    cprint(f"Generating masked images in {output_images_dir}...", "green")
    for mask_file in tqdm(mask_files):
        mask_path = os.path.join(mask_dir, mask_file)
        image_path = os.path.join(image_ori_dir, mask_file)
        
        if not os.path.isfile(image_path):
            cprint(f"Warning: Original image not found for {mask_file}, skipping", "yellow")
            continue
        
        # Read mask and image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path)
        
        if mask is None or image is None:
            continue
        
        # Ensure mask and image have the same size
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create masked image
        mask_bool = (mask > 127).astype(bool)
        bg_color_value = 255 if bg_color == "white" else 0
        bg_image = np.ones_like(image) * bg_color_value
        masked_image = np.where(mask_bool[..., None], image, bg_image)
        
        # Save masked image
        output_path = os.path.join(output_images_dir, mask_file)
        cv2.imwrite(output_path, masked_image)
    
    cprint(f"Generated {len(mask_files)} masked images", "green")


def smooth_and_generate_images(args):
    """
    Smooth masks and generate corresponding masked images for mask_0 and mask_1.
    
    Args:
        args: Arguments containing data_dir and other parameters
    """
    # Check if images_ori_jpg exists, if not create it
    image_ori_dir = os.path.join(args.data_dir, "images_ori")
    image_jpg_dir = os.path.join(args.data_dir, "images_ori_jpg")
    
    if not os.path.exists(image_jpg_dir):
        cprint("Creating images_ori_jpg directory...", "green")
        os.makedirs(image_jpg_dir, exist_ok=True)
        # Convert PNG to JPG
        for fname in os.listdir(image_ori_dir):
            if fname.lower().endswith('.png'):
                png_path = os.path.join(image_ori_dir, fname)
                jpg_fname = fname.replace('.png', '.jpg')
                jpg_path = os.path.join(image_jpg_dir, jpg_fname)
                Image.open(png_path).convert('RGB').save(jpg_path, 'JPEG')
    
    # Load SAM2 model
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    sam2_model_cfg = "sam2_hiera_l.yaml"
    
    cprint("="*60, "cyan")
    cprint("Smoothing masks and generating masked images", "green", attrs=['bold'])
    cprint("-"*60, "cyan")
    
    # Process mask_0 and mask_1
    for mask_name, images_name in [("mask_0", "images_0"), ("mask_1", "images_1")]:
        mask_dir = os.path.join(args.data_dir, mask_name)
        images_dir = os.path.join(args.data_dir, images_name)
        
        if not os.path.exists(mask_dir):
            cprint(f"Warning: {mask_dir} does not exist, skipping", "yellow")
            continue
        
        # Smooth masks
        smooth_masks_in_directory(mask_dir, image_ori_dir, image_jpg_dir, sam2_model_cfg, sam2_checkpoint)
        
        # Generate masked images
        generate_masked_images(mask_dir, image_ori_dir, images_dir, bg_color="white")
    
    cprint("="*60, "cyan")
    cprint("Smoothing and image generation completed!", "green", attrs=['bold'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and resize segmentation maps to match RGB image size")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--skip_smooth', action='store_true', help='Skip mask smoothing and image generation')
    parser.add_argument('--no_pingpong', action='store_true', help='Disable ping-pong heuristic; keep natural order across windows')
    args = parser.parse_args()

    convert_depth_npz_to_png(os.path.join(args.data_dir, "depth"))

    split_seg_maps_to_masks(args, threshold=0.5)
    
    # Smooth masks and generate images_0 and images_1
    if not args.skip_smooth:
        smooth_and_generate_images(args)
