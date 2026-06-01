import open3d as o3d
from open3d import visualization
import numpy as np
import cv2
import os
import sys
from pathlib import Path
from tqdm import tqdm
from termcolor import cprint

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pipeline_config import load_pipeline_config, resolve_pipeline_paths, resolve_preprocess, set_arg_defaults
def create_point_cloud_from_rgbd(rgb_path, depth_path, intrinsic_path, output_path, mask_path=None, depth_scale=1.0, depth_trunc=15.0):
    """
    Create a point cloud from an RGB and a depth image, only within the mask if provided.
    """
    color_raw = o3d.io.read_image(rgb_path)
    
    # The depth images are stored in .npz files.
    depth_data = np.load(depth_path)
    depth_np = depth_data['depth'].astype(np.float32)

    # If mask is provided, load and apply it
    if mask_path is not None and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = (mask > 0).astype(np.float32)
            # Resize mask if needed
            if mask.shape != depth_np.shape:
                mask = cv2.resize(mask, (depth_np.shape[1], depth_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth_np = depth_np * mask  # Zero out depth outside mask
    
    depth_raw = o3d.geometry.Image(depth_np)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

    intrinsics = np.loadtxt(intrinsic_path)
    color_img_np = np.asarray(color_raw)
    height, width, _ = color_img_np.shape
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=width, 
        height=height, 
        fx=intrinsics[0, 0], 
        fy=intrinsics[1, 1], 
        cx=intrinsics[0, 2], 
        cy=intrinsics[1, 2])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsics)

    o3d.io.write_point_cloud(output_path, pcd)
    # Visualize the point cloud
    #visualization.draw_geometries([pcd])

import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to pipeline config YAML')
    parser.add_argument('--object_name', type=str, default=None, help='Object name override used with --config')
    parser.add_argument('--dataset_dir', type=str, default=None, help='Dataset directory')  
    parser.add_argument('--max_depth', type=float, default=None, help='Maximum depth for point cloud generation')
    args = parser.parse_args()

    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        preprocess_cfg = resolve_preprocess(config)
        if args.dataset_dir is None:
            args.dataset_dir = runtime_paths["dataset_dir"]
    else:
        preprocess_cfg = resolve_preprocess({})
    set_arg_defaults(args, {"max_depth": preprocess_cfg["pcd_max_depth"]})
    if args.dataset_dir is None:
        parser.error("--dataset_dir is required when --config is not provided")

    dataset_dir = args.dataset_dir
    rgb_dir = os.path.join(dataset_dir, 'rgb')
    depth_dir = os.path.join(dataset_dir, 'depth')
    mask_dir = os.path.join(dataset_dir, 'masks')
    intrinsic_path = os.path.join(dataset_dir, 'cam_K.txt')
    output_dir = os.path.join(dataset_dir, 'point_clouds')
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of files, sort them to make sure they match
    rgb_files = sorted(os.listdir(rgb_dir))
    depth_files = sorted(os.listdir(depth_dir))
    mask_files = sorted(os.listdir(mask_dir)) if os.path.exists(mask_dir) else None

    # Process one example file
    cprint("="*60, "cyan")
    cprint(f"Generating {len(depth_files)} point cloud from RGB-D data", "green")

    if rgb_files and depth_files:
        # Use tqdm to show progress bar
        for rgb_file, depth_file in tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Processing files"):
            frame_id = os.path.splitext(rgb_file)[0]
            rgb_path = os.path.join(rgb_dir, f"{frame_id}.png")
            depth_path = os.path.join(depth_dir, f"{frame_id}.npz")
            mask_path = os.path.join(mask_dir, f"{frame_id}.png") if mask_files is not None else None
            output_path = os.path.join(output_dir, f"{frame_id}.ply")

            create_point_cloud_from_rgbd(rgb_path, depth_path, intrinsic_path, output_path, mask_path=mask_path, depth_scale=1.0, depth_trunc=args.max_depth)
    else:
        print("No RGB or depth files found in the specified directories.")
