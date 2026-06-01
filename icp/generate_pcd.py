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

from utils.pipeline_config import load_pipeline_config, resolve_pipeline_paths, resolve_icp, set_arg_defaults

def create_point_cloud_from_rgbd(rgb_path, depth_path, intrinsic_path, output_path, mask_path=None, depth_scale=1.0, depth_trunc=3.0, erode_kernel_size=3, erode_iterations=1, downsample_factor=1):
    """
    Create a point cloud from an RGB and a depth image, only within the mask if provided.
    Optionally apply erosion to the mask to reduce noise.
    """
    color_raw = o3d.io.read_image(rgb_path)
    color_img_np = np.asarray(color_raw)
    
    # The depth images are stored in .npz files.
    depth_data = np.load(depth_path)
    depth_np = depth_data['depth']

    # If mask is provided, load and apply it
    if mask_path is not None and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = (mask > 0).astype(np.float32)
            # Resize mask if needed
            if mask.shape != depth_np.shape:
                mask = cv2.resize(mask, (depth_np.shape[1], depth_np.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Erode the mask to suppress boundary noise.
            if erode_kernel_size > 0 and erode_iterations > 0:
                kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=erode_iterations)
            depth_np = depth_np * mask  # Zero out depth outside mask
    
    intrinsics = np.loadtxt(intrinsic_path)

    if downsample_factor > 1:
        new_width = color_img_np.shape[1] // downsample_factor
        new_height = color_img_np.shape[0] // downsample_factor
        color_img_np = cv2.resize(color_img_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        color_raw = o3d.geometry.Image(color_img_np)
        depth_np = cv2.resize(depth_np, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        intrinsics[0, 0] /= downsample_factor
        intrinsics[1, 1] /= downsample_factor
        intrinsics[0, 2] /= downsample_factor
        intrinsics[1, 2] /= downsample_factor

    depth_raw = o3d.geometry.Image(depth_np)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

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
    parser.add_argument('--dataset_dir', type=str, default=None, help='Dataset directory containing rgb, depth, masks, and camera intrinsics')
    parser.add_argument('--erode_kernel_size', type=int, default=None, help='Kernel size for mask erosion (default: 3)')
    parser.add_argument('--erode_iterations', type=int, default=None, help='Number of erosion iterations (default: 1)')
    parser.add_argument('--downsample_factor', type=int, default=1, help='Downsample factor to resize image and depth (default: 1)')
    args = parser.parse_args()

    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        icp_cfg = resolve_icp(config)
        if args.dataset_dir is None:
            args.dataset_dir = runtime_paths["dataset_dir"]
    else:
        icp_cfg = resolve_icp({})
    set_arg_defaults(
        args,
        {
            "erode_kernel_size": icp_cfg["erode_kernel_size"],
            "erode_iterations": icp_cfg["erode_iterations"],
        },
    )
    if args.dataset_dir is None:
        parser.error("--dataset_dir is required when --config is not provided")
    
    dataset_dir = args.dataset_dir
    rgb_dir = os.path.join(dataset_dir, 'rgb')
    depth_dir = os.path.join(dataset_dir, 'depth')
    mask0_dir = os.path.join(dataset_dir, 'mask_0')
    mask1_dir = os.path.join(dataset_dir, 'mask_1')
    intrinsic_path = os.path.join(dataset_dir, 'cam_K.txt')
    output0_dir = os.path.join(dataset_dir, 'point_clouds_0')
    output1_dir = os.path.join(dataset_dir, 'point_clouds_1')
    os.makedirs(output0_dir, exist_ok=True)
    os.makedirs(output1_dir, exist_ok=True)

    # Get a list of files, sort them to make sure they match
    rgb_files = sorted(os.listdir(rgb_dir))
    depth_files = sorted(os.listdir(depth_dir))
    mask0_files = sorted(os.listdir(mask0_dir)) if os.path.exists(mask0_dir) else None
    mask1_files = sorted(os.listdir(mask1_dir)) if os.path.exists(mask1_dir) else None

    # Process files
    cprint("="*60, "cyan")
    cprint(f"Generating {len(depth_files)} point cloud from RGB-D data", "green")
    cprint(f"Erosion parameters: kernel_size={args.erode_kernel_size}, iterations={args.erode_iterations}", "yellow")
    
    if rgb_files and depth_files:
        # Use tqdm to show progress bar
        for rgb_file, depth_file in tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Processing files"):
            frame_id = os.path.splitext(rgb_file)[0]
            rgb_path = os.path.join(rgb_dir, f"{frame_id}.png")
            depth_path = os.path.join(depth_dir, f"{frame_id}.npz")
            mask0_path = os.path.join(mask0_dir, f"{frame_id}.png") if mask0_files is not None else None
            mask1_path = os.path.join(mask1_dir, f"{frame_id}.png") if mask1_files is not None else None
            output0_path = os.path.join(output0_dir, f"{frame_id}.ply")
            output1_path = os.path.join(output1_dir, f"{frame_id}.ply")

            create_point_cloud_from_rgbd(
                rgb_path, depth_path, intrinsic_path, output0_path, 
                mask_path=mask0_path, depth_scale=1.0, depth_trunc=9.0,
                erode_kernel_size=args.erode_kernel_size,
                erode_iterations=args.erode_iterations,
                downsample_factor=args.downsample_factor
            )
            create_point_cloud_from_rgbd(
                rgb_path, depth_path, intrinsic_path, output1_path, 
                mask_path=mask1_path, depth_scale=1.0, depth_trunc=9.0,
                erode_kernel_size=args.erode_kernel_size,
                erode_iterations=args.erode_iterations,
                downsample_factor=args.downsample_factor
            )
    else:
        print("No RGB or depth files found in the specified directories.")
