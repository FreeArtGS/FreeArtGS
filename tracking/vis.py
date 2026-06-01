import os
import random
import numpy as np
from cprint import cprint

def save_visible_trajectories_ply(trajectories, point_indices, rgbs_window, output_dir, window_name, sample_rate=1.0):
    """
    Save visible trajectories as PLY point cloud files, where each point contains its complete trajectory over the entire time window
    
    Args:
        trajectories: numpy array of shape (N, T, 3) 
        point_indices: tuple of (y_indices, x_indices) 
        rgbs_window: list of RGB images for the window
        output_dir: output directory
        window_name: window name
        sample_rate: sample rate
    """
    if len(trajectories) == 0:
        cprint.warn(f"No trajectories to save for {window_name}")
        return
    
    N, T, _ = trajectories.shape
    y_indices, x_indices = point_indices
    
    # Sample points to reduce file size.
    num_points_to_sample = int(N * sample_rate)
    if num_points_to_sample == 0 and N > 0:
        num_points_to_sample = 1
    
    if num_points_to_sample < N:
        sampled_indices = random.sample(range(N), num_points_to_sample)
        sampled_trajectories = trajectories[sampled_indices]
        sampled_y_indices = y_indices[sampled_indices]
        sampled_x_indices = x_indices[sampled_indices]
    else:
        sampled_trajectories = trajectories
        sampled_y_indices = y_indices
        sampled_x_indices = x_indices
    
    # get color for each point (using the color of the first frame)
    colors = []
    rgb_frame_0 = rgbs_window[0]
    img_h, img_w, _ = rgb_frame_0.shape
    
    for y, x in zip(sampled_y_indices, sampled_x_indices):
        if 0 <= y < img_h and 0 <= x < img_w:
            color = rgb_frame_0[y, x]
            colors.append([color[2], color[1], color[0]])  # BGR to RGB
        else:
            colors.append([128, 128, 128])  # default gray
    
    # save method 1: each time frame PLY file (similar to the original function)
    for t in range(T):
        output_path = os.path.join(output_dir, f'visible_traj_{window_name}_frame_{t:04d}.ply')
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(sampled_trajectories)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for traj, color in zip(sampled_trajectories, colors):
                x, y, z = traj[t]
                f.write(f"{x} {y} {z} {color[0]} {color[1]} {color[2]}\n")
    

def save_ply_point_clouds_per_frame(coords_4d, masks_window, camera_matrix_scaled, rgbs_window, output_dir, window_name, sample_rate=0.8):
    """
    Saves a 3D point cloud for each frame in a window as a .ply file.
    The color of the points is sampled from the RGB images.

    Args:
        coords_4d (np.array): The 4D coordinates tensor for a window (B, T, 4, H, W).
        masks_window (list): List of boolean masks for each frame in the window.
        camera_matrix_scaled (np.array): The scaled camera intrinsic matrix.
        rgbs_window (list): List of RGB images (np.array) for the current window.
        output_dir (str): Directory to save the visualization.
        window_name (str): Name for the output file.
        sample_rate (float): Fraction of points to save.
    """
    B, T, C, H, W = coords_4d.shape
    if B == 0 or T < 1:
        cprint.warn(f"Not enough data to save PLY for {window_name}. Shape: {coords_4d.shape}")
        return

    mask_t0 = masks_window[0]
    points_h, points_w = np.where(mask_t0)
    if len(points_h) == 0:
        cprint.warn(f"No points in the initial mask for {window_name}.")
        return

    num_points_to_sample = int(len(points_h) * sample_rate)
    if num_points_to_sample == 0 and len(points_h) > 0:
        num_points_to_sample = 1
    
    if num_points_to_sample < len(points_h):
        sampled_indices = random.sample(range(len(points_h)), num_points_to_sample)
        sampled_points_h = points_h[sampled_indices]
        sampled_points_w = points_w[sampled_indices]
    else:
        sampled_points_h = points_h
        sampled_points_w = points_w

    fx, fy = camera_matrix_scaled[0, 0], camera_matrix_scaled[1, 1]
    cx, cy = camera_matrix_scaled[0, 2], camera_matrix_scaled[1, 2]

    for t in range(T):
        vertices = []
        colors = []
        rgb_frame = rgbs_window[t]
        mask_t = masks_window[t]
        img_h, img_w, _ = rgb_frame.shape

        for h, w in zip(sampled_points_h, sampled_points_w):
            x, y, z, visibility, in_mask = coords_4d[0, t, :, h, w]
            
            if visibility > 0.1:
                # Project to 2D to get color
                if z > 1e-5:
                    u = (x * fx / z) + cx
                    v = (y * fy / z) + cy
                    u_int, v_int = int(round(u)), int(round(v))

                    if 0 <= u_int < img_w and 0 <= v_int < img_h and mask_t[v_int, u_int]:
                        vertices.append([x, y, z]) # Y-up coordinate system
                        color = rgb_frame[v_int, u_int]
                        colors.append([color[2], color[1], color[0]])  # BGR to RGB

        if not vertices:
            continue

        # Write to PLY file for the current frame
        output_path = os.path.join(output_dir, f'traj_{window_name}_frame_{t:04d}.ply')
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for point, color in zip(vertices, colors):
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
                
    #cprint.info(f"Trajectory PLY files for {window_name} saved to {output_dir}")
