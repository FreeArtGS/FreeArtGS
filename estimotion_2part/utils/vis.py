import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d

def create_colored_point_cloud(points, colors):
    """
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array, (3,) single color array, or [r, g, b] list
    Returns:
        o3d.geometry.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Convert colors to numpy array if it's a list
    if isinstance(colors, list):
        colors = np.array(colors)
    
    if colors.ndim == 1:
        colors = np.tile(colors.reshape(1, 3), (len(points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def save_point_cloud_ply(points, colors, save_path, label=""):
    pcd = create_colored_point_cloud(points, colors)
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved {len(points)} {label} points to {save_path}")

def visualize_trajectories_ply(trajectories, save_path, frame_idx=0, color=None, weights=None):
    """
    Save trajectories of a specific frame (default first frame) as ply point cloud file.
    Args:
        trajectories: numpy array, shape (N, T, 3) or (N, T, 4)
        save_path: ply file save path
        frame_idx: which frame to extract points from, default 0
        color: (3,) or None, point cloud color, float[0,1], None uses white
        weights: (N,) or None, weight array, if provided uses weight coloring (blue->red, 0->1)
    """
    points = trajectories[:, frame_idx, :3]
    
    if weights is not None:
        # Color by weights: blue(0) -> red(1)
        weights = np.clip(weights, 0, 1)
        colors = np.zeros((len(weights), 3))
        colors[:, 0] = weights      # Red channel
        colors[:, 2] = 1 - weights  # Blue channel
    elif color is None:
        colors = np.ones_like(points) * 0.9  # Default white
    else:
        colors = np.tile(np.array(color).reshape(1, 3), (points.shape[0], 1))

    save_point_cloud_ply(points, colors, save_path, "trajectory")

def save_transformation_ply(P_orig, P_transformed, Q_target, frame_index, output_dir="vis"):
    """
    Save the three point clouds (original, transformed, target) to PLY files.
    
    Args:
        P_orig: Original points from previous frame (Nx3)
        P_transformed: Transformed points from previous frame (Nx3) 
        Q_target: Target points from current frame (Mx3)
        frame_index: Frame index for file naming
        output_dir: Directory to save PLY files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Only save if 3D points
    if P_orig.shape[1] < 3:
        print("Cannot save PLY files: points are not 3D")
        return
    
    
    # Define colors (RGB values between 0 and 1)
    green_color = [0.0, 1.0, 0.0]  # Green for original
    red_color = [1.0, 0.0, 0.0]    # Red for transformed
    blue_color = [0.0, 0.0, 1.0]   # Blue for target
    
    # Create point clouds
    pcd_orig = create_colored_point_cloud(P_orig, green_color)
    pcd_trans = create_colored_point_cloud(P_transformed, red_color)  
    pcd_target = create_colored_point_cloud(Q_target, blue_color)
    
    # Save individual point clouds
    # orig_file = os.path.join(output_dir, f"frame_{frame_index:05d}_original.ply")
    # trans_file = os.path.join(output_dir, f"frame_{frame_index:05d}_transformed.ply")
    # target_file = os.path.join(output_dir, f"frame_{frame_index:05d}_target.ply")
    # combined_file = os.path.join(output_dir, f"frame_{frame_index:05d}_combined.ply")
    
    # o3d.io.write_point_cloud(orig_file, pcd_orig)
    # o3d.io.write_point_cloud(trans_file, pcd_trans)
    # o3d.io.write_point_cloud(target_file, pcd_target)
    
    # Create combined point cloud
    # pcd_combined = pcd_orig + pcd_trans + pcd_target
    # o3d.io.write_point_cloud(combined_file, pcd_combined)
    
    # print(f"Saved PLY files to {output_dir}:")
    # print(f"  - {os.path.basename(orig_file)} (original points, green)")
    # print(f"  - {os.path.basename(trans_file)} (transformed points, red)")
    # print(f"  - {os.path.basename(target_file)} (target points, blue)")
    # print(f"  - {os.path.basename(combined_file)} (combined view)")


def visualize_transformation(P_orig, Q_target, T1, T2, weights, frame_index, save_ply=False, output_dir="vis", max_points=100000):
    """
    Visualize weighted transformation effect of two transformation matrices on point clouds
    Args:
        P_orig: Original points (N, 3)
        Q_target: Target points (N, 3)
        T1, T2: 4x4 transformation matrices
        weights: (N, 1) weights, one per point
        frame_index: Frame number
        save_ply: Whether to save PLY files
        output_dir: Output directory
        max_points: Maximum number of points for visualization
    """
    N = P_orig.shape[0]
    ones = np.ones((N, 1))
    P_homo = np.hstack([P_orig, ones])  # (N, 4)

    # Calculate weighted transformation for each point
    P_transformed = weights * (P_homo @ T1.T) + (1 - weights) * (P_homo @ T2.T)
    P_transformed = P_transformed[:, :3]

    # Sample points for visualization
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        P_orig_vis = P_orig[idx]
        P_transformed_vis = P_transformed[idx]
        Q_target_vis = Q_target[idx]
        print(f"Sampled {max_points} points from {N} for visualization")
    else:
        P_orig_vis = P_orig
        P_transformed_vis = P_transformed
        Q_target_vis = Q_target

    # Save PLY files
    # if save_ply:
    #     save_transformation_ply(P_orig, P_transformed, Q_target, frame_index, output_dir)

    fig = plt.figure(figsize=(12, 8))
    dim = P_orig.shape[1]
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        s = 5
        ax.scatter(P_orig_vis[:, 0], P_orig_vis[:, 1], P_orig_vis[:, 2],
                   c='green', marker='^', label='Previous Frame Original', alpha=0.7, s=s)
        ax.scatter(P_transformed_vis[:, 0], P_transformed_vis[:, 1], P_transformed_vis[:, 2],
                   c='red', marker='x', label='Weighted Transformed', s=s)
        ax.scatter(Q_target_vis[:, 0], Q_target_vis[:, 1], Q_target_vis[:, 2],
                   c='blue', marker='o', label='Current Frame', alpha=0.6, s=s)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        all_points = np.vstack((P_orig_vis, P_transformed_vis, Q_target_vis))
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
        max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max()
        mid_x = (x_max+x_min) * 0.5
        mid_y = (y_max+y_min) * 0.5
        mid_z = (z_max+z_min) * 0.5
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    elif dim == 2:
        ax = fig.add_subplot(111)
        ax.scatter(P_orig_vis[:, 0], P_orig_vis[:, 1],
                   c='green', marker='^', label='Previous Frame Original', alpha=0.7, s=50)
        ax.scatter(P_transformed_vis[:, 0], P_transformed_vis[:, 1],
                   c='red', marker='x', label='Weighted Transformed')
        ax.scatter(Q_target_vis[:, 0], Q_target_vis[:, 1],
                   c='blue', marker='o', label='Current Frame', alpha=0.6, s=50)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='box')
    else:
        print(f"Visualization for {dim} dimensions is not supported.")
        return
    ax.legend()
    ax.set_title(f'Frame {frame_index} Weighted Transformation')
    plt.show()

def visualize_point_trajectories(point_trajectory_dict, t1, t2, window_index, output_dir=None):
    # Convert point_trajectory_dict for visualization
    if len(point_trajectory_dict) > 0:
        # Prepare data for visualization from point_trajectory_dict
        P_orig_list = []
        Q_target_list = []
        weights_list = []
        
        for point_key, point_data in point_trajectory_dict.items():
            # Use the last trajectory segment for visualization
            last_traj = point_data["trajectories"][-1]  # Shape: (T, 4)
            P_orig_list.append(last_traj[0, :3])  # First frame coordinates
            Q_target_list.append(last_traj[-1, :3])  # Last frame coordinates
            weights_list.append(point_data["segmentation_weights"])  # Use optimized weights
        
        P_orig = np.array(P_orig_list)
        Q_target = np.array(Q_target_list)
        weights = np.array(weights_list).reshape(-1, 1)
        frame_index = window_index
        visualize_transformation(P_orig, Q_target, t1[-1], t2[-1], weights,
                                 frame_index=frame_index, save_ply=True, output_dir=output_dir)

def visualize_part_initialization(trajectories, part_weights, save_path=None, axis=None):
    """
    Visualize part initialization results with optional axis visualization
    
    Args:
        trajectories: numpy array of shape (N, T, 3)
        part_weights: numpy array of shape (N,) with values in [0, 1]
        save_path: Path to save the image, if None then display
        axis: numpy array of shape (6,) representing [point_x, point_y, point_z, dir_x, dir_y, dir_z] or None
    """
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    trajs = np.array(trajectories)
    first_frame = trajs[:, 0, :]  # Use first frame for visualization
    ply_path = os.path.splitext(save_path)[0] + '.ply'
    # visualize_trajectories_ply(trajs, ply_path, weights=part_weights)

    # Create color mapping: weights close to 0 are blue (part 1), close to 1 are red (part 2)
    colors = plt.cm.coolwarm(part_weights)
    
    # Determine subplot layout based on whether axis is provided
    if axis is not None:
        fig = plt.figure(figsize=(18, 5))
        subplot_config = [(131, '3d'), (132, '2d'), (133, '3d')]
        titles = ['3D Part Initialization', '2D Part Initialization (XY plane)', '3D with Rotation Axis']
    else:
        fig = plt.figure(figsize=(12, 5))
        subplot_config = [(121, '3d'), (122, '2d')]
        titles = ['3D Part Initialization', '2D Part Initialization (XY plane)']
    
    for i, (subplot_pos, plot_type) in enumerate(subplot_config):
        if plot_type == '3d':
            ax = fig.add_subplot(subplot_pos, projection='3d')
            scatter = ax.scatter(first_frame[:, 0], first_frame[:, 1], first_frame[:, 2], 
                                c=part_weights, cmap='coolwarm', s=50, alpha=0.7, vmin=0, vmax=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Add axis visualization for the third subplot
            if i == 2 and axis is not None:
                axis_point = axis[:3]
                axis_direction = axis[3:6]
                
                # Normalize axis direction
                axis_dir_norm = np.linalg.norm(axis_direction)
                if axis_dir_norm > 1e-8:
                    axis_direction = axis_direction / axis_dir_norm
                else:
                    axis_direction = np.array([1., 0., 0.])  # Default to X axis
                
                # Calculate appropriate axis length based on point cloud extent
                point_cloud_min = np.min(first_frame, axis=0)
                point_cloud_max = np.max(first_frame, axis=0)
                point_cloud_extent = point_cloud_max - point_cloud_min
                axis_length = np.max(point_cloud_extent) * 0.8  # 80% of the largest dimension
                
                # Add coordinate system reference (origin and coordinate axes)
                origin = np.array([0., 0., 0.])
                coord_axes_length = np.max(point_cloud_extent) * 0.3
                
                # Plot coordinate system axes (X=red, Y=green, Z=blue)
                ax.quiver(origin[0], origin[1], origin[2], coord_axes_length, 0, 0,
                            color='red', alpha=0.5, linewidth=2, label='X-axis')
                ax.quiver(origin[0], origin[1], origin[2], 0, coord_axes_length, 0,
                            color='green', alpha=0.5, linewidth=2, label='Y-axis')
                ax.quiver(origin[0], origin[1], origin[2], 0, 0, coord_axes_length,
                            color='blue', alpha=0.5, linewidth=2, label='Z-axis')
                
                # Calculate point cloud center for reference
                point_cloud_center = np.mean(first_frame, axis=0)
                ax.scatter([point_cloud_center[0]], [point_cloud_center[1]], [point_cloud_center[2]], 
                            c='orange', marker='D', s=200, label='Point Cloud Center', 
                            edgecolors='black', linewidth=1, alpha=0.8)
                
                # Ensure axis passes through or near the point cloud
                axis_start = axis_point - axis_direction * axis_length
                axis_end = axis_point + axis_direction * axis_length
                
                # Plot axis line
                ax.plot([axis_start[0], axis_end[0]], 
                        [axis_start[1], axis_end[1]], 
                        [axis_start[2], axis_end[2]], 
                        'k-', linewidth=4, label='Rotation Axis', alpha=0.8)
                
                # Plot axis point
                ax.scatter([axis_point[0]], [axis_point[1]], [axis_point[2]], 
                            c='black', marker='*', s=300, label='Axis Point', 
                            edgecolors='white', linewidth=2)
                
                # Add arrow to show direction (smaller and more visible)
                ax.quiver(axis_point[0], axis_point[1], axis_point[2],
                            axis_direction[0], axis_direction[1], axis_direction[2],
                            length=axis_length*0.4, color='red', arrow_length_ratio=0.2, 
                            linewidth=3, label='Axis Direction', alpha=0.9)
                
                # Add some reference points along the axis for better visualization
                num_ref_points = 5
                ref_positions = np.linspace(-axis_length*0.5, axis_length*0.5, num_ref_points)
                ref_points = axis_point + np.outer(ref_positions, axis_direction)
                ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2],
                            c='yellow', marker='o', s=50, alpha=0.6, label='Axis Reference')
                
                # Add text annotations for coordinate system understanding
                ax.text(point_cloud_center[0], point_cloud_center[1], point_cloud_center[2] + coord_axes_length*0.5,
                        f'Center: ({point_cloud_center[0]:.2f}, {point_cloud_center[1]:.2f}, {point_cloud_center[2]:.2f})',
                        fontsize=8, ha='center')
                ax.text(axis_point[0], axis_point[1], axis_point[2] + coord_axes_length*0.3,
                        f'Axis: ({axis_point[0]:.2f}, {axis_point[1]:.2f}, {axis_point[2]:.2f})',
                        fontsize=8, ha='center')
                
                ax.legend()
            
            ax.set_title(titles[i])
            if i < 2:  # Only add colorbar for first two subplots to avoid duplication
                plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        
        elif plot_type == '2d':
            ax = fig.add_subplot(subplot_pos)
            scatter2 = ax.scatter(first_frame[:, 0], first_frame[:, 1], 
                                    c=part_weights, cmap='coolwarm', s=50, alpha=0.7, vmin=0, vmax=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(titles[i])
            plt.colorbar(scatter2, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Part initialization visualization saved to: {save_path}")
    else:
        plt.show()
        
    plt.close()



def visualize_seg_map_list(
    seg_map_list,
    trajectory_mask_list=None,
    rgb_list=None,
    mask_list=None,
    save_dir=None,
    prefix="seg_map",
    save_visualizations=True,
):
    """
    Visualize segmentation map list, optionally overlay on RGB images, and save to specified directory with mask region support.
    Args:
        seg_map_list: list of (H, W) numpy arrays
        rgb_list: list of (H, W, 3) numpy arrays or None
        mask_list: list of (H, W) numpy arrays or None, if provided only display segmentation in mask regions
        save_dir: Save directory, if None then don't save
        prefix: Filename prefix
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for idx, seg_map in enumerate(seg_map_list):
        if save_dir is not None:
            np.save(os.path.join(save_dir, f"{prefix}_{idx:05d}.npy"), seg_map)

        if not save_visualizations:
            continue

        plt.figure(figsize=(8, 6))
        show_map = seg_map.copy()
        if mask_list is not None and mask_list[idx] is not None:
            mask = mask_list[idx].astype(bool)
            show_map[~mask] = np.nan
        if rgb_list is not None:
            plt.imshow(rgb_list[idx])
            plt.imshow(show_map, cmap='jet', vmin=0, vmax=1, alpha=0.5)
        else:
            plt.imshow(show_map, cmap='jet', vmin=0, vmax=1)
        plt.title(f"Segmentation Map {idx}")
        plt.axis('off')
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{prefix}_{idx:05d}.png"), bbox_inches='tight')
        plt.close()

        if trajectory_mask_list is not None and trajectory_mask_list[idx] is not None:
            plt.figure(figsize=(8, 6))
            mask = trajectory_mask_list[idx].astype(bool)
            masked_map = seg_map.copy()
            masked_map[~mask] = np.nan
            if rgb_list is not None:
                plt.imshow(rgb_list[idx])
                plt.imshow(masked_map, cmap='jet', vmin=0, vmax=1, alpha=0.5)
            else:
                plt.imshow(masked_map, cmap='jet', vmin=0, vmax=1)
            plt.title(f"Segmentation Map {idx} (Masked)")
            plt.axis('off')
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f"{prefix}_mask_{idx:05d}.png"), bbox_inches='tight')
            plt.close()

