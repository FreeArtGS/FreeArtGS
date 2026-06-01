import open3d as o3d
import numpy as np

def visualize_camera_poses(point_clouds, transforms, start_idx=0, end_idx=None):
    """
    Visualize point clouds in a selected range together with the camera trajectory.
    Geometries are added one by one; press Space to continue.
    """
    if end_idx is None:
        end_idx = len(point_clouds) - 1
    
    # Clamp indices to the valid range.
    start_idx = max(0, min(start_idx, len(point_clouds) - 1))
    end_idx = max(start_idx, min(end_idx, len(point_clouds) - 1))
    
    print(f"Visualizing point clouds from index {start_idx} to {end_idx}")
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Point Clouds and Camera Trajectory (indices {start_idx} to {end_idx})")

    # Add the world coordinate frame.
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(world_frame)

    # Store all geometries to display.
    geometries = []

    # === Fixed frame ===
    # Use the point cloud at start_idx as the reference frame.
    geometries.append(point_clouds[start_idx])

    # Visualize point clouds in the selected range.
    for x in range(start_idx + 1, end_idx + 1):
        cumulative_transform = np.eye(4)
        for i in range(start_idx, x):
            cumulative_transform = transforms[i] @ cumulative_transform  # Keep the matrix multiplication order.

        # Transform frame x into the start-frame coordinate system.
        pcd_x_transformed = point_clouds[x].transform(np.linalg.inv(cumulative_transform))
        geometries.append(pcd_x_transformed)

    # === Camera trajectory ===
    # Initialize camera poses.
    camera_poses = [np.eye(4)]

    # Accumulate camera transforms.
    for i in range(start_idx, end_idx):
        camera_pose = np.dot(camera_poses[-1], np.linalg.inv(transforms[i]))
        camera_poses.append(camera_pose)

    # Create geometry for the camera trajectory.
    for i, pose in enumerate(camera_poses):
        # Camera coordinate frame.
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=[0, 0, 0])
        camera_frame.transform(pose)
        geometries.append(camera_frame)

        # Camera position marker.
        camera_pos = pose[:3, 3]
        camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        camera_sphere.translate(camera_pos)
        camera_sphere.paint_uniform_color([1, 0, 0])  # Red.
        geometries.append(camera_sphere)

        # Connect consecutive camera positions to form the trajectory.
        if i > 0:
            prev_pos = camera_poses[i - 1][:3, 3]
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([prev_pos, camera_pos])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green.
            geometries.append(line)

    # Configure visualization settings.
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # White background.
    opt.point_size = 2.0

    # Index of the next geometry to display.
    current_index = 0

    # Key callback: Space adds the next geometry.
    def next_geometry(vis):
        nonlocal current_index
        if current_index < len(geometries):
            vis.add_geometry(geometries[current_index])
            current_index += 1
            print(f"Added {current_index}/{len(geometries)} geometries")
        else:
            print("All geometries have been added")
        return False

    # Register key callback for the Space key.
    vis.register_key_callback(32, next_geometry)  # 32 is the Space key code.

    # Start with only the world frame and let the user reveal the rest.
    print("Press Space to add point clouds and camera poses step by step...")

    # Run visualization.
    vis.run()
    vis.destroy_window()
import argparse
import os
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='datasets/apple')
    args = parser.parse_args()
    point_clouds0 = []
    point_clouds1 = []
    transforms0 = np.load(os.path.join(args.dataset_dir, "relative_transforms_0.npy"))
    transforms1 = np.load(os.path.join(args.dataset_dir, "relative_transforms_1.npy"))
    for i in range(len(transforms0)):
        point_clouds0.append(o3d.io.read_point_cloud(os.path.join(args.dataset_dir, "point_clouds_0", f"{i:05d}.ply")))
    for i in range(len(transforms1)):
        point_clouds1.append(o3d.io.read_point_cloud(os.path.join(args.dataset_dir, "point_clouds_1", f"{i:05d}.ply")))

# Interactive loop: allow the user to visualize multiple ranges.
    while True:
        # Read user input.
        user_input = input("\nEnter two numbers separated by a space for the point-cloud index range, or 'q' to quit: ")
        
        # Check whether the user wants to quit.
        if user_input.lower() == 'q':
            break
            
        # Parse the input.
        try:
            start, end = map(int, user_input.strip().split())
        except ValueError:
            print("Invalid input. Please enter two integers or 'q' to quit.")
            continue
            
        # Visualize point clouds in the requested range.
        visualize_camera_poses(point_clouds0, transforms0, start, end)
        visualize_camera_poses(point_clouds1, transforms1, start, end)
if __name__ == "__main__":
    main()
