import json
import os
from typing import Dict, Tuple

import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance


CV_TO_GL = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def load_camera_poses_from_json(json_path: str) -> tuple[np.ndarray, list[int]]:
    """Load camera poses from camera_poses.json as 4x4 c2w matrices."""
    with open(json_path, "r") as f:
        frames_data = json.load(f)

    camera_poses_4x4 = []
    image_names = []
    for frame in frames_data:
        if "transform_matrix" not in frame:
            raise ValueError(f"No 'transform_matrix' found in frame: {frame}")
        c2w = np.array(frame["transform_matrix"])
        image_name = int(frame["file_path"].split("/")[-1].split(".")[0])
        image_names.append(image_name)
        camera_poses_4x4.append(c2w)

    camera_poses_4x4 = np.array(camera_poses_4x4)
    return camera_poses_4x4, image_names


def average_se3_transforms(transforms: np.ndarray) -> np.ndarray:
    """Average multiple SE(3) transforms by averaging rotation and translation."""
    from scipy.spatial.transform import Rotation as R_scipy

    rotations = transforms[:, :3, :3]
    translations = transforms[:, :3, 3]

    avg_translation = np.mean(translations, axis=0)

    quats = []
    for i in range(len(rotations)):
        quat = R_scipy.from_matrix(rotations[i]).as_quat()
        quats.append(quat)
    quats = np.array(quats)

    for i in range(1, len(quats)):
        if np.dot(quats[0], quats[i]) < 0:
            quats[i] = -quats[i]

    avg_quat = np.mean(quats, axis=0)
    avg_quat = avg_quat / np.linalg.norm(avg_quat)
    avg_rotation = R_scipy.from_quat(avg_quat).as_matrix()

    avg_transform = np.eye(4)
    avg_transform[:3, :3] = avg_rotation
    avg_transform[:3, 3] = avg_translation
    return avg_transform


def solve_T_align(
    gt_camera_se3: np.ndarray,
    pred_camera_pose_gl: np.ndarray,
    mode: str = "first",
    gt_joint_parameter: Tuple | None = None,
    pred_joint_parameter: Tuple | None = None,
) -> tuple[np.ndarray, int] | np.ndarray:
    """Solve alignment transform from predicted camera poses to GT poses."""
    if mode == "average":
        per_frame_T = np.array(
            [gt_camera_se3[i] @ np.linalg.inv(pred_camera_pose_gl[i]) for i in range(len(gt_camera_se3))]
        )
        return average_se3_transforms(per_frame_T)
    if mode == "best":
        if gt_joint_parameter is None or pred_joint_parameter is None:
            raise ValueError("gt_joint_parameter and pred_joint_parameter are required for mode='best'")

        gt_joint_type, gt_joint_axis, gt_joint_pos, _gt_joint_value = gt_joint_parameter
        pred_joint_type, pred_joint_axis, pred_joint_pos, _pred_joint_value = pred_joint_parameter

        best_error = float("inf")
        best_T = None
        best_idx = 0

        for i in range(len(gt_camera_se3)):
            T_i = gt_camera_se3[i] @ np.linalg.inv(pred_camera_pose_gl[i])

            pred_joint_axis_aligned_i = T_i[:3, :3] @ pred_joint_axis
            pred_joint_axis_aligned_i = pred_joint_axis_aligned_i / np.linalg.norm(pred_joint_axis_aligned_i)

            pred_joint_pos_homo_i = np.append(pred_joint_pos, 1.0)
            pred_joint_pos_aligned_i = (T_i @ pred_joint_pos_homo_i)[:3]

            joint_ori_error = np.arccos(np.abs(np.dot(pred_joint_axis_aligned_i, gt_joint_axis)))

            n = np.cross(pred_joint_axis_aligned_i, gt_joint_axis)
            if np.linalg.norm(n) > 1e-6:
                joint_pos_error = np.abs(np.dot(n, (pred_joint_pos_aligned_i - gt_joint_pos))) / np.linalg.norm(n)
            else:
                joint_pos_error = 0.0

            if gt_joint_type == "prismatic":
                joint_pos_error = 0.0

            combined_error = joint_ori_error + joint_pos_error
            if combined_error < best_error:
                best_error = combined_error
                best_T = T_i
                best_idx = i

        return best_T, best_idx

    return gt_camera_se3[0] @ np.linalg.inv(pred_camera_pose_gl[0])


def visualize_camera_and_joint_axis(
    camera_pose: np.ndarray,
    joint_axis: np.ndarray,
    joint_pos: np.ndarray | None = None,
    axis_length: float = 0.3,
    camera_size: float = 0.1,
    point_cloud: torch.Tensor | np.ndarray | None = None,
    moving_point_cloud: torch.Tensor | np.ndarray | None = None,
    static_point_cloud: torch.Tensor | np.ndarray | None = None,
) -> None:
    """Visualize a camera pose, joint axis, and optional point clouds with Open3D."""
    geometries = []

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries.append(world_frame)

    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=camera_size, origin=[0, 0, 0])
    camera_frame.transform(camera_pose)
    geometries.append(camera_frame)

    camera_position = camera_pose[:3, 3]
    camera_forward = camera_pose[:3, 2]
    camera_up = camera_pose[:3, 1]
    camera_right = camera_pose[:3, 0]

    frustum_depth = camera_size * 1.5
    frustum_width = camera_size * 1.0
    frustum_height = camera_size * 0.75

    apex = camera_position
    bottom_center = apex + camera_forward * frustum_depth
    corners = [
        bottom_center + camera_right * frustum_width / 2 + camera_up * frustum_height / 2,
        bottom_center - camera_right * frustum_width / 2 + camera_up * frustum_height / 2,
        bottom_center - camera_right * frustum_width / 2 - camera_up * frustum_height / 2,
        bottom_center + camera_right * frustum_width / 2 - camera_up * frustum_height / 2,
    ]

    for corner in corners:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([apex, corner])
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7]])
        geometries.append(line_set)

    for i in range(4):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([corners[i], corners[(i + 1) % 4]])
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7]])
        geometries.append(line_set)

    if joint_pos is None:
        joint_pos = np.array([0.0, 0.0, 0.0])

    joint_axis_normalized = joint_axis / np.linalg.norm(joint_axis)
    arrow_start = joint_pos - joint_axis_normalized * axis_length
    arrow_end = joint_pos + joint_axis_normalized * axis_length

    axis_line = o3d.geometry.LineSet()
    axis_line.points = o3d.utility.Vector3dVector([arrow_start, arrow_end])
    axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    axis_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
    geometries.append(axis_line)

    arrow_length = axis_length * 0.2
    arrow_radius = axis_length * 0.05

    arrow1 = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=arrow_radius,
        cone_radius=arrow_radius * 2,
        cylinder_height=arrow_length * 0.7,
        cone_height=arrow_length * 0.3,
    )
    arrow1.paint_uniform_color([1.0, 0.0, 0.0])

    default_direction = np.array([0.0, 0.0, 1.0])
    rotation_axis = np.cross(default_direction, joint_axis_normalized)
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(default_direction, joint_axis_normalized), -1.0, 1.0))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
    elif np.dot(default_direction, joint_axis_normalized) < 0:
        rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        rotation_matrix = np.eye(3)

    transform1 = np.eye(4)
    transform1[:3, :3] = rotation_matrix
    transform1[:3, 3] = joint_pos
    arrow1.transform(transform1)
    geometries.append(arrow1)

    arrow2 = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=arrow_radius,
        cone_radius=arrow_radius * 2,
        cylinder_height=arrow_length * 0.7,
        cone_height=arrow_length * 0.3,
    )
    arrow2.paint_uniform_color([1.0, 0.0, 0.0])

    transform2 = np.eye(4)
    transform2[:3, :3] = rotation_matrix @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    transform2[:3, 3] = joint_pos
    arrow2.transform(transform2)
    geometries.append(arrow2)

    joint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=arrow_radius * 1.5)
    joint_sphere.translate(joint_pos)
    joint_sphere.paint_uniform_color([1.0, 0.5, 0.0])
    geometries.append(joint_sphere)

    if point_cloud is not None:
        pcd = o3d.geometry.PointCloud()
        pcd_points = point_cloud.cpu().numpy() if isinstance(point_cloud, torch.Tensor) else point_cloud
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.paint_uniform_color([0.5, 0.7, 1.0])
        geometries.append(pcd)

    if moving_point_cloud is not None:
        moving_pcd = o3d.geometry.PointCloud()
        moving_points = moving_point_cloud.cpu().numpy() if isinstance(moving_point_cloud, torch.Tensor) else moving_point_cloud
        moving_pcd.points = o3d.utility.Vector3dVector(moving_points)
        moving_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(moving_pcd)

    if static_point_cloud is not None:
        static_pcd = o3d.geometry.PointCloud()
        static_points = static_point_cloud.cpu().numpy() if isinstance(static_point_cloud, torch.Tensor) else static_point_cloud
        static_pcd.points = o3d.utility.Vector3dVector(static_points)
        static_pcd.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(static_pcd)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Camera Pose and Joint Axis Visualization",
        width=1024,
        height=768,
        left=50,
        top=50,
    )


def compute_joint_error(
    gt_joint_parameter: Tuple[str, np.ndarray, np.ndarray, np.ndarray],
    pred_joint_parameter: Tuple[str, np.ndarray, np.ndarray, np.ndarray],
    gt_camera_se3: np.ndarray,
    pred_camera_pose_gl_aligned: np.ndarray,
    dataparser_scale: float = 1.0,
) -> Dict[str, float | bool]:
    gt_joint_type, gt_joint_axis, gt_joint_pos, gt_joint_value = gt_joint_parameter
    pred_joint_type, pred_joint_axis, pred_joint_pos, pred_joint_value = pred_joint_parameter
    pred_joint_axis = pred_joint_axis / np.linalg.norm(pred_joint_axis)

    joint_ori_error = np.arccos(np.abs(np.dot(pred_joint_axis, gt_joint_axis)))

    n = np.cross(pred_joint_axis, gt_joint_axis)
    if np.linalg.norm(n) > 1e-6:
        joint_pos_error = np.abs(np.dot(n, (pred_joint_pos - gt_joint_pos))) / np.linalg.norm(n)
    else:
        joint_pos_error = 0.0

    if gt_joint_type == "prismatic":
        joint_pos_error = 0.0

    def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _state_err_for_pred(pred_vals: np.ndarray) -> tuple[float, float]:
        if gt_joint_type == "revolute":
            diffs = _wrap_to_pi(gt_joint_value - pred_vals)
            off = np.median(diffs)
            residual = _wrap_to_pi(gt_joint_value - (pred_vals + off))
            err = np.mean(np.abs(residual))
            return float(err), float(off)
        off = np.median(gt_joint_value - pred_vals)
        err = np.mean(np.abs(gt_joint_value - (pred_vals + off)))
        return float(err), float(off)

    candidates = [pred_joint_value, -pred_joint_value]
    if gt_joint_type == "prismatic" and dataparser_scale != 1.0:
        scaled = pred_joint_value / dataparser_scale
        candidates += [scaled, -scaled]

    best_err = None
    for cand in candidates:
        err, _off = _state_err_for_pred(cand)
        if best_err is None or err < best_err:
            best_err = err
    joint_state_error = best_err

    pred_camera_rotation = pred_camera_pose_gl_aligned[:, :3, :3]
    pred_camera_translation = pred_camera_pose_gl_aligned[:, :3, 3]
    rotation_error_matrix = pred_camera_rotation @ gt_camera_se3[:, :3, :3].transpose(0, 2, 1)
    cam_rotation_error = np.mean(
        np.arccos(np.clip((np.trace(rotation_error_matrix, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))
    )
    cam_translation_error = np.mean(np.linalg.norm(pred_camera_translation - gt_camera_se3[:, :3, 3], axis=1))

    return {
        "joint orientation error": float(joint_ori_error),
        "joint position error": float(joint_pos_error),
        "joint state error": float(joint_state_error),
        "joint type error": pred_joint_type != gt_joint_type,
        "camera position error": float(cam_translation_error),
        "camera rotation error": float(cam_rotation_error),
    }


def compute_geometry_error(
    gt_full_pcd: np.ndarray,
    gt_moving_pcd: np.ndarray,
    gt_static_pcd: np.ndarray,
    recon_full_pcd: torch.Tensor,
    recon_moving_pcd: torch.Tensor | None = None,
    recon_static_pcd: torch.Tensor | None = None,
    device: str = "cuda:0",
) -> Dict[str, float]:
    """Compute geometry error using aligned point clouds."""

    recon_full_pcd = recon_full_pcd.to(device)
    gt_full_pcd_tensor = torch.from_numpy(gt_full_pcd).to(device).to(recon_full_pcd.dtype)
    bi_chamfer_dist = chamfer_distance(recon_full_pcd[None, ...], gt_full_pcd_tensor[None, ...])

    gt_moving_pcd_tensor = torch.from_numpy(gt_moving_pcd).to(device).to(recon_full_pcd.dtype)
    if recon_moving_pcd is not None:
        recon_moving_pcd = recon_moving_pcd.to(device)
        moving_chamfer_dist = chamfer_distance(recon_moving_pcd[None, ...], gt_moving_pcd_tensor[None, ...])
    else:
        moving_chamfer_dist = torch.tensor([1.0], device=device)

    gt_static_pcd_tensor = torch.from_numpy(gt_static_pcd).to(device).to(recon_full_pcd.dtype)
    if recon_static_pcd is not None:
        recon_static_pcd = recon_static_pcd.to(device)
        static_chamfer_dist = chamfer_distance(recon_static_pcd[None, ...], gt_static_pcd_tensor[None, ...])
    else:
        static_chamfer_dist = torch.tensor([1.0], device=device)

    return {
        "full_chamfer_distance": bi_chamfer_dist[0].item(),
        "moving_chamfer_distance": moving_chamfer_dist[0].item(),
        "static_chamfer_distance": static_chamfer_dist[0].item(),
    }


def save_chamfer_visualizations(
    gt_full_pcd: np.ndarray,
    gt_moving_pcd: np.ndarray,
    gt_static_pcd: np.ndarray,
    recon_full_pcd: torch.Tensor,
    recon_moving_pcd: torch.Tensor | None = None,
    recon_static_pcd: torch.Tensor | None = None,
    out_dir: str = "",
) -> None:
    """Save color-coded GT/reconstruction overlays for the point clouds used in Chamfer evaluation."""

    def to_numpy(points: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(points, torch.Tensor):
            return points.detach().cpu().numpy()
        return np.asarray(points)

    def save_overlay(name: str, gt_np: np.ndarray, recon: torch.Tensor | np.ndarray | None) -> None:
        if recon is None:
            return
        recon_np = to_numpy(recon)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(gt_np.astype(np.float64))
        gt_pcd.paint_uniform_color([0.0, 1.0, 0.0])

        recon_pcd = o3d.geometry.PointCloud()
        recon_pcd.points = o3d.utility.Vector3dVector(recon_np.astype(np.float64))
        recon_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        merged = gt_pcd + recon_pcd
        out_path = os.path.join(out_dir, f"{name}_overlay.ply")
        o3d.io.write_point_cloud(out_path, merged)

    os.makedirs(out_dir, exist_ok=True)
    save_overlay("full", gt_full_pcd, recon_full_pcd)
    save_overlay("moving", gt_moving_pcd, recon_moving_pcd)
    save_overlay("static", gt_static_pcd, recon_static_pcd)
