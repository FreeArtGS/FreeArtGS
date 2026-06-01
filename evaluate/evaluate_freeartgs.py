import os
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from pytorch3d.io import load_ply
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_data import (
    load_gt_joint_from_urdf,
    load_gt_pcd_from_urdf,
    load_joint_values_from_joint_states,
    load_predicted_data,
    resolve_default_partnet_mobility_dir,
)
from eval_utils import (
    CV_TO_GL,
    solve_T_align,
    compute_joint_error,
    compute_geometry_error,
    save_chamfer_visualizations,
    visualize_camera_and_joint_axis,
)
from mask_utils import resolve_psnr_motion_part
from psnr_utils import render_gaussians_and_psnr_gsplat


def _wrap_revolute_values(vals: np.ndarray) -> np.ndarray:
    wrapped = np.remainder(vals, 2.0 * np.pi)
    wrapped[wrapped < 0.0] += 2.0 * np.pi
    return wrapped


def _apply_transform_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return (transform @ points_h.T).T[:, :3]


def _load_and_align_ply(ply_path: str, align_T: np.ndarray) -> torch.Tensor | None:
    if not os.path.exists(ply_path):
        return None
    points, _ = load_ply(ply_path)
    aligned = _apply_transform_to_points(points.cpu().numpy(), align_T.astype(np.float64))
    return torch.from_numpy(aligned).to(torch.float32)


def _apply_transform_to_tensor_points(points: torch.Tensor | None, transform: np.ndarray) -> torch.Tensor | None:
    if points is None:
        return None
    aligned = _apply_transform_to_points(points.detach().cpu().numpy(), transform.astype(np.float64))
    return torch.from_numpy(aligned).to(points.dtype)


def _resolve_alignment(
    align_mode: str,
    gt_camera_se3: np.ndarray,
    pred_camera_pose_gl: np.ndarray,
    gt_joint_parameter: tuple,
    pred_joint_parameter_unaligned: tuple,
    gt_static_pcd: np.ndarray,
    results_dir: str,
    pred_joint_type: str,
    save_icp_overlay: bool,
    icp_overlay_path: str | None,
) -> tuple[np.ndarray, int | None]:
    best_frame_idx = None
    if align_mode == "icp_static":
        align_T_pose, best_frame_idx = solve_T_align(
            gt_camera_se3,
            pred_camera_pose_gl,
            mode="best",
            gt_joint_parameter=gt_joint_parameter,
            pred_joint_parameter=pred_joint_parameter_unaligned,
        )
        type_dir = os.path.join(results_dir, pred_joint_type)
        static_ply_path = os.path.join(type_dir, "static_pcd.ply")
        recon_static = _load_and_align_ply(static_ply_path, np.eye(4, dtype=np.float64))
        if recon_static is None:
            raise FileNotFoundError(f"Missing static point cloud for icp_static alignment: {static_ply_path}")

        recon_static_np_pre = recon_static.cpu().numpy()
        align_T = _compute_icp_align_T_from_pcds(gt_static_pcd, recon_static_np_pre, init_T=align_T_pose)
        if save_icp_overlay:
            out_ply = icp_overlay_path if icp_overlay_path is not None else os.path.join(results_dir, "icp_overlay.ply")
            _save_icp_overlay_ply(gt_static_pcd, recon_static_np_pre, align_T, out_ply)
        return align_T, best_frame_idx

    if align_mode == "best":
        return solve_T_align(
            gt_camera_se3,
            pred_camera_pose_gl,
            mode=align_mode,
            gt_joint_parameter=gt_joint_parameter,
            pred_joint_parameter=pred_joint_parameter_unaligned,
        )
    return solve_T_align(gt_camera_se3, pred_camera_pose_gl, mode=align_mode), best_frame_idx


def _refine_single_geometry_with_icp(
    gt_pcd: np.ndarray,
    recon_pcd: torch.Tensor | None,
) -> tuple[np.ndarray | None, torch.Tensor | None]:
    if recon_pcd is None:
        return None, None
    icp_T = _compute_icp_align_T_from_pcds(
        gt_pcd,
        recon_pcd.detach().cpu().numpy(),
        init_T=np.eye(4, dtype=np.float32),
    )
    recon_refined = _apply_transform_to_tensor_points(recon_pcd, icp_T)
    return icp_T, recon_refined


def _save_cameras_and_pointcloud_vis(gt_camera_se3: np.ndarray,
                                     pred_camera_pose_gl_aligned: np.ndarray,
                                     recon_full_pcd: torch.Tensor,
                                     out_path: str,
                                     recon_moving_pcd: torch.Tensor | None = None,
                                     recon_static_pcd: torch.Tensor | None = None,
                                     gt_joint_axis: np.ndarray | None = None,
                                     gt_joint_pos: np.ndarray | None = None,
                                     pred_joint_axis: np.ndarray | None = None,
                                     pred_joint_pos: np.ndarray | None = None,
                                     gt_full_pcd: np.ndarray | torch.Tensor | None = None,
                                     max_points: int = 50000) -> None:
    gt_centers = gt_camera_se3[:, :3, 3]
    pred_centers = pred_camera_pose_gl_aligned[:, :3, 3]

    pts = recon_full_pcd.cpu().numpy() if isinstance(recon_full_pcd, torch.Tensor) else recon_full_pcd
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.1, c='#7fb3ff', alpha=0.6)

    if recon_moving_pcd is not None:
        mv = recon_moving_pcd.cpu().numpy() if isinstance(recon_moving_pcd, torch.Tensor) else recon_moving_pcd
        if mv.shape[0] > max_points // 4:
            idx = np.random.choice(mv.shape[0], max_points // 4, replace=False)
            mv = mv[idx]
        ax.scatter(mv[:, 0], mv[:, 1], mv[:, 2], s=0.3, c='#ff4d4d', alpha=0.8)
    if recon_static_pcd is not None:
        st = recon_static_pcd.cpu().numpy() if isinstance(recon_static_pcd, torch.Tensor) else recon_static_pcd
        if st.shape[0] > max_points // 4:
            idx = np.random.choice(st.shape[0], max_points // 4, replace=False)
            st = st[idx]
        ax.scatter(st[:, 0], st[:, 1], st[:, 2], s=0.3, c='#33cc66', alpha=0.8)

    if gt_full_pcd is not None:
        gt_pts = gt_full_pcd.cpu().numpy() if isinstance(gt_full_pcd, torch.Tensor) else gt_full_pcd
        if gt_pts.shape[0] > max_points // 2:
            idx = np.random.choice(gt_pts.shape[0], max_points // 2, replace=False)
            gt_pts = gt_pts[idx]
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], s=0.2, c='#888888', alpha=0.5, label='GT pcd')

    ax.plot(gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2], color='k', linewidth=1.0, label='GT traj')
    ax.scatter(gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2], s=4, c='k')
    ax.scatter(gt_centers[0, 0], gt_centers[0, 1], gt_centers[0, 2], s=60, c='k', marker='s', label='GT start')

    ax.plot(pred_centers[:, 0], pred_centers[:, 1], pred_centers[:, 2], color='#ff9900', linewidth=1.0, label='Pred traj')
    ax.scatter(pred_centers[:, 0], pred_centers[:, 1], pred_centers[:, 2], s=4, c='#ff9900')
    ax.scatter(pred_centers[0, 0], pred_centers[0, 1], pred_centers[0, 2], s=60, c='#ff9900', marker='^', label='Pred start')

    all_pts = np.vstack([pts, gt_centers, pred_centers])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    spans = maxs - mins
    span_max = float(spans.max())
    arrow_len = max(span_max * 0.05, 1e-3)

    gt_step = max(1, gt_centers.shape[0] // 30)
    pred_step = max(1, pred_centers.shape[0] // 30)

    for i in range(0, gt_centers.shape[0], gt_step):
        R_i = gt_camera_se3[i, :3, :3]
        t_i = gt_centers[i]
        fwd_i = -R_i[:, 2]
        ax.quiver(t_i[0], t_i[1], t_i[2], fwd_i[0], fwd_i[1], fwd_i[2], length=arrow_len, color='k', arrow_length_ratio=0.25, linewidth=0.5)

    for i in range(0, pred_centers.shape[0], pred_step):
        R_i = pred_camera_pose_gl_aligned[i, :3, :3]
        t_i = pred_centers[i]
        fwd_i = -R_i[:, 2]
        ax.quiver(t_i[0], t_i[1], t_i[2], fwd_i[0], fwd_i[1], fwd_i[2], length=arrow_len, color='#ff9900', arrow_length_ratio=0.25, linewidth=0.5)

    axis_len = arrow_len * 3.0
    if gt_joint_axis is not None and gt_joint_pos is not None:
        a = gt_joint_axis / (np.linalg.norm(gt_joint_axis) + 1e-12)
        p = gt_joint_pos
        p0 = p - a * axis_len
        p1 = p + a * axis_len
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='g', linewidth=2.0, label='GT axis')
    if pred_joint_axis is not None and pred_joint_pos is not None:
        a = pred_joint_axis / (np.linalg.norm(pred_joint_axis) + 1e-12)
        p = pred_joint_pos
        p0 = p - a * axis_len
        p1 = p + a * axis_len
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='#ff9900', linewidth=2.0, linestyle='--', label='Pred axis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best')
    ax.set_title('Cameras and Point Cloud')

    mid = (mins + maxs) / 2.0
    ax.set_xlim(mid[0] - span_max / 2, mid[0] + span_max / 2)
    ax.set_ylim(mid[1] - span_max / 2, mid[1] + span_max / 2)
    ax.set_zlim(mid[2] - span_max / 2, mid[2] + span_max / 2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _compute_icp_align_T_from_pcds(gt_static_pcd: np.ndarray, recon_static_pcd: np.ndarray, max_iter: int = 5000, init_T: np.ndarray | None = None) -> np.ndarray:
    """Multi-scale robust ICP between GT static and reconstructed static PCDs.
    - Uses camera-based align (init_T) if provided
    - Coarse-to-fine: point-to-point then point-to-plane with Tukey loss
    Returns 4x4 float32. Identity on failure.
    """
    gt_np = gt_static_pcd.astype(np.float64)
    rs_np = recon_static_pcd.astype(np.float64)

    # Scene scale from GT bbox (used for voxel sizes)
    mins = gt_np.min(axis=0)
    maxs = gt_np.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))

    # Convert to Open3D
    gt = o3d.geometry.PointCloud(); gt.points = o3d.utility.Vector3dVector(gt_np)
    rs = o3d.geometry.PointCloud(); rs.points = o3d.utility.Vector3dVector(rs_np)

    # Multi-scale settings (coarse -> fine)
    voxel_scales = [diag / 50.0, diag / 100.0, diag / 200.0]
    voxel_scales = [max(v, 1e-3) for v in voxel_scales]
    iters = [min(max_iter // 50, 200), min(max_iter // 25, 300), min(max_iter // 10, 500)]

    T = np.eye(4, dtype=np.float64) if init_T is None else init_T.astype(np.float64)

    for lvl, (vox, n_iter) in enumerate(zip(voxel_scales, iters)):
        gt_ds = gt.voxel_down_sample(vox)
        rs_ds = rs.voxel_down_sample(vox)
        if lvl == 2:
            # estimate normals for point-to-plane at finest level
            rad = vox * 4.0
            gt_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=30))
            rs_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=30))

        threshold = vox * 3.0
        # Use default Open3D estimators (robust kernels may not be supported in this build)
        if lvl < 2:
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        else:
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        reg = o3d.pipelines.registration.registration_icp(
            rs_ds,
            gt_ds,
            threshold,
            T,
            est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(n_iter)),
        )
        if not np.isfinite(reg.transformation).all():
            raise ValueError(f"ICP produced non-finite transform at level {lvl}")
        T = reg.transformation

    T = T.astype(np.float32)
    if not np.isfinite(T).all():
        raise ValueError("ICP returned non-finite alignment transform")
    
    return T


def _save_icp_overlay_ply(gt_static_pcd: np.ndarray, recon_static_pcd: np.ndarray, align_T: np.ndarray, out_path: str) -> None:
    gt_np = gt_static_pcd.astype(np.float64)
    rs_np = recon_static_pcd.astype(np.float64)
    rs_h = np.concatenate([rs_np, np.ones((rs_np.shape[0], 1))], axis=1)
    rs_aligned = (align_T.astype(np.float64) @ rs_h.T).T[:, :3]

    p_gt = o3d.geometry.PointCloud(); p_gt.points = o3d.utility.Vector3dVector(gt_np)
    p_rs = o3d.geometry.PointCloud(); p_rs.points = o3d.utility.Vector3dVector(rs_aligned)
    p_gt.paint_uniform_color([0.0, 1.0, 0.0])
    p_rs.paint_uniform_color([1.0, 0.0, 0.0])
    merged = p_gt + p_rs
    o3d.io.write_point_cloud(out_path, merged)
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FreeArtGS results against PartNet-Mobility ground truth.")
    parser.add_argument("--dataset_dir", type=str, required=False, help="Path to dataset directory (optional; if omitted, use --obj_id to infer datasets/<obj_id>)")
    parser.add_argument("--results_dir", type=str, required=False, help="Path to results directory (exported evaluation data). Default: outputs/<obj_id>")
    parser.add_argument("--motion_part", type=int, default=0, choices=[0, 1], help="Which part is moving: 0 or 1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--align_mode", type=str, default="best", choices=["first", "average", "best", "icp_static"], help="Alignment strategy: 'first'/'average'/'best' (pose-based) or 'icp_static' (point-cloud ICP using static part)")
    parser.add_argument("--no_vis", action="store_true", help="Disable Open3D visualization")
    parser.add_argument("--save_vis", action="store_true", help="Save a PNG visualization of cameras + point cloud to <results_dir>/vis_cameras_pcd.png")
    parser.add_argument("--partnet_mobility_dir", type=str, default=None, help="PartNet-Mobility v0 root directory")
    parser.add_argument("--gt_debug", action="store_true", help="Save GT debug assets to <results_dir>/_gt_debug")
    parser.add_argument("--save_icp_overlay", action="store_true", help="Save ICP overlay point cloud (GT=green, Recon=red)")
    parser.add_argument("--icp_overlay_path", type=str, default=None, help="Output path for ICP overlay PLY (default: <results_dir>/icp_overlay.ply)")
    parser.add_argument("--save_chamfer_vis", action="store_true", help="Save full/moving/static Chamfer overlay point clouds under <results_dir>/chamfer_vis")
    parser.add_argument("--chamfer_vis_dir", type=str, default=None, help="Output directory for Chamfer overlay point clouds (default: <results_dir>/chamfer_vis)")
    # PSNR options
    parser.add_argument("--psnr", action="store_true", help="Compute PSNR by rendering gaussian PLY with eval poses and joint states")
    # Direct GT by id
    parser.add_argument("--obj_id", type=str, help="Object id in PartNet-Mobility (e.g., 100109). If provided with --joint_id, bypass SimDataLoader.")
    parser.add_argument("--joint_id", type=int, help="Joint id for the object (e.g., 0). Required when --obj_id is provided.")
    args = parser.parse_args()

    # Default results_dir to outputs/<obj_id> if not provided
    if not args.results_dir:
        if not args.obj_id:
            raise ValueError("--results_dir not provided. Either provide --results_dir or specify --obj_id to default to outputs/<obj_id>/evaluation_data.")
        args.results_dir = os.path.join("outputs", str(args.obj_id), "evaluation_data")

    # Default dataset_dir to <repo_root>/datasets/<obj_id> if not provided
    if not args.dataset_dir and args.obj_id:
        repo_root = Path(__file__).resolve().parent.parent
        args.dataset_dir = str(repo_root / "datasets" / str(args.obj_id))
    if not args.partnet_mobility_dir:
        args.partnet_mobility_dir = resolve_default_partnet_mobility_dir()

    # 1) Load predicted data first (indices length defines later alignment/truncation)
    (
        pred_joint_type,
        pred_joint_axis,
        pred_joint_pos,
        pred_joint_values,
        pred_camera_poses_4x4,
        pred_indices,
    ) = load_predicted_data(args.results_dir)

    # 2) Load GT data
    gt_camera_se3 = None

    # Prefer loading GT joint from PartNet-Mobility URDF instead of meta JSON
    if args.obj_id is None or args.joint_id is None:
        raise ValueError("--obj_id and --joint_id are required to load GT from PartNet-Mobility URDF")
    gt_joint_values = load_joint_values_from_joint_states(args.dataset_dir)
    gt_frame0_joint_value = float(gt_joint_values[0]) if gt_joint_values is not None and gt_joint_values.size > 0 else None

    gt_joint_type_est, gt_joint_axis_est, gt_joint_pos_est = load_gt_joint_from_urdf(
        args.partnet_mobility_dir, args.obj_id, args.joint_id
    )
    gt_full_pcd, gt_moving_pcd, gt_static_pcd = load_gt_pcd_from_urdf(
        args.partnet_mobility_dir, args.obj_id, args.joint_id, joint_value=gt_frame0_joint_value
    )
    # Place GT point clouds using object_poses.json (object base pose in the scene)
    obj_pose_path = os.path.join(args.dataset_dir, "object_poses.json")
    with open(obj_pose_path, "r") as f:
        obj_poses = json.load(f)
    # Use the first frame as reference; place GT PCD directly at this pose
    pos = np.asarray(obj_poses[0]["position"], dtype=np.float64)
    quat = np.asarray(obj_poses[0]["quaternion"], dtype=np.float64)
    quat = quat / (np.linalg.norm(quat) + 1e-12)
    # Interpret JSON quaternion as [w, x, y, z] and convert to scipy's [x, y, z, w]
    R_obj = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix().astype(np.float64)
    T0 = np.eye(4)
    T0[:3, :3] = R_obj
    T0[:3, 3] = pos
    # Additional world-frame conversion: first Ry(+90°), then Rx(-90°)
    R_y90 = np.array([[0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0]], dtype=np.float64)
    R_xm90 = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, -1.0, 0.0]], dtype=np.float64)
    R_y180 = np.array([[-1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, -1.0]], dtype=np.float64)
    # Order: first Ry(+90°), then Rx(-90°), finally Ry(+180°)
    R_fix = R_y180 @ R_xm90 @ R_y90
    R_obj = R_obj @ R_fix
    def _apply_RT(pts: np.ndarray) -> np.ndarray:
        return (R_obj @ pts.T).T + pos
    gt_full_pcd = _apply_RT(gt_full_pcd)
    gt_moving_pcd = _apply_RT(gt_moving_pcd)
    gt_static_pcd = _apply_RT(gt_static_pcd)
    
    # Also transform GT joint axis/position to world for visualization
    gt_joint_axis_world = (R_obj @ gt_joint_axis_est.astype(np.float64))
    gt_joint_axis_world = gt_joint_axis_world / (np.linalg.norm(gt_joint_axis_world) + 1e-12)
    gt_joint_pos_world = (R_obj @ gt_joint_pos_est.astype(np.float64)) + pos
    # Directly load camera poses from dataset_dir (supports datasets/<obj_id> layout)
    cam_pose_np = np.load(os.path.join(args.dataset_dir, "camera_pose.npy"))
    if cam_pose_np.ndim != 3 or cam_pose_np.shape[1:] != (4, 4):
        raise ValueError(f"camera_pose.npy must have shape (N,4,4), got {cam_pose_np.shape}")
    gt_camera_se3 = cam_pose_np.astype(np.float64)
    gt_indices = list(range(gt_camera_se3.shape[0]))



    # 3) Align frame subsets: use intersection of indices if available, else truncate
    # Map GT frames by index values
    common = [idx for idx in pred_indices if idx in gt_indices]
    gt_camera_se3 = gt_camera_se3[common]
    if gt_joint_values is not None:
        gt_joint_values = gt_joint_values[common]
    # pred_camera_poses_4x4 = pred_camera_poses_4x4[common]

    if pred_joint_type == "revolute" and pred_joint_values is not None:
        pred_joint_values = _wrap_revolute_values(pred_joint_values)
    if gt_joint_values is not None and gt_joint_type_est == "revolute":
        gt_joint_values = _wrap_revolute_values(gt_joint_values)


    # 4) Convert predicted poses to OpenGL convention if needed and normalize axes
    pred_camera_pose_gl = pred_camera_poses_4x4 @ CV_TO_GL
    gt_joint_axis = gt_joint_axis_est / (np.linalg.norm(gt_joint_axis_est) + 1e-12)
    pred_joint_axis = pred_joint_axis / (np.linalg.norm(pred_joint_axis) + 1e-12)

    # 5) Solve alignment
    # Use WORLD-frame GT joint parameters for alignment/metrics
    gt_joint_parameter = (gt_joint_type_est, gt_joint_axis_world, gt_joint_pos_world, gt_joint_values)
    pred_joint_parameter_unaligned = (pred_joint_type, pred_joint_axis, pred_joint_pos, pred_joint_values)
    align_T, best_frame_idx = _resolve_alignment(
        align_mode=args.align_mode,
        gt_camera_se3=gt_camera_se3,
        pred_camera_pose_gl=pred_camera_pose_gl,
        gt_joint_parameter=gt_joint_parameter,
        pred_joint_parameter_unaligned=pred_joint_parameter_unaligned,
        gt_static_pcd=gt_static_pcd,
        results_dir=args.results_dir,
        pred_joint_type=pred_joint_type,
        save_icp_overlay=args.save_icp_overlay,
        icp_overlay_path=args.icp_overlay_path,
    )

    # 6) Apply alignment
    pred_camera_pose_gl_aligned = np.array([align_T @ pose for pose in pred_camera_pose_gl])
    pred_joint_axis_aligned = align_T[:3, :3] @ pred_joint_axis
    pred_joint_axis_aligned = pred_joint_axis_aligned / (np.linalg.norm(pred_joint_axis_aligned) + 1e-12)
    pred_joint_pos_h = np.append(pred_joint_pos, 1.0)
    pred_joint_pos_aligned = (align_T @ pred_joint_pos_h)[:3]

    # 7) Load and align reconstructed point clouds
    recon_full_pcd = _load_and_align_ply(os.path.join(args.results_dir, "surface_pcd.ply"), align_T)
    if recon_full_pcd is None:
        raise FileNotFoundError(f"Missing surface point cloud: {os.path.join(args.results_dir, 'surface_pcd.ply')}")

    type_dir = os.path.join(args.results_dir, pred_joint_type)
    moving_ply = os.path.join(type_dir, "moving_pcd.ply")
    static_ply = os.path.join(type_dir, "static_pcd.ply")
    recon_moving_pcd = _load_and_align_ply(moving_ply, align_T)
    recon_static_pcd = _load_and_align_ply(static_ply, align_T)
    full_icp_T, recon_full_pcd_cd = _refine_single_geometry_with_icp(gt_full_pcd, recon_full_pcd)
    moving_icp_T, recon_moving_pcd_cd = _refine_single_geometry_with_icp(gt_moving_pcd, recon_moving_pcd)
    static_icp_T, recon_static_pcd_cd = _refine_single_geometry_with_icp(gt_static_pcd, recon_static_pcd)

    # 8) Optional visualization
    if not args.no_vis:
        visualize_camera_and_joint_axis(
            camera_pose=pred_camera_pose_gl_aligned[0],
            joint_axis=pred_joint_axis_aligned,
            joint_pos=pred_joint_pos_aligned,
            axis_length=0.3,
            camera_size=0.1,
            point_cloud=recon_full_pcd,
            moving_point_cloud=recon_moving_pcd,
            static_point_cloud=recon_static_pcd,
        )

    # 9) Metrics
    pred_joint_parameter_aligned = (
        pred_joint_type,
        pred_joint_axis_aligned,
        pred_joint_pos_aligned,
        pred_joint_values,
    )
    if gt_joint_values is not None:
        joint_error = compute_joint_error(
            (gt_joint_type_est, gt_joint_axis_world, gt_joint_pos_world, gt_joint_values),
            pred_joint_parameter_aligned,
            gt_camera_se3,
            pred_camera_pose_gl_aligned,
            dataparser_scale=1.0,
        )
    else:
        # Partial metrics without GT joint values
        joint_ori_error = float(np.arccos(np.clip(np.abs(np.dot(pred_joint_axis_aligned, gt_joint_axis_world)), -1.0, 1.0)))
        n = np.cross(pred_joint_axis_aligned, gt_joint_axis_world)
        if np.linalg.norm(n) > 1e-6:
            joint_pos_err = float(
                np.abs(np.dot(n, (pred_joint_pos_aligned - gt_joint_pos_world))) / (np.linalg.norm(n) + 1e-12)
            )
        else:
            joint_pos_err = 0.0
        if gt_joint_type_est == "prismatic":
            joint_pos_err = 0.0
        # Still compute camera pose errors because GT camera poses are available.
        pred_camera_rotation = pred_camera_pose_gl_aligned[:, :3, :3]
        pred_camera_translation = pred_camera_pose_gl_aligned[:, :3, 3]
        rotation_error_matrix = pred_camera_rotation @ gt_camera_se3[:, :3, :3].transpose(0, 2, 1)
        cam_rotation_error = float(np.mean(np.arccos(np.clip((np.trace(rotation_error_matrix, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))))
        cam_translation_error = float(np.mean(np.linalg.norm(pred_camera_translation - gt_camera_se3[:, :3, 3], axis=1)))
        joint_error = {
            "joint orientation error": joint_ori_error,
            "joint position error": joint_pos_err,
            "joint state error": None,
            "joint type error": pred_joint_type != gt_joint_type_est,
            "camera position error": cam_translation_error,
            "camera rotation error": cam_rotation_error,
        }

    geometry_error = compute_geometry_error(
        gt_full_pcd,
        gt_moving_pcd,
        gt_static_pcd,
        recon_full_pcd_cd,
        recon_moving_pcd_cd,
        recon_static_pcd_cd,
        device=args.device,
    )
    if args.save_chamfer_vis:
        chamfer_vis_dir = args.chamfer_vis_dir if args.chamfer_vis_dir is not None else os.path.join(args.results_dir, "chamfer_vis")
        save_chamfer_visualizations(
            gt_full_pcd,
            gt_moving_pcd,
            gt_static_pcd,
            recon_full_pcd_cd,
            recon_moving_pcd_cd,
            recon_static_pcd_cd,
            out_dir=chamfer_vis_dir,
        )

    repo_root = Path(__file__).resolve().parent.parent
    psnr_motion_part, psnr_motion_part_flipped, psnr_motion_part_scores, miou_result = resolve_psnr_motion_part(
        results_dir=args.results_dir,
        dataset_dir=args.dataset_dir,
        object_id=str(args.obj_id),
        repo_root=repo_root,
        cli_motion_part=args.motion_part,
    )

    # Optional PSNR with eval subset and gaussian PLY
    psnr_result = None
    if args.psnr:
        gaussian_dir = f"outputs/{args.obj_id}/splatfacto-art"
        eval_dir = os.path.join(args.dataset_dir, "eval")
        # Use the same align_T (pose-based or ICP) for PSNR rendering
        save_dir = os.path.join(args.results_dir, "psnr_renders")
        axis_dot = float(np.dot(pred_joint_axis_aligned, gt_joint_axis_world))
        axis_sign = 1.0 if axis_dot >= 0.0 else -1.0
        psnr_result = render_gaussians_and_psnr_gsplat(
            gaussian_dir,
            eval_dir,
            args.dataset_dir,
            align_T,
            gt_joint_type_est,
            pred_joint_axis_aligned,
            pred_joint_pos_aligned,
            T0,
            model_joint_value_sign=axis_sign,
            motion_part=psnr_motion_part,
            save_dir=save_dir,
            no_save=False,
        )

    # Optional: save 2D visualization of GT/Pred camera trajectories with point cloud
    if args.save_vis:
        out_img = os.path.join(args.results_dir, "vis_cameras_pcd.png")
        _save_cameras_and_pointcloud_vis(
            gt_camera_se3,
            pred_camera_pose_gl_aligned,
            recon_full_pcd,
            out_img,
            recon_moving_pcd,
            recon_static_pcd,
            gt_joint_axis=gt_joint_axis_world,
            gt_joint_pos=gt_joint_pos_world,
            pred_joint_axis=pred_joint_axis_aligned,
            pred_joint_pos=pred_joint_pos_aligned,
            gt_full_pcd=gt_full_pcd,
        )

    results = {
        "joint_error": joint_error,
        "geometry_error": geometry_error,
        "geometry_icp_transforms": {
            "full": None if full_icp_T is None else full_icp_T.tolist(),
            "moving": None if moving_icp_T is None else moving_icp_T.tolist(),
            "static": None if static_icp_T is None else static_icp_T.tolist(),
        },
        "align_mode": args.align_mode,
        "best_frame_idx": int(best_frame_idx) if best_frame_idx is not None else None,
        "gt_joint_type": gt_joint_type_est,
        "psnr_motion_part": int(psnr_motion_part),
        "psnr_motion_part_flipped": bool(psnr_motion_part_flipped),
    }
    if psnr_motion_part_scores is not None:
        results["psnr_motion_part_scores"] = psnr_motion_part_scores
    if psnr_result is not None:
        results["psnr_eval"] = psnr_result
    if miou_result is not None:
        results["mask_miou"] = miou_result

    # 10) Save
    out_path = os.path.join(args.results_dir, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print("Joint estimation results:", joint_error)
    print("Geometry error results:", geometry_error)
    print("PSNR results:", psnr_result)
    print("mIoU results:", miou_result)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
