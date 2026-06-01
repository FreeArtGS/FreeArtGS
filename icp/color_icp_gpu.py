import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("BLIS_NUM_THREADS", "4")
import json
import argparse
import sys
import numpy as np
import open3d as o3d
import torch
from glob import glob
from termcolor import cprint
from tqdm import tqdm
import cv2
from pathlib import Path


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pipeline_config import (
    apply_section_to_args,
    load_pipeline_config,
    resolve_icp,
    resolve_pipeline_paths,
    set_arg_defaults,
    set_path_defaults,
)

MIN_POINTS_PER_PCD = 50
BAD_MATCH_FITNESS = 0.5
DEBUG_ICP = False
ICP_T_DEVICE = None
TORCH_DEVICE = None
HYBRID_PLANE_WEIGHT = 1.0
HYBRID_POINT_WEIGHT = 0.1
HYBRID_DAMPING = 1e-6
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class RegistrationResultSimple:
    def __init__(self, transformation: np.ndarray, fitness: float, inlier_rmse: float):
        self.transformation = transformation
        self.fitness = float(fitness)
        self.inlier_rmse = float(inlier_rmse)


def resolve_gpu_devices(device_arg: str):
    """Resolve CUDA devices for both Open3D Tensor ICP and Torch math kernels."""
    req = str(device_arg).strip().lower() if device_arg is not None else "auto"
    if req == "auto":
        req = "cuda"
    if not req.startswith("cuda"):
        raise ValueError(f"color_icp_gpu only supports CUDA devices, got '{device_arg}'.")

    if not o3d.core.cuda.is_available():
        raise RuntimeError("Open3D CUDA is not available, but color_icp_gpu requires GPU.")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available, but color_icp_gpu requires GPU.")

    if req == "cuda":
        idx = int(torch.cuda.current_device())
    elif ":" in req:
        idx = int(req.split(":", 1)[1])
    else:
        raise ValueError(f"Invalid CUDA device format: '{req}'")
    if idx < 0:
        raise ValueError(f"Invalid CUDA device index: {idx}")

    torch_device_count = torch.cuda.device_count()
    if idx >= torch_device_count:
        raise RuntimeError(
            f"Requested CUDA device {idx}, but torch sees {torch_device_count} device(s)."
        )

    try:
        torch_device = torch.device(req)
    except Exception as e:
        raise ValueError(f"Invalid CUDA device '{req}': {e}") from e

    o3d_device = o3d.core.Device(f"CUDA:{idx}")
    return o3d_device, torch_device


def _to_tensor_pcd(legacy_pcd, device):
    return o3d.t.geometry.PointCloud.from_legacy(
        legacy_pcd,
        dtype=o3d.core.Dtype.Float32,
        device=device,
    )


def _ensure_tensor_pcd(pcd):
    if isinstance(pcd, o3d.t.geometry.PointCloud):
        return pcd if pcd.device == ICP_T_DEVICE else pcd.to(ICP_T_DEVICE)
    return _to_tensor_pcd(pcd, ICP_T_DEVICE)


def transform_points_homogeneous(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply 4x4 rigid transform to Nx3 points on CUDA."""
    if points.size == 0:
        return points

    pts_t = torch.from_numpy(points).to(device=TORCH_DEVICE, dtype=torch.float64)
    tf_t = torch.tensor(transform, dtype=torch.float64, device=TORCH_DEVICE)
    ones_t = torch.ones((pts_t.shape[0], 1), dtype=pts_t.dtype, device=TORCH_DEVICE)
    pts_h_t = torch.cat((pts_t, ones_t), dim=1)
    pts_w_t = pts_h_t @ tf_t.transpose(0, 1)
    return pts_w_t[:, :3].cpu().numpy()


def rotate_normals(normals: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate Nx3 normals on CUDA."""
    if normals.size == 0:
        return normals

    nrm_t = torch.from_numpy(normals).to(device=TORCH_DEVICE, dtype=torch.float64)
    rot_t = torch.tensor(rotation, dtype=torch.float64, device=TORCH_DEVICE)
    nrm_w_t = nrm_t @ rot_t.transpose(0, 1)
    return nrm_w_t.cpu().numpy()


def _o3d_tensor_to_torch(tensor, dtype=torch.float64):
    out = torch.utils.dlpack.from_dlpack(tensor.contiguous().to_dlpack()).to(TORCH_DEVICE)
    return out.to(dtype=dtype)


def _torch_to_o3d_tensor(tensor):
    tensor = tensor.contiguous()
    return o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def _skew_torch(v):
    zeros = torch.zeros_like(v[:, 0])
    return torch.stack(
        (
            torch.stack((zeros, -v[:, 2], v[:, 1]), dim=1),
            torch.stack((v[:, 2], zeros, -v[:, 0]), dim=1),
            torch.stack((-v[:, 1], v[:, 0], zeros), dim=1),
        ),
        dim=1,
    )


def _so3_exp_torch(omega):
    theta = torch.linalg.norm(omega)
    zero = torch.zeros((), dtype=omega.dtype, device=omega.device)
    wx = torch.stack(
        (
            torch.stack((zero, -omega[2], omega[1])),
            torch.stack((omega[2], zero, -omega[0])),
            torch.stack((-omega[1], omega[0], zero)),
        )
    )
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device)
    if float(theta.detach().cpu()) < 1e-12:
        return eye + wx
    a = torch.sin(theta) / theta
    b = (1.0 - torch.cos(theta)) / (theta * theta)
    return eye + a * wx + b * (wx @ wx)

def save_trajectory_plot(global_poses, valid_mask, output_path, set_idx):
    """
    Visualize the camera trajectory using the requested matplotlib style.
    global_poses: List[np.ndarray] of 4x4 matrices
    valid_mask: List[bool]
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract poses for valid frames.
    valid_poses = [p for i, p in enumerate(global_poses) if valid_mask[i]]
    if not valid_poses:
        print(f"Set {set_idx}: No valid poses to plot.")
        return

    centers = []
    for i, mat in enumerate(valid_poses):
        # R is the rotation matrix and t is the translation vector (camera-to-world).
        R = mat[:3, :3]
        t = mat[:3, 3]
        centers.append(t)

        # 1. Plot camera centers as red scatter points.
        ax.scatter(t[0], t[1], t[2], color='r', s=10, alpha=0.5)
        
        # 2. Draw a direction arrow every 5 frames (blue, camera Z axis / viewing direction).
        if i % 5 == 0:
            scale = 0.05  # Adjust arrow length to match scene scale.
            # mat[:3, 2] is the third column of the rotation matrix, i.e. the camera Z axis.
            ax.quiver(
                t[0], t[1], t[2],
                R[0, 2], R[1, 2], R[2, 2],
                length=scale, color='b', alpha=0.7, arrow_length_ratio=0.3
            )

    # 3. Draw a continuous dashed gray trajectory.
    centers = np.array(centers)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], 
            color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # 4. Force equal axis scales to avoid trajectory distortion.
    # This follows the provided set_axes_equal logic.
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    max_range = max(x_limits[1]-x_limits[0], y_limits[1]-y_limits[0], z_limits[1]-z_limits[0])
    x_mid, y_mid, z_mid = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Sequence {set_idx} Camera Trajectory')
    
    plt.savefig(output_path, dpi=300)
    plt.close()  # Release memory.
    print(f"Trajectory plot saved to {output_path}")
# ================= Helper Functions =================

def compute_adaptive_voxel_size(point_clouds,
                                k: int = 10,
                                sample_points: int = 5000,
                                frames_to_use: int = 10,
                                multiplier: float = 2.0,
                                min_voxel: float = 0.005,
                                max_voxel: float = 0.05) -> float:
    """Estimate a good voxel_size from point density using k-NN distances.
    
    Revised strategy:
    To avoid underestimating density after downsampling, run KD-tree queries on
    the original (or near-original) point-cloud density.
    1) Randomly choose several frames.
    2) For each frame, keep most points to build the KD-tree and preserve density.
    3) Randomly choose a subset of query points.
    4) Measure k-NN distances from those queries in the dense point cloud.
    """
    if not point_clouds:
        return 0.02
        
    # Randomly select up to frames_to_use frames.
    n = len(point_clouds)
    if n <= frames_to_use:
        sel_indices = list(range(n))
    else:
        sel_indices = list(np.linspace(0, n - 1, frames_to_use, dtype=int))

    dists = []
    
    for i in sel_indices:
        pcd = point_clouds[i]
        pts = np.asarray(pcd.points)
        num_pts = pts.shape[0]
        if num_pts < k + 1:
            continue

        # Preserve local density by random subsampling instead of voxel-grid downsampling.
        max_tree_points = 1000000
        if num_pts > max_tree_points:
            indices = np.random.choice(num_pts, max_tree_points, replace=False)
            tree_pts = pts[indices]
        else:
            tree_pts = pts
            
        # Build the KD-tree on the high-density reference cloud.
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(tree_pts)
        kdtree = o3d.geometry.KDTreeFlann(pcd_tmp)
        
        # Randomly sample query points; we do not need many.
        num_queries = min(sample_points, tree_pts.shape[0])
        query_indices = np.random.choice(tree_pts.shape[0], num_queries, replace=False)
        query_pts = tree_pts[query_indices]
        
        query_k = k + 1 
        for q_pt in query_pts:
            ok, idxs, d2 = kdtree.search_knn_vector_3d(q_pt, query_k)
            if ok >= query_k:
                # d2[k] is the squared distance to the k-th neighbor.
                d = float(np.sqrt(d2[k]))
                if np.isfinite(d) and d > 0:
                    dists.append(d)

    if len(dists) == 0:
        return 0.02
    d_med = float(np.median(dists))
    # Because this is estimated on the original density, d_med is smaller, so a multiplier around 2.0 still works well.
    vs = float(np.clip(multiplier * d_med, min_voxel, max_voxel))
    return vs


def refine_registration(source, target, voxel_size, init=None, retry_cnt=0):
    """Try color-aware ICP with coarse, fine, and geometric polish stages."""
    return refine_registration_tensor(source, target, voxel_size, init=init, retry_cnt=retry_cnt)


def refine_registration_hybrid_tensor(
    source_t,
    target_t,
    voxel_size,
    init,
    max_corr_dist,
    max_iteration=None,
):
    """GPU Gauss-Newton ICP with point-to-plane and point-to-point residuals."""
    src_pts = _o3d_tensor_to_torch(source_t.point["positions"])
    tgt_pts_o3d = target_t.point["positions"]
    tgt_pts = _o3d_tensor_to_torch(tgt_pts_o3d)
    tgt_normals = _o3d_tensor_to_torch(target_t.point["normals"])
    tgt_normals = torch.nn.functional.normalize(tgt_normals, dim=1)

    search = o3d.core.nns.NearestNeighborSearch(tgt_pts_o3d)
    search.knn_index()

    transform = np.array(init, dtype=np.float64, copy=True)
    R = torch.tensor(transform[:3, :3], dtype=torch.float64, device=TORCH_DEVICE)
    t = torch.tensor(transform[:3, 3], dtype=torch.float64, device=TORCH_DEVICE)

    max_iteration = max(1, int(max_iteration))
    max_corr_sq = float(max_corr_dist) * float(max_corr_dist)
    plane_scale = float(np.sqrt(max(HYBRID_PLANE_WEIGHT, 0.0)))
    point_scale = float(np.sqrt(max(HYBRID_POINT_WEIGHT, 0.0)))
    if plane_scale <= 0.0 and point_scale <= 0.0:
        plane_scale = 1.0
    damping = float(max(HYBRID_DAMPING, 0.0))
    eye6 = torch.eye(6, dtype=torch.float64, device=TORCH_DEVICE)
    last_rmse = float("inf")
    last_fitness = 0.0

    for _ in range(max_iteration):
        pts_w = src_pts @ R.transpose(0, 1) + t
        query = _torch_to_o3d_tensor(pts_w.to(dtype=torch.float32))
        indices_o3d, dists_o3d = search.knn_search(query, 1)
        indices = _o3d_tensor_to_torch(indices_o3d, dtype=torch.int64).reshape(-1)
        dists = _o3d_tensor_to_torch(dists_o3d).reshape(-1)
        mask = dists <= max_corr_sq
        inlier_count = int(mask.sum().detach().cpu())
        if inlier_count < 6:
            break

        p = pts_w[mask]
        q = tgt_pts[indices[mask]]
        n = tgt_normals[indices[mask]]
        diff = p - q

        plane_res = torch.sum(diff * n, dim=1, keepdim=True)
        plane_j = torch.cat((torch.cross(p, n, dim=1), n), dim=1)

        point_res = diff.reshape(-1, 1)
        point_j = torch.cat(
            (
                -_skew_torch(p),
                torch.eye(3, dtype=torch.float64, device=TORCH_DEVICE)
                .unsqueeze(0)
                .expand(p.shape[0], -1, -1),
            ),
            dim=2,
        ).reshape(-1, 6)

        if plane_scale > 0.0 and point_scale > 0.0:
            J = torch.cat((plane_j * plane_scale, point_j * point_scale), dim=0)
            r = torch.cat((plane_res * plane_scale, point_res * point_scale), dim=0)
        elif point_scale > 0.0:
            J = point_j * point_scale
            r = point_res * point_scale
        else:
            J = plane_j * plane_scale
            r = plane_res * plane_scale

        lhs = J.transpose(0, 1) @ J + damping * eye6
        rhs = -(J.transpose(0, 1) @ r).reshape(6)
        try:
            delta = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            delta = torch.linalg.lstsq(lhs, rhs[:, None]).solution.reshape(6)

        if not torch.isfinite(delta).all():
            break

        dR = _so3_exp_torch(delta[:3])
        dt = delta[3:]
        R = dR @ R
        t = dR @ t + dt

        last_rmse = float(torch.sqrt(torch.mean(dists[mask])).detach().cpu())
        last_fitness = float(inlier_count / max(1, src_pts.shape[0]))
        if float(torch.linalg.norm(delta).detach().cpu()) < 1e-7:
            break

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.detach().cpu().numpy()
    out[:3, 3] = t.detach().cpu().numpy()
    return RegistrationResultSimple(out, last_fitness, last_rmse)


def refine_registration_tensor(source, target, voxel_size, init=None, retry_cnt=0):
    """Open3D Tensor ICP path (CUDA when available). Keeps original 3-stage logic."""
    source_t = _ensure_tensor_pcd(source)
    target_t = _ensure_tensor_pcd(target)

    radius_normal = voxel_size * 2.0
    target_t.estimate_normals(max_nn=100, radius=radius_normal)
    source_t.estimate_normals(max_nn=100, radius=radius_normal)

    trans_init = np.identity(4, dtype=np.float64) if init is None else np.asarray(init, dtype=np.float64)
    trans_init_t = o3d.core.Tensor(trans_init, dtype=o3d.core.Dtype.Float64, device=ICP_T_DEVICE)

    criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=50 if retry_cnt == 0 else 100,
    )

    # 1) Coarse colored ICP
    try:
        coarse_threshold = float(voxel_size) * 5.0
        coarse_result = o3d.t.pipelines.registration.icp(
            source_t,
            target_t,
            coarse_threshold,
            trans_init_t,
            o3d.t.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria,
        )
        trans_curr_t = coarse_result.transformation
    except Exception as e:
        print(f"Tensor Colored ICP failed in coarse registration, falling back to point-to-plane ICP, {e}")
        coarse_threshold = float(voxel_size) * 5.0
        coarse_result = o3d.t.pipelines.registration.icp(
            source_t,
            target_t,
            coarse_threshold,
            trans_init_t,
            o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )
        trans_curr_t = coarse_result.transformation

    # 2) Fine hybrid ICP: point-to-plane anchors depth, point-to-point constrains plane tangents.
    distance_threshold = float(voxel_size) * 3.0
    result_fine = refine_registration_hybrid_tensor(
        source_t,
        target_t,
        voxel_size,
        trans_curr_t.cpu().numpy(),
        distance_threshold,
        max_iteration=50,
    )
    trans_curr_t = o3d.core.Tensor(
        result_fine.transformation,
        dtype=o3d.core.Dtype.Float64,
        device=ICP_T_DEVICE,
    )

    # 3) Ultra-fine hybrid polish with a tighter correspondence threshold.
    fine_threshold = float(voxel_size) * 1.0
    result_ultra = refine_registration_hybrid_tensor(
        source_t,
        target_t,
        voxel_size,
        trans_curr_t.cpu().numpy(),
        fine_threshold,
        max_iteration=50,
    )
    return RegistrationResultSimple(
        transformation=result_ultra.transformation,
        fitness=float(result_ultra.fitness),
        inlier_rmse=float(result_ultra.inlier_rmse),
    )


def pcd_num_points(pcd: o3d.geometry.PointCloud) -> int:
    if isinstance(pcd, o3d.t.geometry.PointCloud):
        if "positions" not in pcd.point:
            return 0
        return int(pcd.point["positions"].shape[0])
    pts = np.asarray(pcd.points)
    return 0 if pts.size == 0 else int(pts.shape[0])


def evaluate_transform_quality(source, target, max_corr_dist, transform):
    """Evaluate fitness / RMSE for a given initial transform without running ICP."""
    source_t = _ensure_tensor_pcd(source)
    target_t = _ensure_tensor_pcd(target)
    transform_t = o3d.core.Tensor(
        np.asarray(transform, dtype=np.float64),
        dtype=o3d.core.Dtype.Float64,
        device=ICP_T_DEVICE,
    )
    result = o3d.t.pipelines.registration.evaluate_registration(
        source_t, target_t, float(max_corr_dist), transform_t
    )
    return float(result.fitness), float(result.inlier_rmse)


def refine_registration_with_init_guard(source, target, voxel_size, max_corr_dist, init=None, retry_cnt=0):
    """Run ICP and report whether final fitness is not worse than the initial fitness."""
    init_transform = np.eye(4, dtype=np.float64) if init is None else np.asarray(init, dtype=np.float64)
    # Use the same correspondence scale as the final ICP polish stage for a fair fitness comparison.
    init_corr_dist = float(voxel_size) * 1.0
    init_fitness, init_rmse = evaluate_transform_quality(source, target, init_corr_dist, init_transform)
    result = refine_registration(source, target, voxel_size, init=init_transform, retry_cnt=retry_cnt)
    improved_or_equal = result.fitness >= (init_fitness - 0.01)
    return result, init_fitness, init_rmse, improved_or_equal


def evaluate_global_pose_against_reference_frame(
    curr_pcd,
    ref_pcd,
    ref_pose,
    curr_pose,
    max_corr_dist,
):
    """Evaluate the current global-pose guess against the reference frame only."""
    if pcd_num_points(ref_pcd) < MIN_POINTS_PER_PCD:
        return None, None

    curr_to_ref = relative_transform_from_global_poses(curr_pose, ref_pose)
    return evaluate_transform_quality(curr_pcd, ref_pcd, max_corr_dist, curr_to_ref)


def assess_motion_plausibility(rel_transform, frame_gap, fit, motion_history_t, motion_history_r, args):
    """Assess whether a relative motion is plausible and return normalized translation/rotation."""
    frame_gap = float(max(1, frame_gap))
    curr_t = np.linalg.norm(rel_transform[:3, 3]) / frame_gap
    curr_r = rotation_angle_deg(rel_transform[:3, :3]) / frame_gap

    is_plausible = True
    if curr_t > args.max_frame_translation or curr_r > args.max_frame_rotation_deg:
        is_plausible = False
    if len(motion_history_t) >= 5:
        t_limit = np.mean(motion_history_t) + 3.0 * np.std(motion_history_t)
        r_limit = np.mean(motion_history_r) + 3.0 * np.std(motion_history_r)
        if fit < 0.90 and (curr_t > max(t_limit, 0.5) or curr_r > max(r_limit, 5.0)):
            is_plausible = False

    return is_plausible, curr_t, curr_r


def is_good_registration(fitness: float) -> bool:
    return fitness >= BAD_MATCH_FITNESS


def preprocess_downsample_and_fpfh(pcd, voxel_size):
    """Voxel downsample, estimate normals, and compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh


def try_ransac_initial_alignment(source, target, voxel_size):
    """Use FPFH + RANSAC for global initialization; return None on failure."""
    src_down, src_fpfh = preprocess_downsample_and_fpfh(source, voxel_size)
    tgt_down, tgt_fpfh = preprocess_downsample_and_fpfh(target, voxel_size)

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000)
    )
    if result.fitness > 0.05:
        return result.transformation
    return None


def random_downsample_to_n(pcd, n):
    """Randomly downsample to exactly n points; keep the cloud unchanged if it already has fewer."""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd
    num_points = pts.shape[0]
    if num_points <= n:
        return pcd

    indices = np.random.choice(num_points, n, replace=False)
    down = o3d.geometry.PointCloud()
    down.points = o3d.utility.Vector3dVector(pts[indices])
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
        down.colors = o3d.utility.Vector3dVector(cols[indices])
    if pcd.has_normals():
        nrm = np.asarray(pcd.normals)
        down.normals = o3d.utility.Vector3dVector(nrm[indices])
    return down


def random_downsample_to_n_tensor(pcd_t, n):
    """Randomly downsample tensor point cloud to exactly n points."""
    num_points = pcd_num_points(pcd_t)
    if num_points <= n:
        return pcd_t
    idx_t = torch.randperm(num_points, device=TORCH_DEVICE, dtype=torch.int64)[:n]
    idx_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(idx_t))
    return pcd_t.select_by_index(idx_o3d)


def rotation_angle_deg(R: np.ndarray) -> float:
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def transform_point_cloud_copy(pcd, transform):
    """Return a copy of the point cloud after applying the rigid transform."""
    out = o3d.geometry.PointCloud()
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return out

    pts_w = transform_points_homogeneous(pts, transform)
    out.points = o3d.utility.Vector3dVector(pts_w)

    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors).copy())

    if pcd.has_normals():
        nrm = np.asarray(pcd.normals)
        nrm_w = rotate_normals(nrm, transform[:3, :3])
        out.normals = o3d.utility.Vector3dVector(nrm_w)

    return out


def relative_transform_from_global_poses(src_pose, tgt_pose):
    """Compute the relative transform src->tgt from global poses."""
    return np.linalg.inv(tgt_pose) @ src_pose


def get_recent_valid_indices(valid_mask, end_idx, window_size):
    """Return recent valid frame indices before end_idx, ordered from earliest to latest."""
    indices = []
    for idx in range(end_idx - 1, -1, -1):
        if valid_mask[idx]:
            indices.append(idx)
            if len(indices) >= window_size:
                break
    indices.reverse()
    return indices


def build_local_map(point_clouds, global_poses, valid_mask, end_idx, window_size, max_points):
    """Build a world-coordinate local map from recent valid frames before end_idx."""
    local_indices = get_recent_valid_indices(valid_mask, end_idx, window_size)
    local_map = o3d.geometry.PointCloud()
    if not local_indices:
        return local_map, local_indices

    all_points = []
    all_colors = []
    all_normals = []
    for idx in local_indices:
        pcd = point_clouds[idx]
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            continue

        pts_w = transform_points_homogeneous(pts, global_poses[idx])
        all_points.append(pts_w)

        if pcd.has_colors():
            all_colors.append(np.asarray(pcd.colors))

        if pcd.has_normals():
            nrm = np.asarray(pcd.normals)
            nrm_w = rotate_normals(nrm, global_poses[idx][:3, :3])
            all_normals.append(nrm_w)

    if not all_points:
        return local_map, local_indices

    local_map.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    if all_colors:
        local_map.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    if all_normals:
        local_map.normals = o3d.utility.Vector3dVector(np.vstack(all_normals))

    if max_points > 0:
        local_map = random_downsample_to_n(local_map, max_points)

    return local_map, local_indices


def build_local_map_tensor(tensor_point_clouds, global_poses, valid_mask, end_idx, window_size, max_points):
    """Build local map in world coordinates directly on tensor/GPU path."""
    local_indices = get_recent_valid_indices(valid_mask, end_idx, window_size)
    if not local_indices:
        return o3d.t.geometry.PointCloud(ICP_T_DEVICE), local_indices

    local_map_t = None
    for idx in local_indices:
        pcd_t = tensor_point_clouds[idx]
        if pcd_num_points(pcd_t) == 0:
            continue

        tf_t = o3d.core.Tensor(
            np.asarray(global_poses[idx], dtype=np.float64),
            dtype=o3d.core.Dtype.Float64,
            device=ICP_T_DEVICE,
        )
        pcd_w = pcd_t.clone()
        pcd_w.transform(tf_t)
        local_map_t = pcd_w if local_map_t is None else local_map_t.append(pcd_w)

    if local_map_t is None:
        return o3d.t.geometry.PointCloud(ICP_T_DEVICE), local_indices

    if max_points > 0:
        local_map_t = random_downsample_to_n_tensor(local_map_t, max_points)

    return local_map_t, local_indices


def predict_pose_constant_velocity(global_poses, valid_indices):
    """Predict the next pose from the last two valid poses; fall back to the latest pose if needed."""
    if not valid_indices:
        return np.eye(4)
    if len(valid_indices) < 2:
        return global_poses[valid_indices[-1]].copy()

    prev_idx = valid_indices[-1]
    prev_prev_idx = valid_indices[-2]
    rel_prev = relative_transform_from_global_poses(
        global_poses[prev_prev_idx], global_poses[prev_idx]
    )
    return global_poses[prev_idx] @ np.linalg.inv(rel_prev)


def write_transforms_json(base_dir, set_idx, poses, cam_K, height, width, valid_mask=None):
    """Overwrite transforms_{set_idx}.json without preserving legacy fields."""
    fl_x = cam_K[0, 0]
    fl_y = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]
    path = os.path.join(base_dir, f"transforms_{set_idx}.json")
    frames = []
    n = len(poses)
    if valid_mask is None:
        valid_mask = [True] * n
    for i in range(n):
        if i < len(valid_mask) and not valid_mask[i]:
            continue
        frames.append({
            'file_path': f"images/{i:05d}.png",
            "mask_path": f"mask_{set_idx}/{i:05d}.png",
            "depth_path": f"depth/{i:05d}.npz",
            'transform_matrix': poses[i].tolist(),
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "h": height,
            "w": width,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        })

    data = {'frames': frames}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Camera poses saved to '{path}' (overwritten)")


# ================= Core Logic =================

def build_registered_point_cloud_from_global_poses(point_clouds, global_poses):
    """Fuse point clouds using per-frame global poses relative to frame 0."""
    merged_pcd = o3d.geometry.PointCloud()
    all_points = []
    all_colors = []
    all_normals = []

    for idx, pcd in enumerate(point_clouds):
        if idx >= len(global_poses): 
            break
            
        # Transform points directly
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            continue
            
        pts = transform_points_homogeneous(pts, global_poses[idx])
        all_points.append(pts)

        if pcd.has_colors():
            all_colors.append(np.asarray(pcd.colors))
            
        if pcd.has_normals():
            nrm = np.asarray(pcd.normals)
            # Normals only rotate
            nrm = rotate_normals(nrm, global_poses[idx][:3, :3])
            all_normals.append(nrm)

    if not all_points:
        return merged_pcd

    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    if all_colors:
        merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    if all_normals:
        merged_pcd.normals = o3d.utility.Vector3dVector(np.vstack(all_normals))
    return merged_pcd


def build_pose_graph(args, point_clouds, tensor_point_clouds=None):
    """Build the pose graph with a local-map frontend."""
    motion_history_t = []
    motion_history_r = []
    window_history = 20
    n = len(point_clouds)
    if n == 0:
        return [], [], []

    odom_transforms = []
    global_poses = [np.eye(4)]
    valid_mask = [False] * n
    valid_mask[0] = pcd_num_points(point_clouds[0]) >= MIN_POINTS_PER_PCD

    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(global_poses[0]))

    max_corr = args.voxel_size * 2.0
    last_good_idx = 0 if valid_mask[0] else -1

    for curr_idx in tqdm(range(1, n), desc="Local Map Odometry"):
        curr_pcd = point_clouds[curr_idx]
        curr_pcd_t = tensor_point_clouds[curr_idx] if tensor_point_clouds is not None else curr_pcd
        curr_pose = global_poses[curr_idx - 1].copy()
        rel_from_ref = np.eye(4)
        is_success = False
        ref_edge_is_verified = False
        prev_edge_is_verified = False
        local_indices = []
        fit = 0.0
        rmse = 0.0
        curr_t = 0.0
        curr_r = 0.0
        ref_idx = curr_idx - 1
        cv_init_fit = None
        cv_init_rmse = None
        cv_init_pose = None
        cv_init_rel_from_ref = None
        cv_init_t = None
        cv_init_r = None

        curr_pts_ok = pcd_num_points(curr_pcd) >= MIN_POINTS_PER_PCD
        if not curr_pts_ok:
            if DEBUG_ICP:
                cprint(f"[LocalMap] {curr_idx} INVALID (insufficient points)", "cyan")
        else:
            if tensor_point_clouds is not None:
                local_map_t, local_indices = build_local_map_tensor(
                    tensor_point_clouds,
                    global_poses,
                    valid_mask,
                    curr_idx,
                    args.local_map_window,
                    args.local_map_max_points,
                )
                local_map = None
            else:
                local_map, local_indices = build_local_map(
                    point_clouds,
                    global_poses,
                    valid_mask,
                    curr_idx,
                    args.local_map_window,
                    args.local_map_max_points,
                )
                local_map_t = _to_tensor_pcd(local_map, ICP_T_DEVICE) if local_indices else None

            if local_indices and pcd_num_points(local_map_t) >= MIN_POINTS_PER_PCD:
                ref_idx = local_indices[-1]
                ref_pcd_t = tensor_point_clouds[ref_idx] if tensor_point_clouds is not None else point_clouds[ref_idx]
                retry = 0
                while retry < 3 and not is_success:
                    if retry == 0:
                        init_guess = predict_pose_constant_velocity(global_poses, local_indices)
                    elif retry == 1:
                        init_guess = global_poses[curr_idx - 1].copy()
                    else:
                        if local_map is None:
                            local_map = local_map_t.to_legacy()
                        init_guess = try_ransac_initial_alignment(curr_pcd, local_map, args.voxel_size)
                        if init_guess is None:
                            init_guess = global_poses[ref_idx].copy()

                    if retry == 0:
                        cv_init_fit, cv_init_rmse = evaluate_global_pose_against_reference_frame(
                            curr_pcd_t,
                            ref_pcd_t,
                            global_poses[ref_idx],
                            init_guess,
                            max_corr,
                        )
                        cv_init_pose = init_guess.copy()
                        cv_init_rel_from_ref = relative_transform_from_global_poses(
                            global_poses[ref_idx], init_guess
                        )
                        frame_gap = float(max(1, curr_idx - ref_idx))
                        cv_init_t = np.linalg.norm(cv_init_rel_from_ref[:3, 3]) / frame_gap
                        cv_init_r = rotation_angle_deg(cv_init_rel_from_ref[:3, :3]) / frame_gap

                    result, init_fit_local, init_rmse_local, fit_not_worse_than_init = refine_registration_with_init_guard(
                        curr_pcd_t,
                        local_map_t,
                        args.voxel_size,
                        max_corr,
                        init=init_guess,
                        retry_cnt=retry,
                    )
                    curr_pose = result.transformation
                    fit = result.fitness
                    rmse = result.inlier_rmse

                    rel_from_ref = relative_transform_from_global_poses(
                        global_poses[ref_idx], curr_pose
                    )
                    is_plausible, curr_t, curr_r = assess_motion_plausibility(
                        rel_from_ref,
                        curr_idx - ref_idx,
                        fit,
                        motion_history_t,
                        motion_history_r,
                        args,
                    )

                    if is_good_registration(fit) and is_plausible and fit_not_worse_than_init:
                        is_success = True
                        ref_edge_is_verified = True
                        prev_edge_is_verified = True
                        valid_mask[curr_idx] = True
                        last_good_idx = curr_idx
                        motion_history_t.append(curr_t)
                        motion_history_r.append(curr_r)
                        if len(motion_history_t) > window_history:
                            motion_history_t.pop(0)
                            motion_history_r.pop(0)
                    else:
                        retry += 1

                    if DEBUG_ICP:
                        color = "green" if is_success else "red"
                        reason = "" if is_plausible else " (MOTION BLOCKED)"
                        if not fit_not_worse_than_init:
                            reason += (
                                f" (FIT DROP {fit:.3f} < init {init_fit_local:.3f}, "
                                f"init_rmse={init_rmse_local:.4f})"
                            )
                        cprint(
                            f"[LocalMap] {curr_idx} ref={ref_idx} map={len(local_indices)} "
                            f"fit={fit:.3f} rmse={rmse:.4f} t={curr_t:.3f} r={curr_r:.1f}{reason}",
                            color,
                        )
                if DEBUG_ICP and not is_success and cv_init_fit is not None and cv_init_rmse is not None:
                    cprint(
                        f"[LocalMap] {curr_idx} failed after 3 retries; "
                        f"constant-velocity init ref-fit={cv_init_fit:.3f} rmse={cv_init_rmse:.4f}",
                        "yellow",
                    )
                if (
                    not is_success
                    and valid_mask[ref_idx]
                    and cv_init_fit is not None
                    and cv_init_fit >= args.init_fallback_fitness
                    and cv_init_pose is not None
                    and cv_init_rel_from_ref is not None
                ):
                    curr_pose = cv_init_pose
                    rel_from_ref = cv_init_rel_from_ref
                    curr_t = 0.0 if cv_init_t is None else cv_init_t
                    curr_r = 0.0 if cv_init_r is None else cv_init_r
                    fit = cv_init_fit
                    rmse = 0.0 if cv_init_rmse is None else cv_init_rmse
                    is_success = True
                    ref_edge_is_verified = True
                    prev_edge_is_verified = (ref_idx == curr_idx - 1)
                    valid_mask[curr_idx] = True
                    last_good_idx = curr_idx
                    motion_history_t.append(curr_t)
                    motion_history_r.append(curr_r)
                    if len(motion_history_t) > window_history:
                        motion_history_t.pop(0)
                        motion_history_r.pop(0)
                    if DEBUG_ICP:
                        cprint(
                            f"[LocalMap] {curr_idx} recovered with constant-velocity init "
                            f"(ref-fit={fit:.3f}, rmse={rmse:.4f}, ref={ref_idx})",
                            "green",
                        )
                if (
                    ref_idx != curr_idx - 1
                    and pcd_num_points(point_clouds[curr_idx - 1]) >= MIN_POINTS_PER_PCD
                ):
                    prev_init_pose = curr_pose.copy()
                    if not is_success and cv_init_pose is not None:
                        prev_init_pose = cv_init_pose.copy()
                    prev_init = relative_transform_from_global_poses(
                        prev_init_pose, global_poses[curr_idx - 1]
                    )
                    prev_result, prev_init_fit, prev_init_rmse, prev_fit_not_worse_than_init = refine_registration_with_init_guard(
                        curr_pcd_t,
                        tensor_point_clouds[curr_idx - 1] if tensor_point_clouds is not None else point_clouds[curr_idx - 1],
                        args.voxel_size,
                        max_corr,
                        init=prev_init,
                    )
                    prev_rel = prev_result.transformation
                    prev_fit = prev_result.fitness
                    prev_rmse = prev_result.inlier_rmse
                    prev_is_plausible, prev_t, prev_r = assess_motion_plausibility(
                        prev_rel,
                        1,
                        prev_fit,
                        motion_history_t,
                        motion_history_r,
                        args,
                    )
                    if DEBUG_ICP:
                        color = "green" if (is_good_registration(prev_fit) and prev_is_plausible and prev_fit_not_worse_than_init) else "yellow"
                        reason = "" if prev_is_plausible else " (MOTION BLOCKED)"
                        if not prev_fit_not_worse_than_init:
                            reason += (
                                f" (FIT DROP {prev_fit:.3f} < init {prev_init_fit:.3f}, "
                                f"init_rmse={prev_init_rmse:.4f})"
                            )
                        cprint(
                            f"[PrevEdge] {curr_idx - 1}->{curr_idx} "
                            f"fit={prev_fit:.3f} rmse={prev_rmse:.4f} t={prev_t:.3f} r={prev_r:.1f}{reason}",
                            color,
                        )
                    if is_good_registration(prev_fit) and prev_is_plausible and prev_fit_not_worse_than_init:
                        prev_edge_is_verified = True
                        if not is_success:
                            curr_pose = global_poses[curr_idx - 1] @ np.linalg.inv(prev_rel)
                            rel_from_ref = relative_transform_from_global_poses(
                                global_poses[ref_idx], curr_pose
                            )
                            fit = prev_fit
                            rmse = prev_rmse
                            curr_t = prev_t
                            curr_r = prev_r
                            is_success = True
                            motion_history_t.append(curr_t)
                            motion_history_r.append(curr_r)
                            if len(motion_history_t) > window_history:
                                motion_history_t.pop(0)
                                motion_history_r.pop(0)
                            if DEBUG_ICP:
                                cprint(
                                    f"[LocalMap] {curr_idx} recovered with prev-edge match "
                                    f"(fit={fit:.3f}, rmse={rmse:.4f}, ref stays {ref_idx})",
                                    "green",
                                )
            elif last_good_idx < 0:
                curr_pose = np.eye(4)
                rel_from_ref = np.eye(4)
                ref_idx = curr_idx - 1
                is_success = True
                ref_edge_is_verified = True
                prev_edge_is_verified = True
                valid_mask[curr_idx] = True
                last_good_idx = curr_idx
                if DEBUG_ICP:
                    cprint(f"[LocalMap] {curr_idx} anchored as the first valid frame", "green")
            elif DEBUG_ICP:
                cprint(f"[LocalMap] {curr_idx} INVALID (empty local map)", "cyan")

        if not is_success:
            if last_good_idx >= 0:
                curr_pose = global_poses[last_good_idx].copy()
                ref_idx = last_good_idx
                rel_from_ref = np.eye(4)
            else:
                curr_pose = np.eye(4)
                ref_idx = 0
                rel_from_ref = np.eye(4)

        global_poses.append(curr_pose)

        rel_prev = relative_transform_from_global_poses(global_poses[curr_idx - 1], global_poses[curr_idx])
        odom_transforms.append(rel_prev)

        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(curr_pose))

        prev_edge_success = is_success and prev_edge_is_verified
        ref_edge_success = is_success and ref_edge_is_verified

        if prev_edge_success:
            try:
                information_prev = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    point_clouds[curr_idx - 1], curr_pcd, max_corr, rel_prev
                )
            except Exception:
                information_prev = np.eye(6)
        else:
            information_prev = np.eye(6) * 1e-9

        if ref_edge_success:
            try:
                information_ref = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    point_clouds[ref_idx], curr_pcd, max_corr, rel_from_ref
                )
            except Exception:
                information_ref = np.eye(6)
        else:
            information_ref = np.eye(6) * 1e-9

        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                curr_idx - 1, curr_idx, rel_prev, information_prev, uncertain=(not prev_edge_success)
            )
        )

        if ref_idx not in (curr_idx - 1, curr_idx):
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    ref_idx, curr_idx, rel_from_ref, information_ref, uncertain=(not ref_edge_success)
                )
            )

    # 2. Loop closure detection and global optimization.
    if args.global_opt:
        # Loop-closure edges (sparse sampling).
        if args.loop_interval and args.loop_interval > 0:
            loop_candidates = range(0, n - args.loop_interval)
            for i in tqdm(loop_candidates, desc="Loop Closure Detection"):
                j = i + args.loop_interval
                if not valid_mask[i] or (j < n and not valid_mask[j]):
                    continue
                # print(f"[PoseGraph] Loop candidate {i}<->{j}") # Reduce clutter
                src = point_clouds[i]
                tgt = point_clouds[j]
                src_t = tensor_point_clouds[i] if tensor_point_clouds is not None else src
                tgt_t = tensor_point_clouds[j] if tensor_point_clouds is not None else tgt

                init = try_ransac_initial_alignment(src, tgt, args.voxel_size)

                result_refined, loop_init_fit, loop_init_rmse, loop_fit_not_worse_than_init = refine_registration_with_init_guard(
                    src_t, tgt_t, args.voxel_size, max_corr, init=init
                )
                if result_refined.fitness < args.bad_match_fitness or not loop_fit_not_worse_than_init:
                    if DEBUG_ICP:
                        reason = f"fit={result_refined.fitness:.3f}"
                        if not loop_fit_not_worse_than_init:
                            reason += f", init_fit={loop_init_fit:.3f}, init_rmse={loop_init_rmse:.4f}"
                        cprint(f"[Loop] {i}<->{j} VALID ({reason}), failed", 'yellow')
                    continue
                

                information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    src, tgt, max_corr, result_refined.transformation
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        i, j, result_refined.transformation, information, uncertain=True
                    )
                )

        # Global optimization.
        print("Running Global Optimization (LevenbergMarquardt)...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_corr,
            edge_prune_threshold=0.25,
            preference_loop_closure=2.0,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

    optimized_poses = [node.pose for node in pose_graph.nodes]
    return odom_transforms, optimized_poses, valid_mask




def process_sequence(base_dir, point_clouds, args, set_idx, cam_K, height, width):
    """Refactored logic to process a single sequence of point clouds."""
    num_files = len(point_clouds)
    if num_files == 0:
        cprint(f"[Set {set_idx}] No point clouds found.", 'yellow')
        return

    # Auto voxel size estimation if unset (<=0)
    # if args.voxel_size <= 0:
    est_vs = compute_adaptive_voxel_size(point_clouds)
    args.voxel_size = float(est_vs)
    print(f"[Auto] Estimated voxel_size from set {set_idx}: {args.voxel_size:.5f}")

    mode_str = "Local Map + Global Optimization" if args.global_opt else "Local Map Odometry"
    print(
        f"[Set {set_idx}] Processing with {mode_str} "
        f"(window={args.local_map_window}, max_points={args.local_map_max_points})..."
    )

    # Cache static frame point clouds as Tensor once to avoid repeated legacy->tensor conversion.
    tensor_point_clouds = [_to_tensor_pcd(pcd, ICP_T_DEVICE) for pcd in point_clouds]

    # Unified processing using build_pose_graph
    transforms, global_poses, valid_mask = build_pose_graph(args, point_clouds, tensor_point_clouds=tensor_point_clouds)

    # Save relative transforms and global poses
    np.save(os.path.join(base_dir, f"relative_transforms_{set_idx}.npy"), np.array(transforms))
    np.save(os.path.join(base_dir, f"global_poses_{set_idx}.npy"), np.array(global_poses))
    print(f"Saved 'relative_transforms_{set_idx}.npy' and 'global_poses_{set_idx}.npy'")
    plot_path = os.path.join(base_dir, f"trajectory_vis_{set_idx}.png")
    save_trajectory_plot(global_poses, valid_mask, plot_path, set_idx)
    # Save registered point cloud (downsampled)
    # Using the same function for both optimized and sequential modes
    reg = build_registered_point_cloud_from_global_poses(
        point_clouds,
        global_poses,
    )
    reg_ds = random_downsample_to_n(reg, args.sample_points)
    out_ply = os.path.join(base_dir, f"registered_{set_idx}_{args.sample_points // 1000}k.ply")
    o3d.io.write_point_cloud(out_ply, reg_ds)
    print(f"Registered point cloud saved to '{out_ply}'")
    
    # Save transforms.json
    write_transforms_json(base_dir, set_idx, global_poses, cam_K, height, width, valid_mask=valid_mask)


# ================= Main =================

def main():
    parser = argparse.ArgumentParser(
        description="Register per-frame point clouds with local-map ICP (GPU-only)"
    )
    parser.add_argument('--config', type=str, default=None, help='Path to pipeline config YAML')
    parser.add_argument('--object_name', type=str, default=None, help='Object name override used with --config')
    parser.add_argument('--dataset_dir', type=str, default=None, help='Dataset directory')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="CUDA device only: cuda | cuda:<id> | auto(=cuda)",
    )
    args = parser.parse_args()
    set_arg_defaults(args, resolve_icp({}))

    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        icp_config = resolve_icp(config)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        set_path_defaults(
            args,
            {
                "dataset_dir": runtime_paths["dataset_dir"],
            },
        )
        apply_section_to_args(args, icp_config)

    if args.dataset_dir is None:
        parser.error("--dataset_dir is required when --config is not provided")
    args.local_map_window = max(1, int(args.local_map_window))
    args.local_map_max_points = int(args.local_map_max_points)
    
    # Override global constants
    global MIN_POINTS_PER_PCD, BAD_MATCH_FITNESS, DEBUG_ICP, ICP_T_DEVICE, TORCH_DEVICE
    global HYBRID_PLANE_WEIGHT, HYBRID_POINT_WEIGHT, HYBRID_DAMPING
    MIN_POINTS_PER_PCD = max(0, int(args.min_points_per_pcd))
    BAD_MATCH_FITNESS = float(args.bad_match_fitness)
    DEBUG_ICP = bool(args.icp_debug)
    HYBRID_PLANE_WEIGHT = float(args.hybrid_plane_weight)
    HYBRID_POINT_WEIGHT = float(args.hybrid_point_weight)
    HYBRID_DAMPING = max(0.0, float(args.hybrid_damping))
    ICP_T_DEVICE, TORCH_DEVICE = resolve_gpu_devices(getattr(args, "device", "cuda"))
    cpu_threads = max(1, int(os.environ.get("ICP_CPU_THREADS", os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, cpu_threads)))
    except RuntimeError:
        pass
    print(f"[GPU] Open3D ICP device: {ICP_T_DEVICE}")
    print(f"[GPU] Torch math device: {TORCH_DEVICE}")
    print(f"[CPU] Thread cap per process: {cpu_threads}")

    # Data
    base_dir = args.dataset_dir
    pcd0_dir = os.path.join(base_dir, "point_clouds_0")
    pcd1_dir = os.path.join(base_dir, "point_clouds_1")
    pcd0_files = sorted(glob(os.path.join(pcd0_dir, "*.ply")))
    pcd1_files = sorted(glob(os.path.join(pcd1_dir, "*.ply")))
    
    cam_K_path = os.path.join(base_dir, "cam_K.txt")
    cam_K = np.loadtxt(cam_K_path)
    
    image_dir = os.path.join(base_dir, "images")

    # Read first image to get dimensions
    img_files = sorted(glob(os.path.join(image_dir, "*.png")))
    height, width = cv2.imread(img_files[0]).shape[:2]

    # Load point clouds
    print("Loading point clouds...")
    point_clouds0 = [o3d.io.read_point_cloud(f) for f in pcd0_files]
    point_clouds1 = [o3d.io.read_point_cloud(f) for f in pcd1_files]

    # Process Set 0
    process_sequence(base_dir, point_clouds0, args, 0, cam_K, height, width)

    # Process Set 1
    process_sequence(base_dir, point_clouds1, args, 1, cam_K, height, width)


if __name__ == "__main__":
    main()    
