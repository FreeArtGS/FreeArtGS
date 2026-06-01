#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pipeline_config import load_pipeline_config, resolve_pipeline_paths


# ------------------------- Linear Algebra Helpers -------------------------

def _project_to_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    return Rp


def _vee_skew(M: np.ndarray) -> np.ndarray:
    return np.array([M[2,1]-M[1,2], M[0,2]-M[2,0], M[1,0]-M[0,1]], dtype=np.float64)


def _axis_from_Rs_pairwise(Rs: np.ndarray) -> np.ndarray:
    """
    Estimate the common axis from pairwise relative rotations:
        A = sum (R_ij - I)^T (R_ij - I)
        u = eigmin(A)
    """
    I = np.eye(3, dtype=np.float64)
    N = Rs.shape[0]
    # Collect all pairwise angles first, then use adaptive thresholds to remove tiny or extreme outliers.
    angle_list = []
    Rij_list = []
    for i in range(N):
        for j in range(i+1, N):
            Rij = Rs[i] @ Rs[j].T
            Rij = _project_to_so3(Rij)
            c = 0.5 * (np.trace(Rij) - 1.0)
            c = float(np.clip(c, -1.0, 1.0))
            ang = float(np.arccos(c))
            Rij_list.append(Rij)
            angle_list.append(ang)
    angle_arr = np.asarray(angle_list, dtype=np.float64)
    A = np.zeros((3, 3), dtype=np.float64)
    # Filter with mean +/- k*sigma within groups that share the same temporal lag d=j-i.
    used = 0
    idx = 0
    k_sigma = 1.0
    # Group angles and relative rotations by lag.
    groups = {}
    idx_map = {}
    for i in range(N):
        for j in range(i+1, N):
            d = j - i
            groups.setdefault(d, {"angles": [], "Rij": []})
            idx_map.setdefault(d, [])
            groups[d]["angles"].append(angle_arr[idx])
            groups[d]["Rij"].append(Rij_list[idx])
            idx_map[d].append(idx)
            idx += 1
    # Apply mean +/- k*sigma filtering to each lag group.
    for d, data in groups.items():
        angs = np.asarray(data["angles"], dtype=np.float64)
        Rijs = data["Rij"]
        mu = float(np.mean(angs))
        sigma = float(np.std(angs))
        if not np.isfinite(sigma) or sigma <= 1e-9:
            sel = np.ones_like(angs, dtype=bool)
        else:
            lo = max(0.0, mu - k_sigma * sigma)
            hi = min(np.pi, mu + k_sigma * sigma)
            sel = (angs >= lo) & (angs <= hi)
        for m, keep in enumerate(sel):
            if not keep:
                continue
            A += (Rijs[m] - I).T @ (Rijs[m] - I)
            used += 1
    print(f"used {used}/{len(Rij_list)} pairs for axis estimation (grouped by lag)")
    if used == 0:
        for Rij in Rij_list:
            A += (Rij - I).T @ (Rij - I)

    w, V = np.linalg.eigh(A)
    u = V[:, int(np.argmin(w))]
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return u / n


def rodrigues_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix about 'axis' (unit) with Rodrigues' formula."""
    ax = axis.astype(np.float32)
    ax_norm = np.linalg.norm(ax)
    if ax_norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    ax = ax / ax_norm
    x, y, z = ax
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )
    return R


class JointRefinementViewer:
    def __init__(
        self,
        pts: np.ndarray,
        cols: Optional[np.ndarray],
        axis: np.ndarray,
        origin: np.ndarray,
        joint_type: str = "revolute",
    ) -> None:
        self.pts = pts
        self.cols = cols if cols is not None else np.ones_like(pts) * 0.7
        self.axis = axis.astype(np.float32)
        self.origin = origin.astype(np.float32)
        self.joint_type = joint_type

        bbox_max = pts.max(0)
        bbox_min = pts.min(0)
        diag_len = float(np.linalg.norm(bbox_max - bbox_min))
        self.axis_step = 0.002 * diag_len
        self.axis_rotate_step = 1.0  # degrees

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.pts)
        self.pcd.colors = o3d.utility.Vector3dVector(self.cols)

        self.axis_arrow = None
        self.arrow_length = diag_len * 0.8

    def _create_arrow(self):
        direction = self.axis
        length = self.arrow_length
        radius = length * 0.015
        
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=radius,
            cone_radius=radius * 2.0,
            cylinder_height=length * 0.8,
            cone_height=length * 0.2,
        )
        arrow.compute_vertex_normals()
        
        # Rotate from +Z to direction
        z = np.array([0, 0, 1])
        d = direction / np.linalg.norm(direction)
        if np.allclose(d, z):
            rot = np.eye(3)
        elif np.allclose(d, -z):
            rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            v = np.cross(z, d)
            s = np.linalg.norm(v)
            c = np.dot(z, d)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))
            
        arrow.rotate(rot, center=(0, 0, 0))
        # Center the arrow on origin
        arrow.translate(self.origin - d * (length * 0.5))
        arrow.paint_uniform_color([1, 0, 0])
        return arrow

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        if o3d is None:
            print("[warn] Open3D not installed, skipping refinement.")
            return self.axis, self.origin

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Joint Refinement (Press 'Enter' or 'Esc' to finish)", width=1280, height=720)
        
        vis.add_geometry(self.pcd)
        self.axis_arrow = self._create_arrow()
        vis.add_geometry(self.axis_arrow)
        
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.arrow_length * 0.1)
        vis.add_geometry(coord)

        def update(vis_):
            vis_.remove_geometry(self.axis_arrow, reset_bounding_box=False)
            self.axis_arrow = self._create_arrow()
            vis_.add_geometry(self.axis_arrow, reset_bounding_box=False)

        def rotate(vis_, axis_rot, angle_deg):
            R = rodrigues_rotation_matrix(axis_rot, np.deg2rad(angle_deg))
            self.axis = (R @ self.axis).astype(np.float32)
            self.axis /= np.linalg.norm(self.axis)
            update(vis_)

        def move(vis_, delta):
            self.origin += delta.astype(np.float32)
            update(vis_)

        # Callbacks
        vis.register_key_callback(ord("Q"), lambda v: rotate(v, np.array([1, 0, 0]), self.axis_rotate_step))
        vis.register_key_callback(ord("W"), lambda v: rotate(v, np.array([1, 0, 0]), -self.axis_rotate_step))
        vis.register_key_callback(ord("E"), lambda v: rotate(v, np.array([0, 1, 0]), self.axis_rotate_step))
        vis.register_key_callback(ord("T"), lambda v: rotate(v, np.array([0, 1, 0]), -self.axis_rotate_step))
        vis.register_key_callback(ord("Y"), lambda v: rotate(v, np.array([0, 0, 1]), self.axis_rotate_step))
        vis.register_key_callback(ord("U"), lambda v: rotate(v, np.array([0, 0, 1]), -self.axis_rotate_step))
        
        vis.register_key_callback(ord("1"), lambda v: move(v, np.array([-self.axis_step, 0, 0])))
        vis.register_key_callback(ord("2"), lambda v: move(v, np.array([self.axis_step, 0, 0])))
        vis.register_key_callback(ord("3"), lambda v: move(v, np.array([0, -self.axis_step, 0])))
        vis.register_key_callback(ord("4"), lambda v: move(v, np.array([0, self.axis_step, 0])))
        vis.register_key_callback(ord("5"), lambda v: move(v, np.array([0, 0, -self.axis_step])))
        vis.register_key_callback(ord("6"), lambda v: move(v, np.array([0, 0, self.axis_step])))
        
        vis.register_key_callback(ord("."), lambda v: move(v, self.axis * self.axis_step))
        vis.register_key_callback(ord(","), lambda v: move(v, -self.axis * self.axis_step))
        
        vis.register_key_callback(ord("["), lambda v: setattr(self, "axis_step", self.axis_step / 1.5))
        vis.register_key_callback(ord("]"), lambda v: setattr(self, "axis_step", self.axis_step * 1.5))

        print("\nJoint Refinement Controls:")
        print("  Q/W, E/T, Y/U: Rotate axis around X, Y, Z")
        print("  1/2, 3/4, 5/6: Move origin along X, Y, Z")
        print("  ./, : Move origin along current axis")
        print("  [/]: Adjust step size")
        print("  Enter/Esc: Finish and use current axis")

        vis.run()
        vis.destroy_window()
        return self.axis, self.origin


def _angle_from_R_and_axis(R: np.ndarray, u: np.ndarray) -> float:
    """
    Compute the signed angle of rotation matrix R around a given axis u:
        sin(theta) = u^T * (1/2 vee(R - R^T))
        cos(theta) = 0.5(tr(R) - 1)
        theta = atan2(sin(theta), cos(theta))
    """
    s = 0.5 * _vee_skew(R - R.T)
    c = 0.5 * (np.trace(R) - 1.0)
    c = float(np.clip(c, -1.0, 1.0))
    s_par = float(u @ s)
    return float(np.arctan2(s_par, c))


def _unwrap_angles(angles: np.ndarray, discont: float = np.pi/2) -> np.ndarray:
    return np.unwrap(angles, discont=discont)

# ------------------------- Robust Line Fitting (RANSAC) -------------------------
def _fit_line_ransac_3d(
    points: np.ndarray,
    num_iters: int = 200,
    dist_thresh: Optional[float] = None,
    min_inliers_ratio: float = 0.4,
    min_pair_sep_ratio: float = 0.05,
    lo_iters: int = 2,
    mad_k: float = 3.0,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a robust 3D line to a point set:
      - Returns (direction_w, point_on_line_p0, inlier_mask).
      - Falls back to PCA over all points when RANSAC fails.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = int(pts.shape[0])
    if n < 2:
        raise AssertionError("Need at least two points for line fitting")
    if n == 2:
        v = pts[1] - pts[0]
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            w = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            w = v / nv
        p0 = pts[0]
        mask = np.ones(n, dtype=bool)
        return w, p0, mask
    # Adaptive threshold based on the bounding-box diagonal.
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if not np.isfinite(diag) or diag < 1e-12:
        diag = 1.0
    if dist_thresh is None:
        dist_thresh = max(1e-5, 0.01 * diag)
    min_pair_sep = max(1e-6, float(min_pair_sep_ratio) * diag)
    rng = np.random.RandomState(int(random_state))
    best_mask = None
    best_count = -1
    # RANSAC sampling.
    for _ in range(int(num_iters)):
        i, j = rng.randint(0, n), rng.randint(0, n)
        if i == j:
            continue
        a = pts[i]
        b = pts[j]
        v = b - a
        if float(np.linalg.norm(v)) < min_pair_sep:
            continue
        nv = float(np.linalg.norm(v))
        if nv < 1e-9:
            continue
        w = v / nv
        # Distance to the line.
        vec = pts - a[None, :]
        proj = (vec @ w)[:, None] * w[None, :]
        perp = vec - proj
        d = np.linalg.norm(perp, axis=1)
        mask = d <= float(dist_thresh)
        cnt = int(mask.sum())
        if cnt > best_count:
            best_count = cnt
            best_mask = mask
            # Early stop when the inlier ratio is very high.
            if cnt >= 0.95 * n:
                break
    # Fall back to PCA if RANSAC fails or finds too few inliers.
    if best_mask is None or best_count < max(3, int(min_inliers_ratio * n)):
        Pm = pts.mean(axis=0)
        X = pts - Pm[None, :]
        C = X.T @ X
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1]
        w = vecs[:, order][:, 0]
        w = w / (np.linalg.norm(w) + 1e-12)
        p0 = Pm - (Pm @ w) * w
        return w, p0, np.ones(n, dtype=bool)
    # Re-estimate direction and p0 from inliers with PCA, then run LO-RANSAC with MAD threshold tightening.
    mask_lo = best_mask.copy()
    w = None
    p0 = None
    for _ in range(max(0, int(lo_iters)) + 1):
        inliers = pts[mask_lo]
        if inliers.shape[0] < 2:
            break
        Pm_in = inliers.mean(axis=0)
        X_in = inliers - Pm_in[None, :]
        C_in = X_in.T @ X_in
        vals, vecs = np.linalg.eigh(C_in)
        order = np.argsort(vals)[::-1]
        w = vecs[:, order][:, 0]
        w = w / (np.linalg.norm(w) + 1e-12)
        p0 = Pm_in - (Pm_in @ w) * w
        # Compute perpendicular distance to the current line and tighten the threshold with MAD.
        vec = pts - p0[None, :]
        proj = (vec @ w)[:, None] * w[None, :]
        perp = vec - proj
        d = np.linalg.norm(perp, axis=1)
        med = float(np.median(d))
        mad = float(1.4826 * np.median(np.abs(d - med)))
        thr = max(1e-6, float(mad_k) * (mad if np.isfinite(mad) and mad > 0 else dist_thresh))
        new_mask = d <= thr
        # Stop early if the mask is unchanged.
        if new_mask.sum() == mask_lo.sum():
            mask_lo = new_mask
            break
        mask_lo = new_mask
    if w is None or p0 is None:
        # Final fallback.
        Pm = pts.mean(axis=0)
        X = pts - Pm[None, :]
        C = X.T @ X
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1]
        w = vecs[:, order][:, 0]
        w = w / (np.linalg.norm(w) + 1e-12)
        p0 = Pm - (Pm @ w) * w
        return w, p0, best_mask
    return w, p0, mask_lo


# ------------------------- Largest Cluster Selection from Radius Graph -------------------------
def _largest_cluster_mask(
    points: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = 3,
    knn_k: int = 3,
) -> np.ndarray:
    """
    Select the largest connected component from a radius graph, similar to DBSCAN, and return a bool mask.
    If eps is None, estimate it adaptively from the median k-nearest-neighbor distance.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return np.ones(1, dtype=bool)
    # Distance matrix.
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.linalg.norm(diff, axis=2)
    # Adaptive eps.
    if eps is None:
        k = max(1, min(int(knn_k), n - 1))
        sorted_d = np.sort(D + np.eye(n) * 1e9, axis=1)
        kth = sorted_d[:, k - 1]
        med_k = float(np.median(kth))
        # Robust scaling factor.
        eps = max(1e-6, 1.5 * med_k)
    # Radius adjacency.
    A = (D <= float(eps))
    np.fill_diagonal(A, False)
    # Connected components in the undirected graph.
    visited = np.zeros(n, dtype=bool)
    best_comp = None
    for i in range(n):
        if visited[i]:
            continue
        # BFS
        queue = [i]
        visited[i] = True
        comp = [i]
        while queue:
            u = queue.pop(0)
            nbrs = np.where(A[u])[0]
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    comp.append(v)
        if best_comp is None or len(comp) > len(best_comp):
            best_comp = comp
    if best_comp is None or len(best_comp) < max(1, int(min_samples)):
        return np.ones(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(best_comp, dtype=int)] = True
    return mask


# ------------------------- Main Algorithm (Revolute / Screw) -------------------------

def _robust_span_from_angles(angles: np.ndarray, iqr_k: float = 1.5) -> float:
    """
    Robust angular span in radians after IQR outlier removal.
    Returns max-min over inliers; falls back to raw max-min when samples are too sparse or IQR is near zero.
    """
    if angles is None:
        return 0.0
    a = np.asarray(angles, dtype=np.float64)
    if a.size == 0:
        return 0.0
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    if a.size < 4:
        return float(a.max() - a.min())
    q1, q3 = np.percentile(a, [25, 75])
    iqr = float(q3 - q1)
    if iqr <= 1e-12:
        return float(a.max() - a.min())
    lo = q1 - iqr_k * iqr
    hi = q3 + iqr_k * iqr
    inliers = a[(a >= lo) & (a <= hi)]
    if inliers.size == 0:
        inliers = a
    return float(inliers.max() - inliers.min())

def estimate_prismatic_if_linear(
    T_list: List[np.ndarray],
) -> Tuple[str, np.ndarray, np.ndarray, List[float], np.ndarray]:
    """
    Estimate a prismatic model when rotation is small and the translation trajectory is nearly collinear.
    Uses RANSAC to robustly fit a line to t_list and improve outlier tolerance.

    Output:
      - joint_type: "prismatic"
      - joint_axis: (3,) unit translation axis from the main direction of t_list
      - joint_origin: (3,) fixed output [0,0,0]
      - joint_params: per-frame distance parameters d_i without zeroing
      - rotation_offset: 4x4 constant alignment containing R_ref and t_ref
    """
    # Normalize T and extract R and t.
    Ts = []
    for T in T_list:
        T = np.asarray(T, dtype=np.float64)
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float64); T4[:3, :4] = T
        elif T.shape == (4, 4):
            T4 = T.copy(); T4[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            raise AssertionError(f"Expected 3x4 or 4x4, got {T.shape}")
        T4[:3, :3] = _project_to_so3(T4[:3, :3])
        Ts.append(T4)
    Ts = np.stack(Ts, axis=0)
    N = Ts.shape[0]
    assert N >= 2, "Need at least two frames"
    R_list = Ts[:, :3, :3]
    t_list = Ts[:, :3, 3]

    # Select the largest spatial cluster first to remove points clearly outside the same cluster.
    cluster_mask = _largest_cluster_mask(t_list, eps=None, min_samples=max(3, int(0.05 * N)), knn_k=3)
    t_cluster = t_list[cluster_mask]
    R_cluster = R_list[cluster_mask]

    # Fit a robust line on the largest cluster with RANSAC to get w, p0, and cluster-relative inliers.
    w, p0, inlier_mask_cluster = _fit_line_ransac_3d(
        t_cluster, num_iters=800, dist_thresh=None, min_inliers_ratio=0.5,
        min_pair_sep_ratio=0.05, lo_iters=2, mad_k=2.5, random_state=0
    )
    # Convert the cluster inlier mask back to the full-frame mask.
    full_inliers = np.zeros(N, dtype=bool)
    idxs = np.where(cluster_mask)[0]
    if inlier_mask_cluster is not None and inlier_mask_cluster.shape[0] == idxs.shape[0]:
        full_inliers[idxs[inlier_mask_cluster]] = True
    else:
        full_inliers[cluster_mask] = True

    # Re-estimate the mean rotation from inliers for better robustness.
    if int(full_inliers.sum()) >= 2:
        S = np.sum(R_list[full_inliers], axis=0)
    else:
        S = np.sum(R_list, axis=0)
    U, _, Vt = np.linalg.svd(S)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt

    # Translation basis: use the closest point p0 on the fitted line directly.
    R_ref = R_mean

    # Distance parameter d_i: projection along w without zeroing.
    d_list = [float(w @ (t_list[i] - p0)) for i in range(N)]

    # Build rotation_offset with R_ref and t_ref=p0; absorb the constant term into t_ref.
    t_ref = p0
    T_ref = np.eye(4, dtype=np.float32)
    T_ref[:3, :3] = R_ref.astype(np.float32)
    T_ref[:3, 3] = t_ref.astype(np.float32)

    joint_type = "prismatic"
    joint_axis = w.astype(np.float32)
    joint_origin = np.zeros(3, dtype=np.float32)  # Output [0,0,0] by design.
    joint_params = [float(x) for x in d_list]
    rotation_offset = T_ref
    return joint_type, joint_axis, joint_origin, joint_params, rotation_offset


def decide_joint_axis_type(
    T_list: List[np.ndarray],
    rot_span_deg: float = 10.0,
    linearity_ratio_thresh: float = 0.05,
) -> Tuple[str, float, float]:
    """
    Only decide which axis/model to use: "revolute" or "prismatic".

    Criteria:
      - Rotation span in degrees is smaller than rot_span_deg.
      - Translation has a dominant principal direction: linearity (lambda2+lambda3)/lambda1 < linearity_ratio_thresh.

    Returns: (model_type, rot_span_deg_val, linearity_ratio)
    """
    if T_list is None or len(T_list) < 2:
        return "revolute", 0.0, 1.0

    # Normalize T and extract R and t.
    Ts = []
    for T in T_list:
        T = np.asarray(T, dtype=np.float64)
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float64); T4[:3, :4] = T
        elif T.shape == (4, 4):
            T4 = T.copy(); T4[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            continue
        T4[:3, :3] = _project_to_so3(T4[:3, :3])
        Ts.append(T4)
    if len(Ts) < 2:
        return "revolute", 0.0, 1.0
    Ts = np.stack(Ts, axis=0)
    R_list = Ts[:, :3, :3]
    t_list = Ts[:, :3, 3]

    # Rotation span relative to the mean rotation.
    S = np.sum(R_list, axis=0)
    U, _, Vt = np.linalg.svd(S)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    angles = []
    for i in range(R_list.shape[0]):
        Rij = _project_to_so3(R_list[i] @ R_mean.T)
        c = 0.5 * (np.trace(Rij) - 1.0)
        c = float(np.clip(c, -1.0, 1.0))
        angles.append(float(np.arccos(c)))
    angles = np.asarray(angles, dtype=np.float64)
    rot_span_val = float(np.rad2deg(_robust_span_from_angles(angles)))

    # Linearity from PCA.
    Pm = t_list.mean(axis=0)
    X = t_list - Pm[None, :]
    C = X.T @ X
    vals, _ = np.linalg.eigh(C)
    vals = vals[np.argsort(vals)[::-1]]
    linearity = float((vals[1] + vals[2]) / (vals[0] + 1e-12)) if float(vals[0]) > 0 else 1.0
    print(f"rot_span_val: {rot_span_val}, linearity: {linearity}")
    use_prismatic = (rot_span_val < float(rot_span_deg)) # or (linearity < float(linearity_ratio_thresh))
    return ("prismatic" if use_prismatic else "revolute"), rot_span_val, linearity




def estimate_revolute_from_poses(T_list: List[np.ndarray]) -> Tuple[str, np.ndarray, np.ndarray, List[float], np.ndarray]:
    """
    Input: a set of 3x4 or 4x4 homogeneous transforms, one T_i per frame.
    Output: (joint_type, axis, origin, joint_params, global_transform)
      - joint_type: "revolute"
      - axis: (3,) unit axis in world coordinates
      - origin: (3,) point on the axis, chosen as the closest point to the origin with u^T p = 0
      - joint_params: per-frame signed angles theta_i, with the first-frame angle fixed to 0
      - global_transform: 4x4 transform satisfying t = (I - R_ref) origin for R_ref and origin
    Notes:
      - Pairwise relative transforms remove any shared rigid transform, whether left- or right-multiplied.
      - The method estimates u, then theta_i, then p, without estimating screw pitch.
    """
    # Normalize input to (N, 4, 4).
    Ts = []
    for T in T_list:
        T = np.asarray(T, dtype=np.float64)
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float64); T4[:3, :4] = T
        elif T.shape == (4, 4):
            T4 = T.copy(); T4[3,:] = np.array([0,0,0,1], dtype=np.float64)
        else:
            raise AssertionError(f"Expected 3x4 or 4x4, got {T.shape}")
        # Project to SE(3).
        T4[:3,:3] = _project_to_so3(T4[:3,:3])
        Ts.append(T4)
    Ts = np.stack(Ts, axis=0)
    N = Ts.shape[0]
    assert N >= 2, "Need at least two frames to estimate fixed-axis rotation"

    R_list = Ts[:, :3, :3].copy()
    t_list = Ts[:, :3, 3].copy()

    # 1) Estimate the common axis u from pairwise relative rotations.
    u = _axis_from_Rs_pairwise(R_list)

    # 2) Get delta theta_ij from R_ij and solve theta_i with theta_0 fixed to 0.
    #    Build LS: for each (i<j), theta_i - theta_j = delta theta_ij.
    pairs = []
    rhs = []
    for i in range(N):
        for j in range(i+1, N):
            Rij = R_list[i] @ R_list[j].T
            Rij = _project_to_so3(Rij)
            dtheta = _angle_from_R_and_axis(Rij, u)
            pairs.append((i, j))
            rhs.append(dtheta)
    rhs = np.array(rhs, dtype=np.float64)

    # Variables theta_1..theta_{N-1}; theta_0 is fixed to 0.
    M = len(pairs)
    A = np.zeros((M, N-1), dtype=np.float64)
    b = rhs.copy()
    for k, (i, j) in enumerate(pairs):
        if i > 0:
            A[k, i-1] = 1.0
        # -θ_j
        if j > 0:
            A[k, j-1] -= 1.0
        # If i==0, the equation is -theta_j = delta theta_ij; if j==0, it is theta_i = delta theta_ij.
    # Least-squares solve.
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    theta = np.zeros(N, dtype=np.float64)
    theta[1:] = x
    theta = _unwrap_angles(theta, discont=np.pi/2)

    # 3) Estimate a point p on the axis for pure rotation without screw pitch:
    #    t_ij ~= P @ (I - R_ij) p aligned to P @ t_ij, solved in the plane perpendicular to u.
    I = np.eye(3, dtype=np.float64)
    # Solve p in the plane perpendicular to u to avoid singularity along u; do not estimate screw pitch.
    P = I - np.outer(u, u)
    A_p = []
    b_p = []
    for (i, j) in pairs:
        Ri = R_list[i]
        Rj = R_list[j]
        tij = t_list[i] - t_list[j]
        # Correct pairwise constraint: P @ (R_j - R_i) p = P @ (t_i - t_j).
        A_p.append(P @ (Rj - Ri))
        b_p.append(P @ tij)
    A_p = np.concatenate(A_p, axis=0)          # (3*M, 3)
    b_p = np.concatenate(b_p, axis=0)          # (3*M,)
    p, *_ = np.linalg.lstsq(A_p, b_p, rcond=None)
    # Force p into the plane perpendicular to u, giving the closest-point parameterization.
    p = p - (u @ p) * u

    # 4) Estimate the constant translation bias c and build the global transform for visualization/alignment.
    #    If observations satisfy t_i ~= c + (I - R_i) p, pairwise constraints remove c, but reconstruction needs it back.
    resid_c = []
    for i in range(N):
        resid_c.append(t_list[i] - ((I - R_list[i]) @ p))
    resid_c = np.stack(resid_c, axis=0)
    c_vec = np.median(resid_c, axis=0)

    # Align R_ref with theta: R_ref = Proj_SO(3)(sum R_u(-theta_i) R_i).
    I3 = np.eye(3, dtype=np.float64)
    ux, uy, uz = u
    K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]], dtype=np.float64)
    M = np.zeros((3, 3), dtype=np.float64)
    for i in range(N):
        th = -float(theta[i])
        c = float(np.cos(th)); s = float(np.sin(th))
        R_u_neg = (c * I3 + s * K + (1.0 - c) * (u[:, None] @ u[None, :]))
        M += R_u_neg @ R_list[i]
    U, _, Vt = np.linalg.svd(M)
    R_ref = U @ Vt
    if np.linalg.det(R_ref) < 0:
        U[:, -1] *= -1
        R_ref = U @ Vt
    t_ref = c_vec + (np.eye(3) - R_ref) @ p
    T_ref = np.eye(4, dtype=np.float32)
    T_ref[:3, :3] = R_ref.astype(np.float32)
    T_ref[:3, 3] = t_ref.astype(np.float32)

    # Output format: origin is the combined pivot o = p + c_perp; keep the axial component of c in rotation_offset.
    c_par = (float(u @ c_vec)) * u
    c_perp = c_vec - c_par
    joint_type = "revolute"
    joint_axis = u.astype(np.float32)
    joint_origin = (p + c_perp).astype(np.float32)
    # Recompute per-frame angles directly from the translation equation:
    # in the plane perpendicular to u, b_i = P (p - (t_i - c)) ~= R(theta_i) a.
    # Here a = P (R_ref p), theta_i = atan2(u dot (a cross b_i), a dot b_i).
    if False:
        joint_params = [float(x) for x in theta]
    else:
        P_ang = I3 - np.outer(u, u)
        a_vec = P_ang @ (R_ref @ p)
        theta_from_t = np.zeros(N, dtype=np.float64)
        for i in range(N):
            b_vec = P_ang @ (p - (t_list[i] - c_vec))
            cross_ab = np.cross(a_vec, b_vec)
            sin_val = float(u @ cross_ab)
            cos_val = float(a_vec @ b_vec)
            theta_from_t[i] = float(np.arctan2(sin_val, cos_val))
        theta_from_t = _unwrap_angles(theta_from_t, discont=np.pi/2)
        joint_params = [float(x) for x in theta_from_t]
    rotation_offset = T_ref                    # Keep the existing name.

    return joint_type, joint_axis, joint_origin, joint_params, rotation_offset


# ------------------------- I/O and Visualization -------------------------

def _read_object_poses(json_path: Path) -> List[np.ndarray]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames: List[np.ndarray] = []
    for item in data:
        T = np.asarray(item["T_obj"], dtype=np.float64)
        if T.shape == (3, 4):
            frames.append(T)
        elif T.shape == (4, 4):
            frames.append(T[:3, :])
        else:
            raise RuntimeError("T_obj must be 3x4 or 4x4")
    return frames


def _visualize_joint_fit(
    T_obj_list: List[np.ndarray],
    joint_type: str,
    joint_axis: np.ndarray,
    joint_origin: np.ndarray,
    joint_params: List[float],
    rotation_offset: np.ndarray,
    save_path: Path,
) -> None:
    # Organize original and fitted points.
    Ts = []
    for T in T_obj_list:
        T = np.asarray(T, dtype=np.float32)
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float32); T4[:3, :4] = T
        else:
            T4 = T.copy(); T4[3,:] = np.array([0,0,0,1], dtype=np.float32)
        Ts.append(T4)
    Ts = np.stack(Ts, axis=0)
    R_list = Ts[:, :3, :3]
    t_list = Ts[:, :3, 3]
    I3 = np.eye(3, dtype=np.float32)

    # Reconstruct fitted poses from joint parameters and rotation_offset.
    R_ref = rotation_offset[:3, :3].astype(np.float32)
    u = joint_axis.astype(np.float32)
    p = joint_origin.astype(np.float32)  # Here p is the combined pivot o.

    def axis_angle_to_R(u: np.ndarray, th: float) -> np.ndarray:
        # Rodrigues
        ux, uy, uz = u
        K = np.array([[0,-uz,uy],[uz,0,-ux],[-uy,ux,0]], dtype=np.float32)
        c = float(np.cos(th)); s = float(np.sin(th))
        return (c * I3 + s * K + (1-c) * (u[:,None] @ u[None,:])).astype(np.float32)

    R_fit = []
    t_fit = []
    if str(joint_type).lower() == "revolute":
        # Revolute: T_fit = [R(θ), (I - R(θ)) p] @ rotation_offset
        o = p
        for th in joint_params:
            R_theta = axis_angle_to_R(u, float(th))
            T_joint = np.eye(4, dtype=np.float32)
            T_joint[:3, :3] = R_theta
            T_joint[:3, 3] = ((I3 - R_theta) @ o).astype(np.float32)
            T_i = T_joint @ rotation_offset
            R_fit.append(T_i[:3, :3])
            t_fit.append(T_i[:3, 3])
    else:
        # Prismatic: T_fit = [I, d u] @ rotation_offset
        for d in joint_params:
            T_joint = np.eye(4, dtype=np.float32)
            T_joint[:3, 3] = (u * float(d)).astype(np.float32)
            T_i = T_joint @ rotation_offset
            R_fit.append(T_i[:3, :3])
            t_fit.append(T_i[:3, 3])
    R_fit = np.stack(R_fit, axis=0); t_fit = np.stack(t_fit, axis=0)

    # Plot.
    def set_axes_equal(ax):
        xs = np.concatenate([t_list[:,0], t_fit[:,0]])
        ys = np.concatenate([t_list[:,1], t_fit[:,1]])
        zs = np.concatenate([t_list[:,2], t_fit[:,2]])
        xr, yr, zr = xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()
        mr = max(float(xr), float(yr), float(zr), 1e-6)
        xm, ym, zm = (xs.max()+xs.min())*0.5, (ys.max()+ys.min())*0.5, (zs.max()+zs.min())*0.5
        ax.set_xlim(xm-0.6*mr, xm+0.6*mr)
        ax.set_ylim(ym-0.6*mr, ym+0.6*mr)
        ax.set_zlim(zm-0.6*mr, zm+0.6*mr)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1,1,1, projection="3d")
    ax.scatter(t_list[:,0], t_list[:,1], t_list[:,2], s=12, label="orig T_obj")
    ax.scatter(t_fit[:,0], t_fit[:,1], t_fit[:,2], s=12, label="fitted T")

    # Mark the first frame for both original and fitted trajectories.
    if t_list.shape[0] > 0:
        p_orig0 = t_list[0]
        p_fit0 = t_fit[0]
        ax.scatter([p_orig0[0]],[p_orig0[1]],[p_orig0[2]], c='tab:red', s=60, marker='*', label='frame 0 (orig)', zorder=5)
        ax.scatter([p_fit0[0]],[p_fit0[1]],[p_fit0[2]], c='tab:orange', s=60, marker='*', label='frame 0 (fit)', zorder=5)
        # Text labels with a small offset to avoid occlusion.
        off = float(max(np.linalg.norm(t_list - t_list.mean(0), axis=1).mean(), 1e-3)) * 0.03
        ax.text(p_orig0[0]+off, p_orig0[1]+off, p_orig0[2]+off, '0', color='tab:red')
        ax.text(p_fit0[0]+off, p_fit0[1]+off, p_fit0[2]+off, '0', color='tab:orange')

    # Draw sampled per-frame coordinate axes.
    N = t_list.shape[0]
    stride = max(1, N // 20)
    scale_axes = max(float(np.linalg.norm(t_list - t_list.mean(0), axis=1).mean()), 1e-3) * 0.2
    colors = ['r', 'g', 'b']
    for i in range(0, N, stride):
        # True frame coordinate axes.
        R_true = R_list[i]
        o_true = t_list[i]
        for k in range(3):
            v = R_true[:, k]
            p0 = o_true
            p1 = o_true + v * scale_axes
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=colors[k], alpha=0.6)
        # Fitted frame coordinate axes.
        Rf = R_fit[i]
        of = t_fit[i]
        for k in range(3):
            v = Rf[:, k]
            p0 = of
            p1 = of + v * scale_axes
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=colors[k], linestyle='--', alpha=0.8)

    # Draw the joint axis.
    scale = max(float(np.linalg.norm(t_list - t_list.mean(0), axis=1).mean()), 1e-3) * 1.5
    p0 = p - u * scale
    p1 = p + u * scale
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k-', lw=2, label='joint axis')
    ax.scatter([p[0]],[p[1]],[p[2]], c='k', s=30, marker='x', label='joint origin')
    # Positive direction arrow along +u.
    arrow_len = 0.6 * scale
    ax.quiver(p[0], p[1], p[2], u[0], u[1], u[2], length=arrow_len, color='k', arrow_length_ratio=0.2, linewidth=1.5)

    title = "Revolute" if str(joint_type).lower() == "revolute" else "Prismatic"
    ax.set_title(f"{title} joint fit: true vs fitted")
    ax.legend(loc="best")
    set_axes_equal(ax)
    plt.tight_layout()
    try:
        fig.savefig(str(save_path))
        backend = str(plt.get_backend()).lower()
        if backend not in ("agg", "cairo", "pdf", "ps", "svg"):
            try:
                plt.show(block=True)
            except Exception:
                pass
    finally:
        plt.close(fig)


# Read the 3DGS point cloud with positions and approximate colors.
def _read_ply_gaussians_rgb(ply_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read positions and approximate colors from 3DGS-compatible PLY.

    Prefers SH DC (f_dc_0..2). Falls back to RGB if present, else None.
    Returns positions (N,3) float32 and colors (N,3) float32 in [0,1] or None.
    """
    import plyfile  # lazy import

    C0 = 0.28209479177387814  # SH DC constant

    ply = plyfile.PlyData.read(str(ply_path))
    v = ply.elements[0].data

    def get_col(name: str) -> Optional[np.ndarray]:
        if name in v.dtype.names:
            return np.asarray(v[name])
        return None

    x = get_col("x"); y = get_col("y"); z = get_col("z")
    if x is None or y is None or z is None:
        raise RuntimeError("PLY missing x/y/z")
    positions = np.stack([x, y, z], axis=-1).astype(np.float32)

    f0 = get_col("f_dc_0"); f1 = get_col("f_dc_1"); f2 = get_col("f_dc_2")
    if f0 is not None and f1 is not None and f2 is not None:
        sh_dc = np.stack([f0, f1, f2], axis=-1).astype(np.float32)
        colors = np.clip(sh_dc * C0 + 0.5, 0.0, 1.0)
        return positions, colors

    r = get_col("red") or get_col("r")
    g = get_col("green") or get_col("g")
    b = get_col("blue") or get_col("b")
    if r is not None and g is not None and b is not None:
        colors = (np.stack([r, g, b], axis=-1).astype(np.float32) / 255.0)
        return positions, colors

    return positions, None


def _visualize_axis_on_gaussians(
    blend_dir: Path,
    joint_axis: np.ndarray,
    joint_origin: np.ndarray,
    save_path: Path,
    max_points: int = 5000,
) -> None:
    ply_path = blend_dir / "object_3dgs.ply"
    if not ply_path.exists():
        return
    pts, cols = _read_ply_gaussians_rgb(ply_path)
    n = pts.shape[0]
    if n == 0:
        return
    # Subsample if needed
    if n > max_points:
        idx = np.random.RandomState(0).choice(n, size=max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx] if cols is not None else None

    # Prepare axis segment across point cloud
    bbox = np.stack([pts.min(0), pts.max(0)], axis=0)
    diag = float(np.linalg.norm(bbox[1] - bbox[0]))
    scale = 0.75 * (diag if diag > 1e-6 else 1.0)
    axis_vec = np.asarray(joint_axis, dtype=np.float32)
    axis_norm = float(np.linalg.norm(axis_vec))
    axis = (axis_vec / axis_norm).astype(np.float32) if axis_norm > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    origin = np.asarray(joint_origin, dtype=np.float32)
    # Shift the visualization anchor along the joint axis toward the projected centroid for better overlap
    proj_vals = (pts - origin) @ axis
    shift = float(np.mean(proj_vals)) if proj_vals.size > 0 else 0.0
    origin_vis = origin + axis * shift
    p0 = origin_vis - axis * scale
    p1 = origin_vis + axis * scale

    # Plot
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # Light point cloud for speed
    if cols is not None:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.2, depthshade=False)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='lightgray', s=0.2, depthshade=False)
    # Axis and origin
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'r-', lw=2.0, label='joint axis')
    ax.scatter([origin_vis[0]], [origin_vis[1]], [origin_vis[2]], c='k', s=30, marker='x', label='joint origin (shifted)')
    # Positive direction arrow
    arrow_len = 0.6 * scale
    ax.quiver(origin_vis[0], origin_vis[1], origin_vis[2], axis[0], axis[1], axis[2], length=arrow_len, color='r', arrow_length_ratio=0.2, linewidth=1.5)

    # View volume
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    x_mid = (xs.max() + xs.min()) * 0.5
    y_mid = (ys.max() + ys.min()) * 0.5
    z_mid = (zs.max() + zs.min()) * 0.5
    max_range = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), float(zs.max() - zs.min()), 1e-6)
    ax.set_xlim(x_mid - 0.6 * max_range, x_mid + 0.6 * max_range)
    ax.set_ylim(y_mid - 0.6 * max_range, y_mid + 0.6 * max_range)
    ax.set_zlim(z_mid - 0.6 * max_range, z_mid + 0.6 * max_range)
    ax.set_title("Joint axis over Gaussian points")
    ax.legend(loc='best')
    plt.tight_layout()
    try:
        fig.savefig(str(save_path))
        backend = str(plt.get_backend()).lower()
        if backend not in ("agg", "cairo", "pdf", "ps", "svg"):
            try:
                plt.show(block=True)
            except Exception:
                pass
    finally:
        plt.close(fig)

# ------------------------- Directory Processing and Main Program -------------------------

def reestimate_params_from_refined_joint(
    T_list: List[np.ndarray],
    joint_type: str,
    axis: np.ndarray,
    origin: np.ndarray,
) -> Tuple[List[float], np.ndarray]:
    """
    Given an adjusted axis and origin, re-estimate joint_params (angles or distances)
    and the global reference transform (rotation_offset).
    """
    Ts = []
    for T in T_list:
        T4 = np.eye(4, dtype=np.float64)
        T4[:3, :4] = T[:3, :4]
        T4[:3, :3] = _project_to_so3(T4[:3, :3])
        Ts.append(T4)
    Ts = np.stack(Ts, axis=0)
    N = Ts.shape[0]
    u = axis.astype(np.float64)
    # Ensure u is normalized
    unorm = np.linalg.norm(u)
    u = u / unorm if unorm > 1e-12 else np.array([1.0, 0.0, 0.0])
    p = origin.astype(np.float64)

    if str(joint_type).lower() == "revolute":
        # 1) Re-estimate angles theta_i
        # Reference is the first frame: theta_0 = 0
        T0 = Ts[0]
        R0 = T0[:3, :3]
        thetas = [0.0]
        for i in range(1, N):
            Ri = Ts[i][:3, :3]
            R_rel = Ri @ R0.T
            R_rel = _project_to_so3(R_rel)
            theta = _angle_from_R_and_axis(R_rel, u)
            thetas.append(theta)
        thetas = _unwrap_angles(np.array(thetas), discont=np.pi/2)

        # 2) Re-estimate rotation_offset (T_offset)
        T_offs = []
        for i in range(N):
            th = -float(thetas[i])  # inverse rotation
            R_inv = rodrigues_rotation_matrix(u, th).astype(np.float64)
            # M(th) = [R_th, (I - R_th)p] => M(th)^-1 = [R_th^T, -R_th^T (I-R_th)p]
            # but simpler: rotate back from origin
            t_inv = p - R_inv @ p
            M_inv = np.eye(4, dtype=np.float64)
            M_inv[:3, :3] = R_inv
            M_inv[:3, 3] = t_inv
            T_offs.append(M_inv @ Ts[i])
        
        # Average the offsets
        R_avg_sum = np.sum([T[:3, :3] for T in T_offs], axis=0)
        U, _, Vt = np.linalg.svd(R_avg_sum)
        R_ref = U @ Vt
        if np.linalg.det(R_ref) < 0:
            U[:, -1] *= -1
            R_ref = U @ Vt
        t_ref = np.mean([T[:3, 3] for T in T_offs], axis=0)
        
        T_ref = np.eye(4, dtype=np.float32)
        T_ref[:3, :3] = R_ref.astype(np.float32)
        T_ref[:3, 3] = t_ref.astype(np.float32)
        return [float(x) for x in thetas], T_ref

    else: # prismatic
        # 1) Re-estimate distances d_i
        T0 = Ts[0]
        t0 = T0[:3, 3]
        ds = [0.0]
        for i in range(1, N):
            ti = Ts[i][:3, 3]
            ds.append(float(np.dot(ti - t0, u)))
        
        # 2) Re-estimate rotation_offset
        T_offs = []
        for i in range(N):
            T_oi = Ts[i].copy()
            T_oi[:3, 3] -= ds[i] * u
            T_offs.append(T_oi)
            
        R_avg_sum = np.sum([T[:3, :3] for T in T_offs], axis=0)
        U, _, Vt = np.linalg.svd(R_avg_sum)
        R_ref = U @ Vt
        if np.linalg.det(R_ref) < 0:
            U[:, -1] *= -1
            R_ref = U @ Vt
        t_ref = np.mean([T[:3, 3] for T in T_offs], axis=0)
        
        T_ref = np.eye(4, dtype=np.float32)
        T_ref[:3, :3] = R_ref.astype(np.float32)
        T_ref[:3, 3] = t_ref.astype(np.float32)
        return [float(x) for x in ds], T_ref


def process_twopart_blend_dir(
    blend_dir: Path,
    use_eval: bool = False,
    visualize: bool = True,
    force_prismatic: bool = False,
    force_revolute: bool = False,
    interactive: bool = False,
) -> Path:
    poses_file = blend_dir / ("object_poses_eval.json" if use_eval else "object_poses_train.json")
    if not poses_file.exists():
        raise FileNotFoundError(f"Missing object poses file: {poses_file}")
    T_list = _read_object_poses(poses_file)

    # Automatically select revolute or prismatic and estimate the joint.
    if force_prismatic:
        model = "prismatic"
    elif force_revolute:
        model = "revolute"
    else:
        model, rot_span_val, linearity = decide_joint_axis_type(
            T_list, rot_span_deg=10, linearity_ratio_thresh=0.05
        )
        print(f"Estimated joint type: {model}")
        print(f"Rot span: {rot_span_val}")
        print(f"Linearity: {linearity}")
    if model == "prismatic":
        joint_type, joint_axis, joint_origin, joint_params, rotation_offset = estimate_prismatic_if_linear(T_list)
    else:
        joint_type, joint_axis, joint_origin, joint_params, rotation_offset = estimate_revolute_from_poses(T_list)

    if interactive and o3d is not None:
        ply_path = blend_dir / "object_3dgs.ply"
        if ply_path.exists():
            print(f"[info] Loading point cloud for refinement: {ply_path}")
            pts, cols = _read_ply_gaussians_rgb(ply_path)
            # Use small subset for faster visualization if huge
            if pts.shape[0] > 50000:
                idx = np.random.choice(pts.shape[0], 50000, replace=False)
                pts = pts[idx]
                if cols is not None:
                    cols = cols[idx]
            
            viewer = JointRefinementViewer(pts, cols, joint_axis, joint_origin, str(joint_type))
            refined_axis, refined_origin = viewer.run()
            
            # Use refined axis/origin to recalculate params
            joint_axis = refined_axis
            joint_origin = refined_origin
            print(f"[info] Refining parameters with adjusted axis: {joint_axis}, origin: {joint_origin}")
            joint_params, rotation_offset = reestimate_params_from_refined_joint(
                T_list, str(joint_type), joint_axis, joint_origin
            )

    out = {
        "joint_type": str(joint_type),
        "axis": [float(x) for x in np.asarray(joint_axis).reshape(-1)],
        "origin": [float(x) for x in np.asarray(joint_origin).reshape(-1)],
        "global_transform": [[float(v) for v in row] for row in np.asarray(rotation_offset, dtype=np.float32)],
        "joint_params": [float(p) for p in joint_params],
        "num_frames": int(len(T_list)),
    }
    out_path = blend_dir / "estimated_joint.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if visualize:
        save_png = blend_dir / "joint_fit.png"
        _visualize_joint_fit(
            T_list,
            str(joint_type),
            np.asarray(joint_axis, dtype=np.float32),
            np.asarray(joint_origin, dtype=np.float32),
            joint_params,
            np.asarray(rotation_offset, dtype=np.float32),
            save_png,
        )
        # Overlay the axis on the 3DGS point cloud.
        save_axis_png = blend_dir / "axis_on_gaussians.png"
        _visualize_axis_on_gaussians(
            blend_dir,
            np.asarray(joint_axis, dtype=np.float32),
            np.asarray(joint_origin, dtype=np.float32),
            save_axis_png,
        )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate fixed-axis (revolute) joint from twopart_blend outputs (NumPy-only)")
    ap.add_argument("--config", type=str, default=None, help="Path to pipeline config YAML")
    ap.add_argument("--outputs-root", type=Path, default=Path("outputs"), help="Root outputs directory")
    ap.add_argument("--object_name", "-on", type=str, default=None, help="Specific object name under outputs to process")
    ap.add_argument("--use-eval", action="store_true", help="Use object_poses_eval.json instead of train")
    ap.add_argument("--no-vis", action="store_true", help="Disable saving joint_fit.png visualization")
    ap.add_argument("--force-prismatic", action="store_true", help="Force prismatic joint even if revolute is detected")
    ap.add_argument("--force-revolute", action="store_true", help="Force revolute joint even if prismatic is detected")
    ap.add_argument("--interactive", "-i", action="store_true", help="Interactively refine the joint axis/origin")
    args = ap.parse_args()

    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        if args.object_name is None:
            args.object_name = runtime_paths["object_name"]
        if args.outputs_root == Path("outputs"):
            args.outputs_root = Path(runtime_paths["output_root"])
    if args.object_name is None:
        ap.error("--object_name is required when --config is not provided")

    outputs_root: Path = args.outputs_root
    assert outputs_root.exists(), f"Outputs root not found: {outputs_root}"

    # Only process the specified object, matching the original script style.
    blend_dir = outputs_root / args.object_name / "twopart_blend"
    if not blend_dir.exists():
        raise FileNotFoundError(f"twopart_blend not found for object: {args.object_name}")

    out_path = process_twopart_blend_dir(
        blend_dir,
        use_eval=args.use_eval,
        visualize=not args.no_vis,
        force_prismatic=args.force_prismatic,
        force_revolute=args.force_revolute,
        interactive=args.interactive,
    )
    print(f"Saved joint estimation to {out_path}")


if __name__ == "__main__":
    main()
