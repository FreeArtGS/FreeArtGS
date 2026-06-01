import os
import sys
from pathlib import Path
import torch    
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from estimotion_2part.utils.vis import visualize_part_initialization
from estimotion_2part.models.trans_models import (
    Trans,
    Optimizer,
    TransformLossComputer,
)
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm


def _stack_points_to_tensor(point_list, device):
    if len(point_list) == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    tensors = []
    for point in point_list:
        tensors.append(point.to(device=device, dtype=torch.float32))
    return torch.stack(tensors, dim=0)
def find_rigid_transform_t(P, Q, weights=None):
    assert P.shape == Q.shape, "Point clouds must have the same shape"
    if weights is None:
        centroid_P = P.mean(dim=0)
        centroid_Q = Q.mean(dim=0)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        H = P_centered.transpose(0, 1) @ Q_centered
    else:
        w = weights / torch.clamp(weights.sum(), min=1e-12)
        centroid_P = (P * w[:, None]).sum(dim=0)
        centroid_Q = (Q * w[:, None]).sum(dim=0)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        H = (P_centered * w[:, None]).transpose(0, 1) @ Q_centered

    U, _, Vh = torch.linalg.svd(H, full_matrices=False)
    R = Vh.transpose(0, 1) @ U.transpose(0, 1)
    if torch.linalg.det(R) < 0:
        Vh = Vh.clone()
        Vh[-1, :] *= -1
        R = Vh.transpose(0, 1) @ U.transpose(0, 1)
    t = centroid_Q - R @ centroid_P
    return R, t


def find_rigid_transform_batch_t(P, Q):
    """
    Batched rigid transform for unweighted correspondences.
    P, Q: [B, N, 3]
    Returns:
        R: [B, 3, 3]
        t: [B, 3]
    """
    centroid_P = P.mean(dim=1, keepdim=True)
    centroid_Q = Q.mean(dim=1, keepdim=True)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = P_centered.transpose(1, 2) @ Q_centered  # [B, 3, 3]

    U, _, Vh = torch.linalg.svd(H, full_matrices=False)
    R = Vh.transpose(1, 2) @ U.transpose(1, 2)
    det_neg = torch.linalg.det(R) < 0
    if bool(det_neg.any()):
        Vh_adj = Vh.clone()
        Vh_adj[det_neg, -1, :] *= -1
        R = Vh_adj.transpose(1, 2) @ U.transpose(1, 2)
    t = centroid_Q.squeeze(1) - (R @ centroid_P.squeeze(1).unsqueeze(-1)).squeeze(-1)
    return R, t

def save_points_as_ply(points, colors, save_path):
    """
    Save 3D points with colors to a PLY file.
    
    Args:
        points: numpy array of shape (N, 3) containing 3D coordinates
        colors: numpy array of shape (N, 3) containing RGB colors in range [0, 255]
        save_path: path to save the PLY file
    """
    assert points.shape[0] == colors.shape[0], "Points and colors must have the same number of entries"
    assert points.shape[1] == 3, "Points must be 3D coordinates"
    assert colors.shape[1] == 3, "Colors must be RGB (3 channels)"
    
    with open(save_path, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write vertex data
        for i in range(points.shape[0]):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                   f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")


def init_part_trans(trajs, point_seg_weights, save_dir=None, max_iter=5):
    """
    Use iterative EM-style optimization to robustly estimate T1_init and T2_init.
    """
    device = trajs.device
    trajs = trajs.to(device=device, dtype=torch.float32)
    point_seg_weights = point_seg_weights.to(device=device, dtype=torch.float32).flatten()
    num_points = int(trajs.shape[0])
    n_sample = min(500, num_points)
    def _fit_rigid(P, Q, weights=None):
        R, t = find_rigid_transform_t(P, Q, weights=weights)
        return R, t

    def apply_T(T, P_xyz):
        return (P_xyz @ T[:3, :3].transpose(0, 1)) + T[:3, 3].unsqueeze(0)

    def sample_indices(indices, n_pick, seed):
        if int(indices.numel()) <= n_pick:
            return indices
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        order = torch.randperm(indices.numel(), generator=rng, device=device)
        return indices[order[:n_pick]]

    low_weight_idx = torch.where(point_seg_weights < 0.4)[0]
    high_weight_idx = torch.where(point_seg_weights > 0.6)[0]
    sampled_low_idx = sample_indices(low_weight_idx, n_sample, seed=42)
    sampled_high_idx = sample_indices(high_weight_idx, n_sample, seed=42)

    def ransac_rigid(P, Q, max_trials=200, tau=None, rng_seed=42):
        P = P.to(device=device, dtype=torch.float32)
        Q = Q.to(device=device, dtype=torch.float32)
        n = int(P.shape[0])
        if n < 3:
            R, t = _fit_rigid(P, Q)
            inliers = torch.ones(n, dtype=torch.bool, device=device)
            return R, t, inliers

        if tau is None:
            delta = Q - P
            delta_sq = (delta * delta).sum(dim=1)
            med_disp_sq = torch.clamp(torch.median(delta_sq), min=1e-18)
            tau_sq = torch.maximum(
                1e-4 * med_disp_sq,
                torch.tensor(1e-8, dtype=torch.float32, device=device),
            )
        else:
            tau_tensor = torch.as_tensor(tau, dtype=torch.float32, device=device)
            tau_sq = torch.clamp(tau_tensor * tau_tensor, min=1e-12)
        tau_sq_value = max(float(tau_sq), 1e-12)

        rng = torch.Generator(device=device)
        rng.manual_seed(rng_seed)
        best_R, best_t, best_inliers = None, None, None
        best_inl_count = 0
        best_mean_res_sq = 1e18

        triplets = torch.randint(0, n, (max_trials, 3), generator=rng, device=device)
        invalid = (
            (triplets[:, 0] == triplets[:, 1])
            | (triplets[:, 0] == triplets[:, 2])
            | (triplets[:, 1] == triplets[:, 2])
        )
        while bool(invalid.any()):
            resample_n = int(invalid.sum())
            triplets[invalid] = torch.randint(0, n, (resample_n, 3), generator=rng, device=device)
            invalid = (
                (triplets[:, 0] == triplets[:, 1])
                | (triplets[:, 0] == triplets[:, 2])
                | (triplets[:, 1] == triplets[:, 2])
            )

        chunk_size = 128
        for start in range(0, max_trials, chunk_size):
            end = min(start + chunk_size, max_trials)
            idx_chunk = triplets[start:end]
            P_chunk = P[idx_chunk]
            Q_chunk = Q[idx_chunk]
            R_chunk, t_chunk = find_rigid_transform_batch_t(P_chunk, Q_chunk)
            pred_chunk = torch.einsum("bij,nj->bni", R_chunk, P) + t_chunk.unsqueeze(1)
            residuals_chunk_sq = ((pred_chunk - Q.unsqueeze(0)) ** 2).sum(dim=2)
            inliers_chunk = residuals_chunk_sq < tau_sq
            ninl_chunk = inliers_chunk.sum(dim=1)

            for j in range(end - start):
                ninl = int(ninl_chunk[j])
                if ninl < 3:
                    continue
                R_lo, t_lo = R_chunk[j], t_chunk[j]
                residuals_all_sq = residuals_chunk_sq[j]
                inliers = inliers_chunk[j]

                for _ in range(3):
                    Pin = P[inliers]
                    Qin = Q[inliers]
                    e_sq = (((Pin @ R_lo.transpose(0, 1)) + t_lo.unsqueeze(0) - Qin) ** 2).sum(dim=1)
                    u_sq = e_sq / tau_sq_value
                    w = (1.0 - torch.clamp(u_sq, 0.0, 1.0)) ** 2
                    if float(w.sum()) < 3.0:
                        break
                    R_lo, t_lo = _fit_rigid(Pin, Qin, weights=w)
                    residuals_all_sq = (((P @ R_lo.transpose(0, 1)) + t_lo.unsqueeze(0) - Q) ** 2).sum(dim=1)
                    inliers = residuals_all_sq < tau_sq
                    ninl = int(inliers.sum())

                mean_res_sq = float(residuals_all_sq[inliers].mean()) if ninl > 0 else 1e18
                if (ninl > best_inl_count) or (ninl == best_inl_count and mean_res_sq < best_mean_res_sq):
                    best_inl_count = ninl
                    best_mean_res_sq = mean_res_sq
                    best_R, best_t, best_inliers = R_lo, t_lo, inliers.clone()
                    if best_inl_count > 0.85 * n:
                        break

            if best_inl_count > 0.85 * n:
                break

        if best_R is None:
            R, t = _fit_rigid(P, Q)
            return R, t, torch.ones(n, dtype=torch.bool, device=device)
        return best_R, best_t, best_inliers

    if sampled_low_idx is None or int(sampled_low_idx.numel()) < 3:
        sampled_low_idx = torch.argsort(point_seg_weights)[:100]
    if sampled_high_idx is None or int(sampled_high_idx.numel()) < 3:
        sampled_high_idx = torch.argsort(point_seg_weights)[-100:]

    P1 = trajs[sampled_low_idx, 0, :3]
    Q1 = trajs[sampled_low_idx, 1, :3]
    R1, t1, _ = ransac_rigid(P1, Q1)
    T1_init = torch.eye(4, dtype=torch.float32, device=device)
    T1_init[:3, :3] = R1
    T1_init[:3, 3] = t1

    P2 = trajs[sampled_high_idx, 0, :3]
    Q2 = trajs[sampled_high_idx, 1, :3]
    R2, t2, _ = ransac_rigid(P2, Q2)
    T2_init = torch.eye(4, dtype=torch.float32, device=device)
    T2_init[:3, :3] = R2
    T2_init[:3, 3] = t2

    P_all = trajs[:, 0, :3]
    Q_all = trajs[:, 1, :3]
    T1_current = T1_init.clone()
    T2_current = T2_init.clone()

    for iteration in range(max_iter):
        Q1_pred = apply_T(T1_current, P_all)
        Q2_pred = apply_T(T2_current, P_all)
        error1_sq = ((Q1_pred - Q_all) ** 2).sum(dim=1)
        error2_sq = ((Q2_pred - Q_all) ** 2).sum(dim=1)
        assign_to_T1 = error1_sq < error2_sq
        assign_to_T2 = ~assign_to_T1

        n_T1 = int(assign_to_T1.sum())
        n_T2 = int(assign_to_T2.sum())
        if n_T1 < 3 or n_T2 < 3:
            print("  Warning: One part has too few points. Stopping iteration.")
            break

        idx_T1 = torch.where(assign_to_T1)[0]
        idx_T2 = torch.where(assign_to_T2)[0]
        if n_T1 > n_sample:
            idx_T1 = sample_indices(idx_T1, n_sample, seed=42 + iteration)
        if n_T2 > n_sample:
            idx_T2 = sample_indices(idx_T2, n_sample, seed=42 + iteration)

        R1_new, t1_new, _ = ransac_rigid(P_all[idx_T1], Q_all[idx_T1])
        T1_new = torch.eye(4, dtype=torch.float32, device=device)
        T1_new[:3, :3] = R1_new
        T1_new[:3, 3] = t1_new

        R2_new, t2_new, _ = ransac_rigid(P_all[idx_T2], Q_all[idx_T2])
        T2_new = torch.eye(4, dtype=torch.float32, device=device)
        T2_new[:3, :3] = R2_new
        T2_new[:3, 3] = t2_new

        T1_diff_sq = ((T1_new - T1_current) ** 2).sum()
        T2_diff_sq = ((T2_new - T2_current) ** 2).sum()
        T1_current = T1_new
        T2_current = T2_new
        if float(T1_diff_sq) < 1e-8 and float(T2_diff_sq) < 1e-8:
            print(f"  Converged at iteration {iteration+1}")
            break

    return T1_current, T2_current

def reinit_part_trans(transform_net, trajs, save_dir=None, epoch=None, factor=1.0):
        """
        use init_part_trans to update T1 and T2 in transform_net based on current weights
        """
        with torch.no_grad():
            current_weights = torch.sigmoid(transform_net.weight).flatten()
            
            # use init_part_trans to reinitialize T1 and T2
            reinit_save_dir = None
            # if save_dir:
            #     reinit_save_dir = os.path.join(save_dir, f'reinit_epoch_{epoch}')

            T1_reinit, T2_reinit = init_part_trans(trajs, current_weights, reinit_save_dir)

            device = next(transform_net.parameters()).device

            # update transform_net T1 and T2
            transform_net.T1[0].data = T1_reinit.to(device=device, dtype=torch.float32)
            transform_net.T2[0].data = T2_reinit.to(device=device, dtype=torch.float32)
            
            return T1_reinit, T2_reinit
        
def joint_reinit_part_trans(transform_net, all_trajectories_for_reinit, save_dir=None):
        """
        Reinitialize the transform matrix for each frame pair during joint optimization.
        """
        with torch.no_grad():
            current_weights = torch.sigmoid(transform_net.weight).flatten()
            
            for frame_pair_id, traj_data_list in all_trajectories_for_reinit.items():
                # Build trajectory arrays and weights for this frame pair.
                trajectories_for_pair = []
                weights_for_pair = []
                
                for traj_data in traj_data_list:
                    point_idx = traj_data['point_idx']
                    traj = traj_data['traj']
                    trajectories_for_pair.append(traj)
                    weights_for_pair.append(current_weights[point_idx])
                
                trajectories_for_pair = torch.stack(
                    [traj.to(dtype=torch.float32) for traj in trajectories_for_pair],
                    dim=0,
                )
                weights_for_pair = torch.stack(weights_for_pair).to(
                    device=trajectories_for_pair.device,
                    dtype=torch.float32,
                )
                
                # Reinitialize the transform matrix for this frame pair with init_part_trans.
                reinit_save_dir = None
                if save_dir:
                    reinit_save_dir = os.path.join(save_dir, f'joint_reinit_epoch_pair_{frame_pair_id}')
                
                    T1_reinit, T2_reinit = init_part_trans(trajectories_for_pair, weights_for_pair, reinit_save_dir)
                    
                    device = next(transform_net.parameters()).device
                    
                    # Update the corresponding frame pair's T1 and T2 in transform_net.
                    if frame_pair_id < transform_net.T1.shape[0]:
                        transform_net.T1[frame_pair_id].data = T1_reinit.to(device=device, dtype=torch.float32)
                        transform_net.T2[frame_pair_id].data = T2_reinit.to(device=device, dtype=torch.float32)
    
                            

def solve_window_transformations(args, trajectories, init_point_seg_weights, point_indices, save_dir, first_frame, valid_seg_masks=None, factor=1.0, global_progress: float = 1.0):
    """
    Solve transformations for a set of trajectories using TransNet.
    Args:
        trajectories: List of trajectories, each trajectory is a numpy array of shape (N, T, 3) where N is the number of points, T is the number of frames.
        window_save_dir: Directory to save the results.
        args: Arguments containing loss type and other parameters.
        init_point_seg_weights: Initial part of the trajectory to use for transformation. Can be:
                  - numpy array of shape (N,) or (N, 1) with 0/1 labels
                  - None for automatic initialization using method specified in args
    """
    cam_K = first_frame["cam_K"]
    feature = first_frame["features"]
    rgb = first_frame["rgbs"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cam_K_tensor = torch.from_numpy(cam_K).to(device=device, dtype=torch.float32)

    trajs_tensor = trajectories
    trajs = trajs_tensor.detach().cpu().numpy()

    init_point_seg_weights_tensor = init_point_seg_weights.flatten().clamp(0.2, 0.8)
    init_point_seg_weights_np = init_point_seg_weights_tensor.detach().cpu().numpy()

    # The valid mask is only used for init_reg.
    point_y, point_x = point_indices
    valid_seg_masks_tensor = valid_seg_masks
    valid_init_mask = valid_seg_masks_tensor[point_y, point_x] == 1
    valid_init_idx = torch.nonzero(valid_init_mask, as_tuple=False).flatten()

    T1_init, T2_init = init_part_trans(
        trajs_tensor,
        init_point_seg_weights_tensor,
        save_dir,
    )

    transform_net = Trans(
        num_points=trajs_tensor.shape[0],
        num_T=1,
        init_weights=init_point_seg_weights_np,
        init_T1=T1_init,
        init_T2=T2_init,
        cam_K=cam_K,
        device=device,
    ).to(device)

    source_points = trajs_tensor[:, 0, :]
    target_points = trajs_tensor[:, 1, :]
    w_id = torch.arange(trajs_tensor.shape[0], device=device)
    T_id = torch.zeros(trajs_tensor.shape[0], dtype=torch.long, device=device)

    # Feature KNN, computed once.
    knn_idx, neighbor_w = compute_neighbors_from_feature_map(feature, rgb, point_indices, device, args)


    optimizer = Optimizer(transform_net, weight_lr=args.lr_w * factor, transform_lr=args.lr_T * factor, args=args)
    w_p = args.w_p / args.projection_normalizer
    w_m = args.w_m / args.scale_obj
    w_e = args.w_e
    w_s = args.w_s
    # Piecewise schedule: first half 0.1->0.2, second half stays 1.0
    # gp = float(np.clip(global_progress, 0.0, 1.0))
    # if gp < 0.3:
    #     w_scale = 0.1 + 0.2 * gp
    # elif gp < 0.6:
    #     w_scale = 1
    # else:
    #      w_scale = 0.5 +  0.5 * gp
    # w_init = args.w_init * w_scale
    if global_progress < 0.5:
        w_init = args.w_init * (0.1 + 0.9 * (global_progress / 0.5))
    else:
        w_init = args.w_init
 

    

    # First stage: optimize segmentation weights only.
    transform_net.T1.requires_grad = False
    transform_net.T2.requires_grad = False

    init_weights_tensor = init_point_seg_weights_tensor
    loss_computer = TransformLossComputer(
        loss_type=args.loss_type,
        cam_k_tensor=cam_K_tensor,
        w_p=w_p,
        w_m=w_m,
        w_e=w_e,
        w_s=w_s,
        w_init=w_init,
        weight_layout="flat",
        allow_normalized_l1=False,
        invalid_loss_mode="raise",
    )
    
    seg_epochs = args.solve_window_epochs // 2
    total_epochs = seg_epochs + args.solve_window_epochs
    pbar = tqdm(total=total_epochs, desc="Solve Window", unit="epoch")
    for epoch in range(seg_epochs):
        optimizer.zero_grad()
        pred_T1, pred_T2, w_sel, blended = transform_net.forward_components(source_points, w_id, T_id)
        total_loss, main_loss = loss_computer.compute_total_loss(
            pred_t1=pred_T1,
            pred_t2=pred_T2,
            w_sel=w_sel,
            blended=blended,
            target_points=target_points,
            source_points=source_points,
            weight_logits=transform_net.weight,
            init_weights_tensor=init_weights_tensor,
            knn_idx=knn_idx,
            neighbor_w=neighbor_w,
            valid_init_idx=valid_init_idx,
        )
        total_loss.backward()
        optimizer.step()
        pbar.set_postfix({
            'Loss': f'{np.sqrt(main_loss.item()):.6f}',
        })
        pbar.update(1)

    # Second stage: joint optimization with transforms unfrozen.
    transform_net.T1.requires_grad = True
    transform_net.T2.requires_grad = True
    optimizer = Optimizer(transform_net, weight_lr=args.lr_w * factor, transform_lr=args.lr_T * factor, args=args)

    epoch_loss_list = []

    for epoch in range(args.solve_window_epochs):
        if epoch > 0 and epoch % 50 == 0 and epoch < 1000:
            reinit_part_trans(transform_net, trajs_tensor, save_dir, epoch, factor)

        optimizer.zero_grad()
        pred_T1, pred_T2, w_sel, blended = transform_net.forward_components(source_points, w_id, T_id)
        total_loss, main_loss = loss_computer.compute_total_loss(
            pred_t1=pred_T1,
            pred_t2=pred_T2,
            w_sel=w_sel,
            blended=blended,
            target_points=target_points,
            source_points=source_points,
            weight_logits=transform_net.weight,
            init_weights_tensor=init_weights_tensor,
            knn_idx=knn_idx,
            neighbor_w=neighbor_w,
            valid_init_idx=valid_init_idx,
        )
        total_loss.backward()
        optimizer.step()

        avg_epoch_loss = main_loss.item()
        epoch_loss_list.append(avg_epoch_loss)
        optimizer.scheduler_step(np.sqrt(avg_epoch_loss))
        # Record gradient history every 100 epochs
        pbar.set_postfix({
            'Loss': f'{np.sqrt(avg_epoch_loss):.6f}',
        })
        pbar.update(1)

        last_loss = avg_epoch_loss
    pbar.close()

    with torch.no_grad():
        w_final = torch.sigmoid(transform_net.weight)
        T1_final = transform_net.T1[0].detach().cpu().numpy()
        T2_final = transform_net.T2[0].detach().cpu().numpy()
        final_point_seg_weights = w_final.flatten()

    if getattr(args, "debug", False) and save_dir and os.path.exists(os.path.dirname(save_dir)):
        init_save_dir = os.path.join(save_dir, 'part_init.png')
        visualize_part_initialization(part_weights=init_point_seg_weights_np, trajectories=trajs, save_path=init_save_dir)
        final_save_dir = os.path.join(save_dir, 'part_final.png')
        visualize_part_initialization(
            part_weights=final_point_seg_weights.detach().cpu().numpy(),
            trajectories=trajs,
            save_path=final_save_dir,
        )

    try:
        del trajs_tensor, transform_net
    except NameError:
        pass
    return T1_final, T2_final, final_point_seg_weights, last_loss

def compute_neighbors_from_point_features(point_trajectory_dict, point_keys, args, device):
    """
    Build point-feature neighbors (KNN or radius neighborhood) and return:
    - knn_idx: LongTensor [N, K_var] or [N, K], with neighbor indices per row.
    - neighbor_w: Tensor [N, K_var] or None, with neighbor weights for radius mode.
    Return (None, None) when features are missing or have incompatible dimensions.
    """
    knn_idx = None
    neighbor_w = None

    if not point_keys:
        return None, None

    first_point_data = point_trajectory_dict[point_keys[0]]
    if 'feature' not in first_point_data:
        return None, None

    features = [point_trajectory_dict[key]['feature'] for key in point_keys]
    feature_tensor = torch.stack(features, dim=0).to(device=device, dtype=torch.float32)

    # Only [N, C] point features are supported.
    if feature_tensor.ndim != 2:
        return None, None

    with torch.no_grad():
        feature_tensor = F.normalize(feature_tensor, dim=1, eps=1e-6)
        
        neighbor_method = getattr(args, 'neighbor_selection_method', 'radius')
        k_neighbors = int(getattr(args, 'knn_k', 1000))
        radius = getattr(args, 'knn_radius', 0.3)
        
        # Compute distances in batches to avoid OOM
        # Estimate batch size based on available memory
        n_points = feature_tensor.shape[0]
        # Assuming we want to limit the distance matrix to ~2GB per batch
        # float32 = 4 bytes, so 2GB = 500M elements, sqrt(500M) ~= 22360
        max_batch_size = 10000  # Conservative batch size
        
        if neighbor_method == 'radius' and radius is not None:
            topk = min(k_neighbors + 1, n_points)
            # Batched computation of top-k nearest neighbors
            all_topk_dists = []
            all_topk_idx = []
            
            for start_idx in range(0, n_points, max_batch_size):
                end_idx = min(start_idx + max_batch_size, n_points)
                batch_features = feature_tensor[start_idx:end_idx]
                
                # Compute distances for this batch against all points
                batch_dists = torch.cdist(batch_features, feature_tensor, p=2)
                
                # Get top-k for this batch
                batch_topk_dists, batch_topk_idx = batch_dists.topk(topk, largest=False)
                all_topk_dists.append(batch_topk_dists)
                all_topk_idx.append(batch_topk_idx)
            
            topk_dists = torch.cat(all_topk_dists, dim=0)
            topk_idx = torch.cat(all_topk_idx, dim=0)
            
            # Drop self-neighbor.
            topk_dists = topk_dists[:, 1:]
            topk_idx = topk_idx[:, 1:]
            mask = topk_dists < radius
            has_any = mask.any(dim=1)
            if (~has_any).any():
                first_col = torch.zeros_like(mask[:, :1])
                first_col[(~has_any), 0] = True
                mask = torch.where(has_any.unsqueeze(1), mask, first_col)
            # Build inverse-distance weights; neighbors outside the radius get 0.
            weights = 1.0 / (topk_dists + 1e-8)
            weights = torch.where(mask, weights, torch.zeros_like(weights))
            # Normalize over each point's neighbors.
            weights_sum = weights.sum(dim=1, keepdim=True)
            no_valid = weights_sum.squeeze(1) == 0
            if no_valid.any():
                # Fall back to the nearest neighbor with weight 1.
                weights[no_valid] = 0
                weights[no_valid, 0] = 1.0
                weights_sum[no_valid] = 1.0
            neighbor_w_pre = weights / weights_sum
            # Use vectorized gather to select the first max_k valid neighbors.
            valid_counts = mask.sum(dim=1)
            max_k = int(torch.clamp(valid_counts.max(), min=1).item())
            # Put out-of-radius distances at infinity so they sort last.
            filtered = topk_dists.masked_fill(~mask, float('inf'))
            if no_valid.any():
                filtered[no_valid, 0] = 0.0
            order = filtered.argsort(dim=1)
            sel_order = order[:, :max_k]
            knn_idx = topk_idx.gather(1, sel_order)
            neighbor_w = neighbor_w_pre.gather(1, sel_order)
        else:
            # Fixed-K nearest neighbors, also using batched computation.
            topk = min(k_neighbors + 1, n_points)
            all_topk_idx = []
            
            for start_idx in range(0, n_points, max_batch_size):
                end_idx = min(start_idx + max_batch_size, n_points)
                batch_features = feature_tensor[start_idx:end_idx]
                
                # Compute distances for this batch against all points
                batch_dists = torch.cdist(batch_features, feature_tensor, p=2)
                
                # Get top-k indices for this batch
                batch_topk_idx = batch_dists.topk(topk, largest=False).indices
                all_topk_idx.append(batch_topk_idx)
            
            knn_idx_with_self = torch.cat(all_topk_idx, dim=0)
            knn_idx = knn_idx_with_self[:, 1:]  # Remove self
            neighbor_w = None  # Uniform weights.

    return knn_idx, neighbor_w

def compute_neighbors_from_feature_map(feature, rgb, point_indices, device, args):
    """
    Sample point features from the image feature map at pixel coordinates and build neighbor indices/weights.
    Supports knn and radius modes. Returns (knn_idx, neighbor_w).
    """
    # Use no_grad (not inference_mode), because returned indices are later used
    # in autograd-tracked loss computation.
    with torch.no_grad():
        if feature is None:
            raise ValueError("Feature map is required for compute_neighbors_from_feature_map")
        feature_tensor = feature.to(device=device, dtype=torch.float32)
        if feature_tensor.ndim == 4 and feature_tensor.shape[0] == 1:
            feature_tensor = feature_tensor.squeeze(0)
        if feature_tensor.ndim != 3:
            raise ValueError(f"Expected feature map with shape (C,H,W), got {tuple(feature_tensor.shape)}")
        target_hw = tuple(rgb.shape[:2])
        if tuple(feature_tensor.shape[-2:]) != target_hw:
            feature_map = F.interpolate(
                feature_tensor.unsqueeze(0),
                size=target_hw,
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        else:
            feature_map = feature_tensor
        _, H, W = feature_map.shape
        y_idx, x_idx = point_indices
        N = int(y_idx.numel())
        if N == 0:
            empty_idx = torch.empty((0, 0), dtype=torch.long, device=device)
            empty_w = torch.empty((0, 0), dtype=torch.float32, device=device)
            return empty_idx, empty_w
        if N == 1:
            single_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
            if getattr(args, 'neighbor_selection_method', 'radius') == 'radius' and getattr(args, 'knn_radius', 0.3) is not None:
                single_w = torch.ones((1, 1), dtype=torch.float32, device=device)
            else:
                single_w = None
            return single_idx, single_w
        point_feats = feature_map[:, y_idx, x_idx].transpose(0, 1).contiguous()
        point_feats = F.normalize(point_feats, dim=1, eps=1e-6)
        
        neighbor_method = getattr(args, 'neighbor_selection_method', 'radius')
        k_neighbors = int(getattr(args, 'knn_k', 1000))
        radius = getattr(args, 'knn_radius', 0.3)
        
        # Compute distances in batches to avoid OOM
        # Only compute top-k for each batch to save memory
        topk = min(k_neighbors + 1, N)
        batch_size = min(5000, N)  # Process 5000 points at a time
        
        topk_dists_list = []
        topk_idx_list = []
        
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            batch_dists = torch.cdist(point_feats[i:end_i], point_feats, p=2)
            batch_topk_dists, batch_topk_idx = batch_dists.topk(topk, largest=False)
            topk_dists_list.append(batch_topk_dists)
            topk_idx_list.append(batch_topk_idx)
            del batch_dists  # Free memory immediately
        
        topk_dists = torch.cat(topk_dists_list, dim=0)
        topk_idx = torch.cat(topk_idx_list, dim=0)
        del topk_dists_list, topk_idx_list

        neighbor_w = None
        if neighbor_method == 'radius' and radius is not None:
            topk_dists = topk_dists[:, 1:]
            topk_idx = topk_idx[:, 1:]
            mask = topk_dists < radius
            has_any = mask.any(dim=1)
            if (~has_any).any():
                first_col = torch.zeros_like(mask[:, :1])
                first_col[(~has_any), 0] = True
                mask = torch.where(has_any.unsqueeze(1), mask, first_col)
            weights = 1.0 / (topk_dists + 1e-8)
            weights = torch.where(mask, weights, torch.zeros_like(weights))
            weights_sum = weights.sum(dim=1, keepdim=True)
            no_valid = weights_sum.squeeze(1) == 0
            if no_valid.any():
                weights[no_valid] = 0
                weights[no_valid, 0] = 1.0
                weights_sum[no_valid] = 1.0
            neighbor_w_pre = weights / weights_sum
            valid_counts = mask.sum(dim=1)
            max_k = int(torch.clamp(valid_counts.max(), min=1).item())
            filtered = topk_dists.masked_fill(~mask, float('inf'))
            if no_valid.any():
                filtered[no_valid, 0] = 0.0
            order = filtered.argsort(dim=1)
            sel_order = order[:, :max_k]
            knn_idx = topk_idx.gather(1, sel_order)
            neighbor_w = neighbor_w_pre.gather(1, sel_order)
        else:
            knn_idx = topk_idx[:, 1:]
            neighbor_w = None

    return knn_idx, neighbor_w

def joint_optimize_segmentation_and_transforms(args, point_trajectory_dict, all_transforms1, all_transforms2, device, cam_K, first_frame_weights=None, factor=1.0, global_progress: float = 1.0):
    """
    Joint optimization (no batching version). Optimize all points in a single forward per epoch.
    """
    point_keys = list(point_trajectory_dict.keys())
    if first_frame_weights is not None:
        init_weights = first_frame_weights.clamp(0.01, 0.99).detach().cpu().numpy()
    else:
        init_weights = []
        for point_key in point_keys:
            point_data = point_trajectory_dict[point_key]
            
            if "segmentation_weights" in point_data and len(point_data["segmentation_weights"]) > 0:
                weight = point_data["segmentation_weights"][-1].reshape(-1)[0].item()
                weight = np.clip(weight, 0.01, 0.99)
            else:
                weight = 0.5
            init_weights.append(weight)

    all_source_points = []
    all_target_points = []
    all_point_to_frame_indices = []
    all_trajectories_for_reinit = {}
    for point_idx, point_key in enumerate(point_keys):
        point_data = point_trajectory_dict[point_key]
        for traj, pair_id in zip(point_data["trajectories"], point_data["pair_ids"]):
            traj_tensor = traj
            assert traj_tensor.shape[0] == 2
            all_source_points.append(traj_tensor[0, :3])
            all_target_points.append(traj_tensor[1, :3])
            all_point_to_frame_indices.append((point_idx, pair_id))
            if pair_id not in all_trajectories_for_reinit:
                all_trajectories_for_reinit[pair_id] = []
            all_trajectories_for_reinit[pair_id].append({'point_idx': point_idx, 'traj': traj_tensor})

    source_points_tensor = _stack_points_to_tensor(all_source_points, device=device)
    target_points_tensor = _stack_points_to_tensor(all_target_points, device=device)
    point_to_frame_indices = torch.tensor(all_point_to_frame_indices, dtype=torch.long, device=device)
    cam_K_tensor = torch.from_numpy(cam_K).to(device=device, dtype=torch.float32)

    num_T = all_transforms1.shape[0] - 1
    w_id = point_to_frame_indices[:, 0].contiguous()
    T_id = point_to_frame_indices[:, 1].contiguous()

    num_points = len(point_keys)
    transform_net = Trans(
        num_points=num_points,
        num_T=num_T,
        init_weights=np.array(init_weights),
        init_T1=all_transforms1[0, 1:, :, :],
        init_T2=all_transforms2[0, 1:, :, :],
        cam_K=cam_K,
        device=device,
    ).to(device)

    # KNN (feature based if available)
    knn_idx = None
    neighbor_w = None
    knn_idx, neighbor_w = compute_neighbors_from_point_features(point_trajectory_dict, point_keys, args, device)

    optimizer = Optimizer(
        transform_net,
        weight_lr=(args.lr_w) / 3 * factor,
        transform_lr=(args.lr_T) / 3 * factor,
        args=args
    )

    w_p = args.w_p / args.projection_normalizer
    w_m = args.w_m / args.scale_obj
    w_e = args.w_e
    w_s = args.w_s
    # Piecewise schedule: first half 0.1->0.2, second half stays 1.0
    # gp = float(np.clip(global_progress, 0.0, 1.0))
    # if gp < 0.3:
    #     w_scale = 0.1 + 0.2 * gp
    # elif gp < 0.6:
    #     w_scale = 5
    # else:
    #      w_scale = 0.5 +  1* gp
    # w_init = args.w_init * w_scale
    # w_init = args.w_init * (0.1 * np.power(10.0, float(global_progress)))
    if global_progress < 0.5:
        w_init = args.w_init * (0.1 + 0.9 * (global_progress / 0.5))
    else:
        w_init = args.w_init
    epochs = args.joint_optimize_epochs
    pbar = tqdm(total=epochs, desc="Joint Optimize", unit="epoch")
    init_weights_tensor = torch.from_numpy(np.asarray(init_weights, dtype=np.float32)).to(device=device)
    loss_computer = TransformLossComputer(
        loss_type=args.loss_type,
        cam_k_tensor=cam_K_tensor,
        w_p=w_p,
        w_m=w_m,
        w_e=w_e,
        w_s=w_s,
        w_init=w_init,
        weight_layout="column",
        allow_normalized_l1=True,
        invalid_loss_mode="l1",
    )

    for epoch in range(epochs):
        if epoch > 0 and epoch % 100 == 0 and epoch < 2000:
            joint_reinit_part_trans(transform_net, all_trajectories_for_reinit)

        optimizer.zero_grad()
        pred_T1, pred_T2, w_sel, blended = transform_net.forward_components(source_points_tensor, w_id, T_id)
        total_loss, main_loss = loss_computer.compute_total_loss(
            pred_t1=pred_T1,
            pred_t2=pred_T2,
            w_sel=w_sel,
            blended=blended,
            target_points=target_points_tensor,
            source_points=source_points_tensor,
            weight_logits=transform_net.weight,
            init_weights_tensor=init_weights_tensor,
            knn_idx=knn_idx,
            neighbor_w=neighbor_w,
            valid_init_idx=None,
        )
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix({
            'Loss': f'{np.sqrt(main_loss.item()):.6f}',
            'Epoch': epoch + 1,
        })
        pbar.update(1)

        optimizer.scheduler_step(np.sqrt(main_loss.item()))
    pbar.close()

    with torch.no_grad():
        final_weights = torch.sigmoid(transform_net.weight).cpu().numpy().flatten()
        final_T1 = transform_net.T1.cpu().numpy()
        final_T2 = transform_net.T2.cpu().numpy()

    if first_frame_weights is not None:
        ref_weights_tensor = first_frame_weights.to(device=device, dtype=torch.float32)
        final_weights_tensor = torch.from_numpy(final_weights.squeeze()).to(ref_weights_tensor.device, dtype=ref_weights_tensor.dtype)
        loss_original = torch.mean((final_weights_tensor - ref_weights_tensor) ** 2)
        loss_reverse = torch.mean(((1 - final_weights_tensor) - ref_weights_tensor) ** 2)
        if loss_reverse < loss_original:
            print("Reversing segmentation weights to match first frame reference.", 'yellow')
            final_weights = 1 - final_weights
            final_T1, final_T2 = final_T2, final_T1

    return final_weights, final_T1, final_T2
