import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F


def convert_trajs_to_segmentation_map(point_seg_weight, point_indices, first_image_dict, fill_method='kmeans'):
    """
    Convert trajectory segmentation results to a segmentation map of size H*W.
    
    Args:
        point_seg_weight: torch.Tensor, shape (N,) or (N,1)
        point_indices: tuple(torch.Tensor, torch.Tensor), (y_indices, x_indices)
    Returns:
        seg_map_t: torch.Tensor, shape (H, W), float32
        trajectory_mask_t: torch.Tensor, shape (H, W), bool
    """
    mask = first_image_dict["masks"]
    H, W = mask.shape
    y_indices, x_indices = point_indices

    weights = point_seg_weight.reshape(-1).to(dtype=torch.float32)
    device = weights.device
    y_indices_t = y_indices.reshape(-1).to(device=device, dtype=torch.long)
    x_indices_t = x_indices.reshape(-1).to(device=device, dtype=torch.long)

    seg_map_t = torch.full((H, W), 0.5, dtype=torch.float32, device=device)
    trajectory_mask_t = torch.zeros((H, W), dtype=torch.bool, device=device)

    valid_mask_idx = (
        (y_indices_t >= 0)
        & (y_indices_t < H)
        & (x_indices_t >= 0)
        & (x_indices_t < W)
    )
    valid_y = y_indices_t[valid_mask_idx]
    valid_x = x_indices_t[valid_mask_idx]
    valid_weights = weights[valid_mask_idx]

    seg_map_t[valid_y, valid_x] = valid_weights
    trajectory_mask_t[valid_y, valid_x] = True

    return seg_map_t, trajectory_mask_t


def _resize_feature_to_hw_torch(feature, H, W, device):
    feature_tensor = feature.to(device=device, dtype=torch.float32)
    if feature_tensor.ndim == 4 and feature_tensor.shape[0] == 1:
        feature_tensor = feature_tensor.squeeze(0)
    if feature_tensor.ndim != 3:
        raise ValueError(f"Expected feature map with shape (C,H,W), got {tuple(feature_tensor.shape)}")
    if tuple(feature_tensor.shape[-2:]) != (H, W):
        feature_tensor = F.interpolate(
            feature_tensor.unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    return feature_tensor.contiguous()


def fill_with_kmeans(args, seg_map_t, trajectory_mask_t, missing_mask_t, feature=None, neighbor_selection_method='radius'):
    H, W = seg_map_t.shape
    device = seg_map_t.device

    missing_y, missing_x = torch.where(missing_mask_t)
    if int(missing_y.numel()) == 0:
        return seg_map_t

    known_y, known_x = torch.where(trajectory_mask_t)
    if int(known_y.numel()) == 0:
        return seg_map_t

    known_values_t = seg_map_t[known_y, known_x]

    if feature is not None:
        resized_feature = _resize_feature_to_hw_torch(feature, H, W, device)
        known_points_t = resized_feature[:, known_y, known_x].transpose(0, 1).contiguous()
        missing_points_t = resized_feature[:, missing_y, missing_x].transpose(0, 1).contiguous()
    else:
        known_points_t = torch.stack([known_y, known_x], dim=1).to(dtype=torch.float32)
        missing_points_t = torch.stack([missing_y, missing_x], dim=1).to(dtype=torch.float32)

    k_neighbors = min(int(args.knn_k), int(known_points_t.shape[0] // 2))
    if k_neighbors == 0:
        return seg_map_t

    distances_t = torch.cdist(missing_points_t, known_points_t)

    if neighbor_selection_method == 'radius':
        distances, nearest_indices = torch.topk(distances_t, k_neighbors, dim=1, largest=False)
        radius = float(args.knn_radius)
        radius_mask = distances < radius

        weights = 1.0 / (distances + 1e-8)
        weights[~radius_mask] = 0.0

        nearest_values = known_values_t[nearest_indices]
        sum_of_weights = weights.sum(dim=1, keepdim=True)

        no_neighbors_mask = sum_of_weights.squeeze(1) == 0
        if bool(no_neighbors_mask.any()):
            weights[no_neighbors_mask] = 0.0
            weights[no_neighbors_mask, 0] = 1.0
            sum_of_weights[no_neighbors_mask] = 1.0

        normalized_weights = weights / sum_of_weights
        assignments_t = (nearest_values * normalized_weights).sum(dim=1)
    elif neighbor_selection_method == 'knn':
        _, nearest_indices = torch.topk(distances_t, k_neighbors, dim=1, largest=False)
        nearest_values = known_values_t[nearest_indices]
        assignments_t = nearest_values.mean(dim=1)
    else:
        raise ValueError(f"unknown method: {neighbor_selection_method}")

    assignments_t = assignments_t.clamp(0.4, 0.6)
    filled_seg_map_t = seg_map_t.clone()
    filled_seg_map_t[missing_y, missing_x] = assignments_t
    return filled_seg_map_t


def fill_missing_segmentation(args, seg_map_t, trajectory_mask_t, valid_mask_t, feature=None, method='kmeans'):
    missing_mask_t = valid_mask_t & (~trajectory_mask_t)
    if not bool(missing_mask_t.any()):
        return seg_map_t

    if method == 'kmeans':
        return fill_with_kmeans(
            args,
            seg_map_t,
            trajectory_mask_t,
            missing_mask_t,
            feature=feature,
            neighbor_selection_method=args.neighbor_selection_method,
        )
    if method == 'none':
        return seg_map_t
    raise ValueError(f"GPU path only supports fill_method='kmeans' or 'none', got {method}")


def pass_segmentation_map(
    args,
    trajectory_maps,
    point_trajectory_dict,
    window_images_dict,
    window_size,
    fill_method='kmeans',
):
    """Tensor-only path: trajectory_maps and per-point weights are expected to be torch tensors."""
    seg_map_list_t = []
    trajectory_mask_list_t = []
    H, W = window_images_dict["rgbs"][0].shape[:2]
    traj_device = trajectory_maps.device
    valid_masks_t = torch.stack(
        [torch.tensor(mask, dtype=torch.bool) for mask in window_images_dict["masks"]],
        dim=0,
    ).to(device=traj_device)
    features_all = window_images_dict.get("features", None)

    point_keys = list(point_trajectory_dict.keys())
    if not point_keys:
        zero_seg = torch.zeros((H, W), dtype=torch.float32, device=traj_device)
        zero_mask = torch.zeros((H, W), dtype=torch.bool, device=traj_device)
        seg_map_list_t = [zero_seg.clone() for _ in range(window_size)]
        trajectory_mask_list_t = [zero_mask.clone() for _ in range(window_size)]
        return (
            seg_map_list_t,
            trajectory_mask_list_t,
            zero_seg,
            zero_mask.to(dtype=torch.float32),
        )

    orig_yx_t = torch.tensor(point_keys, dtype=torch.long, device=traj_device)
    point_weights_t = torch.stack(
        [point_trajectory_dict[k]["segmentation_weights"].reshape(-1)[0] for k in point_keys],
        dim=0,
    ).to(device=traj_device, dtype=torch.float32)

    point_validity_t = torch.zeros((len(point_keys), window_size), dtype=torch.bool, device=traj_device)
    point_validity_t[:, 0] = True
    for i, k in enumerate(point_keys):
        pair_ids = point_trajectory_dict[k]["pair_ids"]
        if not pair_ids:
            continue
        frame_ids = torch.tensor(pair_ids, dtype=torch.long, device=traj_device) + 1
        frame_ids = frame_ids[(frame_ids >= 1) & (frame_ids < window_size)]
        if int(frame_ids.numel()) > 0:
            point_validity_t[i, frame_ids] = True

    for frame_idx in range(window_size):
        seg_map_t = torch.zeros((H, W), dtype=torch.float32, device=traj_device)
        trajectory_mask_t = torch.zeros((H, W), dtype=torch.bool, device=traj_device)

        active_idx_t = torch.where(point_validity_t[:, frame_idx])[0]
        if int(active_idx_t.numel()) > 0:
            active_ys = orig_yx_t[active_idx_t, 0]
            active_xs = orig_yx_t[active_idx_t, 1]
            active_weights = point_weights_t[active_idx_t]

            cur_ys = trajectory_maps[0, frame_idx, 0, active_ys, active_xs].to(dtype=torch.long)
            cur_xs = trajectory_maps[0, frame_idx, 1, active_ys, active_xs].to(dtype=torch.long)
            valid_pos = (cur_ys >= 0) & (cur_ys < W) & (cur_xs >= 0) & (cur_xs < H)

            if bool(valid_pos.any()):
                valid_cur_ys = cur_ys[valid_pos]
                valid_cur_xs = cur_xs[valid_pos]
                valid_active_weights = active_weights[valid_pos]

                trajectory_cnt_t = torch.zeros((H, W), dtype=torch.float32, device=traj_device)
                seg_map_t.index_put_((valid_cur_xs, valid_cur_ys), valid_active_weights, accumulate=True)
                trajectory_cnt_t.index_put_(
                    (valid_cur_xs, valid_cur_ys),
                    torch.ones_like(valid_active_weights),
                    accumulate=True,
                )
                trajectory_mask_t[valid_cur_xs, valid_cur_ys] = True
                nonzero_mask_t = trajectory_cnt_t > 0
                seg_map_t[nonzero_mask_t] /= trajectory_cnt_t[nonzero_mask_t]

        feature = features_all[frame_idx] if features_all is not None else None
        if fill_method != 'none':
            seg_map_t = fill_missing_segmentation(
                args,
                seg_map_t,
                trajectory_mask_t,
                valid_masks_t[frame_idx],
                feature=feature,
                method=fill_method,
            )

        seg_map_list_t.append(seg_map_t)
        trajectory_mask_list_t.append(trajectory_mask_t)

    last_seg_map_t = seg_map_list_t[-1]
    last_trajectory_mask_t = trajectory_mask_list_t[-1].to(dtype=torch.float32)
    return seg_map_list_t, trajectory_mask_list_t, last_seg_map_t, last_trajectory_mask_t

def initialize_part_segmentation(single_image_dict, method='kmeans'):
    """
    Initialize part segmentation for images.
    
    Args:
        rgb_image: numpy array of shape (H, W, 3) RGB image
        depth_image: numpy array of shape (H, W) depth image
        mask: numpy array of shape (H, W) binary mask indicating valid regions
        init_weights: Initial weight map, can be:
                     - numpy array of shape (H, W) with values in [0, 1]
                     - None for automatic initialization
        method: Initialization method when init_weights is None
                - 'kmeans': K-means clustering on RGB+depth features
                - 'random': Random assignment
                - 'depth': Based on depth values
                - 'spatial_half': Based on spatial position (upper/lower half split)
                - 'spatial_lr': Based on spatial position (left/right split)
    
    Returns:
        weight_map: numpy array of shape (H, W) with values in [0, 1]
                   0 means fully belongs to part 1, 1 means fully belongs to part 2
    """
    mask = single_image_dict["masks"]
    rgb_image = single_image_dict["rgbs"]
    depth_image = single_image_dict["depths"]
    cam_K = single_image_dict["cam_K"]
    feature = single_image_dict.get("features", None)
    H, W = mask.shape
    # Initialize weight map
    weight_map = np.zeros((H, W), dtype=np.float32)
    
    if method == 'kmeans':
        # Extract valid pixels
        valid_mask = mask.astype(bool)
        valid_coords = np.where(valid_mask)

        # Prepare features: RGB + depth + spatial coordinates
        y_coords, x_coords = valid_coords
        depth = depth_image[valid_mask]
        
        # Combine features for valid pixels
        if feature is not None:
            # feature is (C, h, w), resize to (C, H, W)
            # Add batch dimension, resize, and remove batch dimension
            feature_tensor = feature.to(dtype=torch.float32)
            if feature_tensor.ndim == 4 and feature_tensor.shape[0] == 1:
                feature_tensor = feature_tensor.squeeze(0)
            feature_tensor = feature_tensor.unsqueeze(0)
            resized_feature_tensor = F.interpolate(feature_tensor, size=(H, W), mode='bilinear', align_corners=False)
            resized_feature = resized_feature_tensor.squeeze(0).numpy() # (C, H, W)
            
            # Extract features for valid pixels
            features = resized_feature[:, y_coords, x_coords].T # (N, C)
            
            # Combine with other features
            # features = np.hstack([
            #     feature_values,
            #     depth_norm[:, np.newaxis],
            #     y_norm[:, np.newaxis],
            #     x_norm[:, np.newaxis]
            # ])
        else:
            y = (y_coords - cam_K[1, 2])/ cam_K[1, 1] * depth
            x = (x_coords - cam_K[0, 2]) / cam_K[0, 0] * depth
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
            depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
            features = np.column_stack([depth_norm, y_norm, x_norm])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Assign labels to weight map
        weight_map[valid_coords] = labels.astype(np.float32)

        # Map 0/1 to 0.2/0.8 for valid pixels
        weight_map[valid_mask] = 0.2 + 0.6 * weight_map[valid_mask]
    
    
    elif method == 'random':
        np.random.seed(42)
        valid_mask = mask.astype(bool)
        random_weights = np.random.randint(0, 2, size=(H, W)).astype(np.float32)
        weight_map = random_weights * valid_mask
        
    elif method == 'depth':
        # Split based on depth values
        valid_mask = mask.astype(bool)
        depth_masked = depth_image * valid_mask
        
        # Calculate median depth of valid pixels
        valid_depths = depth_masked[valid_mask]
        if len(valid_depths) > 0:
            depth_median = np.median(valid_depths)
            weight_map = (depth_image > depth_median).astype(np.float32) * valid_mask
        else:
            weight_map = np.zeros((H, W), dtype=np.float32)
            
    elif method == 'spatial_half':
        # Upper half = 0, lower half = 1
        valid_mask = mask.astype(bool)
        y_coords = np.arange(H).reshape(-1, 1)
        y_median = H // 2
        weight_map = (y_coords > y_median).astype(np.float32) * valid_mask
        
    elif method == 'spatial_lr':
        # Left half = 0, right half = 1
        valid_mask = mask.astype(bool)
        x_coords = np.arange(W).reshape(1, -1)
        x_median = W // 2
        weight_map = (x_coords > x_median).astype(np.float32) * valid_mask
    elif method == 'ones':
        # All ones inside valid mask
        valid_mask = mask.astype(bool)
        weight_map[valid_mask] = 1.0
        
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    # Count pixels in each part
    part1_count = np.sum((weight_map < 0.5) & mask)
    part2_count = np.sum((weight_map >= 0.5) & mask)
    print(f"Auto-initialized parts using '{method}' method: {part1_count} pixels in part 1, {part2_count} pixels in part 2")
    
    return weight_map
