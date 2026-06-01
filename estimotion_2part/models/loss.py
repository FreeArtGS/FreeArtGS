import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_loss(weights, center=0.5, reduction='mean'):
    """
    Binary regularization loss to encourage weights to be either 0 or 1.
    
        
    Returns:
        torch.Tensor: Computed binary regularization loss
    """
    # Compute squared distance from center
    squared_distance = (weights - center) ** 2
    
    # Return negative to encourage weights away from center
    loss = -squared_distance
    
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")
    
    
    def forward(self, weights):
        """
        Forward pass for binary regularization loss.
        
        Args:
            weights (torch.Tensor): Weight values to regularize
            
        Returns:
            torch.Tensor: Binary regularization loss
        """
        return binary_loss(weights, self.center, self.reduction)


def entropy_loss(weights, eps=1e-8, reduction='mean'):
    """
    Information entropy loss for weights to encourage binary values.
    
    Implements: -[w * log(w) + (1-w) * log(1-w)]
    
    This loss function:
    - Has maximum value when w = 0.5 (maximum uncertainty)
    - Has minimum value (approaches 0) when w approaches 0 or 1
    - Encourages weights to be binary by minimizing entropy
    
    Args:
        weights (torch.Tensor): Weight values, should be in range (0, 1)
        eps (float): Small epsilon to avoid log(0). Default: 1e-8
        reduction (str): Reduction method ('mean', 'sum', 'none'). Default: 'mean'
        
    Returns:
        torch.Tensor: Negative entropy loss (minimizing this encourages binary weights)
    """
    # Clamp weights to avoid numerical issues with log
    w_clamped = torch.clamp(weights, eps, 1.0 - eps)
    
    # Compute entropy: -[w * log(w) + (1-w) * log(1-w)]
    entropy = -(w_clamped * torch.log(w_clamped) + (1.0 - w_clamped) * torch.log(1.0 - w_clamped))
    
    if reduction == 'none':
        return entropy
    elif reduction == 'mean':
        return torch.mean(entropy)
    elif reduction == 'sum':
        return torch.sum(entropy)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def consistency_loss(predicted_points, target_points, reduction='mean'):
    """
    Consistency loss to encourage predicted points to match target points.
    
    Args:
        predicted_points (torch.Tensor): Predicted points, shape (B, N, D)
        target_points (torch.Tensor): Target points, shape (B, N, D)
        reduction (str): Reduction method. Default: 'mean'
        
    Returns:
        torch.Tensor: Computed consistency loss
    """
    loss = torch.mean(torch.abs(predicted_points - target_points), dim=-1)
    
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")
    


class ConsistencyLoss(nn.Module):
    """
    A PyTorch module for smoothness regularization loss with KNN support.
    
    This loss encourages neighboring points to have similar weights, promoting spatial smoothness
    in weight assignments. It preserves KNN indices since each point's k nearest neighbors are fixed.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize the smoothness regularization loss.
        
        Args:
            reduction (str): Reduction method ('mean', 'sum', 'none'). Default: 'mean'
        """
        super(ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.knn_idx = None  # Store fixed KNN indices for reuse
    
    def forward(self, batch_weights, batch_knn_idx, current_weights, neighbor_weights=None):
        """
        Forward pass using batch indices for KNN neighbor lookup (smoothness regularization).
        
        This method implements the exact pattern from solve_transformation.py:
        batch_neighbor_weights = current_weights[batch_knn_idx]
        batch_self_weights = batch_weights.unsqueeze(1)
        smooth_reg = torch.mean((batch_self_weights - batch_neighbor_weights) ** 2)
        
        Args:
            batch_weights (torch.Tensor): Current batch weights, shape (batch_size, 1) or (batch_size,)
            batch_knn_idx (torch.Tensor): KNN indices for neighbors, shape (batch_size, K)
            current_weights (torch.Tensor): All weights to index from, shape (N, 1) or (N,)
            neighbor_weights (torch.Tensor, optional): Per-neighbor weights, shape (batch_size, K). If None, uniform.
            
        Returns:
            torch.Tensor: Smoothness regularization loss
        """
        # Store KNN indices (they are fixed for each point)
        self.knn_idx = batch_knn_idx
        
        # Get neighbor weights using KNN indices
        batch_neighbor_weights = current_weights[batch_knn_idx]  # (batch_size, K, 1) if current_weights is (N,1)
        
        # Expand self weights to match neighbor dimensions
        batch_self_weights = batch_weights.unsqueeze(1)  # (batch_size, 1, 1) if batch_weights is (N,1)
        
        # Compute squared differences
        diff_sq = (batch_self_weights - batch_neighbor_weights) ** 2  # (batch_size, K, 1)
        
        if neighbor_weights is None:
            # Unweighted mean as before
            smooth_reg = torch.mean(diff_sq)
            return smooth_reg
        
        # Weighted mean with per-neighbor weights
        if neighbor_weights.dim() == 2:
            neighbor_weights = neighbor_weights.unsqueeze(-1)  # (batch_size, K, 1)
        
        weighted_diff = diff_sq * neighbor_weights  # (batch_size, K, 1)
        if self.reduction == 'sum':
            return torch.sum(weighted_diff)
        
        # Normalize by sum of weights to keep scale stable
        denom = torch.sum(neighbor_weights) + 1e-8
        if self.reduction == 'none':
            # Return per-sample weighted means
            per_sample_denom = torch.sum(neighbor_weights, dim=1, keepdim=True) + 1e-8
            return torch.sum(weighted_diff, dim=1, keepdim=True) / per_sample_denom
        else:  # 'mean'
            return torch.sum(weighted_diff) / denom



def chamfer_loss(pc1, pc2):
    """
    Compute Chamfer distance with random sampling of points to limit memory.
    """
    # Compute distance matrix (manual implementation to ensure gradient support)
    diff = pc1.unsqueeze(1) - pc2.unsqueeze(0)  # (M1, M2, 3)
    dist_matrix = torch.norm(diff, dim=2)  # (M1, M2)
    # Distance from each point to nearest point
    min_dist_pc1, _ = torch.min(dist_matrix, dim=1)
    min_dist_pc2, _ = torch.min(dist_matrix, dim=0)
    # Bidirectional average
    chamfer = torch.mean(min_dist_pc1) + torch.mean(min_dist_pc2)
    return chamfer
    

def reprojection_normal_loss(points, normals, reprojected_points, rho_func='huber', reduction='mean'):
    """
    Simplified projection consistency loss for multi-view geometry.
    
    Implements: L_{project1}(i, j) = sum_{p in I_i} rho(n_i(p) · [reprojected_points - p])
    
    This loss enforces geometric consistency between views using pre-computed reprojected points.
    The reprojected points should be computed externally through the full pipeline:
    T_{ij}^{-1} π_j^{-1}(D_j(π_j(T_{ij} p)))
    
    Args:
        points (torch.Tensor): Original 3D points in view i coordinate system, shape (N, 3)
        normals (torch.Tensor): Surface normals at points in view i, shape (N, 3)
        reprojected_points (torch.Tensor): Reprojected 3D points in view i coordinate system, shape (N, 3)
        rho_func (str): Robust loss function ('huber', 'l1', 'l2'). Default: 'huber'
        reduction (str): Reduction method ('mean', 'sum', 'none'). Default: 'mean'
        
    Returns:
        torch.Tensor: Projection consistency loss
    """
    # Step 1: Compute difference vector
    diff_vector = reprojected_points - points  # (N, 3)
    
    # Step 2: Project onto normal direction: n_i(p) · [reprojected_points - p]
    normal_projection = torch.sum(normals * diff_vector, dim=1)  # (N,)
    
    # Step 3: Apply robust loss function ρ(·)
    if rho_func == 'huber':
        loss_values = F.huber_loss(normal_projection, torch.zeros_like(normal_projection), delta=1.0, reduction='none')
    elif rho_func == 'l1':
        loss_values = torch.abs(normal_projection)
    elif rho_func == 'l2':
        loss_values = normal_projection ** 2
    else:
        raise ValueError(f"Unknown robust loss function: {rho_func}")
    
    # Step 4: Apply reduction
    if reduction == 'none':
        return loss_values
    elif reduction == 'mean':
        return torch.mean(loss_values)
    elif reduction == 'sum':
        return torch.sum(loss_values)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def projection_matching_loss(transformed_pi, pj, K_j, delta=1.0, reduction='mean'):
    """
    Projection matching loss for matched point pairs between views.
    
    Implements: L_{project2}(i, j) = sum_{(transformed_pi, pj) in matchings} Huber(π_j(transformed_pi) - π_j(pj))
    
    This loss enforces geometric consistency for matched point pairs by:
    1. Taking already transformed 3D points in view j coordinate system
    2. Projecting both transformed_pi and pj to view j image plane: π_j
    3. Computing Huber loss between the two projected pixel coordinates
    
    Args:
        transformed_pi (torch.Tensor): Already transformed 3D points in view j coordinate system, shape (N, 3)
        pj (torch.Tensor): Corresponding 3D points in view j coordinate system, shape (N, 3)
        K_j (torch.Tensor): Intrinsic matrix of view j, shape (3, 3)
        delta (float): Huber loss threshold parameter. Default: 1.0
        reduction (str): Reduction method ('mean', 'sum', 'none'). Default: 'mean'
        
    Returns:
        torch.Tensor: Projection matching loss
    """
    device = transformed_pi.device
    N = transformed_pi.shape[0]
    if isinstance(K_j, torch.Tensor):
        K_j = K_j.to(device)
    else:
        K_j = torch.tensor(K_j, dtype=torch.float32, device=device)
    # Step 1: Project transformed 3D points to view j image plane: π_j(transformed_pi)
    # transformed_pi is already in view j coordinate system, shape (N, 3)
    points_i_proj = (K_j @ transformed_pi.T).T  # (N, 3)
    projected_pi = points_i_proj[:, :2] / points_i_proj[:, 2:3]  # (N, 2) normalize by z
    
    # Step 2: Project pj 3D points to view j image plane: π_j(pj)
    # pj is also in view j coordinate system, shape (N, 3)
    points_j_proj = (K_j @ pj.T).T  # (N, 3)
    projected_pj = points_j_proj[:, :2] / points_j_proj[:, 2:3]  # (N, 2) normalize by z
    
    # Step 3: Compute Huber loss between projected pixel coordinates
    pixel_diff = projected_pi - projected_pj  # (N, 2)
    pixel_distance = torch.norm(pixel_diff, dim=1)  # (N,) L2 distance
    
    # Apply Huber loss
    loss_values = F.huber_loss(pixel_distance, torch.zeros_like(pixel_distance), delta=delta, reduction='none')
    
    # Step 4: Apply reduction
    if reduction == 'none':
        return loss_values
    elif reduction == 'mean':
        return torch.mean(loss_values)
    elif reduction == 'sum':
        return torch.sum(loss_values)
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")