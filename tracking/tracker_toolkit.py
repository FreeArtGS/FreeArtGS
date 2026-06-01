import torch
import cv2
import os
import sys
from pathlib import Path
import PIL.Image
import numpy as np
import time
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKING_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRACKING_DIR) not in sys.path:
    sys.path.insert(0, str(TRACKING_DIR))

from termcolor import cprint
from utils.utils import count_parameters
import alltracker.utils.saveload
import alltracker.utils.basic
import alltracker.utils.improc
# Set PyTorch memory management for better memory efficiency
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
    def init_model(self):
        raise NotImplementedError("init_model method must be implemented in subclass")
    def get_track_maps(self, images_dict):
        raise NotImplementedError("get_track_maps method must be implemented in subclass")
    @staticmethod
    def _compute_local_voxel_counts(idx: torch.Tensor, neighbor_radius: int = 1) -> torch.Tensor:
        if idx.numel() == 0:
            return torch.zeros((0,), dtype=torch.int64, device=idx.device)

        idx = idx.to(dtype=torch.long)
        min_idx = idx.min(dim=0).values
        max_idx = idx.max(dim=0).values
        shifted_idx = idx - min_idx + neighbor_radius
        extents = (max_idx - min_idx + 2 * neighbor_radius + 1).to(dtype=torch.long)

        stride_z = torch.tensor(1, dtype=torch.long, device=idx.device)
        stride_y = extents[2]
        stride_x = extents[1] * extents[2]

        unique_voxels, inverse, counts = torch.unique(
            shifted_idx, dim=0, return_inverse=True, return_counts=True
        )
        unique_hash = (
            unique_voxels[:, 0] * stride_x
            + unique_voxels[:, 1] * stride_y
            + unique_voxels[:, 2] * stride_z
        )
        unique_hash_sorted, order = torch.sort(unique_hash)
        counts_sorted = counts[order]

        offsets = torch.tensor(
            [
                [dx, dy, dz]
                for dx in range(-neighbor_radius, neighbor_radius + 1)
                for dy in range(-neighbor_radius, neighbor_radius + 1)
                for dz in range(-neighbor_radius, neighbor_radius + 1)
            ],
            device=idx.device,
            dtype=torch.long,
        )
        offset_hash = (
            offsets[:, 0] * stride_x
            + offsets[:, 1] * stride_y
            + offsets[:, 2] * stride_z
        )

        point_hash = unique_hash[inverse]
        target_hash = point_hash[:, None] + offset_hash[None, :]
        target_hash_flat = target_hash.reshape(-1)

        positions = torch.searchsorted(unique_hash_sorted, target_hash_flat)
        valid = positions < unique_hash_sorted.numel()

        neighbor_counts_flat = torch.zeros_like(target_hash_flat, dtype=counts_sorted.dtype)
        if valid.any():
            valid_positions = positions[valid]
            matched = unique_hash_sorted[valid_positions] == target_hash_flat[valid]
            if matched.any():
                valid_flat_idx = valid.nonzero(as_tuple=False).squeeze(1)
                neighbor_counts_flat[valid_flat_idx[matched]] = counts_sorted[valid_positions[matched]]

        return neighbor_counts_flat.view_as(target_hash).sum(dim=1)

    def get_visible_trajectories(self, args, coords_4d):
        """
        Extract trajectories of points that are visible in all frames of the window.
        
        Args:
            coords_4d: tensor/array of shape (B, T, 5, H, W) where 5 represents
                (x, y, z, visibility, in_mask)
            args.min_depth (float): The minimum average depth for a trajectory to be considered valid.
            args.max_depth (float): The maximum average depth for a trajectory to be considered valid.

        Returns:
            trajectories: torch tensor of shape (N, T, 3) where N is the number of
                points visible in all frames
            point_indices: tuple of (y_indices, x_indices) tensors indicating the
                pixel coordinates of the visible points
        """
        if not torch.is_tensor(coords_4d):
            coords_4d = torch.as_tensor(coords_4d)

        B, T, _, H, W = coords_4d.shape
        
        # visibility from coords_4d (4th channel)
        visibility_from_coords = coords_4d[0, :, 3, :, :]  # Shape: (T, H, W)
        # in_mask from coords_4d (5th channel)
        in_mask_from_coords = coords_4d[0, :, 4, :, :] > 0.5 # Shape: (T, H, W)
        
        # Combine visibility conditions: 
        # 1. Point is in the binary mask for all frames
        # 2. Point has high visibility confidence for all frames
        visibility = visibility_from_coords > args.visibility_threshold
        combined_visibility = visibility & in_mask_from_coords
        always_visible = torch.all(combined_visibility, dim=0)  # Shape: (H, W)
        y_indices, x_indices = torch.where(always_visible)

        if y_indices.numel() == 0:
            empty_xyz = coords_4d.new_zeros((0, T, 3))
            empty_w = coords_4d.new_zeros((0, T))
            empty_idx = torch.zeros((0,), dtype=torch.long, device=coords_4d.device)
            return empty_xyz, (empty_idx, empty_idx.clone()), empty_w, 0.0, 0

        # Extract all required channels for all candidate points in one gather.
        flat_indices = y_indices * W + x_indices
        coords_flat = coords_4d[0].reshape(T, 5, H * W)   # (T, 5, HW)
        coords_points = coords_flat[:, :, flat_indices].permute(2, 0, 1).contiguous()  # (N, T, 5)

        # Filter points based on user-defined depth range.
        depth_values = coords_points[:, :, 2]
        all_in_range = torch.all(
            (depth_values >= args.min_depth) & (depth_values <= args.max_depth),
            dim=1,
        )

        y_indices = y_indices[all_in_range]
        x_indices = x_indices[all_in_range]
        coords_points = coords_points[all_in_range]

        # Keep only x, y, z channels -> (N, T, 3)
        trajectories = coords_points[:, :, :3]

        filter_ratio = 0.0
        filtered_count = int(trajectories.shape[0])

        # Add spatial-distance filtering; keep this robust without early returns or exceptions.
        if trajectories.shape[0] > 0 and T >= 2:
            original_count = int(trajectories.shape[0])
            # Mean adjacent-frame 3D displacement for each trajectory.
            delta_xyz = trajectories[:, 1:, :3] - trajectories[:, :-1, :3]
            distances = torch.linalg.norm(delta_xyz, dim=2).mean(dim=1)

            if distances.numel() > 1:
                mean = distances.mean()
                std = distances.std(unbiased=False)
                std_threshold = 2.0 * std
                valid_mask_std = torch.abs(distances - mean) <= std_threshold
                if valid_mask_std.any():
                    trajectories = trajectories[valid_mask_std]
                    y_indices = y_indices[valid_mask_std]
                    x_indices = x_indices[valid_mask_std]

            if trajectories.shape[0] > 0:
                delta_feat = trajectories[:, -1, :3] - trajectories[:, 0, :3]
                med = delta_feat.median(dim=0).values
                mad = torch.abs(delta_feat - med).median(dim=0).values
                scale = 1.4826 * torch.maximum(mad, torch.full_like(mad, 1e-6))
                Z = (delta_feat - med) / scale
                h = 0.8
                idx = torch.floor(Z / h).to(dtype=torch.long)
                local_cnt = self._compute_local_voxel_counts(idx, neighbor_radius=1)
                min_count = 20
                keep = local_cnt >= min_count
                max_drop_ratio = 0.0005
                min_keep = int((1.0 - max_drop_ratio) * idx.shape[0])
                required_keep = max(3, min_keep)
                if int(keep.sum().item()) < required_keep:
                    if required_keep >= local_cnt.numel():
                        keep = torch.ones_like(local_cnt, dtype=torch.bool)
                    else:
                        thr = torch.topk(local_cnt, k=required_keep, largest=True).values[-1]
                        keep = local_cnt >= thr
                if keep.any():
                    trajectories = trajectories[keep]
                    y_indices = y_indices[keep]
                    x_indices = x_indices[keep]

            filtered_count = int(trajectories.shape[0])
            if original_count > 0:
                filter_ratio = (original_count - filtered_count) / original_count
                cprint(f"  Prefilter: {original_count} -> {filtered_count} trajectories (ratio={filter_ratio:.6f})", "yellow")

        # Keep return signature unchanged: 3rd output is an auxiliary per-point weight tensor.
        aux_weights = torch.ones(
            (trajectories.shape[0], trajectories.shape[1]),
            dtype=trajectories.dtype,
            device=trajectories.device,
        )
        return trajectories[:, :, :3], (y_indices, x_indices), aux_weights, filter_ratio, filtered_count

    def get_4d_traj_maps(self, images_window_dict, traj_maps_e, visconf_maps_e):
        # 1. Get data from the dictionary
        depths_window = images_window_dict["depths"]
        # Prefer eroded masks for trajectory visibility
        masks_window = images_window_dict["masks_eroded"] 
        camera_matrix_scaled = images_window_dict["cam_K"]
        fx, fy = camera_matrix_scaled[0, 0], camera_matrix_scaled[1, 1]
        cx, cy = camera_matrix_scaled[0, 2], camera_matrix_scaled[1, 2]
        # 2. Calculate 3D coordinate tensor
        B, T, _, H, W = traj_maps_e.shape
        depths_tensor = torch.stack([torch.from_numpy(d) for d in depths_window], dim=0).unsqueeze(0).to(traj_maps_e.device)
        masks_tensor = torch.stack([torch.from_numpy(m) for m in masks_window], dim=0).unsqueeze(0).to(traj_maps_e.device)
        # 5D coordinates: x, y, z, visibility, in_mask
        coords_4d = torch.zeros(B, T, 5, H, W, device=traj_maps_e.device, dtype=traj_maps_e.dtype)

        for b in range(B): # B = 1
            for t in range(T):
                pixel_coords = traj_maps_e[b, t]
                visibility_confidence = visconf_maps_e[b, t]
                depth_map_t = depths_tensor[b, t]
                mask_t = masks_tensor[b, t].float()
                u, v = pixel_coords[0], pixel_coords[1]

                # Normalize coordinates for grid_sample
                normalized_u = 2 * u / (W - 1) - 1
                normalized_v = 2 * v / (H - 1) - 1
                grid = torch.stack((normalized_u, normalized_v), dim=2).unsqueeze(0)  # Shape: (1, H, W, 2)
                depth_map_t = depth_map_t.clone()
                depth_map_t[depth_map_t == 0] = float('-inf')
                sampled_depth = torch.nn.functional.grid_sample(
                    depth_map_t.unsqueeze(0).unsqueeze(0),  # Shape: (1, 1, H, W)
                    grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                ).squeeze()
                
                # Sample mask at (u, v)
                in_mask = torch.nn.functional.grid_sample(
                    mask_t.unsqueeze(0).unsqueeze(0),  # Shape: (1, 1, H, W)
                    grid,
                    mode='nearest',
                    padding_mode='border',
                    align_corners=True
                ).squeeze()
                # Unproject pixels to 3D space
                coords_4d[b, t, 0] = (u - cx) * sampled_depth / fx
                coords_4d[b, t, 1] = (v - cy) * sampled_depth / fy
                coords_4d[b, t, 2] = sampled_depth
                if visibility_confidence.dim() == 3:
                    vis_map = visibility_confidence[0]
                elif visibility_confidence.dim() == 2:
                    vis_map = visibility_confidence
                else:
                    # Fall back to all ones.
                    vis_map = torch.ones((H, W), device=traj_maps_e.device, dtype=traj_maps_e.dtype)
                coords_4d[b, t, 3] = vis_map
                coords_4d[b, t, 4] = in_mask 

        return coords_4d


class Alltracker(Tracker):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = self.init_model()

    def init_model(self):
        from alltracker.nets.alltracker import Net
        model = Net(self.cfg.window_len)
        count_parameters(model)
        url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=True)
        model.cuda()
        for n, p in model.named_parameters():
            p.requires_grad = False
        model.eval()
        cprint(f'loaded weights from {url}', "green")
        return model
    
    def get_track_maps(self, images_dict):
        rgbs = images_dict["rgbs_tensor"]
        B,T,C,H,W = rgbs.shape
        assert C == 3
        device = rgbs.device
        assert(B==1)

        grid_xy = alltracker.utils.basic.gridcloud2d(1, H, W, norm=False, device='cuda:0').float() # 1,H*W,2
        grid_xy = grid_xy.permute(0,2,1).reshape(1,1,2,H,W) # 1,1,2,H,W

        f_start_time = time.time()

        flows_e, visconf_maps_e, _, _ = \
            self.model(rgbs[:, self.cfg.query_frame:], iters=self.cfg.inference_iters, sw=None, is_training=False)
        traj_maps_e = flows_e + grid_xy # B,Tf,2,H,W
        if self.cfg.query_frame > 0:
            backward_flows_e, backward_visconf_maps_e, _, _ = \
                self.model(rgbs[:, :self.cfg.query_frame+1].flip([1]), iters=self.cfg.inference_iters, sw=None, is_training=False)
            backward_traj_maps_e = backward_flows_e + grid_xy # B,T,2,H,W, reversed
            backward_traj_maps_e = backward_traj_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
            backward_visconf_maps_e = backward_visconf_maps_e.flip([1])[:, :-1] # flip time and drop the overlapped frame
            traj_maps_e = torch.cat([backward_traj_maps_e, traj_maps_e], dim=1) # B,T,2,H,W
            visconf_maps_e = torch.cat([backward_visconf_maps_e, visconf_maps_e], dim=1) # B,T,2,H,W
        ftime = time.time()-f_start_time

        if self.cfg.verbose:
            cprint(f'finished forward; %.2f seconds / %d frames; %d fps' % (ftime, T, round(T/ftime)), "green")

        return traj_maps_e, visconf_maps_e
    
    def forward_video(self, traj_maps_e, visconf_maps_e, rgbs, framerate, output_dir):
        B,T,C,H,W = rgbs.shape
        assert C == 3
        device = rgbs.device
        assert(B==1)

        # subsample to make the vis more readable
        rate = self.cfg.rate
        trajs_e = traj_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2
        visconfs_e = visconf_maps_e[:,:,:,::rate,::rate].reshape(B,T,2,-1).permute(0,1,3,2) # B,T,N,2

        xy0 = trajs_e[0,0].cpu().numpy()
        colors = alltracker.utils.improc.get_2d_colors(xy0, H, W)

        fn = os.path.basename(self.cfg.image_folder)
        rgb_out_f = os.path.join(output_dir, 'vis_track.mp4')
        
        temp_dir = os.path.join(output_dir)
        alltracker.utils.basic.mkdir(temp_dir)
        vis = []

        frames = self._draw_pts_gpu(rgbs[0].to('cuda:0'), trajs_e[0], visconfs_e[0,:,:,0] > self.cfg.conf_thr,
                            colors, rate=rate, bkg_opacity=self.cfg.bkg_opacity)

        if self.cfg.vstack:
            frames_top = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
            frames = np.concatenate([frames_top, frames], axis=1)
        elif self.cfg.hstack:
            frames_left = rgbs[0].clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3
            frames = np.concatenate([frames_left, frames], axis=2)
        
        
        f_start_time = time.time()
        for ti in range(T):
            temp_out_f = '%s/%03d.jpg' % (temp_dir, ti)
            im = PIL.Image.fromarray(frames[ti])
            im.save(temp_out_f)#, "PNG", subsampling=0, quality=80)
        ftime = time.time()-f_start_time
        if self.cfg.verbose:
            print('finished writing; %.2f seconds / %d frames; %d fps' % (ftime, T, round(T/ftime)))
        
        os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate %d -pattern_type glob -i "./%s/*.jpg" -c:v libx264 -crf 20 -pix_fmt yuv420p %s' % (framerate, temp_dir, rgb_out_f))
        if self.cfg.verbose:
                print(f'rgb video saved to {rgb_out_f}, frames shape: {frames.shape}')
        
        return None
        
    def _draw_pts_gpu(rgbs, trajs, visibs, colormap, rate=1, bkg_opacity=0.5):
        device = rgbs.device
        T, C, H, W = rgbs.shape
        trajs = trajs.permute(1,0,2) # N,T,2
        visibs = visibs.permute(1,0) # N,T
        N = trajs.shape[0]
        colors = torch.tensor(colormap, dtype=torch.float32, device=device)  # [N,3]

        rgbs = rgbs * bkg_opacity # darken, to see the point tracks better
        
        opacity = 1.0
        if rate==1:
            radius = 1
            opacity = 0.9
        elif rate==2:
            radius = 1
        elif rate== 4:
            radius = 2
        elif rate== 8:
            radius = 4
        else:
            radius = 6
        sharpness = 0.15 + 0.05 * np.log2(rate)
        
        D = radius * 2 + 1
        y = torch.arange(D, device=device).float()[:, None] - radius
        x = torch.arange(D, device=device).float()[None, :] - radius
        dist2 = x**2 + y**2
        icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1)  # [D,D]
        icon = icon.view(1, D, D)
        dx = torch.arange(-radius, radius + 1, device=device)
        dy = torch.arange(-radius, radius + 1, device=device)
        disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")  # [D,D]
        for t in range(T):
            mask = visibs[:, t]  # [N]
            if mask.sum() == 0:
                continue
            xy = trajs[mask, t] + 0.5  # [N,2]
            xy[:, 0] = xy[:, 0].clamp(0, W - 1)
            xy[:, 1] = xy[:, 1].clamp(0, H - 1)
            colors_now = colors[mask]  # [N,3]
            N = xy.shape[0]
            cx = xy[:, 0].long()  # [N]
            cy = xy[:, 1].long()
            x_grid = cx[:, None, None] + disp_x  # [N,D,D]
            y_grid = cy[:, None, None] + disp_y  # [N,D,D]
            valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
            x_valid = x_grid[valid]  # [K]
            y_valid = y_grid[valid]
            icon_weights = icon.expand(N, D, D)[valid]  # [K]
            colors_valid = colors_now[:, :, None, None].expand(N, 3, D, D).permute(1, 0, 2, 3)[
                :, valid
            ]  # [3, K]
            idx_flat = (y_valid * W + x_valid).long()  # [K]

            accum = torch.zeros_like(rgbs[t])  # [3, H, W]
            weight = torch.zeros(1, H * W, device=device)  # [1, H*W]
            img_flat = accum.view(C, -1)  # [3, H*W]
            weighted_colors = colors_valid * icon_weights  # [3, K]
            img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
            weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
            weight = weight.view(1, H, W)

            alpha = weight.clamp(0, 1) * opacity
            accum = accum / (weight + 1e-6)  # [3, H, W]
            rgbs[t] = rgbs[t] * (1 - alpha) + accum * alpha
        rgbs = rgbs.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy() # T,H,W,3

        for t in range(T):
            img = np.ascontiguousarray(rgbs[t])
            for n in range(trajs.shape[0]):
                if not visibs[n, t]:
                    x, y = int(trajs[n, t, 0]), int(trajs[n, t, 1])
                    color = tuple([int(c) for c in colors[n].cpu().numpy()])
                    cv2.circle(img, (x, y), radius=radius, color=color, thickness=1, lineType=cv2.LINE_AA)
            rgbs[t] = img

        if bkg_opacity==0.0:
            for t in range(T):
                hsv_frame = cv2.cvtColor(rgbs[t], cv2.COLOR_RGB2HSV)
                saturation_factor = 1.5
                hsv_frame[..., 1] = np.clip(hsv_frame[..., 1] * saturation_factor, 0, 255)
                rgbs[t] = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
        return rgbs
