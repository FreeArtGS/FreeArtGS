import json
import math
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from eval_data import load_joint_values_from_joint_states


def _read_cam_K(cam_k_path: str) -> tuple[float, float, float, float]:
    with open(cam_k_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    mat = []
    for line in lines:
        mat.append([float(x) for x in line.replace(",", " ").split()])
    return float(mat[0][0]), float(mat[1][1]), float(mat[0][2]), float(mat[1][2])


def _load_eval_rgb_paths(eval_dir: str) -> list[str]:
    rgb_dir = Path(eval_dir) / "rgb"
    paths = sorted(str(path) for path in rgb_dir.glob("*.png"))
    if not paths:
        paths = sorted(str(path) for path in rgb_dir.glob("*.jpg"))
    return paths


def _ensure_eval_camera_poses(eval_dir: str, T0: np.ndarray | None = None) -> str:
    pose_npy = os.path.join(eval_dir, "camera_pose.npy")
    from convert_objposes_to_camposes import convert as convert_objposes

    return convert_objposes(eval_dir, output_path=pose_npy, T0=T0)


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis.astype(np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [x * x * one_minus_c + c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, y * y * one_minus_c + c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, z * z * one_minus_c + c],
        ],
        dtype=np.float64,
    )


def _transform_points_rigid(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, pivot: np.ndarray | None = None) -> np.ndarray:
    if pivot is None:
        return (points @ rotation.T) + translation
    return ((points - pivot) @ rotation.T) + pivot + translation


def _compute_psnr(img: np.ndarray, ref: np.ndarray) -> float:
    a = img.astype(np.float32)
    b = ref.astype(np.float32)
    if a.max() > 1.5 or b.max() > 1.5:
        a = a / 255.0
        b = b / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _compute_psnr_masked(img: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float | None:
    a = img.astype(np.float32)
    b = ref.astype(np.float32)
    if a.max() > 1.5 or b.max() > 1.5:
        a = a / 255.0
        b = b / 255.0
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.bool_:
        mask = mask > 0.5 if mask.max() <= 1.0 else mask > 127
    mask = mask.astype(np.bool_)
    valid = int(mask.sum())
    if valid < 1:
        return None
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    mse = float(((a - b) ** 2)[mask3].mean())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _load_articulation_frame_values(articulation_path: str, joint_type: str) -> dict[int, float]:
    if not os.path.exists(articulation_path):
        return {}
    with open(articulation_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames_train") or data.get("frames") or []
    key = "angle_rad" if joint_type == "revolute" else "distance"
    values: dict[int, float] = {}
    for local_idx, frame in enumerate(frames):
        if not isinstance(frame, dict) or key not in frame:
            continue
        frame_idx = int(frame.get("frame_index", local_idx))
        values[frame_idx] = float(frame[key])
    return values


def _estimate_joint_value_alignment_from_train(
    gaussian_dir: str,
    train_dataset_dir: str | None,
    joint_type: str,
    preferred_sign: float = 1.0,
) -> tuple[float | None, float, dict[str, float] | None]:
    if train_dataset_dir is None:
        return None, float(preferred_sign), None
    gt_values = load_joint_values_from_joint_states(train_dataset_dir)
    if gt_values is None or gt_values.size == 0:
        return None, float(preferred_sign), None
    pred_by_frame = _load_articulation_frame_values(
        str(Path(gaussian_dir) / "articulation.json"),
        joint_type,
    )
    if not pred_by_frame:
        return None, float(preferred_sign), None

    common = sorted(idx for idx in pred_by_frame.keys() if 0 <= idx < len(gt_values))
    if not common:
        return None, float(preferred_sign), None

    pred = np.asarray([pred_by_frame[idx] for idx in common], dtype=np.float64)
    gt = np.asarray([gt_values[idx] for idx in common], dtype=np.float64)

    def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
        return (values + np.pi) % (2.0 * np.pi) - np.pi

    candidates = [float(preferred_sign)]
    alt = -float(preferred_sign)
    if alt not in candidates:
        candidates.append(alt)
    for extra in (1.0, -1.0):
        if extra not in candidates:
            candidates.append(extra)

    best_sign = float(preferred_sign)
    best_offset = 0.0
    best_error = None
    for sign in candidates:
        pred_signed = sign * pred
        if joint_type == "revolute":
            pred_fit = np.unwrap(pred_signed)
            gt_fit = np.unwrap(gt)
            offset = float(np.median(gt_fit - pred_fit))
            residual = _wrap_to_pi(gt_fit - (pred_fit + offset))
            err = float(np.mean(np.abs(residual)))
        else:
            offset = float(np.median(gt - pred_signed))
            residual = gt - (pred_signed + offset)
            err = float(np.mean(np.abs(residual)))
        if best_error is None or err < best_error:
            best_error = err
            best_sign = float(sign)
            best_offset = float(offset)

    return best_offset, best_sign, {
        "offset": float(best_offset),
        "sign": float(best_sign),
        "mean_abs_err": float(best_error),
        "frames": float(len(common)),
    }


def render_gaussians_and_psnr_gsplat(
    gaussian_dir: str,
    eval_dir: str,
    train_dataset_dir: str | None,
    align_T: np.ndarray,
    joint_type: str,
    joint_axis_world: np.ndarray,
    joint_pos_world: np.ndarray,
    T0: np.ndarray,
    model_joint_value_sign: float = 1.0,
    motion_part: int = 0,
    sh_degree: int = 3,
    rasterize_mode: str = "classic",
    save_dir: str | None = None,
    refine_motion_delta: bool = True,
    refine_angle_range_deg: float = 10.0,
    refine_trans_rel_range: float = 0.05,
    refine_steps: int = 9,
    no_save: bool = False,
) -> dict:
    from gsplat.project_gaussians import project_gaussians
    from gsplat.rasterize import rasterize_gaussians
    from gsplat.sh import spherical_harmonics
    import cv2
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_raw_multiply

    repo_root = Path(__file__).resolve().parent.parent
    ns_root = repo_root / "reconstruction" / "nerfstudio"
    if str(ns_root) not in sys.path:
        sys.path.append(str(ns_root))
    from nerfstudio.data.utils.dataparsers_utils import apply_left_se3_to_gaussians, load_3dgs_ply  # type: ignore

    gaussian_dir_p = Path(gaussian_dir)
    ply_path = gaussian_dir_p / "object_3dgs.ply"
    weight_path = gaussian_dir_p / "3dgs_part_weight.txt"
    if not ply_path.exists():
        return {"psnr": None, "frames": 0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gauss = load_3dgs_ply(ply_path, device=device, rest_format="auto")
    with torch.no_grad():
        gauss = apply_left_se3_to_gaussians(gauss, torch.tensor(align_T, dtype=torch.float32, device=device), device=device)

    weights = np.loadtxt(str(weight_path)).astype(np.float32)
    moving_mask = (weights <= 0.5) if motion_part == 0 else (weights > 0.5)
    moving_mask = torch.as_tensor(moving_mask, dtype=torch.bool, device=device)

    pose_npy = _ensure_eval_camera_poses(eval_dir, T0)
    c2w = np.load(pose_npy).astype(np.float64)
    rgb_paths = _load_eval_rgb_paths(eval_dir)
    if not rgb_paths:
        return {"psnr": None, "frames": 0}
    fx, fy, cx, cy = _read_cam_K(str(Path(eval_dir) / "cam_K.txt"))
    gt0 = iio.imread(rgb_paths[0])
    H, W = int(gt0.shape[0]), int(gt0.shape[1])

    features_dc = gauss["features_dc"]
    features_rest = gauss["features_rest"]
    if features_rest.ndim == 2:
        n_points, n_coeff = features_rest.shape
        if n_coeff % 3 == 0 and n_coeff > 0:
            features_rest = features_rest.view(n_points, n_coeff // 3, 3)
        else:
            features_rest = features_rest.new_zeros((n_points, 0, 3))
    colors_all = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    opacities_all = torch.sigmoid(gauss["opacities"])
    scales_all = torch.exp(gauss["scales"])
    quats_all = gauss["quats"] / (gauss["quats"].norm(dim=-1, keepdim=True) + 1e-12)
    quats0 = quats_all.clone()
    xyz0 = gauss["means"].clone()

    joint_values = load_joint_values_from_joint_states(eval_dir)
    effective_model_joint_value_sign = float(model_joint_value_sign)
    joint_alignment_info = None
    if joint_values is None:
        joint_values = np.zeros((len(c2w),), dtype=np.float64)
    else:
        joint_value_offset, effective_model_joint_value_sign, joint_alignment_info = _estimate_joint_value_alignment_from_train(
            gaussian_dir,
            train_dataset_dir,
            joint_type,
            preferred_sign=model_joint_value_sign,
        )
        if joint_value_offset is not None:
            joint_values = joint_values - joint_value_offset
    if len(joint_values) != len(c2w):
        count = min(len(joint_values), len(c2w), len(rgb_paths))
        c2w = c2w[:count]
        rgb_paths = rgb_paths[:count]
        joint_values = joint_values[:count]

    background = torch.ones(3, device=device, dtype=torch.float32)
    block_width = 16
    psnr_list = []
    save_pred_dir = save_gt_dir = save_compare_dir = None
    if save_dir is not None and not no_save:
        save_pred_dir = Path(save_dir) / "render"
        save_gt_dir = Path(save_dir) / "gt"
        save_compare_dir = Path(save_dir) / "compare"
        os.makedirs(save_pred_dir, exist_ok=True)
        os.makedirs(save_gt_dir, exist_ok=True)
        os.makedirs(save_compare_dir, exist_ok=True)

    total_bases = int(colors_all.shape[1])
    auto_sh_degree = int(round(math.sqrt(total_bases) - 1))
    if (auto_sh_degree + 1) * (auto_sh_degree + 1) != total_bases:
        auto_sh_degree = 0 if colors_all.shape[1] <= 1 else sh_degree

    best_global_delta = 0.0
    if refine_motion_delta and bool(moving_mask.any().item()):
        pivot = joint_pos_world.astype(np.float64)
        axis_w = (effective_model_joint_value_sign * joint_axis_world).astype(np.float64)

        if joint_type == "revolute":
            angle_max = float(refine_angle_range_deg) * math.pi / 180.0
            delta_list = [float(x) for x in np.linspace(-angle_max, angle_max, max(3, int(refine_steps)))]
        else:
            xyz_mov0 = xyz0[moving_mask]
            if xyz_mov0.numel() > 0:
                xyz_min = xyz_mov0.amin(dim=0)
                xyz_max = xyz_mov0.amax(dim=0)
                diag = float(torch.linalg.norm(xyz_max - xyz_min).item())
            else:
                diag = 0.0
            trans_max = (refine_trans_rel_range * diag) if diag > 0 else 0.0
            delta_list = [0.0] if trans_max <= 0 else [float(x) for x in np.linspace(-trans_max, trans_max, max(3, int(refine_steps)))]

        best_err = None
        for delta in delta_list:
            total_err = 0.0
            count = 0
            for i in range(len(c2w)):
                xyz_try = xyz0.clone()
                quats_try = quats0.clone()

                value = float(joint_values[i]) + float(delta)
                if joint_type == "revolute":
                    rotation = _axis_angle_to_matrix(axis_w, value)
                    translation = np.zeros(3, dtype=np.float64)
                else:
                    rotation = np.eye(3, dtype=np.float64)
                    translation = (axis_w * value).astype(np.float64)
                pts = xyz_try[moving_mask].detach().cpu().numpy()
                pts_new = _transform_points_rigid(pts, rotation, translation, pivot=pivot)
                xyz_try[moving_mask] = torch.tensor(pts_new, device=device, dtype=xyz_try.dtype)
                if joint_type == "revolute":
                    rotation_t = torch.tensor(rotation, dtype=torch.float32, device=device)
                    q_rot_wxyz = matrix_to_quaternion(rotation_t[None, ...])
                    q_mov = quats_try[moving_mask]
                    q_mov_wxyz = torch.stack([q_mov[..., 3], q_mov[..., 0], q_mov[..., 1], q_mov[..., 2]], dim=-1)
                    q_out_wxyz = quaternion_raw_multiply(q_rot_wxyz.repeat(q_mov_wxyz.shape[0], 1), q_mov_wxyz)
                    q_out_xyzw = torch.stack([q_out_wxyz[..., 1], q_out_wxyz[..., 2], q_out_wxyz[..., 3], q_out_wxyz[..., 0]], dim=-1)
                    quats_try = quats_try.clone()
                    quats_try[moving_mask] = q_out_xyzw / (q_out_xyzw.norm(dim=-1, keepdim=True) + 1e-12)

                c2w_i = c2w[i]
                rotation = c2w_i[:3, :3].copy()
                translation = c2w_i[:3, 3:4].copy()
                rotation_inv = rotation.T
                translation_inv = -rotation_inv @ translation
                viewmat = np.eye(4, dtype=np.float64)
                viewmat[:3, :3] = rotation_inv
                viewmat[:3, 3:4] = translation_inv
                view3x4 = torch.tensor(viewmat[:3, :], dtype=torch.float32, device=device)

                xys_d, depths_d, radii_d, conics_d, comp_d, num_tiles_hit_d, _ = project_gaussians(
                    xyz_try, scales_all, 1, quats_try, view3x4, float(fx), float(fy), float(cx), float(cy), H, W, block_width,
                )
                if (radii_d.sum()).item() == 0:
                    continue
                cam_center = torch.tensor(c2w_i[:3, 3], dtype=torch.float32, device=device)
                viewdirs = xyz_try - cam_center
                if auto_sh_degree <= 0 or colors_all.shape[1] <= 1:
                    rgbs_d = torch.sigmoid(features_dc)
                else:
                    rgbs_d = spherical_harmonics(int(auto_sh_degree), viewdirs, colors_all)
                    rgbs_d = torch.clamp(rgbs_d + 0.5, min=0.0)
                opacities_d = opacities_all * comp_d[:, None] if rasterize_mode == "antialiased" else opacities_all
                rgb_d = rasterize_gaussians(
                    xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d, rgbs_d, opacities_d, H, W, block_width, background=background,
                )
                pred_d = torch.clamp(rgb_d, max=1.0)

                gt_i = iio.imread(rgb_paths[i])
                if gt_i.ndim == 2:
                    gt_i = np.repeat(gt_i[..., None], 3, axis=2)
                if gt_i.shape[1] != W or gt_i.shape[0] != H:
                    gt_i = cv2.resize(gt_i, (W, H), interpolation=cv2.INTER_AREA)
                gt_t = torch.tensor(gt_i, dtype=torch.float32, device=device) / 255.0
                mask_path = Path(eval_dir) / "masks" / f"{Path(rgb_paths[i]).stem}.png"
                mask_img = None
                if mask_path.exists():
                    mask_img = iio.imread(str(mask_path))
                    if mask_img.ndim == 3 and mask_img.shape[2] > 1:
                        mask_img = mask_img[:, :, 0]
                    if mask_img.shape[0] != H or mask_img.shape[1] != W:
                        mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
                if mask_img is not None:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    mask_img = cv2.erode((mask_img > 0).astype(np.uint8), kernel, iterations=1)
                    mask_t = torch.tensor(mask_img > 0, dtype=torch.float32, device=device)[..., None]
                    denom = mask_t.sum()
                    if denom.item() < 0.5:
                        err = float(torch.mean((pred_d - gt_t) ** 2).item())
                    else:
                        diff = (pred_d - gt_t) * mask_t
                        err = (diff.square().sum() / (3.0 * denom)).item()
                else:
                    err = float(torch.mean((pred_d - gt_t) ** 2).item())

                total_err += err
                count += 1

            if count > 0:
                mean_err = total_err / count
                if best_err is None or mean_err < best_err:
                    best_err = mean_err
                    best_global_delta = float(delta)

    for i in range(len(c2w)):
        xyz_frame = xyz0.clone()
        quats_frame = quats0.clone()
        if moving_mask is not None:
            value = float(joint_values[i]) + float(best_global_delta)
            axis_w = (effective_model_joint_value_sign * joint_axis_world).astype(np.float64)
            if joint_type == "revolute":
                rotation = _axis_angle_to_matrix(axis_w, value)
                translation = np.zeros(3, dtype=np.float64)
            else:
                rotation = np.eye(3, dtype=np.float64)
                translation = (axis_w * value).astype(np.float64)
            pivot = joint_pos_world.astype(np.float64)
            pts = xyz_frame[moving_mask].detach().cpu().numpy()
            pts_new = _transform_points_rigid(pts, rotation, translation, pivot=pivot)
            xyz_frame[moving_mask] = torch.tensor(pts_new, device=device, dtype=xyz_frame.dtype)
            if joint_type == "revolute":
                rotation_t = torch.tensor(rotation, dtype=torch.float32, device=device)
                q_rot_wxyz = matrix_to_quaternion(rotation_t[None, ...])
                q_mov = quats_frame[moving_mask]
                q_mov_wxyz = torch.stack([q_mov[..., 3], q_mov[..., 0], q_mov[..., 1], q_mov[..., 2]], dim=-1)
                q_out_wxyz = quaternion_raw_multiply(q_rot_wxyz.repeat(q_mov_wxyz.shape[0], 1), q_mov_wxyz)
                q_out_xyzw = torch.stack([q_out_wxyz[..., 1], q_out_wxyz[..., 2], q_out_wxyz[..., 3], q_out_wxyz[..., 0]], dim=-1)
                quats_frame = quats_frame.clone()
                quats_frame[moving_mask] = q_out_xyzw / (q_out_xyzw.norm(dim=-1, keepdim=True) + 1e-12)

        c2w_i = c2w[i]
        rotation = c2w_i[:3, :3].copy()
        translation = c2w_i[:3, 3:4].copy()
        rotation_edit = np.diag([1.0, -1.0, -1.0])
        rotation = rotation @ rotation_edit
        rotation_inv = rotation.T
        translation_inv = -rotation_inv @ translation
        viewmat = np.eye(4, dtype=np.float64)
        viewmat[:3, :3] = rotation_inv
        viewmat[:3, 3:4] = translation_inv
        view3x4 = torch.tensor(viewmat[:3, :], dtype=torch.float32, device=device)

        if refine_motion_delta and bool(moving_mask.any().item()):
            gt_ref = iio.imread(rgb_paths[i])
            if gt_ref.ndim == 2:
                gt_ref = np.repeat(gt_ref[..., None], 3, axis=2)
            if gt_ref.shape[1] != W or gt_ref.shape[0] != H:
                gt_ref = cv2.resize(gt_ref, (W, H), interpolation=cv2.INTER_AREA)
            stem_ref = Path(rgb_paths[i]).stem
            mask_path_ref = Path(eval_dir) / "masks" / f"{stem_ref}.png"
            mask_ref = None
            if mask_path_ref.exists():
                mask_ref = iio.imread(str(mask_path_ref))
                if mask_ref.ndim == 3 and mask_ref.shape[2] > 1:
                    mask_ref = mask_ref[:, :, 0]
                if mask_ref.shape[0] != H or mask_ref.shape[1] != W:
                    mask_ref = cv2.resize(mask_ref, (W, H), interpolation=cv2.INTER_NEAREST)

            with torch.no_grad():
                pivot = joint_pos_world.astype(np.float64)
                axis_w = (effective_model_joint_value_sign * joint_axis_world).astype(np.float64)

                if joint_type == "revolute":
                    angle_max = float(refine_angle_range_deg) * math.pi / 180.0
                    deltas = [float(x) for x in np.linspace(-angle_max, angle_max, max(3, int(refine_steps)))]
                else:
                    xyz_mov = xyz_frame[moving_mask]
                    if xyz_mov.numel() > 0:
                        xyz_min = xyz_mov.amin(dim=0)
                        xyz_max = xyz_mov.amax(dim=0)
                        diag = float(torch.linalg.norm(xyz_max - xyz_min).item())
                    else:
                        diag = 0.0
                    trans_max = (refine_trans_rel_range * diag) if diag > 0 else 0.0
                    deltas = [0.0] if trans_max <= 0 else [float(x) for x in np.linspace(-trans_max, trans_max, max(3, int(refine_steps)))]

                gt_t = torch.tensor(gt_ref, dtype=torch.float32, device=device) / 255.0
                mask_t = None
                if mask_ref is not None:
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    mask_ref = cv2.erode((mask_ref > 0).astype(np.uint8), kernel, iterations=1)
                    mask_t = torch.tensor(mask_ref > 0, dtype=torch.float32, device=device)[..., None]

                best_err = None
                best_xyz = None
                best_quats = None
                quats_base = quats_all.clone()
                for delta in deltas:
                    xyz_try = xyz_frame.clone()
                    if joint_type == "revolute":
                        rotation_delta = _axis_angle_to_matrix(axis_w, float(delta))
                        translation_delta = np.zeros(3, dtype=np.float64)
                    else:
                        rotation_delta = np.eye(3, dtype=np.float64)
                        translation_delta = (axis_w * float(delta)).astype(np.float64)
                    pts = xyz_try[moving_mask].detach().cpu().numpy()
                    pts_new = _transform_points_rigid(pts, rotation_delta, translation_delta, pivot=pivot)
                    xyz_try[moving_mask] = torch.tensor(pts_new, device=device, dtype=xyz_try.dtype)

                    quats_try = quats_base
                    if joint_type == "revolute":
                        rotation_t = torch.tensor(rotation_delta, dtype=torch.float32, device=device)
                        q_rot_wxyz = matrix_to_quaternion(rotation_t[None, ...])
                        q_mov = quats_base[moving_mask]
                        q_mov_wxyz = torch.stack([q_mov[..., 3], q_mov[..., 0], q_mov[..., 1], q_mov[..., 2]], dim=-1)
                        q_out_wxyz = quaternion_raw_multiply(q_rot_wxyz.repeat(q_mov_wxyz.shape[0], 1), q_mov_wxyz)
                        q_out_xyzw = torch.stack([q_out_wxyz[..., 1], q_out_wxyz[..., 2], q_out_wxyz[..., 3], q_out_wxyz[..., 0]], dim=-1)
                        quats_try = quats_try.clone()
                        quats_try[moving_mask] = q_out_xyzw / (q_out_xyzw.norm(dim=-1, keepdim=True) + 1e-12)

                    xys_d, depths_d, radii_d, conics_d, comp_d, num_tiles_hit_d, _ = project_gaussians(
                        xyz_try,
                        scales_all,
                        1,
                        quats_try,
                        view3x4,
                        float(fx),
                        float(fy),
                        float(cx),
                        float(cy),
                        H,
                        W,
                        block_width,
                    )
                    if (radii_d.sum()).item() == 0:
                        continue
                    cam_center = torch.tensor(c2w_i[:3, 3], dtype=torch.float32, device=device)
                    viewdirs = xyz_try - cam_center
                    if auto_sh_degree <= 0 or colors_all.shape[1] <= 1:
                        rgbs_d = torch.sigmoid(features_dc)
                    else:
                        rgbs_d = spherical_harmonics(int(auto_sh_degree), viewdirs, colors_all)
                        rgbs_d = torch.clamp(rgbs_d + 0.5, min=0.0)
                    opacities_d = opacities_all * comp_d[:, None] if rasterize_mode == "antialiased" else opacities_all
                    rgb_d = rasterize_gaussians(
                        xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d, rgbs_d, opacities_d, H, W, block_width, background=background,
                    )
                    pred_d = torch.clamp(rgb_d, max=1.0)
                    if mask_t is not None:
                        denom = mask_t.sum()
                        if denom.item() < 0.5:
                            err = float(torch.mean((pred_d - gt_t) ** 2).item())
                        else:
                            diff = (pred_d - gt_t) * mask_t
                            err = (diff.square().sum() / (3.0 * denom)).item()
                    else:
                        err = float(torch.mean((pred_d - gt_t) ** 2).item())

                    if best_err is None or err < best_err:
                        best_err = err
                        best_xyz = xyz_try
                        best_quats = quats_try

                if best_xyz is not None:
                    xyz_frame = best_xyz
                    quats_all = best_quats

        xys, depths, radii, conics, comp, num_tiles_hit, _ = project_gaussians(
            xyz_frame, scales_all, 1, quats_frame, view3x4, float(fx), float(fy), float(cx), float(cy), H, W, block_width,
        )
        if (radii.sum()).item() == 0:
            continue

        cam_center = torch.tensor(c2w_i[:3, 3], dtype=torch.float32, device=device)
        viewdirs = xyz_frame - cam_center
        if auto_sh_degree <= 0 or colors_all.shape[1] <= 1:
            rgbs = torch.sigmoid(features_dc)
        else:
            rgbs = spherical_harmonics(int(auto_sh_degree), viewdirs, colors_all)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)

        opacities = opacities_all * comp[:, None] if rasterize_mode == "antialiased" else opacities_all
        rgb = rasterize_gaussians(
            xys, depths, radii, conics, num_tiles_hit, rgbs, opacities, H, W, block_width, background=background,
        )
        pred = torch.clamp(rgb, max=1.0).detach().cpu().numpy()
        pred = (pred * 255.0 + 0.5).astype(np.uint8)

        if save_pred_dir is not None:
            image_id = Path(rgb_paths[i]).stem
            out_path = save_pred_dir / f"{image_id}.png"
            iio.imwrite(str(out_path), pred)

        gt = iio.imread(rgb_paths[i])
        if gt.ndim == 2:
            gt = np.repeat(gt[..., None], 3, axis=2)
        if gt.shape[-1] > 3:
            gt = gt[..., :3]
        if gt.shape[1] != W or gt.shape[0] != H:
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_AREA)

        gt_save = gt
        if gt_save.dtype != np.uint8:
            gt_float = gt_save.astype(np.float32)
            if gt_float.max() <= 1.5:
                gt_float = gt_float * 255.0
            gt_save = np.clip(gt_float + 0.5, 0, 255).astype(np.uint8)
        if save_gt_dir is not None and save_compare_dir is not None:
            image_id = Path(rgb_paths[i]).stem
            gt_path = save_gt_dir / f"{image_id}.png"
            compare_path = save_compare_dir / f"{image_id}.png"
            compare = np.concatenate([gt_save, pred], axis=1)
            iio.imwrite(str(gt_path), gt_save)
            iio.imwrite(str(compare_path), compare)

        stem = Path(rgb_paths[i]).stem
        mask_path_png = Path(eval_dir) / "masks" / f"{stem}.png"
        mask_img = iio.imread(str(mask_path_png)) if mask_path_png.exists() else None
        if mask_img is not None:
            if mask_img.ndim == 3 and mask_img.shape[2] > 1:
                mask_img = mask_img[:, :, 0]
            if mask_img.shape[0] != H or mask_img.shape[1] != W:
                mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
            psnr = _compute_psnr_masked(pred, gt, mask_img)
            psnr_list.append(_compute_psnr(pred, gt) if psnr is None else psnr)
        else:
            psnr_list.append(_compute_psnr(pred, gt))

    if not psnr_list:
        return {"psnr": None, "frames": 0}
    result = {"psnr": float(np.mean(psnr_list)), "frames": len(psnr_list)}
    if joint_alignment_info is not None:
        result["joint_alignment"] = joint_alignment_info
    return result
