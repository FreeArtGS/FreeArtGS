import argparse
import importlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    import open3d as o3d  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Please install open3d: pip install open3d") from e


from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
GSPLAT_AVAILABLE = True



C0 = 0.28209479177387814  # SH DC constant


def read_ply_gaussians(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read positions and colors from 3DGS-compatible PLY.

    Prefers SH DC coefficients (f_dc_0..2) if available; otherwise, falls back to 'colors' or white.
    Returns (N,3) float32 positions and (N,3) float32 colors in [0,1].
    """
    import plyfile  # lazy import

    ply = plyfile.PlyData.read(str(ply_path))
    v = ply.elements[0].data
    names = list(v.dtype.names)

    def get_col(name: str) -> Optional[np.ndarray]:
        if name in v.dtype.names:
            return np.asarray(v[name])
        return None

    x = get_col("x"); y = get_col("y"); z = get_col("z")
    if x is None or y is None or z is None:
        raise RuntimeError("PLY missing x/y/z")
    positions = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Try SH DC first
    f0 = get_col("f_dc_0"); f1 = get_col("f_dc_1"); f2 = get_col("f_dc_2")
    if f0 is not None and f1 is not None and f2 is not None:
        sh_dc = np.stack([f0, f1, f2], axis=-1).astype(np.float32)
        colors = sh_dc * C0 + 0.5
        colors = np.clip(colors, 0.0, 1.0)
    else:
        # Fallback to uint8 colors
        col = get_col("colors")
        if col is not None:
            colors = (col.astype(np.float32) / 255.0).reshape(-1, 3)
        else:
            colors = np.ones_like(positions, dtype=np.float32) * 0.8

    return positions.astype(np.float32), colors.astype(np.float32)


def read_part_weights(txt_path: Path) -> Optional[np.ndarray]:
    if not txt_path.exists():
        return None
    w = np.loadtxt(str(txt_path)).astype(np.float32)
    if w.ndim == 0:
        w = np.array([float(w)], dtype=np.float32)
    return w.reshape(-1)


def read_articulation(json_path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    axis = np.asarray(meta.get("joint_axis_world", meta.get("joint_axis", [0, 0, 1])), dtype=np.float32)
    origin = np.asarray(meta.get("joint_origin_world", meta.get("joint_origin", [0, 0, 0])), dtype=np.float32)
    joint_type_raw = str(meta.get("joint_type", "revolute")).strip().lower()
    jt_map = {
        "hinge": "revolute",
        "rotational": "revolute",
        "rotation": "revolute",
        "rev": "revolute",
        "slider": "prismatic",
        "linear": "prismatic",
        "translate": "prismatic",
        "translation": "prismatic",
        "translational": "prismatic",
        "pris": "prismatic",
    }
    joint_type = jt_map.get(joint_type_raw, joint_type_raw)
    n = np.linalg.norm(axis)
    axis = axis if n < 1e-8 else axis / n
    return axis, origin, joint_type


def rodrigues_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """3x3 rotation matrix about 'axis' (unit) with Rodrigues' formula."""
    ax = axis.astype(np.float32)
    ax = ax / (np.linalg.norm(ax) + 1e-8)
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


def _axis_angle_to_quat_xyzw(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Return quaternion (xyzw) for rotation storing axis-angle on given device."""
    assert axis.shape == (3,)
    axis_norm = torch.linalg.norm(axis)
    if torch.isnan(axis_norm) or axis_norm < 1e-8:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=axis.dtype, device=axis.device)
    axis_unit = axis / axis_norm.clamp_min(1e-8)
    half = angle * 0.5
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)
    return torch.stack([axis_unit[0] * sin_half, axis_unit[1] * sin_half, axis_unit[2] * sin_half, cos_half], dim=0)


def _quat_mul_xyzw(q_left: torch.Tensor, q_right: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication (xyzw format): result = q_left * q_right."""
    if q_left.ndim == 1:
        q_left = q_left.unsqueeze(0)
    if q_right.ndim == 1:
        q_right = q_right.unsqueeze(0)
    x1, y1, z1, w1 = q_left.unbind(dim=-1)
    x2, y2, z2, w2 = q_right.unbind(dim=-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    out = torch.stack([x, y, z, w], dim=-1)
    if out.shape[0] == 1:
        return out.squeeze(0)
    return out


class ArticulationViewer:
    def __init__(
        self,
        input_dir: Path,
        threshold: float = 0.5,
        motion_part: int = 1,
        step_deg: float = 2.0,
        step_trans: Optional[float] = None,
        soft: bool = False,
        use_gsplat_render: bool = False,
        record: bool = False,
        record_interval: float = 1.0,
    ) -> None:
        self.input_dir = input_dir
        self.threshold = float(threshold)
        self.motion_part = int(motion_part)
        self.step_deg = float(step_deg)
        self.soft = bool(soft)
        self.use_gsplat_render = bool(use_gsplat_render) and GSPLAT_AVAILABLE
        self.record_enabled = bool(record)
        self.record_interval = float(record_interval)
        self._last_record_time: Optional[float] = None

        # ensure nerfstudio utilities accessible for gaussian export
        module_name = "nerfstudio.data.utils.dataparsers_utils"
        try:
            ns_utils = importlib.import_module(module_name)
        except ModuleNotFoundError:
            recon_root = Path(__file__).resolve().parent
            if str(recon_root) not in sys.path:
                sys.path.append(str(recon_root))
            ns_utils = importlib.import_module(module_name)
        self._load_3dgs_ply = getattr(ns_utils, "load_3dgs_ply")
        self._save_gaussians_to_ply = getattr(ns_utils, "save_gaussians_to_ply")

        # export bookkeeping
        self.export_root = self.input_dir / "viewer_exports"
        self.export_images = self.export_root / "screenshots"
        self.export_gaussians = self.export_root / "gaussians"
        self.export_record_gs = self.export_root / "record_gsplat"
        for path in (self.export_root, self.export_images, self.export_gaussians, self.export_record_gs):
            path.mkdir(parents=True, exist_ok=True)
        self._save_counter = 0

        ply_path = input_dir / "object_3dgs.ply"
        art_json = input_dir / "articulation.json"
        weights_path = input_dir / "3dgs_part_weight.txt"
        if not ply_path.exists():
            raise FileNotFoundError(f"Missing PLY: {ply_path}")
        if not art_json.exists():
            raise FileNotFoundError(f"Missing articulation.json: {art_json}")

        pts, cols = read_ply_gaussians(ply_path)
        self.points0 = pts  # [N,3]
        self.colors = cols
        self.axis, self.origin, self.joint_type = read_articulation(art_json)
        self.weights = read_part_weights(weights_path)
        if self.weights is None or self.weights.shape[0] != self.points0.shape[0]:
            # fallback: all move if motion_part==1; none move if 0
            base = 1.0 if self.motion_part == 1 else 0.0
            self.weights = np.full((self.points0.shape[0],), base, dtype=np.float32)
        self.part_mask = self.weights >= 0.5

        self._refine_axis_origin()
        # Save initial values for comparison
        self.axis_init = self.axis.copy()
        self.origin_init = self.origin.copy()

        # cache gaussian parameters for export
        self.gauss_base = self._load_gaussians(ply_path)

        # arrow orientation tracking (True -> origin -axis-> origin+axis)
        self.axis_arrow_forward = True

        # default translation step for prismatic joints based on scene scale
        bbox_min = self.points0.min(0)
        bbox_max = self.points0.max(0)
        diag_len = float(np.linalg.norm(bbox_max - bbox_min))
        if diag_len < 1e-6:
            diag_len = 1.0
        self.step_trans = float(step_trans) if step_trans is not None else 0.02 * diag_len
        self.axis_step = 0.002 * diag_len
        self.axis_rotate_step = 0.1
        self.axis_radius_scale = 0.03
        self.axis_head_scale = 0.2
        self.axis_head_ratio = 2.5

        self.angle_deg = 0.0
        self._build_scene()

    def _build_scene(self) -> None:
        # point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points0)
        self.pcd.colors = o3d.utility.Vector3dVector(self.colors)

        # axis indicator arrow
        length = float(np.linalg.norm(self.points0.max(0) - self.points0.min(0)))
        if length < 1e-6:
            length = 1.0
        self.scene_center = (self.points0.max(0) + self.points0.min(0)) * 0.5
        p0, p1 = self._arrow_endpoints(length)
        radius = max(self.axis_radius_scale * length, length * 0.005)
        self.axis_arrow = self._create_arrow_mesh(p0, p1, radius)

        # world coordinate frame (origin and axes)
        coord_size = 0.1
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_size, origin=[0.0, 0.0, 0.0])
        # explicit origin marker at world origin
        self.origin_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02 * length)
        self.origin_marker.paint_uniform_color([1.0, 1.0, 0.0])  # yellow
        # (origin is already at [0,0,0])

    def _load_gaussians(self, ply_path: Path) -> Optional[Dict[str, torch.Tensor]]:
        try:
            gauss = self._load_3dgs_ply(ply_path, device=torch.device("cpu"), rest_format="auto")
        except Exception as exc:
            print(f"[warn] Failed to load gaussian parameters for export: {exc}")
            return None
        out: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for key, val in gauss.items():
                if isinstance(val, torch.Tensor):
                    out[key] = val.detach().clone().cpu()
                else:  # should not happen, but keep reference
                    out[key] = torch.as_tensor(val, dtype=torch.float32)
        return out

    def _refine_axis_origin(self) -> None:
        axis_norm = float(np.linalg.norm(self.axis))
        if axis_norm < 1e-8 or self.points0.shape[0] == 0:
            return
        axis_unit = self.axis / axis_norm
        pts = self.points0
        proj = pts @ axis_unit
        if self.joint_type == "prismatic":
            perp = pts - np.outer(proj, axis_unit)
            mean_perp = perp.mean(axis=0)
            mean_proj = proj.mean()
            self.origin = (mean_perp + mean_proj * axis_unit).astype(np.float32)
        elif self.joint_type == "revolute":
            mean_proj = proj.mean()
            current_proj = float(self.origin @ axis_unit)
            delta = (mean_proj - current_proj) * axis_unit
            self.origin = (self.origin + delta).astype(np.float32)
        self.axis = axis_unit.astype(np.float32)

    def _print_axis_diff(self) -> None:
        """Calculate and print the difference between current axis/origin and initial ones."""
        # Angle difference
        dot = np.clip(np.dot(self.axis, self.axis_init), -1.0, 1.0)
        angle_diff = np.rad2deg(np.arccos(dot))

        # Shortest distance between two lines in 3D:
        # L1: P1 + t*D1 (Initial)
        # L2: P2 + s*D2 (Current)
        p1, d1 = self.origin_init, self.axis_init
        p2, d2 = self.origin, self.axis

        cross = np.cross(d1, d2)
        norm_cross = np.linalg.norm(cross)

        if norm_cross < 1e-8:
            # Parallel lines: dist = |(P2-P1) x D1| / |D1|
            dist = np.linalg.norm(np.cross(p2 - p1, d1)) / (np.linalg.norm(d1) + 1e-12)
        else:
            # Skew lines: dist = |(P2-P1) . (D1 x D2)| / |D1 x D2|
            dist = abs(np.dot(p2 - p1, cross)) / norm_cross

        print(f"[diff] Angle: {angle_diff:.4f}°, Dist: {dist:.6f}")

    @staticmethod
    def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
        d = direction.astype(np.float64)
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-8:
            return np.eye(3, dtype=np.float64)
        d = d / norm_d
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.allclose(d, z, atol=1e-6):
            return np.eye(3, dtype=np.float64)
        if np.allclose(d, -z, atol=1e-6):
            return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
        v = np.cross(z, d)
        s = np.linalg.norm(v)
        c = float(z @ d)
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float64,
        )
        r = np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))
        return r

    def _arrow_endpoints(self, length: float) -> Tuple[np.ndarray, np.ndarray]:
        half = 0.6 * length
        if self.axis_arrow_forward:
            start = self.origin - self.axis * half
            end = self.origin + self.axis * half
        else:
            start = self.origin + self.axis * half
            end = self.origin - self.axis * half
        return start.astype(np.float32), end.astype(np.float32)

    def _create_arrow_mesh(self, start: np.ndarray, end: np.ndarray, radius: float) -> o3d.geometry.TriangleMesh:
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            length = 1.0
        base_radius = max(radius, length * 0.005)
        head_frac = float(np.clip(self.axis_head_scale, 0.05, 0.6))
        cone = max(length * head_frac, length * 0.01)
        cone = min(cone, length * 0.9)
        cyl = max(length - cone, length * 0.02)
        cone_radius = max(base_radius * float(np.clip(self.axis_head_ratio, 1.1, 6.0)), base_radius * 1.05)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=max(base_radius, length * 0.01),
            cone_radius=cone_radius,
            cylinder_height=cyl,
            cone_height=cone,
        )
        arrow.compute_vertex_normals()
        rot = self._rotation_from_z(direction)
        arrow.rotate(rot, center=np.zeros(3))
        # shift so base aligns with start
        arrow.translate(start)
        arrow.paint_uniform_color([0.1, 0.9, 0.1])
        return arrow

    def _current_gaussians(self) -> Dict[str, torch.Tensor]:
        if self.gauss_base is None:
            raise RuntimeError("Gaussian parameters unavailable for export.")
        gauss: Dict[str, torch.Tensor] = {k: v.clone() for k, v in self.gauss_base.items()}
        means = gauss["means"]
        quats = gauss["quats"]

        axis_t = torch.tensor(self.axis, dtype=means.dtype, device=means.device)
        origin_t = torch.tensor(self.origin, dtype=means.dtype, device=means.device)
        weights_np = self._mask_for_motion()
        weights_t = torch.as_tensor(weights_np, dtype=means.dtype, device=means.device)

        if self.joint_type == "revolute":
            angle_rad_value = float(np.deg2rad(self.angle_deg))
            if abs(angle_rad_value) > 1e-8:
                if self.soft:
                    for idx in range(means.shape[0]):
                        wi = float(weights_np[idx])
                        if wi <= 1e-6:
                            continue
                        local_angle = angle_rad_value * wi
                        if abs(local_angle) <= 1e-8:
                            continue
                        R_np = rodrigues_rotation_matrix(self.axis, local_angle)
                        R = torch.tensor(R_np, dtype=means.dtype, device=means.device)
                        shifted = means[idx] - origin_t
                        means[idx] = (R @ shifted) + origin_t
                        quat_delta = _axis_angle_to_quat_xyzw(axis_t, torch.tensor(local_angle, dtype=means.dtype, device=means.device))
                        quats[idx] = _quat_mul_xyzw(quat_delta, quats[idx])
                else:
                    mask = weights_t > 0.5
                    if mask.any():
                        R_np = rodrigues_rotation_matrix(self.axis, angle_rad_value)
                        R = torch.tensor(R_np, dtype=means.dtype, device=means.device)
                        shifted = means[mask] - origin_t
                        means[mask] = (shifted @ R.T) + origin_t
                        delta = _axis_angle_to_quat_xyzw(axis_t, torch.tensor(angle_rad_value, dtype=means.dtype, device=means.device)).unsqueeze(0)
                        quats_sel = quats[mask]
                        quats[mask] = _quat_mul_xyzw(delta.expand_as(quats_sel), quats_sel)

        elif self.joint_type == "prismatic":
            dist_value = float(self.angle_deg)
            if abs(dist_value) > 1e-8:
                axis_norm = torch.linalg.norm(axis_t)
                if axis_norm > 1e-8:
                    axis_unit = axis_t / axis_norm
                    if self.soft:
                        disp = axis_unit[None, :] * (weights_t[:, None] * dist_value)
                        means = means + disp
                    else:
                        mask = weights_t > 0.5
                        if mask.any():
                            disp = axis_unit * dist_value
                            means[mask] = means[mask] + disp
                gauss["means"] = means

        gauss["means"] = means
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gauss["quats"] = quats
        return gauss

    def _capture_snapshot(self, vis: o3d.visualization.Visualizer) -> None:
        tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._save_counter:03d}"
        img_path = self.export_images / f"{tag}.png"
        img_part_path = self.export_images / f"{tag}_parts.png"
        ply_path = self.export_gaussians / f"{tag}.ply"
        meta_path = self.export_root / f"{tag}.json"

        vis.capture_screen_image(str(img_path), do_render=True)
        print(f"[info] Saved screenshot to {img_path}")

        original_colors = np.asarray(self.pcd.colors).copy()
        part_colors = self._part_highlight_colors()
        self.pcd.colors = o3d.utility.Vector3dVector(part_colors)
        vis.update_geometry(self.pcd)
        vis.update_renderer()
        vis.capture_screen_image(str(img_part_path), do_render=True)
        print(f"[info] Saved part-highlight screenshot to {img_part_path}")
        self.pcd.colors = o3d.utility.Vector3dVector(original_colors)
        vis.update_geometry(self.pcd)
        vis.update_renderer()

        # GSplat rendering if enabled
        gsplat_render_path = None
        if self.use_gsplat_render and self.gauss_base is not None:
            try:
                gauss = self._current_gaussians()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Move to device
                gauss_render = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gauss.items()}

                params = vis.get_view_control().convert_to_pinhole_camera_parameters()
                intrinsic = np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64)
                extrinsic = np.asarray(params.extrinsic, dtype=np.float64)  # w2c

                H = params.intrinsic.height
                W = params.intrinsic.width
                fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                cx, cy = intrinsic[0, 2], intrinsic[1, 2]

                # Use w2c directly as the view matrix; do not flip or invert it again.
                viewmat = extrinsic
                view3x4 = torch.tensor(viewmat[:3, :], dtype=torch.float32, device=device)
                # Prepare gaussians
                means = gauss_render["means"]
                scales = gauss_render["scales"]
                quats = gauss_render["quats"]
                features_dc = gauss_render["features_dc"]
                features_rest = gauss_render["features_rest"]
                opacities = gauss_render["opacities"]

                colors_all = torch.cat((features_dc[:, None, :], features_rest), dim=1)
                opacities_all = torch.sigmoid(opacities)
                scales_all = torch.exp(scales)
                quats_all = quats / (quats.norm(dim=-1, keepdim=True) + 1e-12)

                # Project and rasterize
                BLOCK_WIDTH = 16
                background = torch.ones(3, device=device, dtype=torch.float32)
                xys, depths, radii, conics, comp, num_tiles_hit, _ = project_gaussians(
                    means, scales_all, 1, quats_all, view3x4, fx, fy, cx, cy, H, W, BLOCK_WIDTH
                )
                cam_center = torch.tensor(extrinsic[:3, 3], dtype=torch.float32, device=device)
                viewdirs = means - cam_center
                # Assume SH degree 3 or auto-detect
                total_bases = colors_all.shape[1]
                sh_degree = int(round(math.sqrt(total_bases) - 1)) if total_bases > 1 else 0
                if sh_degree > 0:
                    rgbs = spherical_harmonics(sh_degree, viewdirs, colors_all)
                    rgbs = torch.clamp(rgbs + 0.5, min=0.0)
                else:
                    rgbs = torch.sigmoid(features_dc)
                opacities_render = opacities_all * comp[:, None]
                rgb = rasterize_gaussians(
                    xys, depths, radii, conics, num_tiles_hit, rgbs, opacities_render, H, W, BLOCK_WIDTH, background=background
                )
                pred = torch.clamp(rgb, max=1.0).detach().cpu().numpy()
                pred_img = (pred * 255.0).astype(np.uint8)
                gsplat_img_path = self.export_images / f"{tag}_gsplat.png"
                import imageio.v3 as iio
                iio.imwrite(str(gsplat_img_path), pred_img)
                print(f"[info] Saved GSplat render to {gsplat_img_path}")
                gsplat_render_path = gsplat_img_path.name
            except Exception as exc:
                print(f"[warn] GSplat rendering failed: {exc}")

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "angle_deg": float(self.angle_deg),
            "joint_type": self.joint_type,
            "joint_axis": self.axis.tolist(),
            "joint_origin": self.origin.tolist(),
            "soft_mode": bool(self.soft),
            "motion_part": int(self.motion_part),
            "threshold": float(self.threshold),
            "screenshot": img_path.name,
            "screenshot_parts": img_part_path.name,
            "gaussian_ply": None,
            "gsplat_render": gsplat_render_path,
        }
        self._save_counter += 1

    def _capture_gsplat_only(self, vis: o3d.visualization.Visualizer, out_dir: Optional[Path] = None) -> None:
        tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._save_counter:03d}"
        target_dir = out_dir if out_dir is not None else self.export_record_gs
        if self.use_gsplat_render and self.gauss_base is not None:
            try:
                gauss = self._current_gaussians()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                gauss_render = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gauss.items()}

                params = vis.get_view_control().convert_to_pinhole_camera_parameters()
                intrinsic = np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64)
                extrinsic = np.asarray(params.extrinsic, dtype=np.float64)

                H = params.intrinsic.height
                W = params.intrinsic.width
                fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                cx, cy = intrinsic[0, 2], intrinsic[1, 2]

                viewmat = extrinsic
                view3x4 = torch.tensor(viewmat[:3, :], dtype=torch.float32, device=device)

                means = gauss_render["means"]
                scales = gauss_render["scales"]
                quats = gauss_render["quats"]
                features_dc = gauss_render["features_dc"]
                features_rest = gauss_render["features_rest"]
                opacities = gauss_render["opacities"]

                colors_all = torch.cat((features_dc[:, None, :], features_rest), dim=1)
                opacities_all = torch.sigmoid(opacities)
                scales_all = torch.exp(scales)
                quats_all = quats / (quats.norm(dim=-1, keepdim=True) + 1e-12)

                BLOCK_WIDTH = 16
                background = torch.ones(3, device=device, dtype=torch.float32)
                xys, depths, radii, conics, comp, num_tiles_hit, _ = project_gaussians(
                    means, scales_all, 1, quats_all, view3x4, fx, fy, cx, cy, H, W, BLOCK_WIDTH
                )
                cam_center = torch.tensor(extrinsic[:3, 3], dtype=torch.float32, device=device)
                viewdirs = means - cam_center
                total_bases = colors_all.shape[1]
                sh_degree = int(round(math.sqrt(total_bases) - 1)) if total_bases > 1 else 0
                if sh_degree > 0:
                    rgbs = spherical_harmonics(sh_degree, viewdirs, colors_all)
                    rgbs = torch.clamp(rgbs + 0.5, min=0.0)
                else:
                    rgbs = torch.sigmoid(features_dc)
                opacities_render = opacities_all * comp[:, None]
                rgb = rasterize_gaussians(
                    xys, depths, radii, conics, num_tiles_hit, rgbs, opacities_render, H, W, BLOCK_WIDTH, background=background
                )
                pred = torch.clamp(rgb, max=1.0).detach().cpu().numpy()
                pred_img = (pred * 255.0).astype(np.uint8)
                gsplat_img_path = target_dir / f"{tag}_gsplat.png"
                import imageio.v3 as iio
                iio.imwrite(str(gsplat_img_path), pred_img)
                print(f"[info] Saved GSplat render to {gsplat_img_path}")
                self._save_counter += 1
            except Exception as exc:
                print(f"[warn] GSplat rendering failed: {exc}")

    def _mask_for_motion(self) -> np.ndarray:
        w = self.weights.copy()
        if self.motion_part == 0:
            w = 1.0 - w
        if self.soft:
            return w  # [0,1] weights for soft displacement
        return (w >= self.threshold).astype(np.float32)

    def _transform_points(self) -> np.ndarray:
        pts = self.points0
        w = self._mask_for_motion()
        if self.joint_type == "revolute":
            angle = np.deg2rad(self.angle_deg)
            if not self.soft:
                mask = w > 0.5
                if not np.any(mask):
                    return pts
                R = rodrigues_rotation_matrix(self.axis, angle)
                shifted = pts[mask] - self.origin[None, :]
                rotated = (R @ shifted.T).T + self.origin[None, :]
                out = pts.copy()
                out[mask] = rotated
                return out
            else:
                out = pts.copy()
                shifted = pts - self.origin[None, :]
                for i in range(out.shape[0]):
                    wi = float(w[i])
                    if wi <= 1e-6:
                        continue
                    Ri = rodrigues_rotation_matrix(self.axis, angle * wi)
                    out[i] = (Ri @ shifted[i]) + self.origin
                return out
        elif self.joint_type == "prismatic":
            dist = self.angle_deg
            if not self.soft:
                mask = w > 0.5
                if not np.any(mask):
                    return pts
                t = self.axis * dist
                out = pts.copy()
                out[mask] = pts[mask] + t[None, :]
                return out
            else:
                # per-point displacement
                disp = (dist * w)[:, None] * self.axis[None, :]
                return pts + disp
        else:
            return pts

    def _part_highlight_colors(self) -> np.ndarray:
        colors = np.zeros((self.points0.shape[0], 3), dtype=np.float32)
        colors[~self.part_mask] = np.array([0.2, 0.2, 1.0], dtype=np.float32)
        colors[self.part_mask] = np.array([1.0, 0.1, 0.1], dtype=np.float32)
        return colors

    def _update_axis_arrow(self, vis: o3d.visualization.Visualizer) -> None:
        # remove existing arrow and rebuild with updated origin/axis
        vis.remove_geometry(self.axis_arrow, reset_bounding_box=False)
        length = float(np.linalg.norm(self.points0.max(0) - self.points0.min(0)))
        if length < 1e-6:
            length = 1.0
        p0, p1 = self._arrow_endpoints(length)
        radius = max(self.axis_radius_scale * length, length * 0.005)
        self.axis_arrow = self._create_arrow_mesh(p0, p1, radius)
        vis.add_geometry(self.axis_arrow, reset_bounding_box=False)
        vis.update_geometry(self.axis_arrow)

    def _rotate_axis(self, axis_of_rotation: np.ndarray, angle_rad: float) -> None:
        R = rodrigues_rotation_matrix(axis_of_rotation, angle_rad)
        self.axis = (R @ self.axis).astype(np.float32)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self._print_axis_diff()

    def run(self) -> None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Articulation Viewer", width=1280, height=720)
        vis.add_geometry(self.pcd)
        vis.add_geometry(self.axis_arrow)
        vis.add_geometry(self.coord_frame)
        vis.add_geometry(self.origin_marker)

        ctr = vis.get_view_control()
        if hasattr(self, "scene_center"):
            ctr.set_lookat(self.scene_center.tolist())

        def refresh():
            new_pts = self._transform_points()
            self.pcd.points = o3d.utility.Vector3dVector(new_pts)
            vis.update_geometry(self.pcd)
            vis.update_renderer()

        def refresh_axis(vis_):
            self._update_axis_arrow(vis_)
            vis_.update_renderer()

        def move_origin(vis_, direction: np.ndarray) -> bool:
            delta = direction * self.axis_step
            self.origin = (self.origin + delta).astype(np.float32)
            refresh_axis(vis_)
            refresh()
            self._print_axis_diff()
            return True

        def cb_axis_forward(vis_):
            self.origin = (self.origin + self.axis * self.axis_step).astype(np.float32)
            refresh_axis(vis_)
            refresh()
            self._print_axis_diff()
            return True

        def cb_axis_backward(vis_):
            self.origin = (self.origin - self.axis * self.axis_step).astype(np.float32)
            refresh_axis(vis_)
            refresh()
            self._print_axis_diff()
            return True

        def cb_axis_radius_up(vis_):
            self.axis_radius_scale = min(self.axis_radius_scale * 1.2, 0.5)
            print(f"[info] axis_radius_scale -> {self.axis_radius_scale:.5f}")
            refresh_axis(vis_)
            return True

        def cb_axis_radius_down(vis_):
            self.axis_radius_scale = max(self.axis_radius_scale / 1.2, 0.002)
            print(f"[info] axis_radius_scale -> {self.axis_radius_scale:.5f}")
            refresh_axis(vis_)
            return True

        def cb_axis_head_up(vis_):
            self.axis_head_scale = min(self.axis_head_scale * 1.2, 0.6)
            print(f"[info] axis_head_scale -> {self.axis_head_scale:.5f}")
            refresh_axis(vis_)
            return True

        def cb_axis_head_down(vis_):
            self.axis_head_scale = max(self.axis_head_scale / 1.2, 0.05)
            print(f"[info] axis_head_scale -> {self.axis_head_scale:.5f}")
            refresh_axis(vis_)
            return True

        def cb_axis_head_ratio_up(vis_):
            self.axis_head_ratio = min(self.axis_head_ratio * 1.2, 6.0)
            print(f"[info] axis_head_ratio -> {self.axis_head_ratio:.5f}")
            refresh_axis(vis_)
            return True

        def cb_axis_head_ratio_down(vis_):
            self.axis_head_ratio = max(self.axis_head_ratio / 1.2, 1.05)
            print(f"[info] axis_head_ratio -> {self.axis_head_ratio:.5f}")
            refresh_axis(vis_)
            return True

        def cb_axis_flip(vis_):
            self.axis_arrow_forward = not self.axis_arrow_forward
            print(f"[info] axis_arrow_forward -> {self.axis_arrow_forward}")
            refresh_axis(vis_)
            return True

        def cb_axis_move_x_neg(vis_):
            return move_origin(vis_, np.array([-1.0, 0.0, 0.0], dtype=np.float32))

        def cb_axis_move_x_pos(vis_):
            return move_origin(vis_, np.array([1.0, 0.0, 0.0], dtype=np.float32))

        def cb_axis_move_y_neg(vis_):
            return move_origin(vis_, np.array([0.0, -1.0, 0.0], dtype=np.float32))

        def cb_axis_move_y_pos(vis_):
            return move_origin(vis_, np.array([0.0, 1.0, 0.0], dtype=np.float32))

        def cb_axis_move_z_neg(vis_):
            return move_origin(vis_, np.array([0.0, 0.0, -1.0], dtype=np.float32))

        def cb_axis_move_z_pos(vis_):
            return move_origin(vis_, np.array([0.0, 0.0, 1.0], dtype=np.float32))

        def cb_axis_step_up(vis_):
            self.axis_step *= 1.5
            print(f"[info] axis_step -> {self.axis_step:.6f}")
            return True

        def cb_axis_step_down(vis_):
            self.axis_step = max(self.axis_step / 1.5, 1e-5)
            print(f"[info] axis_step -> {self.axis_step:.6f}")
            return True

        def cb_inc(vis_):
            step = self.step_deg if self.joint_type == "revolute" else self.step_trans
            self.angle_deg += step
            refresh()
            return True

        def cb_dec(vis_):
            step = self.step_deg if self.joint_type == "revolute" else self.step_trans
            self.angle_deg -= step
            refresh()
            return True

        def cb_reset(vis_):
            self.angle_deg = 0.0
            refresh()
            return True

        def cb_toggle_part(vis_):
            self.motion_part = 1 - self.motion_part
            refresh()
            return True

        def cb_thresh_up(vis_):
            if self.soft:
                return True
            self.threshold = min(0.99, self.threshold + 0.05)
            refresh()
            return True

        def cb_thresh_down(vis_):
            if self.soft:
                return True
            self.threshold = max(0.01, self.threshold - 0.05)
            refresh()
            return True

        def cb_toggle_soft(vis_):
            self.soft = not self.soft
            refresh()
            return True

        def cb_save(vis_):
            refresh()
            self._capture_snapshot(vis_)
            return True

        def cb_rotate_axis_x_pos(vis_):
            self._rotate_axis(np.array([1.0, 0.0, 0.0]), np.deg2rad(self.axis_rotate_step))
            refresh_axis(vis_)
            refresh()
            return True

        def cb_rotate_axis_x_neg(vis_):
            self._rotate_axis(np.array([1.0, 0.0, 0.0]), np.deg2rad(-self.axis_rotate_step))
            refresh_axis(vis_)
            refresh()
            return True

        def cb_rotate_axis_y_pos(vis_):
            self._rotate_axis(np.array([0.0, 1.0, 0.0]), np.deg2rad(self.axis_rotate_step))
            refresh_axis(vis_)
            refresh()
            return True

        def cb_rotate_axis_y_neg(vis_):
            self._rotate_axis(np.array([0.0, 1.0, 0.0]), np.deg2rad(-self.axis_rotate_step))
            refresh_axis(vis_)
            refresh()
            return True

        def cb_rotate_axis_z_pos(vis_):
            self._rotate_axis(np.array([0.0, 0.0, 1.0]), np.deg2rad(self.axis_rotate_step))
            refresh_axis(vis_)
            refresh()
            return True

        def cb_rotate_axis_z_neg(vis_):
            self._rotate_axis(np.array([0.0, 0.0, 1.0]), np.deg2rad(-self.axis_rotate_step))
            refresh_axis(vis_)
            refresh()
            return True

        for k in ("D", "d"):
            vis.register_key_callback(ord(k), cb_inc)
        for k in ("A", "a"):
            vis.register_key_callback(ord(k), cb_dec)
        for k in ("R", "r"):
            vis.register_key_callback(ord(k), cb_reset)
        for k in ("M", "m"):
            vis.register_key_callback(ord(k), cb_toggle_part)
        for k in ("K", "k"):
            vis.register_key_callback(ord(k), cb_thresh_up)
        for k in ("J", "j"):
            vis.register_key_callback(ord(k), cb_thresh_down)
        for k in ("S", "s"):
            vis.register_key_callback(ord(k), cb_toggle_soft)
        for k in ("P", "p"):
            vis.register_key_callback(ord(k), cb_save)
        for k in (",", "<"):
            vis.register_key_callback(ord(k), cb_axis_backward)
        for k in (".", ">"):
            vis.register_key_callback(ord(k), cb_axis_forward)
        vis.register_key_callback(ord("["), cb_axis_step_down)
        vis.register_key_callback(ord("]"), cb_axis_step_up)
        vis.register_key_callback(ord("-"), cb_axis_radius_down)
        vis.register_key_callback(ord("="), cb_axis_radius_up)
        vis.register_key_callback(ord("7"), cb_axis_head_down)
        vis.register_key_callback(ord("8"), cb_axis_head_up)
        vis.register_key_callback(ord("9"), cb_axis_head_ratio_down)
        vis.register_key_callback(ord("0"), cb_axis_head_ratio_up)
        vis.register_key_callback(ord("1"), cb_axis_move_x_neg)
        vis.register_key_callback(ord("2"), cb_axis_move_x_pos)
        vis.register_key_callback(ord("3"), cb_axis_move_y_neg)
        vis.register_key_callback(ord("4"), cb_axis_move_y_pos)
        vis.register_key_callback(ord("5"), cb_axis_move_z_neg)
        vis.register_key_callback(ord("6"), cb_axis_move_z_pos)
        vis.register_key_callback(ord("F"), cb_axis_flip)
        vis.register_key_callback(ord("f"), cb_axis_flip)
        vis.register_key_callback(ord("Q"), cb_rotate_axis_x_pos)
        vis.register_key_callback(ord("W"), cb_rotate_axis_x_neg)
        vis.register_key_callback(ord("E"), cb_rotate_axis_y_pos)
        vis.register_key_callback(ord("T"), cb_rotate_axis_y_neg)
        vis.register_key_callback(ord("Y"), cb_rotate_axis_z_pos)
        vis.register_key_callback(ord("U"), cb_rotate_axis_z_neg)

        if self.record_enabled:
            self._last_record_time = time.monotonic()

            def _record_callback(vis_):
                now = time.monotonic()
                if self._last_record_time is None:
                    self._last_record_time = now
                    return False
                if now - self._last_record_time >= max(self.record_interval, 1e-3):
                    self._last_record_time = now
                    self._capture_gsplat_only(vis_, out_dir=self.export_record_gs)
                return False

            vis.register_animation_callback(_record_callback)
            print(f"[info] Recording enabled. Interval: {self.record_interval:.2f} s")

        unit = "deg" if self.joint_type == "revolute" else "units"
        step = self.step_deg if self.joint_type == "revolute" else self.step_trans
        print(f"Joint type: {self.joint_type}. Step: {step:.4f} {unit}. Axis step: {self.axis_step:.4f}")
        print("Controls: D/d(+), A/a(-), R(reset), M(toggle motion_part), J/K(threshold -/+), S(toggle soft/hard), P(save capture)")
        print("Axis controls: </,(axis back), >./(axis forward), [ (step down), ] (step up)")
        print("Axis sizing: -(thinner), =(thicker), 7(head shorter), 8(head longer), 9(head narrower), 0(head wider)")
        print("Axis flip: F/f(toggle arrow direction)")
        print("Origin controls: 1(-X), 2(+X), 3(-Y), 4(+Y), 5(-Z), 6(+Z)")
        print("Axis rotation: Q/W(X +/-), E/T(Y +/-), Y/U(Z +/-)")

        vis.run()
        vis.destroy_window()


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive visualization of 3DGS articulation")
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory with object_3dgs.ply and articulation.json")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for hard mask (ignored in soft mode)")
    ap.add_argument("--motion_part", type=int, default=0, choices=[0, 1], help="Which part moves initially")
    ap.add_argument("--step-deg", type=float, default=2.0, help="Angle step in degrees for A/D")
    ap.add_argument("--soft", action="store_true", help="Use soft per-point blending instead of hard mask")
    ap.add_argument("--use-gsplat-render", action="store_true", help="Use GSplat for rendering instead of Open3D point cloud")
    ap.add_argument("--record", action="store_true", help="Enable automatic periodic recording of views")
    ap.add_argument("--record-interval", type=float, default=0.1, help="Recording interval in seconds")
    args = ap.parse_args()

    viewer = ArticulationViewer(
        input_dir=args.input_dir,
        threshold=args.threshold,
        motion_part=args.motion_part,
        step_deg=args.step_deg,
        soft=args.soft,
        use_gsplat_render=args.use_gsplat_render,
        record=args.record,
        record_interval=args.record_interval,
    )
    viewer.run()


if __name__ == "__main__":
    main()
