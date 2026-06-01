from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

try:
    import open3d as o3d  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("Please install open3d: pip install open3d") from e


C0 = 0.28209479177387814  # SH DC constant


def _hat(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-8:
        # first-order
        return np.eye(3, dtype=np.float32) + _hat(w)
    axis = (w / theta).astype(np.float32)
    K = _hat(axis)
    s = np.sin(theta)
    c = np.cos(theta)
    R = np.eye(3, dtype=np.float32) + s * K + (1.0 - c) * (K @ K)
    return R.astype(np.float32)


def so3_log(R: np.ndarray) -> np.ndarray:
    # Clamp trace-based computation for numerical stability
    tr = float(np.trace(R))
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    # Handle near-pi case more carefully
    if abs(np.pi - theta) < 1e-4:
        # From diagonal of (R+I)/2
        A = (R + np.eye(3, dtype=np.float32)) * 0.5
        axis = np.array([
            np.sqrt(max(A[0, 0], 0.0)),
            np.sqrt(max(A[1, 1], 0.0)),
            np.sqrt(max(A[2, 2], 0.0)),
        ], dtype=np.float32)
        # Choose signs using off-diagonals
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-8:
            return np.zeros(3, dtype=np.float32)
        axis = axis / axis_norm
        return (axis * theta).astype(np.float32)
    # General case
    skew = (R - R.T) * (0.5 / np.sin(theta))
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float32)
    return (axis * theta).astype(np.float32)


def read_ply_gaussians_rgb(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read positions and colors from 3DGS-compatible PLY as RGB point cloud.

    Prefers SH DC coefficients (f_dc_0..2) if available; otherwise falls back to RGB if present,
    then finally to white.
    Returns (N,3) float32 positions and (N,3) float32 colors in [0,1].
    """
    import plyfile  # lazy import

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

    # Prefer SH DC
    f0 = get_col("f_dc_0"); f1 = get_col("f_dc_1"); f2 = get_col("f_dc_2")
    if f0 is not None and f1 is not None and f2 is not None:
        sh_dc = np.stack([f0, f1, f2], axis=-1).astype(np.float32)
        colors = np.clip(sh_dc * C0 + 0.5, 0.0, 1.0)
    else:
        # Try RGB
        r = get_col("red") or get_col("r")
        g = get_col("green") or get_col("g")
        b = get_col("blue") or get_col("b")
        if r is not None and g is not None and b is not None:
            colors = (np.stack([r, g, b], axis=-1).astype(np.float32) / 255.0)
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


def read_object_poses(obj_json: Path) -> List[np.ndarray]:
    """Read list of per-frame object transforms T_obj (3x4) from object_poses_train.json."""
    with obj_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames: List[np.ndarray] = []
    for item in data:
        T = np.asarray(item["T_obj"], dtype=np.float32)
        if T.shape == (3, 4):
            frames.append(T)
        elif T.shape == (4, 4):
            frames.append(T[:3, :])
        else:
            raise RuntimeError("T_obj must be 3x4 or 4x4")
    return frames


class TwoPartBlendViewer:
    def __init__(
        self,
        input_dir: Path,
        threshold: float = 0.5,
        motion_part: int = 1,
        soft: bool = True,
        map_dataparser: bool = False,
        invert: bool = False,
    ) -> None:
        self.input_dir = input_dir
        self.threshold = float(threshold)
        self.motion_part = int(motion_part)
        self.soft = bool(soft)
        self.apply_ns_transform = bool(map_dataparser)
        self.invert_transform = bool(invert)

        # Always load base PLY in original world coordinates
        ply_path = input_dir / "object_3dgs.ply"
        weights_path = input_dir / "3dgs_part_weight.txt"
        objposes_path = input_dir / "object_poses_train.json"
        if not ply_path.exists():
            raise FileNotFoundError(f"Missing PLY: {ply_path}")
        if not objposes_path.exists():
            raise FileNotFoundError(f"Missing object_poses_train.json: {objposes_path}")

        # Cache dataparser transform if present
        self.dp_meta = None
        dp_json = input_dir / "dataparser_transforms.json"
        if dp_json.exists():
            with dp_json.open("r", encoding="utf-8") as f:
                self.dp_meta = json.load(f)

        pts, cols = read_ply_gaussians_rgb(ply_path)
        # Optionally map raw (world-frame) PLY into nerfstudio normalized space using dataparser_transforms.json
        if self.apply_ns_transform and self.dp_meta is not None and (ply_path == input_dir / "object_3dgs.ply"):
            T = np.asarray(self.dp_meta.get("transform", np.eye(3, 4).tolist()), dtype=np.float32)
            s = float(self.dp_meta.get("scale", 1.0))
            Rn = (T[:, :3] * s).astype(np.float32)
            tn = (T[:, 3] * s).astype(np.float32)
            pts = (Rn @ pts.T).T + tn[None, :]

        self.points0 = pts.astype(np.float32)  # [N,3]
        self.colors = cols  # [N,3] in [0,1]
        self.weights = read_part_weights(weights_path)
        if self.weights is None or self.weights.shape[0] != self.points0.shape[0]:
            # default: all move if motion_part==1; none move if 0
            base = 1.0 if self.motion_part == 1 else 0.0
            self.weights = np.full((self.points0.shape[0],), base, dtype=np.float32)

        # Subsample for visualization to reduce computation
        target_n = 50000
        n_pts = int(self.points0.shape[0])
        if n_pts > target_n:
            rng = np.random.default_rng(42)
            self.sample_idx = rng.choice(n_pts, size=target_n, replace=False)
            self.points0 = self.points0[self.sample_idx]
            self.colors = self.colors[self.sample_idx]
            if self.weights is not None and self.weights.shape[0] == n_pts:
                self.weights = self.weights[self.sample_idx]
        else:
            self.sample_idx = np.arange(n_pts, dtype=np.int64)

        self.frames_T_obj = read_object_poses(objposes_path)  # list of [3,4]
        self.num_frames = len(self.frames_T_obj)
        self.frame_idx = 0

        self._build_scene()

    def _build_scene(self) -> None:
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points0)
        self.pcd.colors = o3d.utility.Vector3dVector(self.colors)

        # world coordinate frame for reference
        bbox = np.array([self.points0.min(0), self.points0.max(0)], dtype=np.float32)
        diag = float(np.linalg.norm(bbox[1] - bbox[0]))
        size = 0.5 * (diag if diag > 1e-6 else 1.0)
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])

        # text labels are not supported; rely on console prints

    def _mask_for_motion(self) -> np.ndarray:
        w = self.weights.copy()
        if self.motion_part == 0:
            w = 1.0 - w
        if self.soft:
            return w
        return (w >= self.threshold).astype(np.float32)

    def _transform_points(self) -> np.ndarray:
        pts = self.points0
        w = self._mask_for_motion()  # [N]
        T = self.frames_T_obj[self.frame_idx]  # [3,4]
        R = T[:, :3].astype(np.float32)
        t = T[:, 3].astype(np.float32)
        if self.invert_transform:
            R_inv = R.T
            t = (-(R_inv @ t)).astype(np.float32)
            R = R_inv.astype(np.float32)

        if not self.soft:
            mask = w > 0.5
            if not np.any(mask):
                return pts
            out = pts.copy()
            out[mask] = (R @ pts[mask].T).T + t[None, :]
            return out
        else:
            # SO(3) geodesic blend for rotation, linear for translation
            v = so3_log(R)  # axis-angle vector (axis * angle)
            out = pts.copy().astype(np.float32)
            # Rotate each point by exp(w_i * v), translate by w_i * t
            # For speed, precompute rotated points incrementally if needed; simple loop is fine here
            for i in range(out.shape[0]):
                wi = float(w[i])
                if wi <= 1e-8:
                    continue
                Ri = so3_exp(v * wi)
                out[i] = (Ri @ pts[i]) + (wi * t)
            return out

    def run(self) -> None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="TwoPartBlend Viewer", width=1280, height=720)
        vis.add_geometry(self.pcd)
        vis.add_geometry(self.coord_frame)

        def refresh():
            new_pts = self._transform_points()
            self.pcd.points = o3d.utility.Vector3dVector(new_pts)
            vis.update_geometry(self.pcd)
            vis.update_renderer()
            # diagnostics
            T = self.frames_T_obj[self.frame_idx]
            R = T[:, :3].astype(np.float32)
            if self.invert_transform:
                R = R.T
            ang = float(np.linalg.norm(so3_log(R)))
            t = T[:, 3].astype(np.float32)
            if self.invert_transform:
                t = -R @ t
            tnorm = float(np.linalg.norm(t))
            print(
                f"Frame {self.frame_idx+1}/{self.num_frames} | motion_part={self.motion_part} | mode={'soft' if self.soft else 'hard'} | thr={self.threshold:.2f} | map_dp={'Y' if self.apply_ns_transform else 'N'} | inv={'Y' if self.invert_transform else 'N'} | |R|~{ang:.3f} rad | |t|={tnorm:.4f}",
                end='\r',
            )

        def cb_next(vis_):
            self.frame_idx = (self.frame_idx + 1) % self.num_frames
            refresh()
            return True

        def cb_prev(vis_):
            self.frame_idx = (self.frame_idx - 1) % self.num_frames
            refresh()
            return True

        def cb_reset(vis_):
            self.frame_idx = 0
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

        def cb_toggle_ns(vis_):
            # Toggle applying dataparser transform on the fly for base PLY
            self.apply_ns_transform = not self.apply_ns_transform
            base_ply = self.input_dir / "object_3dgs.ply"
            pts, _ = read_ply_gaussians_rgb(base_ply)
            if self.apply_ns_transform and self.dp_meta is not None:
                T = np.asarray(self.dp_meta.get("transform", np.eye(3, 4).tolist()), dtype=np.float32)
                s = float(self.dp_meta.get("scale", 1.0))
                Rn = (T[:, :3] * s).astype(np.float32)
                tn = (T[:, 3] * s).astype(np.float32)
                pts = (Rn @ pts.T).T + tn[None, :]
            # Apply the same subsampling indices to keep size consistent
            if hasattr(self, "sample_idx"):
                pts = pts[self.sample_idx]
            self.points0 = pts.astype(np.float32)
            refresh()
            return True

        def cb_toggle_inv(vis_):
            self.invert_transform = not self.invert_transform
            refresh()
            return True

        vis.register_key_callback(ord("D"), cb_next)
        vis.register_key_callback(ord("A"), cb_prev)
        vis.register_key_callback(ord("R"), cb_reset)
        vis.register_key_callback(ord("M"), cb_toggle_part)
        vis.register_key_callback(ord("K"), cb_thresh_up)
        vis.register_key_callback(ord("J"), cb_thresh_down)
        vis.register_key_callback(ord("S"), cb_toggle_soft)
        vis.register_key_callback(ord("N"), cb_toggle_ns)
        vis.register_key_callback(ord("I"), cb_toggle_inv)

        print("Controls: D(next), A(prev), R(reset0), M(toggle part), J/K(thr -/+ hard), S(soft/hard), N(map dataparser), I(invert T)")

        refresh()
        vis.run()
        vis.destroy_window()


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive visualization for twopart_blend outputs")
    ap.add_argument("--input-dir", type=Path, required=True, help="twopart_blend output folder containing object_3dgs.ply and object_poses_train.json")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for hard mask (ignored in soft mode)")
    ap.add_argument("--motion_part", type=int, default=0, choices=[0, 1], help="Which part moves initially")
    ap.add_argument("--hard", action="store_true", help="Use hard mask (default soft)")
    ap.add_argument("--map-dataparser", action="store_true", help="Map raw PLY to nerfstudio normalized space using dataparser_transforms.json")
    ap.add_argument("--invert", action="store_true", help="Apply inverse of each frame transform T_obj")
    args = ap.parse_args()

    viewer = TwoPartBlendViewer(
        input_dir=args.input_dir,
        threshold=args.threshold,
        motion_part=args.motion_part,
        soft=not args.hard,
        map_dataparser=args.map_dataparser,
        invert=args.invert,
    )
    viewer.run()


if __name__ == "__main__":
    main()


