#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pipeline_config import get_section, load_pipeline_config, resolve_pipeline_paths

try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit("open3d is not installed. Please install it first: pip install open3d")


# Find the mask path for a given part_id. Prefer mask_{pid}; do not fall back to a shared masks directory.
def find_mask_path(dataset_dir: Path, image_rel_path: str, part_id: int) -> Optional[Path]:
    img_path = Path(image_rel_path)
    candidates = []
    # Support both images and images_* naming schemes.
    candidates.append(Path(str(img_path).replace("images", f"mask_{part_id}")))
    for cand in candidates:
        p = dataset_dir / cand
        if p.exists():
            return p
    return None


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def find_depth_path(dataset_dir: Path, image_rel_path: str) -> Optional[Path]:
    # images/foo.png -> depth/foo.npy or depth/foo.npz
    img_path = Path(image_rel_path)
    depth_rel_stem = img_path.with_suffix("")
    npy_candidate = Path(str(img_path).replace("images", "depth")).with_suffix(".npy")
    npz_candidate = Path(str(img_path).replace("images", "depth")).with_suffix(".npz")
    p_npy = dataset_dir / npy_candidate
    p_npz = dataset_dir / npz_candidate
    if p_npy.exists():
        return p_npy
    if p_npz.exists():
        return p_npz
    return None


def load_depth(depth_path: Path) -> np.ndarray:
    if depth_path.suffix == ".npy":
        d = np.load(depth_path)
    elif depth_path.suffix == ".npz":
        npz = np.load(depth_path)
        # common keys: 'arr_0', 'depth'
        if "depth" in npz:
            d = npz["depth"]
        else:
            # fallback to first array
            first_key = list(npz.keys())[0]
            d = npz[first_key]
    else:
        raise ValueError(f"Unsupported depth format: {depth_path}")
    d = np.asarray(d)
    if d.ndim == 3:
        # Keep a single channel.
        d = d[..., 0]
    return d


def backproject(depth: np.ndarray,
                rgb: np.ndarray,
                fx: float, fy: float, cx: float, cy: float,
                c2w: np.ndarray,
                pixel_stride: int = 4,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    # Sampling grid.
    ys = np.arange(0, h, pixel_stride)
    xs = np.arange(0, w, pixel_stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth[grid_y, grid_x]
    valid = np.isfinite(z) & (z > 0)
    if mask is not None:
        # Apply the mask to sampled points.
        if mask.dtype != np.bool_:
            mask_bool = mask > 0
        else:
            mask_bool = mask
        valid &= mask_bool[grid_y, grid_x]
    if valid.sum() == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # Floating-point coordinates used for projection.
    u_f = grid_x[valid].astype(np.float32)
    v_f = grid_y[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (u_f - cx) / fx * z
    y = (v_f - cy) / fy * z
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # N x 4

    # Transform into world coordinates.
    pts_world_h = pts_cam @ c2w.T  # N x 4
    pts_world = pts_world_h[:, :3]

    # Colors in [0, 1], sampled with integer pixel indices.
    u_i = grid_x[valid].astype(np.int64)
    v_i = grid_y[valid].astype(np.int64)
    cols = rgb[v_i, u_i, :].astype(np.float32) / 255.0
    return pts_world.astype(np.float32), cols


def get_intrinsics(meta: Dict, frame: Dict) -> Tuple[float, float, float, float, int, int]:
    def pick(key: str):
        return float(frame[key]) if key in frame else float(meta[key])

    fx = pick("fl_x") if ("fl_x" in frame or "fl_x" in meta) else None
    fy = pick("fl_y") if ("fl_y" in frame or "fl_y" in meta) else None
    cx = pick("cx") if ("cx" in frame or "cx" in meta) else None
    cy = pick("cy") if ("cy" in frame or "cy" in meta) else None
    w = int(frame["w"]) if "w" in frame else int(meta.get("w", 0))
    h = int(frame["h"]) if "h" in frame else int(meta.get("h", 0))

    if None in (fx, fy, cx, cy) or h == 0 or w == 0:
        raise ValueError("Missing camera intrinsics: fl_x, fl_y, cx, cy, w, and h are required")
    return fx, fy, cx, cy, w, h


def process_part(dataset_dir: Path,
                 part_id: int,
                 out_name: Optional[str] = None,
                 frame_stride: int = 1,
                 pixel_stride: int = 4,
                 max_points: int = 2_000_000,
                 depth_scale: float = 1.0,
                 seed: int = 0,
                 visualize: bool = False) -> Path:
    np.random.seed(seed)

    tjson = dataset_dir / f"transforms_{part_id}.json"
    if not tjson.exists():
        raise FileNotFoundError(f"Could not find {tjson}")

    meta = load_json(tjson)
    frames: List[Dict] = meta.get("frames", [])
    if len(frames) == 0:
        raise ValueError(f"No frames found in {tjson}")

    if out_name is None:
        out_name = f"sparse_pc_part{part_id}.ply"
    ply_path = dataset_dir / out_name

    all_pts = []
    all_cols = []
    camera_poses = []  # Store valid c2w poses.

    for i, fr in enumerate(frames[::max(1, frame_stride)]):
        try:
            fx, fy, cx, cy, w, h = get_intrinsics(meta, fr)
        except Exception as e:
            print(f"[warn] Invalid intrinsics for frame {i}; skipping: {e}")
            continue

        img_rel = fr["file_path"]
        img_path = dataset_dir / img_rel

        depth_path = find_depth_path(dataset_dir, img_rel)

        depth = load_depth(depth_path)

        depth = depth.astype(np.float32) * float(depth_scale)

        # Load RGB image.
        rgb = np.array(Image.open(img_path).convert("RGB"))
        if rgb.shape[0] != depth.shape[0] or rgb.shape[1] != depth.shape[1]:
            # Resize the image to match the depth resolution.
            rgb = np.array(Image.fromarray(rgb).resize((depth.shape[1], depth.shape[0]), Image.BILINEAR))

        # Load the segmentation mask from mask_{part_id}.
        mask_path = find_mask_path(dataset_dir, img_rel, part_id)
        mask_img = Image.open(mask_path).convert("L")
        if (mask_img.size[1], mask_img.size[0]) != depth.shape:
            # PIL reports size as (W, H), while depth.shape is (H, W).
            mask_img = mask_img.resize((depth.shape[1], depth.shape[0]), Image.NEAREST)
        mask_np = np.array(mask_img)
        mask_np = mask_np > 0  # Convert to boolean.

        # Pose.
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float32)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, np.array([0, 0, 0, 1], dtype=np.float32)])
        if c2w.shape != (4, 4):
            print(f"[warn] Unexpected pose shape; skipping: {c2w.shape}")
            continue
        
        # # Store w2c.
        # w2c = np.linalg.inv(c2w)
        # fr["transform_matrix"] = w2c.tolist()
        camera_poses.append(c2w)

        pts, cols = backproject(depth, rgb, fx, fy, cx, cy, c2w, pixel_stride=pixel_stride, mask=mask_np)
        if pts.shape[0] == 0:
            continue
        all_pts.append(pts)
        all_cols.append(cols)

    if len(all_pts) == 0:
        raise RuntimeError("No points were generated. Depth may be missing, or intrinsics/poses may be invalid.")

    P = np.concatenate(all_pts, axis=0)
    C = np.concatenate(all_cols, axis=0)

    # Downsample to cap the number of points.
    if max_points is not None and P.shape[0] > max_points:
        idx = np.random.choice(P.shape[0], size=max_points, replace=False)
        P = P[idx]
        C = C[idx]

    # Write the PLY file; colors must stay in [0, 1].
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    # Clamp color range.
    C = np.clip(C, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float64))

    if visualize:
        geometries = [pcd]
        for pose in camera_poses:
            # Create one coordinate frame for each camera pose.
            frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            frame_mesh.transform(pose)
            geometries.append(frame_mesh)
        
        print("[info] Visualizing point cloud and camera poses...")
        o3d.visualization.draw_geometries(geometries)

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write PLY: {ply_path}")

    # Back up transforms_*.json and write back ply_file_path.
    backup = tjson.with_name(tjson.stem + "_ori.json")
    if not backup.exists():
        try:
            with open(tjson, "rb") as rf, open(backup, "wb") as wf:
                wf.write(rf.read())
        except Exception as e:
            print(f"[warn] Failed to back up {tjson}: {e}")

    meta["ply_file_path"] = ply_path.name
    save_json(tjson, meta)

    print(f"[info] Generated point cloud: {ply_path}\n[info] Wrote ply_file_path back to {tjson}")
    return ply_path


def main():
    parser = argparse.ArgumentParser(description="Generate a point cloud from transforms_*.json and depth, then write back ply_file_path")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline config YAML")
    parser.add_argument("--object_name", type=str, default=None, help="Object name override used with --config")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--part_ids", type=int, nargs="*", default=[0, 1], help="List of part IDs to process")
    parser.add_argument("--frame_stride", type=int, default=1, help="Frame sampling stride")
    parser.add_argument("--pixel_stride", type=int, default=4, help="Pixel sampling stride")
    parser.add_argument("--max_points", type=int, default=2_000_000, help="Maximum number of points")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Depth unit scale (for example, use 0.001 for mm->m)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Visualize the point cloud and camera poses")

    args = parser.parse_args()

    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        if args.dataset_dir is None:
            args.dataset_dir = runtime_paths["dataset_dir"]
    if args.dataset_dir is None:
        parser.error("--dataset_dir is required when --config is not provided")

    dataset_dir = Path(args.dataset_dir)

    for pid in args.part_ids:
        print(f"==== Processing part {pid} ====")
        process_part(dataset_dir,
                     part_id=pid,
                     out_name=f"sparse_pc_part{pid}.ply",
                     frame_stride=args.frame_stride,
                     pixel_stride=args.pixel_stride,
                     max_points=args.max_points,
                     depth_scale=args.depth_scale,
                     seed=args.seed,
                     visualize=args.visualize)


if __name__ == "__main__":
    main()
