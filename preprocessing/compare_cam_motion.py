#!/usr/bin/env python3
import os
import json
import argparse
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pipeline_config import load_pipeline_config, resolve_pipeline_paths

def load_poses(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    def frame_id(item):
        fn = os.path.basename(item["file_path"])
        stem = os.path.splitext(fn)[0]
        try:
            return int(stem)
        except:
            return stem
    data = sorted(data, key=frame_id)

    Rs, ts = [], []
    for item in data:
        T = item["transform"]  # 3x4
        R = np.array([row[:3] for row in T], dtype=np.float64)
        t = np.array([row[3] for row in T], dtype=np.float64)
        Rs.append(R)
        ts.append(t)
    return np.stack(Rs), np.stack(ts)

def rotation_angle_deg(R1, R2):
    dR = R2 @ R1.T
    tr = np.trace(dR)
    val = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))

def _robust_inlier_mask(values, z_thresh=3.5, iqr_k=1.5):
    """Return boolean mask indicating inliers using MAD-based robust z-score with IQR fallback."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return np.ones_like(x, dtype=bool)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 1e-12:
        # 0.674489... is the inverse CDF of normal at 0.75; scales MAD to sigma
        z = 0.6744897501960817 * np.abs(x - med) / mad
        mask = z <= z_thresh
    else:
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        if iqr <= 1e-12:
            return np.ones_like(x, dtype=bool)
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        mask = (x >= lo) & (x <= hi)
    return mask

def compute_metrics(Rs, ts):
    n = len(ts)
    if n < 2:
        return {
            "num_frames": n,
            "trans_path": 0.0,
            "trans_net": 0.0,
            "trans_avg_step": 0.0,
            "rot_path_deg": 0.0,
        }
    diffs = ts[1:] - ts[:-1]
    step_trans = np.linalg.norm(diffs, axis=1)
    step_rot = np.array([rotation_angle_deg(Rs[i], Rs[i + 1]) for i in range(n - 1)], dtype=np.float64)

    mask_t = _robust_inlier_mask(step_trans)
    mask_r = _robust_inlier_mask(step_rot)

    if np.any(mask_t):
        trans_path = float(np.sum(step_trans[mask_t]))
        trans_avg_step = float(np.mean(step_trans[mask_t]))
    else:
        # fallback: no filtering
        trans_path = float(np.sum(step_trans))
        trans_avg_step = float(trans_path / (n - 1))

    trans_net = float(np.linalg.norm(ts[-1] - ts[0]))

    if np.any(mask_r):
        rot_path = float(np.sum(step_rot[mask_r]))
    else:
        rot_path = float(np.sum(step_rot))

    return {
        "num_frames": n,
        "trans_path": trans_path,
        "trans_net": trans_net,
        "trans_avg_step": trans_avg_step,
        "rot_path_deg": float(rot_path),
    }

def main():
    parser = argparse.ArgumentParser(description="Compare camera motion between two parts.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline config YAML")
    parser.add_argument("--object_name", type=str, default=None, help="Object name override used with --config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base directory containing part_0/ and part_1/")
    parser.add_argument("--part0", type=str, default=None, help="Path to part_0 transforms_train.json")
    parser.add_argument("--part1", type=str, default=None, help="Path to part_1 transforms_train.json")
    args = parser.parse_args()

    if args.config:
        config = load_pipeline_config(args.config, object_name_override=args.object_name)
        runtime_paths = resolve_pipeline_paths(config, object_name_override=args.object_name)
        if args.output_dir is None:
            args.output_dir = runtime_paths["output_dir"]
    if args.output_dir is None:
        parser.error("--output_dir is required when --config is not provided")

    part0_path = args.part0 or os.path.join(args.output_dir, "part_0", "transforms_train.json")
    part1_path = args.part1 or os.path.join(args.output_dir, "part_1", "transforms_train.json")

    Rs0, ts0 = load_poses(part0_path)
    Rs1, ts1 = load_poses(part1_path)

    m0 = compute_metrics(Rs0, ts0)
    m1 = compute_metrics(Rs1, ts1)

    print("== part_0 ==")
    print(f"frames: {m0['num_frames']}")
    print(f"translation_path: {m0['trans_path']:.6f}")
    print(f"translation_net:  {m0['trans_net']:.6f}")
    print(f"avg_step:         {m0['trans_avg_step']:.6f}")
    print(f"rotation_path(deg): {m0['rot_path_deg']:.6f}")
    print("")

    print("== part_1 ==")
    print(f"frames: {m1['num_frames']}")
    print(f"translation_path: {m1['trans_path']:.6f}")
    print(f"translation_net:  {m1['trans_net']:.6f}")
    print(f"avg_step:         {m1['trans_avg_step']:.6f}")
    print(f"rotation_path(deg): {m1['rot_path_deg']:.6f}")
    print("")

    if m0["trans_path"] > m1["trans_path"]:
        verdict = "part_0 camera moves more based on translation path length"
    elif m0["trans_path"] < m1["trans_path"]:
        verdict = "part_1 camera moves more based on translation path length"
    else:
        verdict = "Both parts have the same translation path length"

    print("Verdict:", verdict)

    # Write the selected moving-part index to base/motion_part.txt.
    # 0 means part_0, 1 means part_1; ties default to part_0.
    winner_idx = 0 if m0["trans_path"] >= m1["trans_path"] else 1
    motion_txt = os.path.join(args.output_dir, "motion_part.txt")
    try:
        with open(motion_txt, "w") as f:
            f.write(f"{winner_idx}\n")
        print(f"choose motion part: {winner_idx}")
    except Exception as e:
        print(f"Failed to write motion_part.txt: {e}")

if __name__ == "__main__":
    main()
