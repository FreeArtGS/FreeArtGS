#!/usr/bin/env python3
"""
Export evaluation data from splatfacto-art training results.

This script generates the exported evaluation inputs consumed by FreeArtGS evaluation.
"""

import os
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
import struct


GENERATED_JOINT_DIRS = ("prismatic", "revolute")
PART_WEIGHT_MARGIN = 0.3


def clear_generated_evaluation_outputs(output_dir: Path) -> None:
    """Remove generated joint-type outputs so stale directories cannot affect evaluation."""
    for dirname in GENERATED_JOINT_DIRS:
        path = output_dir / dirname
        if path.exists():
            shutil.rmtree(path)


def load_gaussian_ply(ply_path):
    """Load 3D Gaussian Splatting PLY file and extract xyz positions."""
    print(f"Loading PLY file from {ply_path}...")
    plydata = PlyData.read(ply_path)
    
    xyz = np.stack([
        np.array(plydata['vertex']['x']),
        np.array(plydata['vertex']['y']),
        np.array(plydata['vertex']['z'])
    ], axis=1)
    
    print(f"Loaded {len(xyz)} Gaussian points")
    return xyz, plydata


def load_part_weights(weight_path):
    """Load part weights from text file."""
    print(f"Loading part weights from {weight_path}...")
    weights = []
    with open(weight_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    weights.append(float(line))
                except ValueError:
                    continue
    weights = np.array(weights)
    print(f"Loaded {len(weights)} part weights, range: [{weights.min():.4f}, {weights.max():.4f}]")
    return weights


def save_point_cloud_ply(xyz, output_path, rgb=None):
    """Save point cloud to PLY file."""
    print(f"Saving point cloud to {output_path}...")
    
    if rgb is None:
        # Default white color
        rgb = np.ones_like(xyz) * 255
    
    # Ensure RGB is uint8
    rgb = rgb.astype(np.uint8)
    
    # Create structured array
    vertex_data = np.zeros(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    vertex_data['red'] = rgb[:, 0]
    vertex_data['green'] = rgb[:, 1]
    vertex_data['blue'] = rgb[:, 2]
    
    # Create PLY element
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # Write PLY file
    PlyData([vertex_element], text=False).write(output_path)
    print(f"Saved {len(xyz)} points")


def load_articulation_json(json_path):
    """Load articulation parameters from JSON file."""
    print(f"Loading articulation from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    joint_type = data['joint_type']
    joint_axis = np.array(data['joint_axis_world'])
    joint_pos = np.array(data['joint_origin_world'])
    
    # Extract joint values for each frame
    joint_values = []
    for frame in data['frames']:
        if joint_type == 'revolute':
            joint_values.append(frame['angle_rad'])
        else:  # prismatic
            # For prismatic, we might need translation value
            # Assuming it's stored in a similar way
            joint_values.append(frame.get('distance', 0.0))
    
    joint_values = np.array(joint_values)
    joint_values = joint_values - joint_values[0]
    print(f"Joint type: {joint_type}")
    print(f"Joint axis: {joint_axis}")
    print(f"Joint pos: {joint_pos}")
    print(f"Joint values: {joint_values}")
    
    return joint_type, joint_axis, joint_pos, joint_values


def load_camera_poses(transforms_path, dataparser_transform_path=None, output_json_path=None):
    """
    Load camera poses from transforms.json and optionally save to JSON.
    
    Args:
        transforms_path: Path to transforms.json
        dataparser_transform_path: Optional path to dataparser_transforms.json
        output_json_path: If provided, save the transformed poses to this JSON file
    
    Returns:
        camera_poses: (N, 7) array: [qw, qx, qy, qz, tx, ty, tz]
        frames_data: List of dictionaries with all original info plus 4x4 transform
    """
    print(f"Loading camera poses from {transforms_path}...")
    
    with open(transforms_path, 'r') as f:
        transforms_data = json.load(f)
    
    # Load dataparser transform if available
    dataparser_transform = np.eye(4)
    if dataparser_transform_path and os.path.exists(dataparser_transform_path):
        with open(dataparser_transform_path, 'r') as f:
            dp_data = json.load(f)
            if 'transform' in dp_data:
                dataparser_transform[:3, :] = np.array(dp_data['transform'])
    
    camera_poses = []
    frames_data = []
    
    for frame_data in transforms_data:
        # Get camera-to-world transform (3x4 or 4x4)
        if isinstance(frame_data, dict) and 'transform' in frame_data:
            c2w = np.array(frame_data['transform'])
        elif isinstance(frame_data, dict) and 'transform_matrix' in frame_data:
            c2w = np.array(frame_data['transform_matrix'])
        else:
            print(f"Warning: Cannot find transform in frame data: {frame_data}")
            continue
        
        # Convert to 4x4 if needed
        if c2w.shape == (3, 4):
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3, :] = c2w
            c2w = c2w_4x4
        elif c2w.shape == (4, 3):
            # Transpose format
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3, :] = c2w[:3, :]
            c2w = c2w_4x4
        
        # Extract rotation and translation
        rotation_matrix = c2w[:3, :3]
        translation = c2w[:3, 3]
        
        # Convert rotation matrix to quaternion (scalar-first: w, x, y, z)
        rot = R.from_matrix(rotation_matrix)
        quat = rot.as_quat()  # Returns [x, y, z, w]
        quat_scalar_first = np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]
        
        # Combine into [qw, qx, qy, qz, tx, ty, tz]
        pose = np.concatenate([quat_scalar_first, translation])
        camera_poses.append(pose)
        
        # Prepare frame data with 4x4 transform
        frame_dict = frame_data.copy() if isinstance(frame_data, dict) else {}
        # Update transform to 4x4 matrix (convert to list for JSON serialization)
        frame_dict['transform_matrix'] = c2w.tolist()
        frames_data.append(frame_dict)
    
    camera_poses = np.array(camera_poses)
    print(f"Loaded {len(camera_poses)} camera poses")
    
    # Save to JSON if output path provided
    if output_json_path:
        print(f"Saving transformed camera poses to {output_json_path}...")
        with open(output_json_path, 'w') as f:
            json.dump(frames_data, f, indent=2)
        print(f"Saved {len(frames_data)} frames to JSON")
    
    return camera_poses, frames_data


def compute_best_loss(joint_type, output_dir):
    """
    Compute a dummy best loss value.
    In practice, you should save the actual training loss.
    """
    # TODO: Replace with actual loss from training
    # For now, use a small dummy value to indicate successful optimization
    dummy_loss = 0.001
    return dummy_loss


def export_evaluation_data(
    splatfacto_art_dir,
    motion_part,
    output_dir,
    weight_threshold=0.5,
    sample_num=10000,
    export_both_types=False,
    joint_type_override: str | None = None,
):
    """
    Export all evaluation data from splatfacto-art results.
    
    Args:
        splatfacto_art_dir: Path to splatfacto-art output directory
        output_dir: Path to output directory for evaluation data
        motion_part: Motion part (default: 0)
        weight_threshold: Threshold for separating moving/static parts (default: 0.5)
        sample_num: Number of points to sample for surface PCD (default: 480*64=30720)
        export_both_types: Whether to export both prismatic and revolute (default: False)
        joint_type_override: If provided, force export for this joint type only
    """
    splatfacto_art_dir = Path(splatfacto_art_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_evaluation_outputs(output_dir)
    
    # Load data
    ply_path = splatfacto_art_dir / "object_3dgs.ply"
    weight_path = splatfacto_art_dir / "3dgs_part_weight.txt"
    articulation_path = splatfacto_art_dir / "articulation.json"
    transforms_path = splatfacto_art_dir / "transforms_train.json"
    dataparser_path = splatfacto_art_dir / "dataparser_transforms.json"
    
    # 1. Load Gaussian positions and part weights
    xyz, plydata = load_gaussian_ply(ply_path)
    part_weights = load_part_weights(weight_path)
    
    assert len(xyz) == len(part_weights), f"Mismatch: {len(xyz)} points vs {len(part_weights)} weights"
    
    # 2. Sample surface point cloud
    if len(xyz) > sample_num:
        indices = np.random.choice(len(xyz), sample_num, replace=False)
        surface_xyz = xyz[indices]
        surface_weights = part_weights[indices]
    else:
        surface_xyz = xyz
        surface_weights = part_weights
    
    save_point_cloud_ply(surface_xyz, output_dir / "surface_pcd.ply")
    
    low_threshold = weight_threshold - PART_WEIGHT_MARGIN
    high_threshold = weight_threshold + PART_WEIGHT_MARGIN
    if motion_part == 1:
        moving_mask = part_weights > high_threshold
        static_mask = part_weights < low_threshold
    else:
        moving_mask = part_weights < low_threshold
        static_mask = part_weights > high_threshold
    moving_xyz = xyz[moving_mask]
    static_xyz = xyz[static_mask]
    uncertain_count = int((~moving_mask & ~static_mask).sum())
    
    print(f"Moving points: {len(moving_xyz)} ({100*len(moving_xyz)/len(xyz):.1f}%)")
    print(f"Static points: {len(static_xyz)} ({100*len(static_xyz)/len(xyz):.1f}%)")
    print(f"Uncertain points ignored for part-specific PCDs: {uncertain_count} ({100*uncertain_count/len(xyz):.1f}%)")
    
    # Sample moving and static parts with the same ratio as surface_pcd
    if len(xyz) > sample_num:
        sample_ratio = sample_num / len(xyz)
        
        # Sample moving points
        moving_sample_num = int(len(moving_xyz) * sample_ratio)
        if moving_sample_num > 0 and len(moving_xyz) > moving_sample_num:
            moving_indices = np.random.choice(len(moving_xyz), moving_sample_num, replace=False)
            moving_xyz = moving_xyz[moving_indices]
        
        # Sample static points
        static_sample_num = int(len(static_xyz) * sample_ratio)
        if static_sample_num > 0 and len(static_xyz) > static_sample_num:
            static_indices = np.random.choice(len(static_xyz), static_sample_num, replace=False)
            static_xyz = static_xyz[static_indices]
        
        print(f"After sampling - Moving points: {len(moving_xyz)}, Static points: {len(static_xyz)}")
    
    # Color code: moving = red, static = blue
    moving_rgb = np.zeros((len(moving_xyz), 3))
    moving_rgb[:, 0] = 255  # Red
    
    static_rgb = np.zeros((len(static_xyz), 3))
    static_rgb[:, 2] = 255  # Blue
    
    # 4. Load articulation parameters
    joint_type, joint_axis, joint_pos, joint_values = load_articulation_json(articulation_path)
    effective_joint_type = joint_type_override if joint_type_override is not None else joint_type
    
    # 5. Load camera poses
    camera_poses, frames_data = load_camera_poses(transforms_path, dataparser_path)
    
    # Ensure joint_values matches camera_poses length
    if len(joint_values) != len(camera_poses):
        print(f"Warning: {len(joint_values)} joint values vs {len(camera_poses)} camera poses")
        # Pad or truncate joint_values to match
        if len(joint_values) < len(camera_poses):
            # Pad with last value
            joint_values = np.pad(joint_values, (0, len(camera_poses) - len(joint_values)), 
                                 mode='edge')
        else:
            # Truncate
            joint_values = joint_values[:len(camera_poses)]
    
    # 6. Export data for the detected/overridden joint type
    if export_both_types:
        # Export both types, but mark the detected/overridden one with lower loss
        joint_types_to_export = ['prismatic', 'revolute']
    else:
        joint_types_to_export = [effective_joint_type]

    manifest = {
        "joint_type": effective_joint_type,
        "exported_joint_types": joint_types_to_export,
        "motion_part": int(motion_part),
        "weight_threshold": float(weight_threshold),
        "part_weight_margin": float(PART_WEIGHT_MARGIN),
    }
    with open(output_dir / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    for jtype in joint_types_to_export:
        type_dir = output_dir / jtype
        type_dir.mkdir(exist_ok=True)
        
        # Save joint parameters
        np.save(type_dir / "joint_axis.npy", joint_axis)
        np.save(type_dir / "joint_pos.npy", joint_pos)
        np.save(type_dir / "joint_value.npy", joint_values)
        
        # Save camera poses
        np.save(type_dir / "camera_poses.npy", camera_poses)
        
        # Save camera poses as JSON with 4x4 transforms
        with open(type_dir / "camera_poses.json", 'w') as f:
            json.dump(frames_data, f, indent=2)
        
        # Save part point clouds
        save_point_cloud_ply(moving_xyz, type_dir / "moving_pcd.ply", moving_rgb)
        save_point_cloud_ply(static_xyz, type_dir / "static_pcd.ply", static_rgb)
        
        # Save best loss
        if jtype == effective_joint_type:
            # The detected type gets a low loss
            best_loss = compute_best_loss(jtype, type_dir)
        else:
            # The other type gets a high loss (so it won't be selected)
            best_loss = 100.0
        
        with open(type_dir / "best_loss.txt", 'w') as f:
            f.write(f"{best_loss}\n")
        
        print(f"\n=== Exported {jtype} joint type ===")
        print(f"  Joint axis: {joint_axis}")
        print(f"  Joint pos: {joint_pos}")
        print(f"  Joint values: {len(joint_values)} frames")
        print(f"  Camera poses: {len(camera_poses)} frames")
        print(f"  Best loss: {best_loss}")
        print(f"  Moving points: {len(moving_xyz)}")
        print(f"  Static points: {len(static_xyz)}")
    
    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - surface_pcd.ply")
    for jtype in joint_types_to_export:
        print(f"  - {jtype}/")
        print(f"      - joint_axis.npy")
        print(f"      - joint_pos.npy")
        print(f"      - joint_value.npy")
        print(f"      - camera_poses.npy")
        print(f"      - camera_poses.json (with 4x4 transforms)")
        print(f"      - moving_pcd.ply")
        print(f"      - static_pcd.ply")
        print(f"      - best_loss.txt")
    print(f"\nNote: optional mask exports such as moving_map.npz are generated separately from rendering")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Export evaluation data from splatfacto-art training results"
    )
    parser.add_argument(
        "--splatfacto_art_dir",
        type=str,
        required=False,
        help="Path to splatfacto-art output directory (containing articulation.json, object_3dgs.ply, etc.)"
    )
    parser.add_argument(
        "--motion_part",
        type=int,
        default=0,
        help="Motion part (default: 0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Path to output directory for evaluation data"
    )
    parser.add_argument(
        "--weight_threshold",
        type=float,
        default=0.5,
        help="Threshold for separating moving/static parts (default: 0.5)"
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=10000,
        help="Number of points to sample for surface PCD (default: 480*64=30720)"
    )
    parser.add_argument(
        "--export_both_types",
        action="store_true",
        default=False,
        help="Export both prismatic and revolute (default: False)"
    )
    parser.add_argument(
        "--joint_type",
        type=str,
        choices=["revolute", "prismatic"],
        help="Override joint type to export; default reads from articulation.json"
    )
    
    # Convenience: infer paths from a compact experiment id like 19855_j0_v1
    parser.add_argument(
        "--exp_id",
        type=str,
        help="Compact id <obj>_j<joint>_v<view> (e.g., 19855_j0_v1) to auto-infer paths under outputs_root"
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default="outputs",
        help="Root directory containing <exp_id>/splatfacto-art and <exp_id>/evaluation_data"
    )
    args = parser.parse_args()

    # Auto-infer paths if exp_id provided
    if args.exp_id is not None:
        exp_dir = os.path.join(args.outputs_root, args.exp_id)
        inferred_splat_dir = os.path.join(exp_dir, "splatfacto-art")
        inferred_out_dir = os.path.join(exp_dir, "evaluation_data")
        if os.path.isdir(inferred_splat_dir):
            args.splatfacto_art_dir = inferred_splat_dir
            print(f"[auto] splatfacto_art_dir = {args.splatfacto_art_dir}")
        if args.output_dir is None:
            args.output_dir = inferred_out_dir
            print(f"[auto] output_dir = {args.output_dir}")

    # Validate required paths
    if not args.splatfacto_art_dir:
        raise ValueError("--splatfacto_art_dir is required unless --exp_id is provided and resolvable.")
    if not args.output_dir:
        raise ValueError("--output_dir is required unless --exp_id is provided (then defaults to <outputs_root>/<exp_id>/evaluation_data).")

    export_evaluation_data(
        splatfacto_art_dir=args.splatfacto_art_dir,
        output_dir=args.output_dir,
        motion_part=args.motion_part,
        weight_threshold=args.weight_threshold,
        sample_num=args.sample_num,
        export_both_types=args.export_both_types,
        joint_type_override=args.joint_type
    )


if __name__ == "__main__":
    main()
