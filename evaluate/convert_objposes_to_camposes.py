import os
import json
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R


def _read_object_poses_as_T_list(object_poses_path: str) -> List[np.ndarray]:
    with open(object_poses_path, "r") as f:
        frames = json.load(f)

    T_list: List[np.ndarray] = []
    for item in frames:
        position = np.asarray(item.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
        quat = np.asarray(item.get("quaternion", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        if quat.shape[0] != 4:
            raise ValueError("Quaternion must have 4 elements.")
        q = quat / (np.linalg.norm(quat) + 1e-12)
        R_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_mat
        T[:3, 3] = position
        T_list.append(T)
    return T_list


def _read_first_camera_c2w(camera_params_path: str) -> np.ndarray:
    """Read the first camera c2w (4x4) from camera_params.json if possible; otherwise identity."""
    with open(camera_params_path, "r") as f:
        data = json.load(f)
    return np.array(data[0]["model_matrix"])


def convert(dataset_dir: str, output_path: str | None = None, T0: np.ndarray | None = None) -> str:
    obj_path = os.path.join(dataset_dir, "object_poses.json")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"object_poses.json not found under {dataset_dir}")

    T_list = _read_object_poses_as_T_list(obj_path)
    if len(T_list) == 0:
        raise ValueError("No frames in object_poses.json")

    T0 = T_list[0] if T0 is None else T0
    # Use first camera from camera_params.json as base C0 if available
    cam_params_path = os.path.join(dataset_dir, "camera_params.json")
    C0 = _read_first_camera_c2w(cam_params_path)
    poses = []
    for T_obj in T_list:

        # Keep object fixed at frame 0 by applying inv(T_i) relative to T0, then apply initial camera C0:
        # c2w_i = T0 @ inv(T_obj_i) @ C0
        c2w = T0 @ np.linalg.inv(T_obj) @ C0

        poses.append(c2w)
    poses_np = np.asarray(poses, dtype=np.float64)

    if output_path is None:
        output_path = os.path.join(dataset_dir, "camera_pose.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, poses_np)
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert object_poses.json to camera_pose.npy (c2w per frame)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory, e.g., datasets/100109")
    parser.add_argument("--output_path", type=str, default=None, help="Output .npy path; default: <dataset_dir>/camera_pose.npy")
    args = parser.parse_args()

    convert(args.dataset_dir, args.output_path)


if __name__ == "__main__":
    main()
