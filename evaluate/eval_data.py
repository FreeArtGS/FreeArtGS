import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF

from data_utils import descendant_links, sample_urdf_pcd
from eval_utils import load_camera_poses_from_json


def infer_predicted_joint_type(results_dir: str) -> str:
    manifest_path = os.path.join(results_dir, "export_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing export manifest: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    joint_type = manifest.get("joint_type")
    if joint_type not in {"prismatic", "revolute"}:
        raise ValueError(f"Invalid joint_type in {manifest_path}: {joint_type}")
    return joint_type


def load_predicted_data(results_dir: str) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    pred_joint_type = infer_predicted_joint_type(results_dir)
    type_dir = os.path.join(results_dir, pred_joint_type)

    pred_joint_axis = np.load(os.path.join(type_dir, "joint_axis.npy")).astype(np.float64)
    pred_joint_pos = np.load(os.path.join(type_dir, "joint_pos.npy")).astype(np.float64)
    pred_joint_values = np.load(os.path.join(type_dir, "joint_value.npy")).astype(np.float64)

    cam_json = os.path.join(type_dir, "camera_poses.json")
    pred_camera_poses_4x4, pred_indices = load_camera_poses_from_json(cam_json)
    return pred_joint_type, pred_joint_axis, pred_joint_pos, pred_joint_values, pred_camera_poses_4x4, pred_indices


def resolve_partnet_urdf(partnet_mobility_dir: str, obj_id: str) -> Tuple[Path, Path]:
    base = Path(partnet_mobility_dir).expanduser().resolve()
    cand1 = base / "dataset" / str(obj_id) / "mobility.urdf"
    cand2 = base / "datasets" / str(obj_id) / "mobility.urdf"
    urdf_path = cand1 if cand1.exists() else cand2
    urdf_path = urdf_path.expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found for object {obj_id} under {partnet_mobility_dir}/dataset[s]/")
    return urdf_path, urdf_path.parent


def _build_joint_cfg(robot: URDF, joint_name: str, joint_value: float | None) -> Dict[str, float] | None:
    if joint_value is None:
        return None
    cfg: Dict[str, float] = {}
    actuated_joint_names = getattr(robot, "actuated_joint_names", []) or []
    for name in actuated_joint_names:
        cfg[str(name)] = 0.0
    cfg[joint_name] = float(joint_value)
    return cfg


def load_gt_pcd_from_urdf(
    partnet_mobility_dir: str,
    obj_id: str,
    joint_id: int,
    joint_value: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    urdf_path, urdf_dir = resolve_partnet_urdf(partnet_mobility_dir, obj_id)
    robot = URDF.load(urdf_path, mesh_dir=urdf_dir)
    joint_name = f"joint_{int(joint_id)}"
    cfg = _build_joint_cfg(robot, joint_name, joint_value)
    moving_links = descendant_links(robot, joint_name)
    link_map = robot.link_map
    full_link_list = link_map.keys()
    gt_full_pcd = sample_urdf_pcd(urdf_dir, full_link_list, robot, final_pts_num=10000, cfg=cfg)
    gt_moving_pcd = sample_urdf_pcd(urdf_dir, moving_links, robot, final_pts_num=10000, cfg=cfg)
    static_links = [link for link in full_link_list if link not in moving_links]
    gt_static_pcd = sample_urdf_pcd(urdf_dir, static_links, robot, final_pts_num=10000, cfg=cfg)
    return gt_full_pcd, gt_moving_pcd, gt_static_pcd


def load_gt_joint_from_urdf(partnet_mobility_dir: str, obj_id: str, joint_id: int) -> Tuple[str, np.ndarray, np.ndarray]:
    urdf_path, urdf_dir = resolve_partnet_urdf(partnet_mobility_dir, obj_id)
    robot = URDF.load(urdf_path, mesh_dir=urdf_dir)
    joint_name = f"joint_{int(joint_id)}"
    if joint_name not in robot.joint_map:
        raise KeyError(f"Joint '{joint_name}' not found in URDF for object {obj_id}")
    joint = robot.joint_map[joint_name]

    jt = getattr(joint, "joint_type", None) or getattr(joint, "type", None)
    if jt is None:
        jt = "revolute"
    jt = str(jt).lower()
    if jt in ["revolute", "continuous", "hinge"]:
        gt_joint_type = "revolute"
    elif jt in ["prismatic", "slider"]:
        gt_joint_type = "prismatic"
    else:
        gt_joint_type = "revolute"

    axis_local = np.array(getattr(joint, "axis", [1.0, 0.0, 0.0]), dtype=np.float64)
    child = getattr(joint, "child", None)
    child_name = getattr(child, "name", child)
    T_child = robot.get_transform(child_name)
    R_child = T_child[:3, :3].astype(np.float64)
    t_child = T_child[:3, 3].astype(np.float64)
    axis_urdf = R_child @ axis_local
    pos_urdf = t_child

    rotate_back = R.from_euler("zyx", [90, 0, -90], degrees=True).as_matrix().astype(np.float64)
    gt_joint_axis = rotate_back @ axis_urdf
    gt_joint_pos = rotate_back @ pos_urdf
    return gt_joint_type, gt_joint_axis, gt_joint_pos


def _extract_joint_state_value(item) -> float:
    known_keys = ["angle_rad", "angle", "value", "qpos", "state", "theta", "translation"]

    if isinstance(item, (int, float)):
        return float(item)

    if isinstance(item, (list, tuple)):
        if not item:
            return 0.0
        first = item[0]
        if isinstance(first, (int, float)):
            return float(first)
        for value in item:
            nested = _extract_joint_state_value(value)
            if isinstance(nested, (int, float)):
                return float(nested)
        return 0.0

    if isinstance(item, dict):
        for key in known_keys:
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        joint = item.get("joint")
        if isinstance(joint, dict):
            for key in known_keys:
                value = joint.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        for value in item.values():
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    return 0.0


def load_joint_values_from_joint_states(dataset_dir: str) -> np.ndarray | None:
    joint_states_path = os.path.join(dataset_dir, "joint_states.json")
    if not os.path.exists(joint_states_path):
        return None
    with open(joint_states_path, "r") as f:
        data = json.load(f)

    values: List[float] = []
    if isinstance(data, list):
        values = [_extract_joint_state_value(item) for item in data]
    elif isinstance(data, dict):
        seq = None
        for top_key in ["frames", "values", "joint_values", "joint_states"]:
            if top_key in data and isinstance(data[top_key], list):
                seq = data[top_key]
                break
        if seq is not None:
            values = [_extract_joint_state_value(item) for item in seq]
        else:
            items = sorted((int(k), v) for k, v in data.items() if str(k).isdigit())
            values = [_extract_joint_state_value(v) for _, v in items]
    else:
        return None

    arr = np.asarray(values, dtype=np.float64)
    return arr


def resolve_default_partnet_mobility_dir() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    preferred = repo_root / "data" / "partnet-mobility-v0"
    legacy = repo_root / "data" / "video2art_data" / "partnet-mobility-v0"
    if preferred.exists():
        return str(preferred)
    if legacy.exists():
        return str(legacy)
    raise FileNotFoundError(
        "PartNet-Mobility directory not found. Expected one of: "
        f"{preferred} or {legacy}"
    )
