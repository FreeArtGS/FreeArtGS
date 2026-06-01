import argparse
import copy
import json
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


TRACKING_DEFAULTS = {
    "debug": False,
    "window_size": 10,
    "min_depth": 0.1,
    "max_depth": 20.0,
    "min_inlier_ratio": 0.3,
    "min_trajectory_count": 50,
    "visibility_threshold": 0.7,
    "confidence_threshold": 0.7,
    "loss_type": "normalized_huber",
    "init_method": "kmeans",
    "seg_fill_method": "kmeans",
    "lr_T": 0.0001,
    "lr_w": 0.01,
    "filter_ratio_threshold": 0.15,
    "joint_optimize_epochs": 200,
    "solve_window_epochs": 100,
    "neighbor_selection_method": "radius",
    "knn_k": 3000,
    "knn_radius": 0.3,
    "downscale_factor": 2,
    "pingpong": True,
    "include_turn_frame": False,
    "tracker_config": "tracking/configs/alltracker.yaml",
    "w_s": 10.0,
    "w_init": 10.0,
    "w_p": 0.0,
    "w_m": 200.0,
    "w_e": -0.01,
}

PREPROCESS_DEFAULTS = {
    "segment_prompt": None,
    "track_sam": True,
    "interactive_mask": False,
    "skip_segment": True,
    "bg_color": "white",
    "visualize_features": False,
    "feature_model_name": "dinov3_vitl16",
    "feature_image_size": 720,
    "feature_patch_size": 16,
    "feature_checkpoint_path": "checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "pcd_max_depth": 15.0,
}

ICP_DEFAULTS = {
    "erode_kernel_size": 3,
    "erode_iterations": 1,
    "voxel_size": 0,
    "sample_points": 100000,
    "local_map_window": 8,
    "local_map_max_points": 50000,
    "init_fallback_fitness": 0.9,
    "hybrid_plane_weight": 1.0,
    "hybrid_point_weight": 0.1,
    "hybrid_damping": 1e-6,
    "max_frame_translation": 2.0,
    "max_frame_rotation_deg": 12.0,
    "global_opt": True,
    "loop_interval": 0,
    "min_points_per_pcd": 50,
    "bad_match_fitness": 0.5,
    "icp_debug": False,
}


def load_pipeline_config(
    config_path: str,
    object_name_override: Optional[str] = None,
) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Pipeline config must be a mapping: {config_path}")
    config = copy.deepcopy(config)
    config.setdefault("paths", {})
    if object_name_override:
        config["paths"]["object_name"] = object_name_override
    return config


def resolve_pipeline_paths(
    config: Mapping[str, Any],
    object_name_override: Optional[str] = None,
) -> Dict[str, str]:
    paths_cfg = dict(config.get("paths", {}) or {})
    object_name = object_name_override or paths_cfg.get("object_name")
    if not object_name:
        raise ValueError("object_name must be provided by config or CLI override")

    original_data_dir = str(paths_cfg.get("original_data_dir", "examples"))
    dataset_root = str(paths_cfg.get("dataset_root", "datasets"))
    output_root = str(paths_cfg.get("output_root", "outputs"))

    original_object_dir = str(Path(original_data_dir) / object_name)
    dataset_dir = str(Path(dataset_root) / object_name)
    output_dir = str(Path(output_root) / object_name)
    tracking_dir = str(Path(dataset_dir) / "tracking")

    return {
        "object_name": object_name,
        "original_data_dir": original_data_dir,
        "dataset_root": dataset_root,
        "output_root": output_root,
        "original_object_dir": original_object_dir,
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "tracking_dir": tracking_dir,
    }


def get_section(config: Mapping[str, Any], section: str) -> Dict[str, Any]:
    value = config.get(section, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return dict(value)


def resolve_tracking(config: Mapping[str, Any]) -> Dict[str, Any]:
    tracking = dict(TRACKING_DEFAULTS)
    tracking_cfg = get_section(config, "tracking")
    tracking.update(tracking_cfg)
    if "no_pingpong" in tracking_cfg:
        tracking["pingpong"] = not bool(tracking_cfg["no_pingpong"])
    tracking.pop("no_pingpong", None)
    return tracking


def resolve_preprocess(config: Mapping[str, Any]) -> Dict[str, Any]:
    preprocess = dict(PREPROCESS_DEFAULTS)
    preprocess.update(get_section(config, "preprocess"))
    return preprocess


def resolve_icp(config: Mapping[str, Any]) -> Dict[str, Any]:
    icp = dict(ICP_DEFAULTS)
    icp.update(get_section(config, "icp"))
    return icp


def apply_section_to_args(
    args: argparse.Namespace,
    section: Mapping[str, Any],
    key_map: Optional[Mapping[str, str]] = None,
    preserve_existing: Optional[Iterable[str]] = None,
) -> argparse.Namespace:
    key_map = dict(key_map or {})
    preserve_existing = set(preserve_existing or [])
    for key, value in section.items():
        attr = key_map.get(key, key)
        current = getattr(args, attr, None)
        if attr in preserve_existing and current not in (None, ""):
            continue
        if value is not None:
            setattr(args, attr, value)
    return args


def set_path_defaults(
    args: argparse.Namespace,
    defaults: Mapping[str, Any],
) -> argparse.Namespace:
    for attr, value in defaults.items():
        current = getattr(args, attr, None)
        if current in (None, ""):
            setattr(args, attr, value)
    return args


def set_arg_defaults(
    args: argparse.Namespace,
    defaults: Mapping[str, Any],
) -> argparse.Namespace:
    for attr, value in defaults.items():
        current = getattr(args, attr, None) if hasattr(args, attr) else None
        if not hasattr(args, attr) or current in (None, ""):
            setattr(args, attr, value)
    return args


def bool_to_string(value: bool) -> str:
    return "True" if value else "False"


def to_shell_value(value: Any) -> str:
    if isinstance(value, bool):
        return bool_to_string(value)
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def to_shell_name(key: str) -> str:
    return key.upper().replace("-", "_")


def build_shell_exports(
    config_path: str,
    object_name_override: Optional[str] = None,
    stage: Optional[str] = None,
) -> Dict[str, str]:
    config = load_pipeline_config(config_path, object_name_override=object_name_override)
    runtime = resolve_pipeline_paths(config)
    exports = {to_shell_name(key): to_shell_value(value) for key, value in runtime.items()}
    if stage:
        section = get_section(config, stage)
        for key, value in section.items():
            exports[to_shell_name(key)] = to_shell_value(value)
    return exports


def emit_shell_exports(exports: Mapping[str, str]) -> None:
    for key in sorted(exports):
        print(f"{key}={shlex.quote(exports[key])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit resolved pipeline config values for shell consumption.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--object_name", default=None, help="Optional object_name CLI override.")
    parser.add_argument("--stage", default=None, help="Optional stage section to export.")
    parser.add_argument("--format", choices=["shell", "json"], default="shell")
    args = parser.parse_args()

    exports = build_shell_exports(args.config, object_name_override=args.object_name, stage=args.stage)
    if args.format == "json":
        print(json.dumps(exports, ensure_ascii=False, indent=2))
    else:
        emit_shell_exports(exports)


if __name__ == "__main__":
    main()
