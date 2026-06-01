import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def load_eval_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Evaluation config must be a mapping: {config_path}")
    return dict(config)


def resolve_object_id(object_name: str) -> str:
    return str(object_name).split("_", 1)[0]


def extract_joint_ids(config: Mapping[str, Any]) -> Dict[str, Any]:
    joint_ids = config.get("joint_ids", config)
    if not isinstance(joint_ids, dict):
        raise ValueError("Evaluation joint config must be a mapping or contain a 'joint_ids' mapping")
    return dict(joint_ids)


def resolve_joint_id(config: Mapping[str, Any], object_name: str) -> int:
    object_id = resolve_object_id(object_name)
    joint_ids = extract_joint_ids(config)

    for key in (object_name, object_id):
        if key in joint_ids:
            return int(joint_ids[key])

    key_hints = [f"'{object_name}'"]
    if object_id != object_name:
        key_hints.append(f"'{object_id}'")

    raise ValueError(
        "No joint_id configured for object "
        f"'{object_name}'. Add {', '.join(key_hints)} to the evaluation config."
    )


def build_exports(config_path: str, object_name: str) -> Dict[str, str]:
    config = load_eval_config(config_path)
    object_id = resolve_object_id(object_name)
    joint_id = resolve_joint_id(config, object_name)
    return {
        "OBJECT_NAME": object_name,
        "OBJECT_ID": object_id,
        "JOINT_ID": str(joint_id),
        "DATASET_DIR": str(Path("datasets") / object_id),
        "EVAL_CONFIG_PATH": config_path,
    }


def emit_shell_exports(exports: Mapping[str, str]) -> None:
    for key in sorted(exports):
        print(f"{key}={shlex.quote(exports[key])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve evaluation joint_id for an object.")
    parser.add_argument("--config", required=True, help="Path to evaluation joint config YAML/JSON.")
    parser.add_argument("--object_name", required=True, help="Object or experiment name to resolve.")
    parser.add_argument("--format", choices=["shell", "json"], default="shell")
    args = parser.parse_args()

    exports = build_exports(args.config, args.object_name)
    if args.format == "json":
        print(json.dumps(exports, ensure_ascii=False, indent=2))
    else:
        emit_shell_exports(exports)


if __name__ == "__main__":
    main()
