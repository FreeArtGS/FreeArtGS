#!/bin/bash

set -euo pipefail

object_name=""
joint_id=""
eval_config_path="configs/eval_joint_ids.yaml"

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --eval_config)
      eval_config_path="$2"
      shift
      shift
      ;;
    --object_name | -on)
      object_name="$2"
      shift
      shift
      ;;
    --joint_id)
      joint_id="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

if [[ -z "$object_name" ]]; then
  echo "Usage: $0 --object_name <name> [--joint_id <id>] [--eval_config configs/eval_joint_ids.yaml]"
  exit 1
fi

if [[ -z "$joint_id" ]]; then
  eval "$(python utils/eval_joint_config.py --config "$eval_config_path" --object_name "$object_name")"
  obj_prefix="$OBJECT_ID"
  dataset_dir="$DATASET_DIR"
  joint_id="$JOINT_ID"
else
  obj_prefix=${object_name%%_*}
  dataset_dir="datasets/$obj_prefix"
fi

motion_part=$(cat outputs/$object_name/motion_part.txt)
python evaluate/export_evaluation_data.py --exp_id $object_name --motion_part $motion_part

results_dir="outputs/$object_name/evaluation_data"

# Convert object_poses.json -> camera_pose.npy for compatibility with loaders/tools
python evaluate/convert_objposes_to_camposes.py \
  --dataset_dir "$dataset_dir" \
  --output_path "$dataset_dir/camera_pose.npy"

# Evaluate
python evaluate/evaluate_freeartgs.py \
  --dataset_dir "$dataset_dir" \
  --results_dir "$results_dir" \
  --obj_id "$obj_prefix" \
  --joint_id "$joint_id" \
  --psnr \
  --save_icp_overlay \
  --motion_part $motion_part \
  --save_vis --no_vis --align_mode average --save_chamfer_vis
