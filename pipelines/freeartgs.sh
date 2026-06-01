#!/bin/bash

set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

config_path=""
object_name=""
preprocess_only=false
tracking_only=false
reconstruction_only=false

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config)
      config_path="$2"
      shift
      shift
      ;;
    --object_name | -on)
      object_name="$2"
      shift
      shift
      ;;
    --preprocess_only | -po)
      preprocess_only=true
      shift
      ;;
    --tracking_only | -to)
      tracking_only=true
      shift
      ;;
    --reconstruction_only | -ro)
      reconstruction_only=true
      shift
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

if [[ -z "$config_path" ]]; then
  echo "Usage: $0 --config <config.yaml> [--object_name <name>] [--preprocess_only|--tracking_only|--reconstruction_only]"
  exit 1
fi

helper_cmd=(python utils/pipeline_config.py --config "$config_path")
if [[ -n "$object_name" ]]; then
  helper_cmd+=(--object_name "$object_name")
fi
eval "$("${helper_cmd[@]}")"

run_preprocess() {
  python preprocessing/split_eval_subset.py --config "$config_path" --object_name "$OBJECT_NAME"
  bash pipelines/preprocess.sh --config "$config_path" --object_name "$OBJECT_NAME"
}

run_tracking() {
  bash pipelines/tracking_dynamic_2part.sh --config "$config_path" --object_name "$OBJECT_NAME"
}

run_reconstruction() {
  bash pipelines/icp.sh --config "$config_path" --object_name "$OBJECT_NAME"
  bash pipelines/train_parts.sh --config "$config_path" --object_name "$OBJECT_NAME"
  python preprocessing/compare_cam_motion.py --config "$config_path" --object_name "$OBJECT_NAME" --output_dir "$OUTPUT_DIR"
  motion_part=$(cat "$OUTPUT_DIR/motion_part.txt")
  bash pipelines/twopart_blend.sh --config "$config_path" --object_name "$OBJECT_NAME" --motion_part "$motion_part"
  bash pipelines/splatfacto_art.sh --config "$config_path" --object_name "$OBJECT_NAME" --motion_part "$motion_part"
  bash pipelines/evaluate_freeartgs.sh -on "$OBJECT_NAME"
}

if [[ "$preprocess_only" = false && "$tracking_only" = false && "$reconstruction_only" = false ]]; then
  run_preprocess
  run_tracking
  run_reconstruction
elif [[ "$preprocess_only" = true ]]; then
  run_preprocess
elif [[ "$tracking_only" = true ]]; then
  run_tracking
elif [[ "$reconstruction_only" = true ]]; then
  run_reconstruction
fi
