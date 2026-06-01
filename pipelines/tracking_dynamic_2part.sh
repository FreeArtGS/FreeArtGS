#!/bin/bash

set -euo pipefail

config_path=""
object_name=""

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
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

if [[ -z "$config_path" ]]; then
  echo "Usage: $0 --config <config.yaml> [--object_name <name>]"
  exit 1
fi

helper_cmd=(python utils/pipeline_config.py --config "$config_path" --stage tracking)
if [[ -n "$object_name" ]]; then
  helper_cmd+=(--object_name "$object_name")
fi
eval "$("${helper_cmd[@]}")"
DEBUG="${DEBUG:-False}"

python estimotion_2part/track_2part.py \
  --config "$config_path" \
  --input_dir "$DATASET_DIR" \
  --output_dir "$TRACKING_DIR"

python estimotion_2part/utils/segmap_video.py \
  --config "$config_path" \
  --object_name "$OBJECT_NAME" \
  --tracking_dir "$TRACKING_DIR"

