#!/bin/bash

set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

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

helper_cmd=(python utils/pipeline_config.py --config "$config_path" --stage train_parts)
if [[ -n "$object_name" ]]; then
  helper_cmd+=(--object_name "$object_name")
fi
eval "$("${helper_cmd[@]}")"

trans_penalty=$(echo "$PENALTY_SCALE * $TRANS_PENALTY_FACTOR" | bc -l)
rot_penalty=$(echo "$PENALTY_SCALE * $ROT_PENALTY_FACTOR" | bc -l)

python reconstruction/generate_ply.py --config "$config_path" --dataset_dir "$DATASET_DIR"

ns-train splatfacto-depth \
    --data "$DATASET_DIR" \
    --enable-frame-filtering "$ENABLE_FRAME_FILTERING" \
    --output-dir "$OUTPUT_DIR/part_0" \
    --experiment-name "part_0" \
    --pipeline.model.fg-mask-lambda "$FG_MASK_LAMBDA" \
    --pipeline.model.sensor-depth-lambda "$SENSOR_DEPTH_LAMBDA" \
    --pipeline.model.camera-optimizer.trans-l2-penalty "$trans_penalty" \
    --pipeline.model.camera-optimizer.rot-l2-penalty "$rot_penalty" \
    --vis "$VIS" \
    --specialize-output-dir True \
    --pipeline.model.sh-degree "$SH_DEGREE" \
    --max-num-iterations "$MAX_NUM_ITERATIONS" \
    part-data \
    --part_id 0
    
ns-export gaussian-splat \
    --load-config "$OUTPUT_DIR/part_0/config.yml" \
    --output-dir "$OUTPUT_DIR/part_0" \
    --world-frame True
    
ns-export cameras \
    --load-config "$OUTPUT_DIR/part_0/config.yml" \
    --output-dir "$OUTPUT_DIR/part_0" \
    --optimized True \
    --world-frame True

ns-train splatfacto-depth \
    --data "$DATASET_DIR" \
    --enable-frame-filtering "$ENABLE_FRAME_FILTERING" \
    --output-dir "$OUTPUT_DIR/part_1" \
    --experiment-name "part_1" \
    --vis "$VIS" \
    --specialize-output-dir True \
    --pipeline.model.fg-mask-lambda "$FG_MASK_LAMBDA" \
    --pipeline.model.sensor-depth-lambda "$SENSOR_DEPTH_LAMBDA" \
    --pipeline.model.camera-optimizer.trans-l2-penalty "$trans_penalty" \
    --pipeline.model.camera-optimizer.rot-l2-penalty "$rot_penalty" \
    --pipeline.model.sh-degree "$SH_DEGREE" \
    --max-num-iterations "$MAX_NUM_ITERATIONS" \
    part-data \
    --part_id 1

ns-export gaussian-splat \
    --load-config "$OUTPUT_DIR/part_1/config.yml" \
    --output-dir "$OUTPUT_DIR/part_1" \
    --world-frame True

ns-export cameras \
    --load-config "$OUTPUT_DIR/part_1/config.yml" \
    --output-dir "$OUTPUT_DIR/part_1" \
    --optimized True \
    --world-frame True
