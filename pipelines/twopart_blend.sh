#!/bin/bash

set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

config_path=""
object_name=""
motion_part=""

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
    --motion_part | -mp)
      motion_part="$2"
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
  echo "Usage: $0 --config <config.yaml> [--object_name <name>] --motion_part <0|1>"
  exit 1
fi

helper_cmd=(python utils/pipeline_config.py --config "$config_path" --stage twopart_blend)
if [[ -n "$object_name" ]]; then
  helper_cmd+=(--object_name "$object_name")
fi
eval "$("${helper_cmd[@]}")"

if [[ -z "$motion_part" ]]; then
  motion_part=0
fi

trans_penalty=$(echo "$PENALTY_SCALE * $TRANS_PENALTY_FACTOR" | bc -l)
rot_penalty=$(echo "$PENALTY_SCALE * $ROT_PENALTY_FACTOR" | bc -l)

ns-train twopart-blend \
    --data "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR/twopart_blend" \
    --experiment-name "twopart_blend" \
    --enable-frame-filtering "$ENABLE_FRAME_FILTERING" \
    --specialize-output-dir True \
    --vis "$VIS" \
    --pipeline.model.obj-pose-opt-enabled "$OBJ_POSE_OPT_ENABLED" \
    --pipeline.model.trainable-weights "$TRAINABLE_WEIGHTS" \
    --pipeline.model.camera-optimizer.mode "$CAMERA_OPTIMIZER_MODE" \
    --pipeline.model.enable-gs-refinement "$ENABLE_GS_REFINEMENT" \
    --pipeline.model.fg-mask-lambda "$FG_MASK_LAMBDA" \
    --pipeline.model.sensor-depth-lambda "$SENSOR_DEPTH_LAMBDA" \
    --pipeline.model.sh-degree "$SH_DEGREE" \
    --max-num-iterations "$MAX_NUM_ITERATIONS" \
    --pipeline.model.camera-optimizer.trans-l2-penalty "$trans_penalty" \
    --pipeline.model.camera-optimizer.rot-l2-penalty "$rot_penalty" \
    --pipeline.model.obj-trans-l2-penalty "$OBJ_TRANS_L2_PENALTY" \
    --pipeline.model.obj-rot-l2-penalty "$OBJ_ROT_L2_PENALTY" \
    twopart-blend-data \
    --data "$DATASET_DIR" \
    --part0_dir "$OUTPUT_DIR/part_0" \
    --part1_dir "$OUTPUT_DIR/part_1" \
    --output-dir "$OUTPUT_DIR/twopart_blend" \
    --motion_part "$motion_part"

ns-export gaussian-splat \
    --load-config "$OUTPUT_DIR/twopart_blend/config.yml" \
    --output-dir "$OUTPUT_DIR/twopart_blend" \
    --world-frame True

ns-export cameras \
    --load-config "$OUTPUT_DIR/twopart_blend/config.yml" \
    --output-dir "$OUTPUT_DIR/twopart_blend" \
    --optimized True \
    --world-frame True
