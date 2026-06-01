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

helper_cmd=(python utils/pipeline_config.py --config "$config_path" --stage splatfacto_art)
if [[ -n "$object_name" ]]; then
  helper_cmd+=(--object_name "$object_name")
fi
eval "$("${helper_cmd[@]}")"

if [[ -z "$motion_part" ]]; then
  motion_part=0
fi

trans_penalty=$(echo "$PENALTY_SCALE * $TRANS_PENALTY_FACTOR" | bc -l)
rot_penalty=$(echo "$PENALTY_SCALE * $ROT_PENALTY_FACTOR" | bc -l)

python preprocessing/estimate_joint.py --config "$config_path" --object_name "$OBJECT_NAME"

ns-train splatfacto-art \
    --data "$DATASET_DIR" \
    --pipeline.model.enable-gs-refinement "$ENABLE_REFINE" \
    --pipeline.model.optimize-joint-axis "$OPTIMIZE_JOINT_AXIS" \
    --pipeline.model.optimize-joint-origin "$OPTIMIZE_JOINT_ORIGIN" \
    --pipeline.model.optimize-joint-params "$OPTIMIZE_JOINT_PARAMS" \
    --pipeline.model.origin-adjustment-weight "$ORIGIN_ADJUSTMENT_WEIGHT" \
    --pipeline.model.axis-adjustment-weight "$AXIS_ADJUSTMENT_WEIGHT" \
    --pipeline.model.fg-mask-lambda "$FG_MASK_LAMBDA" \
    --pipeline.model.sh-degree "$SH_DEGREE" \
    --pipeline.model.sensor-depth-lambda "$SENSOR_DEPTH_LAMBDA" \
    --pipeline.model.camera-optimizer.trans-l2-penalty "$trans_penalty" \
    --pipeline.model.camera-optimizer.rot-l2-penalty "$rot_penalty" \
    --vis "$VIS" \
    --specialize-output-dir True \
    --max-num-iterations "$MAX_NUM_ITERATIONS" \
    --output-dir "$OUTPUT_DIR/splatfacto-art" \
    splatfacto-art-data \
    --twopart-blend-dir "$OUTPUT_DIR/twopart_blend" \
    --motion_part "$motion_part" \
    --data-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR/splatfacto-art"

ns-export gaussian-splat \
    --load-config "$OUTPUT_DIR/splatfacto-art/config.yml" \
    --output-dir "$OUTPUT_DIR/splatfacto-art" \
    --world-frame True

ns-export cameras \
    --load-config "$OUTPUT_DIR/splatfacto-art/config.yml" \
    --output-dir "$OUTPUT_DIR/splatfacto-art" \
    --optimized True \
    --world-frame True

ns-export articulation \
    --load-config "$OUTPUT_DIR/splatfacto-art/config.yml" \
    --output-dir "$OUTPUT_DIR/splatfacto-art" \
    --world-frame True
