#!/bin/bash

set -euo pipefail

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

object_name=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
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
if [ -z "$object_name" ]; then
  object_name="drawer"
fi


ns-train splatfacto \
  --vis tensorboard \
  --output-dir outputs/$object_name \
  --specialize-output-dir True \
  --pipeline.model.camera-optimizer.mode SO3xR3  \
  nerfstudio-data \
  --data datasets/$object_name

ns-export gaussian-splat \
  --load-config outputs/$object_name/config.yml \
  --output-dir outputs/$object_name/
