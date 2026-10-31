#!/usr/bin/env bash
# ==============================================================================
# TensorRT Engine Builder for Jetson Nano (FP16 Maxwell sm_53)
# ==============================================================================
# Usage:
#   ./build_engine.sh models/obico_spaghetti.onnx models/obico_spaghetti.engine
#   ./build_engine.sh models/yolov8_failure_640.onnx models/yolov8_failure_640.engine
# ==============================================================================

set -e

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [ ! -f "$TRTEXEC" ]; then
    echo "[ERROR] trtexec not found at $TRTEXEC"
    exit 1
fi

ONNX_FILE="${1:-models/obico_spaghetti.onnx}"
ENGINE_FILE="${2:-${ONNX_FILE%.*}.engine}"

if [ ! -f "$ONNX_FILE" ]; then
    echo "[ERROR] ONNX file not found: $ONNX_FILE"
    echo "Usage: $0 <input.onnx> [output.engine]"
    exit 1
fi

mkdir -p "$(dirname "$ENGINE_FILE")"

echo "============================================================"
echo " Building TensorRT FP16 Engine on Jetson Nano"
echo "  Input ONNX   : $ONNX_FILE"
echo "  Output Engine: $ENGINE_FILE"
echo "  Target GPU   : Tegra X1 / Maxwell (sm_53) FP16"
echo "============================================================"
echo "[INFO] Running trtexec... (This may take 1-3 minutes)"

$TRTEXEC \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    --fp16 \
    --workspace=512

echo "============================================================"
echo "[SUCCESS] TensorRT Engine successfully built: $ENGINE_FILE"
ls -lh "$ENGINE_FILE"
echo "============================================================"
