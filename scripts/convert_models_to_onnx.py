#!/usr/bin/env python3
"""
Model Conversion Script: PyTorch (.pt) -> ONNX for TensorRT
============================================================
Exports failure detection models (Obico, YOLOv8, YOLOv5) to optimized,
fixed-shape ONNX format on the host machine with GPU acceleration.

Usage:
    .venv/bin/python scripts/convert_models_to_onnx.py --model all
    .venv/bin/python scripts/convert_models_to_onnx.py --model yolov8 --imgsz 640
    .venv/bin/python scripts/convert_models_to_onnx.py --model obico
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
MODELS_DIR = ROOT_DIR / "models"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Remote URLs
OBICO_ONNX_URL = "https://www.obico.io/static_pub/ml-models/model-weights-5a6b1be1fa.onnx"
YOLOV8_PT_URL = "https://media.githubusercontent.com/media/VaibhavMugulavalli/3DprintfailureYOLO/main/best.pt"
YOLOV5_PT_URL = "https://huggingface.co/Javiai/3dprintfails-yolo5vs/resolve/main/model_torch.pt"


def download_file(url: str, dest_path: Path, desc: str):
    """Download a file with progress reporting."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[INFO] {desc} already exists at {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB). Skipping download.")
        return

    print(f"[INFO] Downloading {desc} from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        total_size = int(response.info().get('Content-Length', 0))
        block_size = 1024 * 1024  # 1MB
        downloaded = 0
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded / total_size * 100
                print(f"\r  Progress: {downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB ({percent:.1f}%)", end="")
            else:
                print(f"\r  Downloaded: {downloaded / 1e6:.1f} MB", end="")
        print()
    print(f"[SUCCESS] Saved to {dest_path}")


def convert_obico():
    """Download and prepare the Obico Spaghetti Detection ONNX model."""
    print("\n" + "=" * 60)
    print(" [1/3] Preparing Obico Spaghetti Detection Model")
    print("=" * 60)
    dest_onnx = MODELS_DIR / "obico_spaghetti.onnx"
    download_file(OBICO_ONNX_URL, dest_onnx, "Obico ONNX model")
    
    # Verify ONNX model
    import onnx
    model = onnx.load(str(dest_onnx))
    onnx.checker.check_model(model)
    inputs = [f"{inp.name}: {[dim.dim_value for dim in inp.type.tensor_type.shape.dim]}" for inp in model.graph.input]
    outputs = [f"{out.name}: {[dim.dim_value for dim in out.type.tensor_type.shape.dim]}" for out in model.graph.output]
    print(f"[SUCCESS] Obico ONNX model verified:")
    print(f"  Inputs : {inputs}")
    print(f"  Outputs: {outputs}")
    return dest_onnx


def convert_yolov8(imgsz: int = 640):
    """Download and export YOLOv8 Failure Detection model to ONNX."""
    print("\n" + "=" * 60)
    print(f" [2/3] Converting YOLOv8 Failure Model (imgsz={imgsz})")
    print("=" * 60)
    pt_path = WEIGHTS_DIR / "yolov8_failure.pt"
    download_file(YOLOV8_PT_URL, pt_path, "YOLOv8 weights (best.pt)")

    from ultralytics import YOLO
    import shutil

    print(f"[INFO] Loading YOLOv8 model from {pt_path}...")
    model = YOLO(str(pt_path))
    print(f"[INFO] Model classes: {model.names}")

    print(f"[INFO] Exporting to ONNX format (opset 11, fixed shape {imgsz}x{imgsz}, simplified)...")
    exported_path = model.export(
        format="onnx",
        opset=11,
        imgsz=imgsz,
        simplify=True,
        dynamic=False
    )
    dest_onnx = MODELS_DIR / f"yolov8_failure_{imgsz}.onnx"
    shutil.move(exported_path, dest_onnx)
    print(f"[SUCCESS] Exported YOLOv8 ONNX model to: {dest_onnx}")
    return dest_onnx


def convert_yolov5(imgsz: int = 640):
    """Download and export YOLOv5 Failure Detection model to ONNX."""
    print("\n" + "=" * 60)
    print(f" [3/3] Converting YOLOv5 Failure Model (imgsz={imgsz})")
    print("=" * 60)
    pt_path = WEIGHTS_DIR / "yolov5_failure.pt"
    download_file(YOLOV5_PT_URL, pt_path, "YOLOv5 weights (model_torch.pt)")

    from ultralytics import YOLO
    import shutil

    print(f"[INFO] Loading YOLOv5 model via Ultralytics from {pt_path}...")
    model = YOLO(str(pt_path))
    print(f"[INFO] Model classes: {model.names}")

    print(f"[INFO] Exporting to ONNX format (opset 11, fixed shape {imgsz}x{imgsz}, simplified)...")
    exported_path = model.export(
        format="onnx",
        opset=11,
        imgsz=imgsz,
        simplify=True,
        dynamic=False
    )
    dest_onnx = MODELS_DIR / f"yolov5_failure_{imgsz}.onnx"
    shutil.move(exported_path, dest_onnx)
    print(f"[SUCCESS] Exported YOLOv5 ONNX model to: {dest_onnx}")
    return dest_onnx


def main():
    parser = argparse.ArgumentParser(description="Export 3D Print Failure Detection Models to ONNX")
    parser.add_argument(
        "--model",
        choices=["all", "obico", "yolov8", "yolov5"],
        default="all",
        help="Model to export (default: all)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default: 640)"
    )
    args = parser.parse_args()

    results = []
    if args.model in ("all", "obico"):
        results.append(convert_obico())
    if args.model in ("all", "yolov8"):
        results.append(convert_yolov8(args.imgsz))
    if args.model in ("all", "yolov5"):
        results.append(convert_yolov5(args.imgsz))

    print("\n" + "=" * 60)
    print(" [DONE] All requested ONNX models successfully prepared:")
    for path in results:
        size_mb = path.stat().st_size / 1e6
        print(f"  - {path.name} ({size_mb:.2f} MB)")
    print("\nNext step on Jetson Nano:")
    print("  Copy these .onnx files to Jetson and run:")
    print("  /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16")
    print("=" * 60)


if __name__ == "__main__":
    main()
