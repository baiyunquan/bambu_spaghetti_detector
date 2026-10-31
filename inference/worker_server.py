#!/usr/bin/env python3
"""
GPU Inference Worker Server (Unix Domain Socket)
=================================================
Runs in JetPack Python 3.6 environment with full CUDA/TensorRT acceleration.
Listens on /tmp/spaghetti_trt.sock for raw JPEG frames, executes TensorRT
inference with GPU NMS, and returns JSON detection results.
"""

import os
import sys
import time
import json
import socket
import struct
import argparse
from pathlib import Path

# Mandatory Jetson Nano OpenBLAS setting
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"

# Add current folder to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_preprocess import GPUPreprocessor
from postprocess import GPUPostProcessor
from trt_detector import TensorRTDetector


SOCKET_PATH = "/tmp/spaghetti_trt.sock"


def recv_exact(sock, num_bytes: int) -> bytes:
    """Reads exactly num_bytes from a socket."""
    data = bytearray()
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def detect_frame(detector, preprocessor, postprocessor, model_type, jpeg_bytes):
    """Use the same complete path for startup warmup and client requests."""
    tensor, meta = preprocessor.decode_and_preprocess(jpeg_bytes)
    outputs = detector.infer(tensor)
    if model_type == "obico":
        return postprocessor.process_obico(outputs, meta, detector.class_names)
    return postprocessor.process_yolo(outputs, meta, detector.class_names)


def warmup_pipeline(detector, preprocessor, postprocessor, model_type):
    """Finish lazy image/CUDA/NMS initialization before publishing readiness."""
    import cv2
    import numpy as np
    import torch

    # Cover nominal 720p and the actual A1's 1536x1080 stream. The latter
    # triggers a different OpenCV resize/thread-pool path on this Jetson;
    # prewarming only 720p still leaves a multi-second first real request.
    for source_h, source_w in ((720, 1280), (1080, 1536)):
        image = np.random.RandomState(0).randint(0, 256, (source_h, source_w, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("Failed to encode warmup JPEG")
        jpeg_bytes = encoded.tobytes()
        del image, encoded
        for _ in range(3):
            detect_frame(detector, preprocessor, postprocessor, model_type, jpeg_bytes)

    # The model may return no candidates for the warmup image. Force the full
    # nonempty GPU NMS + coordinate conversion path without affecting decisions.
    tensor, meta = preprocessor.decode_and_preprocess(jpeg_bytes)
    device = tensor.device
    if model_type == "obico":
        boxes = torch.tensor([[[[0.1, 0.1, 0.5, 0.5]],
                               [[0.12, 0.12, 0.48, 0.48]]]], device=device)
        scores = torch.ones((1, 2, 1), device=device)
        postprocessor.process_obico([boxes, scores], meta, detector.class_names)
    else:
        outputs = torch.zeros((1, 4 + len(detector.class_names), 2), device=device)
        outputs[0, :4, :] = torch.tensor([[100., 102.], [100., 102.],
                                         [50., 50.], [50., 50.]], device=device)
        outputs[0, 4, :] = 1.0
        postprocessor.process_yolo([outputs], meta, detector.class_names)
    if tensor.is_cuda:
        torch.cuda.synchronize(device)


def run_worker(engine_path: str, model_type: str = "obico", conf_thresh: float = 0.35, iou_thresh: float = 0.45):
    print("=" * 60)
    print(f" [INFO] Starting Spaghetti TensorRT Worker Daemon")
    print(f"  Engine Path  : {engine_path}")
    print(f"  Model Type   : {model_type}")
    print(f"  Confidence   : {conf_thresh}")
    print(f"  Socket Path  : {SOCKET_PATH}")
    print("=" * 60)

    # Clean existing socket file
    if os.path.exists(SOCKET_PATH):
        try:
            os.remove(SOCKET_PATH)
        except OSError:
            pass

    # Initialize Engine & Processors
    detector = TensorRTDetector(engine_path, model_type=model_type)
    input_h, input_w = detector.get_input_resolution()
    print(f"[INFO] Engine loaded. Input resolution: {input_w}x{input_h}")

    preprocessor = GPUPreprocessor(target_size=(input_h, input_w))
    postprocessor = GPUPostProcessor(conf_thresh=conf_thresh, iou_thresh=iou_thresh)

    print("[INFO] Warming up JPEG preprocessing, TensorRT and GPU NMS...", flush=True)
    warmup_pipeline(detector, preprocessor, postprocessor, model_type)

    # Bind UDS socket
    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(SOCKET_PATH)
    server_sock.listen(5)
    os.chmod(SOCKET_PATH, 0o777)
    print("[INFO] Full pipeline warmup complete. Listening for connections...", flush=True)

    try:
        while True:
            conn, _ = server_sock.accept()
            try:
                while True:
                    # 1. Read 4-byte header (payload length in big-endian)
                    header = recv_exact(conn, 4)
                    if not header:
                        break
                    payload_len = struct.unpack(">I", header)[0]

                    # 2. Read JPEG bytes
                    jpeg_bytes = recv_exact(conn, payload_len)
                    if not jpeg_bytes:
                        break

                    t0 = time.time()
                    try:
                        detections = detect_frame(
                            detector, preprocessor, postprocessor, model_type, jpeg_bytes)

                        inference_time = (time.time() - t0) * 1000.0
                        response = {
                            "status": "ok",
                            "detections": detections,
                            "inference_time_ms": round(inference_time, 2)
                        }
                    except Exception as e:
                        response = {
                            "status": "error",
                            "error": str(e),
                            "detections": []
                        }

                    # 6. Send JSON response
                    resp_data = json.dumps(response).encode("utf-8")
                    resp_header = struct.pack(">I", len(resp_data))
                    conn.sendall(resp_header + resp_data)

            except Exception as e:
                # Client disconnected or read error
                pass
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\n[INFO] Worker shutting down...")
    finally:
        server_sock.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        print("[INFO] Worker stopped.")


def main():
    parser = argparse.ArgumentParser(description="TensorRT Spaghetti Detection UDS Worker")
    parser.add_argument("--engine", type=str, required=True, help="Path to .engine plan file")
    parser.add_argument("--type", type=str, default="obico", choices=["obico", "yolov8", "yolov5"], help="Model architecture")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--test-dry", action="store_true", help="Perform engine load test and exit")
    args = parser.parse_args()

    if args.test_dry:
        print(f"[TEST] Loading engine {args.engine}...")
        detector = TensorRTDetector(args.engine, model_type=args.type)
        print(f"[TEST] Engine loaded successfully. Input: {detector.input_shape}")
        sys.exit(0)

    run_worker(
        engine_path=args.engine,
        model_type=args.type,
        conf_thresh=args.conf,
        iou_thresh=args.iou
    )


if __name__ == "__main__":
    main()
